import argparse

import torch
import os
import numpy as np
from models import Deeplab
import torch.nn.functional as F
from data import Freiburg, FreiburgTest
from utils import get_composed_augmentations, get_logger
from torch.utils.data import DataLoader
import time
from utils import transforms as T
from PIL import ImageFile
from utils import FocalLoss
import random

class SelfTrain:
    def __init__(self, args, seg_net, ema_net, optimizer_seg, device, logger):
        self.args = args
        self.seg_net = seg_net
        self.ema_net = ema_net
        self.num_classes = self.args.num_classes
        self.optimizer_seg = optimizer_seg
        self.objective_vectors = torch.zeros([self.num_classes, 256]).to(device)
        self.objective_vectors_num = torch.zeros([self.num_classes]).to(device)
        self.logger = logger
        self.seg_loss = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_index, reduction='mean')
        # self.seg_loss = FocalLoss(gamma=2)
        self.scale_rate = 4
        self.device = device
        self.class_names = [
            "road,parking",
            "ground,sidewalk",
            "building,",
            'curb',
            'fence',
            'pole,traffic light,traffic sign',
            'vegetation',
            'terrain',
            'sky',
            'person,rider',
            'car,truck,bus,train',
            'motorcycle,bicycle',
            'ignore']
        self.counter = 0
        self.pole_correct = 0
        self.pole_acc_list = []

    def step(self, source_image, source_label, target_image, target_image_full, target_lp_soft, target_lp_hard,
             target_weak_params, target_label):
        source_out = self.seg_net(source_image)
        scaled_size = (int(source_label.size()[1]/self.scale_rate), int(source_label.size()[2]/self.scale_rate))
        source_out = torch.nn.Upsample(
            size=(source_label.size()[1], source_label.size()[2]),
            mode='bilinear', align_corners=True)(source_out['out'])

        loss_source = self.seg_loss(source_out, source_label)
        loss_source.backward()
        # pole_index = (target_label == 5).nonzero(as_tuple=False)
        # print(pole_index)
        if self.args.proto_rectify:
            threshold_arg = torch.nn.Upsample(size=scaled_size, mode='bilinear', align_corners=True)(target_lp_soft)
        else:
            threshold_arg = torch.nn.Upsample(size=scaled_size, mode='bilinear', align_corners=True)(target_lp_hard.unsqueeze(1).float()).long()

        if self.args.ema:
            with torch.no_grad():
                ema_out = self.ema_net(target_image_full)
            ema_out['feat'] = F.interpolate(ema_out['feat'], size=(int(target_image_full.shape[2]/4), int(target_image_full.shape[3]/4)), mode='bilinear', align_corners=True)
            ema_out['out'] = F.interpolate(ema_out['out'], size=(
            int(target_image_full.shape[2] / 4), int(target_image_full.shape[3] / 4)), mode='bilinear',
                                            align_corners=True)

        target_out = self.seg_net(target_image)
        target_out['out'] = F.interpolate(target_out['out'], size=threshold_arg.shape[2:], mode='bilinear', align_corners=True)
        target_out['feat'] = F.interpolate(target_out['feat'], size=threshold_arg.shape[2:], mode='bilinear',
                                           align_corners=True)

        loss = torch.Tensor([0]).to(self.device)
        batch, _, w, h = threshold_arg.shape
        if self.args.proto_rectify:
            weights = self.get_prototype_weight(ema_out['feat'], target_weak_params=target_weak_params)
            # print(weights.size()[1])
            # print(weights[0, :, 0, 0].size())
            # if self.counter % 20 == 0:
            # target_label = F.interpolate(target_label.float(), size=target_out['out'].shape[2:], mode='nearest')
            # pole_index = (target_label == 5).nonzero(as_tuple=False)
            # a = random.randint(0, len(pole_index)-1)
            # pole_index = pole_index[a]
            # print('real class is ' + self.class_names[int(target_label[5].cpu().numpy())])
            # print('real class is ' + self.class_names[5])
            # pole_correct = 0
            # for m in range(len(pole_index)):
            #     highest_weight = np.argmax(weights[pole_index[m][0], :, pole_index[m][2], pole_index[m][3]].cpu().numpy())
            #     if highest_weight == 5:
            #         pole_correct += 1
            # pole_acc = pole_correct/len(pole_index)
            # print('acc: ' + str(pole_acc))
            # print('highest weight: ' + str(self.class_names[highest_weight]))
            # self.pole_acc_list.append(pole_acc)
            # for i in range(weights.size()[1]):
            #     print(self.class_names[i], ' weight: {}'.format(weights[pole_index[0], i, pole_index[2], pole_index[3]].cpu().numpy()))
            hard_lp = F.interpolate(target_lp_hard.float(), size=target_out['out'].shape[2:], mode='nearest').long()
            # hard_lp = torch.nn.Upsample(size=scaled_size, mode='nearest', align_corners=True)(
            #     target_lp_hard.unsqueeze(1).float()).long()
            rectified = weights * threshold_arg
            threshold_arg = rectified.max(1, keepdim=True)[1]   # index of max value (among all channels), namely the prediction.
            # next three lines maybe not useful for my case
            rectified = rectified / rectified.sum(1, keepdim=True)
            argmax = rectified.max(1, keepdim=True)[0]   # the max value of the normalized prediction
            threshold_arg[argmax < self.args.train_thred] = self.args.ignore_index
            threshold_arg = torch.where(hard_lp != self.args.ignore_index, hard_lp, threshold_arg)

        loss_target_pseudo = self.seg_loss(target_out['out'], threshold_arg.reshape([batch, w, h]))

        if self.args.rce:
            rce = self.rce(target_out['out'], threshold_arg.reshape([batch, w, h]).clone())     # why reshape
            loss_target_pseudo = self.args.rce_alpha * loss_target_pseudo + self.args.rce_beta * rce

        loss_target_pseudo.backward()
        self.optimizer_seg.step()

        if self.args.moving_prototype: # update prototype
            ema_vectors, ema_ids = self.calculate_mean_vector(ema_out['feat'].detach(), ema_out['out'].detach())
            for t in range(len(ema_ids)):
                self.update_objective_SingleVector(ema_ids[t], ema_vectors[t].detach(), start_mean=False)

        if self.args.ema: #update ema model
            for param_q, param_k in zip(self.seg_net.parameters(), self.ema_net.parameters()):
                param_k.data = param_k.data.clone() * 0.999 + param_q.data.clone() * (1. - 0.999)
            for buffer_q, buffer_k in zip(self.seg_net.buffers(), self.ema_net.buffers()):
                buffer_k.data = buffer_q.data.clone()
        self.counter += 1
        return loss_target_pseudo, loss_source

    def get_prototype_weight(self, feat, target_weak_params):
        feat = self.full2weak(feat, target_weak_params)
        feat_proto_distance = self.feat_prototype_distance(feat)
        feat_nearest_proto_distance, feat_nearest_proto = feat_proto_distance.min(dim=1, keepdim=True)

        feat_proto_distance = feat_proto_distance - feat_nearest_proto_distance
        weight = F.softmax(-feat_proto_distance * self.args.proto_temperature, dim=1)
        return weight

    def full2weak(self, feat, target_weak_params):
        tmp = []
        for i in range(feat.shape[0]):
            h, w = target_weak_params['RandomSized'][0][i], target_weak_params['RandomSized'][1][i]
            feat_ = F.interpolate(feat[i:i+1], size=[int(h/4), int(w/4)], mode='bilinear', align_corners=True)
            y1, y2, x1, x2 = target_weak_params['RandomCrop'][0][i], target_weak_params['RandomCrop'][1][i], target_weak_params['RandomCrop'][2][i], target_weak_params['RandomCrop'][3][i]
            y1, th, x1, tw = int(y1/4), int((y2-y1)/4), int(x1/4), int((x2-x1)/4)
            feat_ = feat_[:, :, y1:y1+th, x1:x1+tw]
            if target_weak_params['RandomHorizontallyFlip'][i]:
                inv_idx = torch.arange(feat_.size(3)-1,-1,-1).long().to(feat_.device)
                feat_ = feat_.index_select(3,inv_idx)
            tmp.append(feat_)
        feat = torch.cat(tmp, 0)
        return feat

    def feat_prototype_distance(self, feat):
        N, C, H, W = feat.shape
        feat_proto_distance = -torch.ones((N, self.num_classes, H, W)).to(feat.device)
        for i in range(self.num_classes):
            feat_proto_distance[:, i, :, :] = torch.norm(self.objective_vectors[i].reshape(-1, 1, 1).expand(-1, H, W) - feat, 2, dim=1,)
        return feat_proto_distance

    def rce(self, prediction, label):
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.clamp(prediction, min=1e-7, max=1.0)
        mask = (label != self.args.ignore_index).float()
        label_one_hot = torch.nn.functional.one_hot(label, self.num_classes+1).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot.permute(0, 3, 1, 2)[:, :-1, :, :], min=1e-4, max=1.0)
        rce = -(torch.sum(prediction * torch.log(label_one_hot), dim=1) * mask).sum() / (mask.sum() + 1e-6)
        return rce

    def calculate_mean_vector(self, feat_cls, outputs, labels=None, thresh=None):
        outputs_softmax = F.softmax(outputs, dim=1)
        if thresh is None:
            thresh = -1
        conf = outputs_softmax.max(dim=1, keepdim=True)[0]
        mask = conf.ge(thresh)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = self.process_label(outputs_argmax.float())
        if labels is None:
            outputs_pred = outputs_argmax
        else:
            labels_expanded = self.process_label(labels)
            outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred * mask, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.num_classes):
                if scale_factor[n][t].item() == 0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t] * mask[n]
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def process_label(self, label):
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, self.num_classes + 1, w, h).to(self.device)
        id = torch.where(label < self.num_classes, label, torch.Tensor([self.args.ignore_index]).to(self.device))
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1

    def update_objective_SingleVector(self, id, vector, name='moving_average', start_mean=True):
        if vector.sum().item() == 0:
            return
        if start_mean and self.objective_vectors_num[id].item() < 100:
            name = 'mean'
        if name == 'moving_average':
            self.objective_vectors[id] = self.objective_vectors[id] * (1 - self.args.proto_momentum) + self.args.proto_momentum * vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
        elif name == 'mean':
            self.objective_vectors[id] = self.objective_vectors[id] * self.objective_vectors_num[id] + vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors[id] = self.objective_vectors[id] / self.objective_vectors_num[id]
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
            pass
        else:
            raise NotImplementedError('no such updating way of objective vectors {}'.format(name))


