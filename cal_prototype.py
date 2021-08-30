import os
import logging
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from models import thermal_semantic_segmentation_models, semantic_segmentation_models, Deeplab
import visdom
from data import Cityscapes, TrainTDataset, Freiburg, Kitti
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from options import calc_proto_parse
from tqdm import tqdm
from PIL import ImageFile


def calc_prototype(args):
    ## create dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    MODEL_ROOT_PATH = './checkpoints/semantic_segmentation'
    visualizer = visdom.Visdom(env='thermal semantic segmentation')

    single_transform = T.Compose([
        T.Resize(size=(256, 512)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(args.normalize, args.normalize)
    ])

    double_transform = T.Compose([
        T.RandomResizedCrop(size=(256, 512), ratio=(1.5, 8 / 3.), scale=(0.5, 1.)),
        # it return an image of size 256x512
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(args.normalize, args.normalize)
    ])

    source_train_transform = double_transform
    target_train_transform = single_transform

    if args.dataset == 'flir':
        dataset = TrainTDataset('datasets/target_dataset', transforms=target_train_transform)
    elif args.dataset == 'freiburg_ir':
        dataset = Freiburg('datasets/freiburg', split='train', domain='IR', transforms=target_train_transform,
                                  with_label=False)
    else:
        raise ValueError('target dataset does not exist.')

    train_target_loader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    if args.net_mode == 'one_channel':
        net = Deeplab(torch.nn.BatchNorm2d, num_classes=13, num_channels=1, freeze_bn=False, get_feat=True).to(device)
    elif args.net_mode == 'three_channels':
        net = semantic_segmentation_models.deeplabv2_resnet101(num_classes=args.num_classes,
                                                                               pretrained_backbone=False).to(device)
    else:
        raise ValueError('net mode does not exist.')

    load_checkpoint = torch.load(os.path.join(MODEL_ROOT_PATH, args.checkpoint_name))
    net.load_state_dict(load_checkpoint['sem_net_state_dict'])
    class_features = Class_Features(device=device, numbers=args.num_classes)

    for epoch in range(args.epochs):
        for i, data_i in enumerate(train_target_loader):

            image = data_i['img'].to(device)

            net.eval()

            with torch.no_grad():
                out = net(image)
                vectors, ids = class_features.calculate_mean_vector(out['feat'], out['out'])  # out[1] is feature, out[0] is output.
                #vectors, ids = class_features.calculate_mean_vector_by_output(feat_cls, output, model)
                for t in range(len(ids)):
                    class_features.update_objective_SingleVector(ids[t], vectors[t].detach().cpu(), 'mean')
            if i % 10 == 0:
                print('epoch [{}], prototype calculation process: [{}/{}]'.format(epoch, i, len(train_target_loader)))

        save_path = os.path.join(args.root, 'prototypes', "prototypes_on_{}_from_{}".format(args.target_dataset, args.checkpoint_name))
        print('saving prototypes......')
        torch.save(class_features.objective_vectors, save_path)


class Class_Features:
    def __init__(self, device, numbers = 13):
        self.class_numbers = numbers
        self.class_features = [[] for i in range(self.class_numbers)]
        self.num = np.zeros(numbers)
        self.device = device
        self.objective_vectors = torch.zeros([self.class_numbers, 256])     # 256 is the number of features
        self.objective_vectors_num = torch.zeros([self.class_numbers])
        self.proto_momentum = 0.9999

    def calculate_mean_vector_by_output(self, feat_cls, outputs, model):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = model.process_label(outputs_argmax.float())
        outputs_pred = outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def calculate_mean_vector(self, feat_cls, outputs, labels_val=None):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)

        outputs_argmax = self.process_label(outputs_argmax.float())
        if labels_val is None:
            outputs_pred = outputs_argmax
        else:
            labels_expanded = self.process_label(labels_val)
            outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def process_label(self, label):  # turn the label obtained by softmax into one-hot form, so that it can be used as indicator function.
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, self.class_numbers + 1, w, h).to(self.device)
        id = torch.where(label < self.class_numbers, label, torch.Tensor([self.class_numbers]).to(self.device))
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1

    def update_objective_SingleVector(self, id, vector, name='moving_average', start_mean=True):
        if vector.sum().item() == 0:
            return
        if start_mean and self.objective_vectors_num[id].item() < 100:
            name = 'mean'
        if name == 'moving_average':
            self.objective_vectors[id] = self.objective_vectors[id] * (1 - self.proto_momentum) + self.proto_momentum * vector.squeeze()
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


def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    file_path = os.path.join(logdir, 'run.log')
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


if __name__ == "__main__":
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    calc_proto_parse().add_argument('-root', type=str, default='')
    args_ = calc_proto_parse().parse_args()
    args_.checkpoint_name = '256_freiburg_rgb2ir_segmentation.pth'

    calc_prototype(args_)
