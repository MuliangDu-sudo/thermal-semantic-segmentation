import os
import logging
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import fliplr

from options import pseudo_generation_parse

from tqdm import tqdm
from data import *
from models import Deeplab
from utils import transforms as T, freiburg_prediction_visualize, freiburg_palette
from torch.utils.data import DataLoader
from PIL import ImageFile


def main(args):
    MODEL_ROOT_PATH = './checkpoints/semantic_segmentation'
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    net = Deeplab(torch.nn.BatchNorm2d, num_classes=13, num_channels=1, freeze_bn=False, get_feat=True).to(device)
    load_checkpoint = torch.load(os.path.join(MODEL_ROOT_PATH, args.checkpoint_name))
    net.load_state_dict(load_checkpoint['sem_net_state_dict'])

    train_transform = T.Compose([
        T.Resize((512, 256)),
        T.ToTensor(),
    ])

    if args.dataset == 'cityscapes_translation':
        dataset = CityscapesTranslation('datasets/dataset', data_folder='translation',
                                               transforms=train_transform)
    elif args.dataset == 'cityscapes':
        dataset = Cityscapes('datasets/dataset', transforms=train_transform)

    elif args.dataset == 'freiburg_ir':
        dataset = Freiburg('datasets/freiburg', split='train', domain='IR', transforms=train_transform,
                                  with_label=True)
    elif args.dataset == 'freiburg_rgb':
        dataset = Freiburg('datasets/freiburg', split='train', domain='RGB', transforms=train_transform,
                                  grayscale=args.grayscale, with_label=True)
    elif args.dataset == 'freiburg_translation':
        dataset = Freiburg('datasets/freiburg', split='train', domain='RGB', transforms=train_transform,
                                  with_label=True, segmentation_mode=True, translation_name=args.translation_name)
    else:
        raise ValueError('dataset does not exist.')

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                pin_memory=True,
                                drop_last=True)

    with torch.no_grad():
        generate_pl(net, dataloader, device, args)


def generate_pl(net, dataloader, device, args):
    net.eval()
    torch.cuda.empty_cache()

    pseudo_save_path = os.path.join(args.root, 'pseudo_labels', args.pseudo_type, args.checkpoint_name.replace('.pth', ''))
    if not os.path.exists(pseudo_save_path):
        os.makedirs(pseudo_save_path)

    for data_i in dataloader:
        images = data_i['img'].to(device)
        filename = data_i['img_path']

        out = net(images)

        if args.soft:
            threshold_arg = F.softmax(out['out'], dim=1)
            for k in range(images.shape[0]):
                name = os.path.basename(filename[k])
                np.save(os.path.join(pseudo_save_path, name.replace('.png', '.npy')), threshold_arg[k].cpu().numpy())
        else:
            if args.flip:
                flip_out = net(fliplr(images))
                flip_out['out'] = F.interpolate(F.softmax(flip_out['out'], dim=1), size=images.size()[2:], mode='bilinear', align_corners=True)
                out['out'] = F.interpolate(F.softmax(out['out'], dim=1), size=images.size()[2:], mode='bilinear', align_corners=True)
                out['out'] = (out['out'] + fliplr(flip_out['out'])) / 2
            confidence, pseudo = out['out'].max(1, keepdim=True)

            for k in range(images.shape[0]):
                name = os.path.basename(filename[k])

                pseudo_rgb = freiburg_prediction_visualize(pseudo[k], freiburg_palette())
                Image.fromarray(np.squeeze(pseudo[k].cpu().numpy().astype(np.uint8), axis=0)).save(os.path.join(pseudo_save_path, name))

                pseudo_rgb.save(os.path.join(pseudo_save_path, name[:-4] + '_color.png'))
                np.save(os.path.join(pseudo_save_path, name.replace('.png', '_conf.npy')), confidence[k, 0].cpu().numpy().astype(np.float16))


if __name__ == "__main__":
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument('--root', type=str, default='/data/data_bank/muliang_gp/Prototypical', help='pseudo label update thred')
    parser.add_argument('--soft', default=True, help='save soft pseudo label')
    parser.add_argument('--flip', default=False)
    parser.add_argument('-checkpoint_name', default='256_freiburg_rgb2ir_segmentation.pth')
    parser.add_argument('-batch_size', default=4)
    parser.add_argument('--dataset', default='freiburg_ir')
    parser.add_argument('-pseudo_type', default='soft')

    args_ = parser.parse_args()

    main(args_)