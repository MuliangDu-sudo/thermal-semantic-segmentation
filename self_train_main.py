import argparse

import torch
import os
import numpy as np
from models import Deeplab
import torch.nn.functional as F
from data import Freiburg, FreiburgTest
from segmentation_evaluate import seg_validate
from utils import get_composed_augmentations, get_logger, AverageMeter, ProgressMeter
from torch.utils.data import DataLoader
import time
from utils import transforms as T
from PIL import ImageFile
from self_training import SelfTrain


def main(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    train_transform = T.Compose([
        T.RandomResizedCrop(size=(256, 512), ratio=(1.5, 8 / 3.), scale=(0.5, 1.)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    val_transform = T.Compose([
        T.Resize((512, 256)),
        T.ToTensor(),
        ])

    source_dataset = Freiburg(args=args, root='datasets/freiburg', split='train', domain='RGB', translation_name=args.translation_name,
                              segmentation_mode=True, transforms=train_transform)
    target_dataset = Freiburg(args=args, root='datasets/freiburg', split='train', domain='IR', segmentation_mode=True,
                              self_train=args.self_train, augmentations=get_composed_augmentations(args))
    target_val_dataset = FreiburgTest(args=args, root='datasets/freiburg', split='test', domain='IR', transforms=val_transform,
                                  with_label=True)

    train_source_loader = DataLoader(source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    train_target_loader = DataLoader(target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    target_val_dataloader = DataLoader(target_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
                                drop_last=False)

    seg_net = Deeplab(torch.nn.BatchNorm2d, num_classes=13, num_channels=1, freeze_bn=False, get_feat=True).to(device)

    restart_epoch = 0
    best_score = 0
    lowest_val_loss = 1000

    if args.load_model:
        load_checkpoint = torch.load(os.path.join(args.model_root_path, args.checkpoint_name))
        restart_epoch = load_checkpoint['epoch'] + 1
        print('loading trained model. start from epoch {}. Last validation loss is {}'.format(restart_epoch, lowest_val_loss))
        seg_net.load_state_dict(load_checkpoint['sem_net_state_dict'])
        #best_score = load_checkpoint['best_score']
        logger.info('successfully loaded model {}. Resume from epoch {}. Best score is {}'.format(args.checkpoint_name, restart_epoch, best_score))

    ema_net = Deeplab(torch.nn.BatchNorm2d, num_classes=13, num_channels=1, freeze_bn=False, get_feat=True).to(device)
    ema_net.load_state_dict(seg_net.state_dict().copy())

    optimizer_seg = torch.optim.Adam(seg_net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_seg, 'min', verbose=True)
    self_training = SelfTrain(args, seg_net, ema_net, optimizer_seg, device, logger)

    objective_vectors = torch.load(os.path.join(args.root,
                                                'prototypes_on_{}_from_{}'.format(opt.tgt_dataset, opt.model_name)))
    self_training.objective_vectors = torch.Tensor(objective_vectors).to(0)

    for epoch in range(restart_epoch, restart_epoch+args.epochs):

        pseudo_loss = AverageMeter('pseudo_loss', ':3.4f')
        s_loss = AverageMeter('s_loss', ':3.4f')
        iteration_length = len(train_target_loader)
        progress = ProgressMeter(iteration_length, [pseudo_loss, s_loss], prefix="Epoch: [{}]".format(epoch))
        i = 0

        for target_data, source_data in zip(train_target_loader, train_source_loader):

            target_image = target_data['image'].to(device)
            target_image_full = target_data['image_full'].to(device)
            target_weak_params = target_data['weak_params']

            target_label_hard = target_data['label_hard'].to(device) if 'label_hard' in target_data.keys() else None
            target_label_soft = target_data['label_soft'].to(device) if 'label_soft' in target_data.keys() else None

            source_image = source_data['image'].to(device)
            source_label = source_data['label'].long().to(device)

            start_ts = time.time()
            seg_net.train()
            ema_net.train()

            optimizer_seg.zero_grad()

            loss_target_pseudo, loss_source = self_training.step(source_image, source_label, target_image, target_image_full,
                                                                 target_label_soft, target_label_hard, target_weak_params)
            pseudo_loss.update(loss_target_pseudo.item(), target_image.size(0))
            s_loss.update(loss_source.item(), target_image.size(0))
            if i % 10 == 0:
                progress.display(i, logger)

            if i % 500 == 0 or i == len(train_target_loader)-1:
                mean_iu, val_loss, class_iou = seg_validate(args, seg_net, target_val_dataloader, self_training.seg_loss, device)
                fmt_str = 'target test dataset mean iou score: ' + str(mean_iu)
                logger.info(fmt_str)
                print(fmt_str)
                for k, v in class_iou.items():
                    fmt_str = 'target set class {}: {}'.format(k, v)
                    logger.info(fmt_str)
                    print(fmt_str)

            i += 1
            args.iter_counter += 1
        torch.save({
            'epoch': epoch,
            'sem_net_state_dict': seg_net.state_dict(),
            'val_loss': lowest_val_loss,
        }, os.path.join(args.model_root_path, args.new_checkpoint_name))


if __name__ == "__main__":
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument('--root', type=str, default='/data/data_bank/muliang_gp/Prototypical', help='pseudo label update thred')
    parser.add_argument('--resize', default=1200, help='random resize image')
    parser.add_argument('--rcrop', type=tuple, default=(512, 256), help='rondom crop size')
    parser.add_argument('--hflip', type=float, default=0.5, help='random flip probility')
    parser.add_argument('--proto_rectify', default=True)
    parser.add_argument('--load_model', type=bool, default=True, help='whether to load trained model')
    parser.add_argument('-checkpoint_name', default='256_freiburg_rgb2ir_segmentation.pth')
    parser.add_argument('-new_checkpoint_name', default='256_freiburg_rgb2ir_onlytarget_segmentation.pth')
    parser.add_argument('-batch_size', default=4)
    parser.add_argument('--use_saved_pseudo', type=bool, default=True, help='whether to use saved pseudo')
    parser.add_argument('--self_train', type=bool, default=True, help='whether to train with self-training')
    parser.add_argument('--path_soft', type=str, default='')
    parser.add_argument('-pseudo_type', default='soft')
    parser.add_argument('-translation_name', type=str, default='freiburg_rgb2ir_130epochs')
    parser.add_argument('--model_root_path', type=str, default='./checkpoints/semantic_segmentation')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', default=0.000001)
    parser.add_argument('--num_classes', default=13)
    parser.add_argument('--ignore_index', default=12)
    parser.add_argument('--ema', default=True)
    parser.add_argument('--proto_temperature', type=float, default=1.0)
    parser.add_argument("--train_thred", default=0, type=float)
    parser.add_argument("--rce", default=True, type=bool)
    parser.add_argument("--rce_alpha", default=0.1, type=float, help="loss weight for symmetry cross entropy loss")
    parser.add_argument("--rce_beta", default=1.0, type=float, help="loss weight for symmetry cross entropy loss")
    parser.add_argument('--moving_prototype', default=True)
    parser.add_argument("--proto_momentum", default=0.0001, type=float)
    parser.add_argument("--visualize_prediction", default='save_one')
    parser.add_argument('--iter_counter', default=0)
    parser.add_argument('--baseline', default=False)
    parser.add_argument('--generator_type', default=None)
    args_ = parser.parse_args()
    args_.path_soft = os.path.join(args_.root, 'pseudo_labels', args_.pseudo_type, args_.checkpoint_name.replace('.pth', ''))

    args_.logdir = os.path.join('logs', 'self-training', args_.new_checkpoint_name.replace('.pth', ''))
    if not os.path.exists(args_.logdir):
        os.makedirs(args_.logdir)
    logger_ = get_logger(args_.logdir)

    main(args_, logger_)