import torch
from utils import loss, ImagePool
from utils import transforms as T
from data import Cityscapes, TrainTDataset, Freiburg
from torch.utils.data import DataLoader
from models import discriminators, generators, semantic_segmentation_models, thermal_semantic_segmentation_models
from itertools import chain
from train import train
from options import train_parse
from torchvision import transforms as TT
import visdom
import os
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

visualizer = visdom.Visdom(env='thermal semantic segmentation')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ROOT_PATH = './checkpoints'


source_train_transform = T.Compose([
    T.RandomResizedCrop(size=(256, 512), ratio=(1.5, 8 / 3.), scale=(0.5, 1.)),  # it return an image of size 256x512
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

target_train_transform = TT.Compose([
    TT.RandomResizedCrop(size=(256, 512), ratio=(1.5, 8 / 3.), scale=(0.5, 1.)),
    TT.RandomHorizontalFlip(),
    TT.ToTensor(),
    #TT.Normalize((0.5,), (0.5,))
])


def main(args):

    # data loading
    source_dataset = Cityscapes('datasets/source_dataset', transforms=source_train_transform)
    target_dataset = Freiburg('datasets/freiburg', split='train', domain='IR', transforms=target_train_transform,
                              with_label=False)
    # target_dataset = TrainTDataset('datasets/target_dataset', transforms=target_train_transform)
    train_source_loader = DataLoader(source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    train_target_loader = DataLoader(target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    # networks
    net_g_s2t = generators.unet_256(ngf=64, input_nc=3, output_nc=1).to(device)
    net_g_t2s = generators.unet_256(ngf=64, input_nc=1, output_nc=3).to(device)
    net_d_s = discriminators.NLayerDiscriminator(input_nc=3).to(device)
    net_d_t = discriminators.NLayerDiscriminator(input_nc=1).to(device)
    net_seg_s = semantic_segmentation_models.deeplabv2_resnet101().to(device)
    net_seg_t = thermal_semantic_segmentation_models.deeplabv2_resnet101_thermal(pretrained_backbone=False).to(device)

    restart_epoch = 0
    if args.load_model:
        load_checkpoint = torch.load(os.path.join(MODEL_ROOT_PATH, args.checkpoint_name))
        restart_epoch = load_checkpoint['epoch']
        print('loading trained model. start from epoch {}.'.format(restart_epoch))
        net_g_s2t.load_state_dict(load_checkpoint['net_g_s2t_state_dict'])
        net_g_t2s.load_state_dict(load_checkpoint['net_g_t2s_state_dict'])
        net_d_s.load_state_dict(load_checkpoint['net_d_s_state_dict'])
        net_d_t.load_state_dict(load_checkpoint['net_d_t_state_dict'])
        net_seg_s.load_state_dict(load_checkpoint['net_seg_s_state_dict'])
        net_seg_t.load_state_dict(load_checkpoint['net_seg_t_state_dict'])
    # create image buffer to store previously generated images
    fake_s_pool = ImagePool(50)
    fake_t_pool = ImagePool(50)

    # define optimizers
    all_params_g = chain(net_g_s2t.parameters(), net_g_t2s.parameters())
    optimizer_g = torch.optim.Adam(all_params_g, lr=0.001)
    all_params_d = chain(net_d_s.parameters(), net_d_t.parameters())
    optimizer_d = torch.optim.Adam(all_params_d, lr=0.001)

    # define loss
    gan_loss_func = loss.LeastSquaresGenerativeAdversarialLoss()
    cycle_loss_func = torch.nn.L1Loss()
    identity_loss_func = torch.nn.L1Loss()
    sem_loss_func = loss.SemanticConsistency().to(device)
    loss_dict = {'g_s2t': [], 'g_t2s': [], 'd_s': [], 'd_t': [], 'cycle_s': [], 'cycle_t': []}
    epoch_counter_ratio = []
    print("--------START TRAINING--------")
    for epoch in range(restart_epoch, restart_epoch+10):
        print("--------EPOCH {}--------".format(epoch))
        train(args, train_source_loader, train_target_loader, net_g_s2t, net_g_t2s, net_d_s, net_d_t, net_seg_s, net_seg_t,
              gan_loss_func, cycle_loss_func, identity_loss_func, sem_loss_func, optimizer_g, optimizer_d, fake_s_pool,
              fake_t_pool, device, epoch, visualizer, loss_dict, epoch_counter_ratio)

        torch.save({
            'epoch': epoch,
            'net_g_s2t_state_dict': net_g_s2t.state_dict(),
            'net_g_t2s_state_dict': net_g_t2s.state_dict(),
            'net_d_s_state_dict': net_d_s.state_dict(),
            'net_d_t_state_dict': net_d_t.state_dict(),
            'net_seg_s_state_dict': net_seg_s.state_dict(),
            'net_seg_t_state_dict': net_seg_t.state_dict()
        }, os.path.join(MODEL_ROOT_PATH, args.checkpoint_name))


if __name__ == '__main__':
    args_ = train_parse().parse_args()
    main(args_)

