import torch
from utils import loss, ImagePool, set_requires_grad
from utils import transforms as TT
from utils import triple_transforms as TTT
from data import Cityscapes, TrainTDataset, Freiburg
from torch.utils.data import DataLoader
from models import discriminators, generators, semantic_segmentation_models, thermal_semantic_segmentation_models, Canny
from itertools import chain
from train_gan import train
from options import train_parse
from torchvision import transforms as T
import visdom
import os
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

visualizer = visdom.Visdom(env='thermal semantic segmentation')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ROOT_PATH = './checkpoints'


def main(args):
    single_transform = T.Compose([
        T.RandomResizedCrop(size=(256, 512), ratio=(1.5, 8 / 3.), scale=(0.5, 1.)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    double_transform = TT.Compose([
        TT.RandomResizedCrop(size=(256, 512), ratio=(1.5, 8 / 3.), scale=(0.5, 1.)),
        # it return an image of size 256x512
        TT.RandomHorizontalFlip(),
        TT.ToTensor(),
        TT.Normalize(args.normalize, args.normalize)
    ])

    # triple transform only used for items with contour original image
    triple_transform = TTT.Compose([
        TTT.RandomResizedCrop(size=(256, 512), ratio=(1.5, 8 / 3.), scale=(0.5, 1.)),
        # it return an image of size 256x512
        TTT.RandomHorizontalFlip(),
        TTT.ToTensor(),
        TTT.Normalize(args.normalize, args.normalize)
    ])

    source_train_transform = double_transform
    target_train_transform = single_transform
    # data loading
    if args.source_dataset == 'Cityscapes':
        source_dataset = Cityscapes('datasets/source_dataset', transforms=source_train_transform)
    elif args.source_dataset == 'freiburg_rgb':
        source_dataset = Freiburg('datasets/freiburg', split='train', domain='RGB', transforms=source_train_transform,
                                  with_label=True, grayscale=args.grayscale)
    else:
        raise ValueError('source dataset does not exist.')

    if args.target_dataset == 'flir':
        target_dataset = TrainTDataset('datasets/target_dataset', transforms=target_train_transform)
    elif args.target_dataset == 'freiburg_ir':
        target_dataset = Freiburg('datasets/freiburg', split='train', domain='IR', transforms=target_train_transform,
                                  with_label=False)
    else:
        raise ValueError('target dataset does not exist.')

    train_source_loader = DataLoader(source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    train_target_loader = DataLoader(target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    # networks
    net_g_s2t = generators.unet_256(ngf=64, input_nc=args.s2t_input_nc, output_nc=args.t2s_input_nc).to(device)
    net_d_t = discriminators.NLayerDiscriminator(input_nc=args.t2s_input_nc).to(device)
    canny_thermal = Canny(device, threshold=1, batch_size=args.batch_size).to(device)
    canny_rgb = Canny(device, threshold=2, batch_size=args.batch_size).to(device)
    set_requires_grad(canny_thermal, False)
    set_requires_grad(canny_rgb, False)

    canny = {'thermal': canny_thermal, 'rgb': canny_rgb}

    restart_epoch = 0
    if args.load_model:
        load_checkpoint = torch.load(os.path.join(MODEL_ROOT_PATH, args.checkpoint_name))
        restart_epoch = load_checkpoint['epoch']
        print('loading trained model. start from epoch {}.'.format(restart_epoch))
        net_g_s2t.load_state_dict(load_checkpoint['net_g_s2t_state_dict'])
        net_d_t.load_state_dict(load_checkpoint['net_d_t_state_dict'])
    # create image buffer to store previously generated images
    fake_s_pool = ImagePool(50)
    fake_t_pool = ImagePool(50)

    # define optimizers
    optimizer_g = torch.optim.Adam(net_g_s2t.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(net_d_t.parameters(), lr=args.lr)

    # define loss
    gan_loss_func = loss.LeastSquaresGenerativeAdversarialLoss()
    contour_loss_func = torch.nn.L1Loss()
    loss_dict = {'g_s2t': [], 'd_t': [], 'con_s2t': []}
    epoch_counter_ratio = []
    print("--------START TRAINING--------")
    for epoch in range(restart_epoch, restart_epoch+args.num_epoch):
        print("--------EPOCH {}--------".format(epoch))
        train(args, train_source_loader, train_target_loader, net_g_s2t, net_d_t, canny, gan_loss_func,
              contour_loss_func, optimizer_g, optimizer_d, fake_s_pool, fake_t_pool, device, epoch, visualizer,
              loss_dict, epoch_counter_ratio)

        torch.save({
            'epoch': epoch,
            'net_g_s2t_state_dict': net_g_s2t.state_dict(),
            'net_d_t_state_dict': net_d_t.state_dict()
        }, os.path.join(MODEL_ROOT_PATH, args.checkpoint_name))


if __name__ == '__main__':
    args_ = train_parse().parse_args()
    main(args_)

