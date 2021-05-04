import torch
from utils import loss, ImagePool
from utils import transforms as T
from data import Cityscapes, TrainTDataset
from torch.utils.data import DataLoader
from models import discriminators, generators, semantic_segmentation_models, thermal_semantic_segmentation_models
from itertools import chain
from train import train, predict
from options import train_parse
from torchvision import transforms as TT


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

source_train_transform = T.Compose([
    T.RandomResizedCrop(size=(512, 256), ratio=(1.5, 8 / 3.), scale=(0.5, 1.)),  # it return an image of size 256x512
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    #T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

target_train_transform = TT.Compose([
    TT.RandomResizedCrop(size=(256, 512), ratio=(1.5, 8 / 3.), scale=(0.5, 1.)),
    TT.RandomHorizontalFlip(),
    TT.ToTensor(),
    #TT.Normalize((0.5,), (0.5,))
])


def main():

    # data loading
    source_dataset = Cityscapes('datasets/source_dataset', transforms=source_train_transform)
    target_dataset = TrainTDataset('datasets/target_dataset', transforms=target_train_transform)
    train_source_loader = DataLoader(source_dataset, batch_size=1,
                                     shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    train_target_loader = DataLoader(target_dataset, batch_size=1,
                                     shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    # networks
    net_g_s2t = generators.unet_256(ngf=64, input_nc=3, output_nc=1).to(device)
    net_g_t2s = generators.unet_256(ngf=64, input_nc=1, output_nc=3).to(device)
    net_d_s = discriminators.NLayerDiscriminator(input_nc=3).to(device)
    net_d_t = discriminators.NLayerDiscriminator(input_nc=1).to(device)
    net_seg_s = semantic_segmentation_models.deeplabv2_resnet101().to(device)
    net_seg_t = thermal_semantic_segmentation_models.deeplabv2_resnet101_thermal(pretrained_backbone=False).to(device)

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

    for epoch in range(10):

        train(train_source_loader, train_target_loader, net_g_s2t, net_g_t2s, net_d_s, net_d_t, net_seg_s, net_seg_t,
              gan_loss_func, cycle_loss_func, identity_loss_func, sem_loss_func, optimizer_g, optimizer_d, fake_s_pool,
              fake_t_pool, device)

        #torch.save()


if __name__ == '__main__':
    #args_ = train_parse().parse_args()
    main()

