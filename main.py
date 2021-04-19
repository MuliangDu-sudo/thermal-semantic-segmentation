import torch
from utils import train_transform, loss, ImagePool
from data import SDataset, TDataset
from torch.utils.data import DataLoader
from models import discriminators, generators, semantic_segmentation_models
from itertools import chain
from train import train, predict
from options import train_parse


def main(args):

    # data loading
    source_dataset = SDataset(transform=args.transform)
    target_dataset = TDataset(transform=args.transform)
    train_source_loader = DataLoader(source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    train_target_loader = DataLoader(target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    # networks
    net_g_s2t = generators.GeneratorS2T().to(args.device)
    net_g_t2s = generators.GeneratorT2S().to(args.device)
    net_d_s = discriminators.DiscriminatorS().to(args.device)
    net_d_t = discriminators.DiscriminatorT().to(args.device)
    net_seg_s = semantic_segmentation_models.SegNetwork().to(args.device)
    net_seg_t = semantic_segmentation_models.SegNetwork().to(args.device)

    # create image buffer to store previously generated images
    fake_s_pool = ImagePool(args.pool_size)
    fake_t_pool = ImagePool(args.pool_size)

    # define optimizers
    all_params_g = chain(net_g_s2t.parameters(), net_g_t2s.parameters())
    optimizer_g = torch.optim.Adam(all_params_g, lr=args.lr)
    all_params_d = chain(net_d_s.parameters(), net_d_t.parameters())
    optimizer_d = torch.optim.Adam(all_params_d, lr=args.lr)

    # define loss
    gan_loss_func = loss.LeastSquaresGenerativeAdversarialLoss()
    cycle_loss_func = torch.nn.L1Loss()
    identity_loss_func = torch.nn.L1Loss()
    sem_loss_func = loss.ContourLoss()

    for epoch in range(args.epoch_num):

        train(args, train_source_loader, train_target_loader, net_g_s2t, net_g_t2s, net_d_s, net_d_t, gan_loss_func,
              cycle_loss_func, identity_loss_func, sem_loss_func, optimizer_g, optimizer_d, fake_s_pool, fake_t_pool)

        torch.save()


if __name__ == '__main__':
    args_ = train_parse().parse_args()
    main(args_)

