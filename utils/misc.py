from typing import Optional, List
import torch
import random
import os
from torch import nn
from torch.nn import init
import functools
import numpy as np

class AverageMeter(object):
    r"""Computes and stores the average and current value.

    Examples::

        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """
    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def set_requires_grad(net, requires_grad=False):
    """
    Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    """
    for param in net.parameters():
        param.requires_grad = requires_grad


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Args:
        net (torch.nn.Module): network to be initialized
        init_type (str): the name of an initialization method. Choices includes: ``normal`` |
            ``xavier`` | ``kaiming`` | ``orthogonal``
        init_gain (float): scaling factor for normal, xavier and orthogonal.

    'normal' is used in the original CycleGAN paper. But xavier and kaiming might
    work better for some applications.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class ImagePool:
    """An image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.

    Args:
        pool_size (int): the size of image buffer, if pool_size=0, no buffer will be created

    """

    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Args:
            images (torch.Tensor): the latest generated images from the generator

        Returns:
            By 50/100, the buffer will return input images.
            By 50/100, the buffer will return images previously stored in the buffer,
            and insert the current images to the buffer.

        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


def cityscapes_txt(root, data_folder, split):
    """

    :param root: str, root directory
    :param data_folder: str, image(leftImg8bit) or label(gtFine_labelIds)
    :param split: str, train, eval, test
    :return: txt file of files paths
    """
    im_dir: str = os.path.join(root, data_folder, split)
    list_file = open(r"{}/image_list/{}_{}.txt".format(root, data_folder, split), "w+")
    if data_folder == 'leftImg8bit' or data_folder == 'translation':
        print('im here')
        for dirpath, dirnames, filenames in os.walk(im_dir):
            for filename in filenames:
                list_file.write(os.path.join(dirpath, filename)+'\n')
    elif data_folder == 'gtFine_labelIds':
        for dirpath, dirnames, filenames in os.walk(im_dir):
            for filename in filenames:
                if filename.endswith('gtFine_labelIds.png'):
                    list_file.write(os.path.join(dirpath, filename)+'\n')
    list_file.close()


def flir_txt(root, split, data_folder='images'):
    """

    :param root: str. path to the target dataset folder.
    :param split: str. train or test.
    :param data_folder: image or label. Only for the test set.
    :return:
    """
    if split == 'train':
        im_dir: str = os.path.join(root, split)
    elif split == 'test':
        im_dir: str = os.path.join(root, split, data_folder)
    else:
        raise ValueError('path does not exist.')

    if split == 'train':
        list_file = open(r"{}/image_list/train.txt".format(root), "w+")
    else:
        list_file = open(r"{}/image_list/test_{}.txt".format(root, data_folder), "w+")
    for dirpath, dirnames, filenames in os.walk(im_dir):
        for filename in filenames:
            list_file.write(os.path.join(dirpath, filename)+'\n')
    list_file.close()


def plot_loss(epoch_counter_ratio, losses, vis):
    plot_data = {'X': epoch_counter_ratio, 'Y': [], 'legend': list(losses.keys())}
    plot_data['Y'] = [losses[k] for k in plot_data['legend']]
    x = np.array(plot_data['X'])
    # or x = np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1)
    y = np.array(plot_data['Y']).transpose()
    vis.line(
        X=x,
        Y=y,
        opts={
            'title': ' loss over time',
            'legend': plot_data['legend'],
            'xlabel': 'epoch',
            'ylabel': 'loss'},
        win='loss')