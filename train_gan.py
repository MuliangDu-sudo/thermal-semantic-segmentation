from utils import AverageMeter, set_requires_grad, ProgressMeter, plot_loss
from torchvision import transforms as T
import numpy as np
import torch
from PIL import Image
import time


class Denormalize(T.Normalize):
    """DeNormalize a tensor image with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will denormalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = input[channel] * std[channel] + mean[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.

    """
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        super().__init__((-mean / std).tolist(), (1 / std).tolist())


class NormalizeAndTranspose:
    """
    First, normalize a tensor image with mean and standard deviation.
    Then, convert the shape (H x W x C) to shape (C x H x W).
    """
    def __init__(self, domain, mean):
        self.mean = np.array(mean, dtype=np.float32)
        self.domain = domain

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.asarray(image, np.float32)
            if self.domain == 'source':
                # change to BGR
                image = image[:, :, ::-1]
            # normalize
            image -= self.mean
            image = image.transpose((2, 0, 1)).copy()
        elif isinstance(image, torch.Tensor):
            if self.domain == 'source':
                # change to BGR
                image = image[:, :, [2, 1, 0]]
            # normalize
            image -= torch.from_numpy(self.mean).to(image.device)
            image = image.permute((2, 0, 1))
        else:
            raise NotImplementedError(type(image))
        return image


def tensor_transmit(domain):
    denormalize = ()
    mean = ()
    if domain == 'source':
        denormalize = (0.5, 0.5, 0.5)
        mean = (104.00698793, 116.66876762, 122.67891434)
    elif domain == 'target':
        denormalize = (0.5,)
        mean = (116.66876762,)
    cycle_gan_tensor_to_segmentation_tensor = T.Compose([
        Denormalize(denormalize, denormalize),
        T.Lambda(lambda image: image.mul(255).permute((1, 2, 0))),
        NormalizeAndTranspose(domain=domain, mean=mean)
    ])

    return cycle_gan_tensor_to_segmentation_tensor


def predict(image, model, device, domain):

    image = tensor_transmit(domain)(image.squeeze(dim=0))
    image = image.unsqueeze(dim=0).to(device)
    prediction = model(image)

    return torch.nn.Upsample(size=(256, 512), mode='bilinear', align_corners=True)(prediction)


def train(args, s_data, t_data, g_s2t, d_t, canny, gan_loss_func, contour_loss_func, optim_g, optim_d,
          fake_s_pool, fake_t_pool, device, epoch, vis, loss_dict, epoch_counter_ratio):
    """
    :param args: parser
    :param s_data: source train dataloader
    :param t_data: target train dataloader
    :param g_s2t: source to target generator
    :param g_t2s: target to source generator
    :param d_s: source discriminator
    :param d_t: target discriminator
    :param gan_loss_func: gan loss function
    :param cycle_loss_func: reconstruction loss function
    :param identity_loss_func: identity loss function
    :param sem_loss_func: semantic consistency loss function
    :param optim_g: generator optimizer
    :param optim_d: discriminator optimizer
    :param fake_s_pool: source image pool for discriminator
    :param fake_t_pool: target image pool for discriminator
    :param device:
    :return:
    """
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_g = AverageMeter('g_s2t', ':3.4f')
    losses_d_t = AverageMeter('d_s', ':3.4f')
    losses_contour_s2t = AverageMeter('con_s2t', ':3.4f')

    iteration_length = min(len(s_data), len(t_data))
    progress = ProgressMeter(
        iteration_length,
        [batch_time, data_time, losses_g, losses_d_t, losses_contour_s2t],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()

    i = 0
    for s, t in zip(s_data, t_data):
        real_s = s[0].to(device)
        real_t = t[0].float().to(device)
        label_s = s[1].float().to(device)

        data_time.update(time.time() - end)

        # forward pass
        fake_t = g_s2t(real_s)

        # Optimizing generators
        # discriminators require no gradients
        set_requires_grad(d_t, False)

        optim_g.zero_grad()
        # gan loss d_s(g_s2t(s))
        loss_g = gan_loss_func(d_t(fake_t), real=True)

        if args.with_contour:
            contour_s_ori = T.Grayscale()(s[0]).to(device)
            contour_real_s = canny['rgb'](contour_s_ori).detach()
            contour_fake_t = canny['thermal'](fake_t).detach()

            loss_contour_s2t = contour_loss_func(contour_real_s, contour_fake_t)
            loss_g = loss_g + loss_contour_s2t
            losses_contour_s2t.update(loss_contour_s2t.item(), real_s.size(0))

        loss_g.backward()
        optim_g.step()

        # Optimize discriminator
        set_requires_grad(d_t, True)
        optim_d.zero_grad()
        # Calculate GAN loss for discriminator D_T
        fake_t_ = fake_t_pool.query(fake_t.detach())
        loss_d_t = 0.5 * (gan_loss_func(d_t(real_t), True) + gan_loss_func(d_t(fake_t_), False))
        loss_d_t.backward()
        optim_d.step()

        # measure elapsed time
        losses_g.update(loss_g.item(), real_s.size(0))
        losses_d_t.update(loss_d_t.item(), real_s.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)
            vis.images(real_s[0], win='real_s', padding=2, opts=dict(title='real_s', caption='real_s'))
            vis.images(fake_t[0], win='fake_t', padding=2, opts=dict(title='fake_t', caption='fake_t'))
            #vis.images(rec_s, win='rec_s', padding=2, opts=dict(title='rec_s', caption='rec_s'))
            #vis.images(real_t, win='real_t', padding=2, opts=dict(title='real_t', caption='real_t'))
            #vis.images(fake_s, win='fake_s', padding=2, opts=dict(title='fake_s', caption='fake_s'))
            #vis.images(rec_t, win='rec_t', padding=2, opts=dict(title='rec_t', caption='rec_t'))
            loss_dict['g_s2t'].append(loss_g.item())
            loss_dict['d_t'].append(loss_d_t.item())
            if args.with_contour:
                vis.images(contour_real_s[0], win='contour_real_s', padding=2, opts=dict(title='contour_real_s', caption='contour_real_s'))
                vis.images(contour_fake_t[0], win='contour_fake_t', padding=2, opts=dict(title='contour_fake_t', caption='contour_fake_t'))
                loss_dict['con_s2t'].append(loss_contour_s2t.item())
            epoch_counter_ratio.append(epoch+i/iteration_length)
            plot_loss(epoch_counter_ratio, loss_dict, vis)
            # print(loss_dict)
        i += 1
