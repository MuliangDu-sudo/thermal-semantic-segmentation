import math
import numbers
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf

from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask, mask1=None, lpsoft=None):
        params = {}

        if mask1 is not None:
            mask1 = Image.fromarray(mask1, mode="L")
        if lpsoft is not None:
            lpsoft = torch.from_numpy(lpsoft)
            lpsoft = F.interpolate(lpsoft.unsqueeze(0), size=[img.size[1], img.size[0]], mode='bilinear', align_corners=True)[0]
        self.PIL2Numpy = True

        if img.size != mask.size:
            print(img.size, mask.size)
        assert img.size == mask.size
        if mask1 is not None:
            assert (img.size == mask1.size)
        for a in self.augmentations:
            img, mask, mask1, lpsoft, params = a(img, mask, mask1, lpsoft, params)

        if self.PIL2Numpy:
            img, mask = np.array(img), np.array(mask, dtype=np.uint8)
            if mask1 is not None:
                mask1 = np.array(mask1, dtype=np.uint8)
        return img, mask, mask1, lpsoft, params


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, mask1=None, lpsoft=None, params=None):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            if mask1 is not None:
                mask1 = ImageOps.expand(mask1, border=self.padding, fill=0)

        assert img.size == mask.size
        if mask1 is not None:
            assert (img.size == mask1.size)
        w, h = img.size
        tw, th = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            if lpsoft is not None:
                lpsoft = F.interpolate(lpsoft.unsqueeze(0), size=[th, tw], mode='bolinear', align_corners=True)[0]
            if mask1 is not None:
                return (
                        img.resize((tw, th), Image.BILINEAR),
                        mask.resize((tw, th), Image.NEAREST),
                        mask1.resize((tw, th), Image.NEAREST),
                        lpsoft
                    )
            else:
                    return (
                        img.resize((tw, th), Image.BILINEAR),
                        mask.resize((tw, th), Image.NEAREST),
                        None,
                        lpsoft
                    )

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        params['RandomCrop'] = (y1, y1 + th, x1, x1 + tw)
        if lpsoft is not None:
            lpsoft = lpsoft[:, y1:y1 + th, x1:x1 + tw]
        if mask1 is not None:
            return (
                img.crop((x1, y1, x1 + tw, y1 + th)),
                mask.crop((x1, y1, x1 + tw, y1 + th)),
                mask1.crop((x1, y1, x1 + tw, y1 + th)),
                lpsoft,
                params
            )
        else:
            return (
                img.crop((x1, y1, x1 + tw, y1 + th)),
                mask.crop((x1, y1, x1 + tw, y1 + th)),
                None,
                lpsoft,
                params
            )


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask, mask1=None, lpsoft=None, params=None):
        assert img.size == mask.size
        if mask1 is not None:
            assert (img.size == mask1.size)

        prop = 1.0 * img.size[0] / img.size[1]
        w = int(random.uniform(0.5, 1.5) * self.size)
        #w = self.size
        h = int(w/prop)
        params['RandomSized'] = (h, w)
        # h = int(random.uniform(0.5, 2) * self.size[1])

        img, mask = (
            img.resize((w, h), Image.BILINEAR),
            mask.resize((w, h), Image.NEAREST),
        )
        if mask1 is not None:
            mask1 = mask1.resize((w, h), Image.NEAREST)
        if lpsoft is not None:
            lpsoft = F.interpolate(lpsoft.unsqueeze(0), size=[h, w], mode='bilinear', align_corners=True)[0]

        return img, mask, mask1, lpsoft, params


class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask, mask1=None, lpsoft=None, params=None):
        if random.random() < self.p:
            params['RandomHorizontallyFlip'] = True
            if lpsoft is not None:
                inv_idx = torch.arange(lpsoft.size(2)-1,-1,-1).long()  # C x H x W
                lpsoft = lpsoft.index_select(2,inv_idx)
            if mask1 is not None:
                return (
                    img.transpose(Image.FLIP_LEFT_RIGHT),
                    mask.transpose(Image.FLIP_LEFT_RIGHT),
                    mask1.transpose(Image.FLIP_LEFT_RIGHT),
                    lpsoft,
                    params
                )
            else:
                return (
                    img.transpose(Image.FLIP_LEFT_RIGHT),
                    mask.transpose(Image.FLIP_LEFT_RIGHT),
                    None,
                    lpsoft,
                    params
                )
        else:
            params['RandomHorizontallyFlip'] = False
        return img, mask, mask1, lpsoft, params

def get_composed_augmentations(args):
    return Compose([RandomSized(args.resize),
                    RandomCrop(args.rcrop),
                    RandomHorizontallyFlip(args.hflip)])