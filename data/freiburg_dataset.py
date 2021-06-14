import os
from .base_dataset import BaseDataset
from utils import freiburg_txt
from PIL import Image
import torch
import numpy as np
from torch.utils import data


class Freiburg(data.Dataset):

    def __init__(self, root, split, domain, transforms, with_label, **kwargs):
        """
        :param root: str. root path to the dataset.
        :param split: str. train or test.
        :param domain: str. RGB or IR.
        :param transforms: transforms pipeline.
        :param kwargs: others.
        :param with_label: bool. whether returns label
        """

        assert split in ['train', 'test']

        data_list_file = os.path.join(root, "image_list", "{}_{}_data.txt".format(split, domain))
        label_list_file = os.path.join(root, "image_list", "{}_{}_label.txt".format(split, domain))
        if not (os.path.exists(data_list_file) and os.path.exists(label_list_file)):
            freiburg_txt(root, split, domain)
        self.data_list = self.parse_file(data_list_file)
        self.label_list = self.parse_file(label_list_file)
        self.split = split
        self.domain = domain
        self.transforms = transforms
        self.with_label = with_label

    def parse_file(self, file_name):
        """Parse file to image list

        Args:
            file_name (str): The path of data file

        Returns:
            List of image path
        """

        with open(file_name, "r") as f:
            data_list = [line.strip() for line in f.readlines()]
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):

        image_name = self.data_list[item]
        label_name = self.label_list[item]
        if self.domain == 'IR':
            image = np.array(Image.open(os.path.join(image_name)).resize((960, 320), Image.BICUBIC),  dtype=np.float32)
            image = image[:, 150:850]
            # normalize IR data (is in range 0, 2**16 --> crop to relevant range(20800, 27000))
            minval = 21800
            maxval = 25000

            image[image < minval] = minval
            image[image > maxval] = maxval

            image = (image - minval) / (maxval - minval)
            image = Image.fromarray(image)
        elif self.domain == 'RGB':
            image = np.array(Image.open(os.path.join(image_name)).convert('RGB').resize((960, 320), Image.BICUBIC),
                             dtype=np.float32)
            image = image[:, 150:850, :]
            image = Image.fromarray(image)
        else:
            raise ValueError('Not a valid domain.')

        if self.with_label:
            label = np.array(Image.open(os.path.join(label_name)).resize((960, 320), Image.NEAREST),  dtype=np.uint8)
            label = Image.fromarray(label, mode='L')
            image, label = self.transforms(image, label)
            return image, np.array(label, dtype=np.int64)
        else:
            image = self.transforms(image)
            return image




