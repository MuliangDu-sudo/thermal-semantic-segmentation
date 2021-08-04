import os
from utils import kitti_txt
from PIL import Image, ImageOps
import torch
import numpy as np
from torch.utils import data
from torchvision import transforms as T
import glob


def parse_file(file_name):
    """Parse file to image list

    Args:
        file_name (str): The path of data file

    Returns:
        List of image path
    """

    with open(file_name, "r") as f:
        data_list = [line.strip() for line in f.readlines()]
    return data_list


class Kitti(data.Dataset):

    def __init__(self, root, transforms, grayscale=False, translation_mode=False,
                 translation_name='translation'):
        """
        :param root: str. root path to the dataset.
        :param split: str. train or test.
        :param domain: str. RGB or IR.
        :param transforms: transforms pipeline.
        :param with_label: bool. whether returns label
        """

        data_list_file = os.path.join(root, "image_list", "kitti_data.txt")

        if not os.path.exists(data_list_file):
            kitti_txt(root)
            data_list_file = os.path.join(root, "image_list", "kitti_data.txt")

        self.data_list = parse_file(data_list_file)
        self.transforms = transforms
        self.grayscale = grayscale
        self.translation_mode = translation_mode
        self.translation_name = translation_name

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):

        image_name = self.data_list[item]

        if self.grayscale:
            image = ImageOps.grayscale(Image.open(os.path.join(image_name)).convert('RGB'))

        else:
            image = Image.open(os.path.join(image_name)).convert('RGB')
        image = self.transforms(image)
        return_item = image
        if self.translation_mode:
            translation_name = self.translation_name + image_name
            return_item = image, translation_name
        return return_item
