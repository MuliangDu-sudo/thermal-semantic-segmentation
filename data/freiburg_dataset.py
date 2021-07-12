import os
from .base_dataset import BaseDataset
from utils import freiburg_txt
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


class Freiburg(data.Dataset):

    def __init__(self, root, split, domain, transforms, with_label, grayscale=False, translation_mode=False,
                 translation_name='translation', segmentation_mode=False):
        """
        :param root: str. root path to the dataset.
        :param split: str. train or test.
        :param domain: str. RGB or IR.
        :param transforms: transforms pipeline.
        :param with_label: bool. whether returns label
        """

        assert split in ['train', 'test']

        data_list_file = os.path.join(root, "image_list", "{}_{}_data.txt".format(split, domain))
        label_list_file = os.path.join(root, "image_list", "{}_{}_label.txt".format(split, domain))
        if not (os.path.exists(data_list_file) and os.path.exists(label_list_file)):
            freiburg_txt(root, split, domain)
        self.data_list = parse_file(data_list_file)
        self.label_list = parse_file(label_list_file)
        self.split = split
        self.domain = domain
        self.transforms = transforms
        self.with_label = with_label
        self.grayscale = grayscale
        self.translation_mode = translation_mode
        self.translation_name = translation_name
        self.segmentation_mode = segmentation_mode
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):

        image_name = self.data_list[item]
        label_name = self.label_list[item]
        if self.domain == 'IR' and not self.segmentation_mode:
            image = np.array(Image.open(os.path.join(image_name)).resize((960, 320), Image.BICUBIC),  dtype=np.float32)
            image = image[:, 150:850]
            # normalize IR data (is in range 0, 2**16 --> crop to relevant range(20800, 27000))
            minval = 21800
            maxval = 25000

            image[image < minval] = minval
            image[image > maxval] = maxval

            image = (image - minval) / (maxval - minval)
            image = Image.fromarray(image)
        elif self.domain == 'RGB' and not self.segmentation_mode:
            if self.grayscale:
                image = np.array(ImageOps.grayscale(Image.open(os.path.join(image_name)).convert('RGB')).resize((960, 320), Image.BICUBIC),
                                 dtype=np.float32)
                image = image[:, 150:850]
            else:
                image = np.array(Image.open(os.path.join(image_name)).convert('RGB').resize((960, 320), Image.BICUBIC),
                             dtype=np.float32)
                image = image[:, 150:850, :]
            image = Image.fromarray(np.uint8(image))

        elif self.segmentation_mode:
            image_name = image_name.replace(str(self.split), self.translation_name)
            image = Image.open(os.path.join(image_name))

        else:
            raise ValueError('Not a valid domain.')

        if self.with_label:
            label = np.array(Image.open(os.path.join(label_name)).resize((960, 320), Image.NEAREST), dtype=np.uint8)
            label = label[:, 150:850]
            label = Image.fromarray(label, mode='L')
            image, label = self.transforms(image, label)
            return_item = image, np.array(label, dtype=np.int64)
        else:
            return_item = self.transforms(image), 0

        if self.translation_mode:
            image = self.transforms(image)
            translation_name = image_name.replace(str(self.split), self.translation_name)
            return_item = image, translation_name

        return return_item


class FreiburgTest(Freiburg):

    def __init__(self, root, split, domain, transforms, with_label, grayscale=False, transform_label=True):
        super(FreiburgTest, self).__init__(root, split, domain, transforms, with_label, grayscale)
        self.transform_label = transform_label

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
            if self.grayscale:
                image = np.array(ImageOps.grayscale(Image.open(os.path.join(image_name)).convert('RGB')).resize((960, 320), Image.BICUBIC),
                                 dtype=np.float32)
                image = image[:, 150:850]
            else:
                image = np.array(Image.open(os.path.join(image_name)).convert('RGB').resize((960, 320), Image.BICUBIC),
                                 dtype=np.float32)
                image = image[:, 150:850, :]
            image = Image.fromarray(np.uint8(image))
        else:
            raise ValueError('Not a valid domain.')

        if self.with_label:
            label = np.load(os.path.join(label_name))
            label = Image.fromarray(label).resize((960, 320), Image.NEAREST)
            label = Image.fromarray(np.array(label)[:, 150:850])
            if self.transform_label:
                image, label = self.transforms(image, label)
                label = np.array(label, dtype=np.int64)
            else:
                image = self.transforms(image)
                label = T.ToTensor()(label)
            return image, label

        else:
            image = self.transforms(image)
            return image


class FreiburgT2S(data.Dataset):
    def __init__(self, folder, transforms, root='datasets/freiburg/t2s/'):
        self.translation_files = glob.glob(root + folder + '*_translation.jpg', recursive=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.translation_files)

    def __getitem__(self, item):
        image_name = self.translation_files[item]
        label_name = image_name.replace('translation.jpg', 'groundtruth.png')
        image = Image.open(os.path.join(image_name)).convert('RGB')
        label = Image.open(os.path.join(label_name))
        image, label = self.transforms(image, label)
        return image, np.array(label, dtype=np.int64)


class FreiburgTranslation(data.Dataset):
    def __init__(self, folder, transforms, root='datasets/freiburg/translations'):
        print(folder)
        self.translation_files = glob.glob(root + folder + '*_translation.jpg', recursive=True)
        self.transforms = transforms

    def __len__(self):
        print(len(self.translation_files))
        return len(self.translation_files)

    def __getitem__(self, item):
        image_name = self.translation_files[item]
        label_name = image_name.replace('_translation.jpg', '_groundtruth.png')
        image = Image.open(os.path.join(image_name))
        label = Image.open(os.path.join(label_name))
        image, label = self.transforms(image, label)
        return image, np.array(label, dtype=np.int64)
