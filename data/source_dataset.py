import os
from .base_dataset import BaseDataset
from utils import cityscapes_txt
from PIL import Image
import torch
import numpy as np


class Cityscapes(BaseDataset):
    """`Cityscapes <https://www.cityscapes-dataset.com/>`_ is a real-world semantic segmentation dataset collected
    in driving scenarios.

    Args:
        root (str): Root directory of dataset. e.g. datasets/source_dataset
        split (str, optional): The dataset split, supports ``train``, or ``val``.
        data_folder (str, optional): Sub-directory of the image. Default: 'leftImg8bit'.
        label_folder (str, optional): Sub-directory of the label. Default: 'gtFine_labelIds'.
        mean (seq[float]): mean BGR value. Normalize the image if not None. Default: None.
        transforms (callable, optional): A function/transform that  takes in  (PIL image, label) pair \
            and returns a transformed version. E.g, :class:`~common.vision.transforms.segmentation.Resize`.

    .. note:: You need to download Cityscapes manually.
        Ensure that there exist following files in the `root` directory before you using this class.
        ::
            leftImg8bit/
                train/
                val/
                test/
            gtFine_labelIds/
                train/
                val/
                test/
    """

    CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
               'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle']

    # ID_TO_TRAIN_ID = {
    #     7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
    #     19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
    #     26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
    # }

    ID_TO_TRAIN_ID = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
        19: 5, 20: 5, 21: 6, 22: 7, 23: 8, 24: 9, 25: 9,
        26: 10, 27: 10, 28: 10, 31: 10, 32: 11, 33: 11
    }

    TRAIN_ID_TO_COLOR = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
                                  (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                                  (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                                  (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
                                  (0, 0, 230), (119, 11, 32), [0, 0, 0]]
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/08745e798b16483db4bf/?dl=1"),
    ]
    EVALUATE_CLASSES = CLASSES

    def __init__(self, root, split='train', data_folder='leftImg8bit', label_folder='gtFine_labelIds', **kwargs):
        assert split in ['train', 'val']

        if not os.path.exists(os.path.join(root, "image_list", "{}_{}.txt".format(data_folder, split))):
            cityscapes_txt(root, data_folder, split)
        data_list_file = os.path.join(root, "image_list", "{}_{}.txt".format(data_folder, split))
        self.split = split
        super(Cityscapes, self).__init__(root, Cityscapes.CLASSES, data_list_file, data_list_file,
                                         os.path.join(data_folder, split), os.path.join(label_folder, split),
                                         id_to_train_id=Cityscapes.ID_TO_TRAIN_ID,
                                         train_id_to_color=Cityscapes.TRAIN_ID_TO_COLOR, **kwargs)
        self.ignore_label = 12
    def parse_label_file(self, label_list_file):
        with open(label_list_file, "r") as f:
            label_list = [line.strip().replace("leftImg8bit", "gtFine_labelIds") for line in f.readlines()]
        return label_list


class CityscapesTranslation(BaseDataset):
    """`Cityscapes <https://www.cityscapes-dataset.com/>`_ is a real-world semantic segmentation dataset collected
    in driving scenarios.

    Args:
        root (str): Root directory of dataset. e.g. datasets/source_dataset
        split (str, optional): The dataset split, supports ``train``, or ``val``.
        data_folder (str, optional): Sub-directory of the image. Default: 'leftImg8bit'.
        label_folder (str, optional): Sub-directory of the label. Default: 'gtFine_labelIds'.
        mean (seq[float]): mean BGR value. Normalize the image if not None. Default: None.
        transforms (callable, optional): A function/transform that  takes in  (PIL image, label) pair \
            and returns a transformed version. E.g, :class:`~common.vision.transforms.segmentation.Resize`.

    .. note:: You need to download Cityscapes manually.
        Ensure that there exist following files in the `root` directory before you using this class.
        ::
            leftImg8bit/
                train/
                val/
                test/
            gtFine_labelIds/
                train/
                val/
                test/
    """

    CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
               'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle']

    # ID_TO_TRAIN_ID = {
    #     7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
    #     19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
    #     26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
    # }

    ID_TO_TRAIN_ID = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
        19: 5, 20: 5, 21: 6, 22: 7, 23: 8, 24: 9, 25: 9,
        26: 10, 27: 10, 28: 10, 31: 10, 32: 11, 33: 11
    }

    TRAIN_ID_TO_COLOR = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
                                  (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                                  (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                                  (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
                                  (0, 0, 230), (119, 11, 32), [0, 0, 0]]
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/08745e798b16483db4bf/?dl=1"),
    ]
    EVALUATE_CLASSES = CLASSES

    def __init__(self, root, split='train', data_folder='translation', label_folder='gtFine_labelIds', **kwargs):
        assert split in ['train', 'val']

        if not os.path.exists(os.path.join(root, "image_list", "{}_{}.txt".format(data_folder, split))):
            cityscapes_txt(root, data_folder, split)
        data_list_file = os.path.join(root, "image_list", "{}_{}.txt".format(data_folder, split))
        self.split = split
        super(CityscapesTranslation, self).__init__(root, CityscapesTranslation.CLASSES, data_list_file, data_list_file,
                                         os.path.join(data_folder, split), os.path.join(label_folder, split),
                                         id_to_train_id=CityscapesTranslation.ID_TO_TRAIN_ID,
                                         train_id_to_color=CityscapesTranslation.TRAIN_ID_TO_COLOR, **kwargs)
        self.ignore_label = 12

    def parse_label_file(self, label_list_file):
        with open(label_list_file, "r") as f:
            label_list = [line.strip().replace("translation", "gtFine_labelIds") for line in f.readlines()]
        return label_list

    def __getitem__(self, index):
        input_dict = {}
        image_name = self.data_list[index]
        image = Image.open(os.path.join(image_name))
        label_name = self.label_list[index]
        label = Image.open(os.path.join(label_name))
        input_dict['image'], label = self.transforms(image, label)
        # remap label
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        label = np.asarray(label, np.int64)
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.int64)
        if self.id_to_train_id:
            for k, v in self.id_to_train_id.items():
                label_copy[label == k] = v
        input_dict['label'] = label_copy
        return input_dict
