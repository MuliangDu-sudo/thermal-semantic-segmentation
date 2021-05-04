from torch.utils.data import Dataset
import os
from utils import flir_txt
from PIL import Image


class TrainTDataset(Dataset):

    def __init__(self, root, transforms):
        if not os.path.exists(os.path.join(root, "image_list", "train.txt")):
            flir_txt(root, 'train')
        data_list_file = os.path.join(root, "image_list", "train.txt")
        self.data_list = self.parse_data_file(data_list_file)
        self.root = root
        self.transform = transforms

    def parse_data_file(self, file_name):
        """Parse file to image list

        Args:
            file_name (str): The path of data file

        Returns:
            List of image path
        """

        with open(file_name, "r") as f:
            data_list = [line.strip() for line in f.readlines()]
        return data_list

    def __getitem__(self, index):
        image_name = self.data_list[index]
        image = Image.open(os.path.join(image_name))
        image = self.transform(image)

        return image

    def __len__(self):
        return len(self.data_list)


class TestTDataset(Dataset):

    def __init__(self):
        pass

    def parse_label_file(self, label_list_file):
        with open(label_list_file, "r") as f:
            label_list = [line.strip().replace("jpeg", "png") for line in f.readlines()]
        return label_list

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass