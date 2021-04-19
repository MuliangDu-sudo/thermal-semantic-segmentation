from abc import ABC

from torch.utils import data


class BaseDataset(data.Dataset):
    def __init__(self, transform=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass