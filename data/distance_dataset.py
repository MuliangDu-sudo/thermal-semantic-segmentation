from torch.utils.data import Dataset
import os
from data import parse_file
import numpy as np
from PIL import Image
from torchvision import transforms as T


class DistanceDataset(Dataset):

    def __init__(self, root, translate_name, split='train', domain='IR'):
        ori_list_file = os.path.join(root, "image_list", "{}_{}_data.txt".format(split, domain))
        self.data_list = parse_file(ori_list_file)
        self.translate_name = translate_name
        self.split = split
        self.domain = domain
        self.transform = T.Compose([
            T.ToTensor(),
            # T.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        ori_name = self.data_list[item]
        trans_name = ori_name.replace(self.split, self.translate_name).replace('ir_aligned', 'rgb')
        if self.domain == 'IR':
            image = np.array(Image.open(os.path.join(ori_name)).resize((960, 320), Image.BICUBIC),  dtype=np.float32)
            image = image[:, 150:850]
            # normalize IR data (is in range 0, 2**16 --> crop to relevant range(20800, 27000))
            minval = 21800
            maxval = 25000

            image[image < minval] = minval
            image[image > maxval] = maxval

            image = (image - minval) / (maxval - minval)
            ori_image = Image.fromarray(image)
        elif self.domain == 'RGB':

            image = np.array(Image.open(os.path.join(ori_name)).convert('RGB').resize((960, 320), Image.BICUBIC),
                         dtype=np.float32)
            image = image[:, 150:850, :]
            ori_image = Image.fromarray(np.uint8(image))
        else:
            raise ValueError('Not a valid domain.')
        trans_image = Image.open(os.path.join(trans_name))
        return self.transform(ori_image), self.transform(trans_image)