import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from data import DistanceDataset
from PIL import ImageFile
from utils import AverageMeter
from tqdm import tqdm
import visdom
from options import translation_parse
from pytorch_msssim import ssim
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

trans_args = translation_parse().parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
visualizer = visdom.Visdom(env='translation vs original')

# random select data for evaluation
validation_split = .2
shuffle_dataset = True
random_seed = 42
distance_dataset = DistanceDataset('datasets/freiburg', translate_name=trans_args.checkpoint_name.replace('.pth', ''))
# Creating data indices for training and validation splits:
dataset_size = len(distance_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
val_sampler = SubsetRandomSampler(val_indices)

distance_dataloader = DataLoader(distance_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True,
                                 sampler=val_sampler, drop_last=True)

distance_func = torch.nn.L1Loss()
distances = AverageMeter('distance', ':3.4f')
ssim_scores = []

for i, data in enumerate(tqdm(distance_dataloader)):
    ori_image = data[0].to(device)
    # trans_image = torch.flip(data[1].to(device), dims=[0])
    trans_image = data[1].to(device)
    distance = distance_func(ori_image, trans_image)
    ssim_score = np.array(ssim(ori_image, trans_image, data_range=1, size_average=True).cpu())
    distances.update(distance.item(), ori_image.size(0))
    ssim_scores = np.append(ssim_scores, ssim_score)
    if i % 5 == 0:
        visualizer.images(ori_image[0], win='original{}'.format(i / 5), padding=2,
                          opts=dict(title='original{}'.format(i / 5), caption='original{}'.format(i / 5)))
        visualizer.images(trans_image[0], win='translation{}'.format(i / 5), padding=2,
                          opts=dict(title='translation{}'.format(i / 5), caption='translation{}'.format(i / 5)))

# model is selected in translation_parse().
print('Model: ' + str(trans_args.checkpoint_name.replace('.pth', '')))
print('L1 distance: ' + str(distances.avg))
print('SSIM score: ' + str(np.mean(ssim_scores)))

