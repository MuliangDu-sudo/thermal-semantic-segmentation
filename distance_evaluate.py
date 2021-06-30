import torch
from torch.utils.data import DataLoader
from data import DistanceDataset
from PIL import ImageFile
from utils import AverageMeter, ProgressMeter


ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

distance_dataset = DistanceDataset('datasets/freiburg', translate_name='translation')
distance_dataloader = DataLoader(distance_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

distance_func = torch.nn.L1Loss()
distances = AverageMeter('distance', ':3.4f')

for i, data in enumerate(distance_dataloader):
    ori_image = data[0].to(device)
    trans_image = data[1].to(device)
    distance = distance_func(ori_image, trans_image)
    distances.update(distance.item(), ori_image.size(0))

print(distances.avg)

