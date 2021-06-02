from models import thermal_semantic_segmentation_models
import torch
from utils import transforms as T
import os
from PIL import ImageFile
from utils import AverageMeter, set_requires_grad, ProgressMeter, plot_loss
import time
from data import CityscapesTranslation
from torch.utils.data import DataLoader
import visdom
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from utils.eval_tools import evaluate
from options import seg_parse


MODEL_ROOT_PATH = './checkpoints/semantic_segmentation'


def seg_validate(args, sem_net, val_data, loss_func, device, vis):
    print('validating...')
    val_loss = AverageMeter('val_loss', ':3.4f')
    sem_net.eval()
    prediction_list, label_list = [], []
    random_id = np.random.choice(len(val_data), args.num_samples_show)
    i = 0
    for item in val_data:
        image = item[0].to(device)
        label = item[1].to(device)
        # print(image[0].shape)
        # print(label[0].unsqueeze(dim=0).shape)
        outputs = sem_net(image)
        outputs = torch.nn.Upsample(size=(256, 512), mode='bilinear', align_corners=True)(outputs)
        loss = loss_func(outputs, label)

        predictions = outputs.max(1)[1].squeeze_(1).cpu().numpy()
        label_list.append(label.cpu().numpy())
        prediction_list.append(predictions)
        val_loss.update(loss.item(), image.size(0))
        if i in random_id:
            vis.images(image[0], win='image [{}]'.format(i), padding=2, opts=dict(title='image [{}]'.format(i), caption='image [{}]'.format(i)))
            vis.images(np.uint8(label[0].cpu().numpy()), win='label [{}]'.format(i), padding=2, opts=dict(title='label [{}]'.format(i), caption='label [{}]'.format(i)))
            vis.images(np.uint8(predictions[0]), win='prediction [{}]'.format(i), padding=2, opts=dict(title='prediction [{}]'.format(i), caption='prediction [{}]'.format(i)))
        i += 1

    label_list = np.concatenate(label_list)
    prediction_list = np.concatenate(prediction_list)
    acc, acc_cls, mean_iu, fwavacc = evaluate(prediction_list, label_list, 19)
    return mean_iu, val_loss.avg


def seg_evaluation(args):
    print('evaluating...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualizer = visdom.Visdom(env='thermal semantic segmentation')

    train_transform = T.Compose([
        T.RandomResizedCrop(size=(256, 512), ratio=(1.5, 8 / 3.), scale=(0.5, 1.)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        # T.Normalize((0.5,), (0.5,))
    ])

    validation_split = .2
    shuffle_dataset = True
    random_seed = 42
    source_dataset = CityscapesTranslation('datasets/source_dataset', data_folder='translation',
                                           transforms=train_transform)
    # Creating data indices for training and validation splits:
    dataset_size = len(source_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    valid_sampler = SubsetRandomSampler(val_indices)
    val_dataloader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
                                drop_last=True, sampler=valid_sampler)

    net = thermal_semantic_segmentation_models.deeplabv2_resnet101_thermal(pretrained_backbone=False).to(device)
    load_checkpoint = torch.load(os.path.join(MODEL_ROOT_PATH, args.checkpoint_name))
    net.load_state_dict(load_checkpoint['sem_net_state_dict'])

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    mean_iu, avg_loss = seg_validate(args, net, val_dataloader, loss_function, device, visualizer)
    print('mean iou score: [{}]. val_loss: [{}]'.format(mean_iu, avg_loss))


if __name__ == '__main__':
    args_ = seg_parse().parse_args()
    seg_evaluation(args_)


