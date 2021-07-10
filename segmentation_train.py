from models import thermal_semantic_segmentation_models, semantic_segmentation_models
import torch
from utils import transforms as T
import os
from PIL import ImageFile
from utils import AverageMeter, set_requires_grad, ProgressMeter, plot_loss
import time
from data import CityscapesTranslation, Cityscapes, Freiburg
from torch.utils.data import DataLoader
import visdom
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from utils.eval_tools import evaluate
from options import seg_parse
from PIL import ImageFile
from segmentation_evaluate import seg_validate


ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_ROOT_PATH = './checkpoints/semantic_segmentation'
if not os.path.exists(MODEL_ROOT_PATH):
    os.makedirs(MODEL_ROOT_PATH)


def seg_train(sem_net, data, loss_func, optim, device, vis, epoch, loss_dict):

    train_loss = AverageMeter('train_loss', ':3.4f')
    iteration_length = len(data)
    progress = ProgressMeter(iteration_length, [train_loss], prefix="Epoch: [{}]".format(epoch))

    #epoch_counter_ratio = []
    sem_net.train()
    i = 0
    for item in data:
        image = item[0].to(device)
        label = item[1].to(device)
        # print('label in main.')
        # print(torch.unique(label))
        optim.zero_grad()
        prediction = sem_net(image)
        prediction = torch.nn.Upsample(size=(256, 512), mode='bilinear', align_corners=True)(prediction)
        loss = loss_func(prediction, label)
        loss.backward()
        optim.step()

        train_loss.update(loss.item(), image.size(0))

        if i % 10 == 0:
            progress.display(i)
            loss_dict['train_loss'].append(loss.item())
            loss_dict['epoch_counter_ratio'].append(epoch + i / iteration_length)
            vis.line(X=np.array(loss_dict['epoch_counter_ratio']),
                     Y=np.array(loss_dict['train_loss']).transpose(),
                     opts={
                    'title': ' loss over time',
                    'xlabel': 'epoch',
                    'ylabel': 'loss'}, win='loss')
        i += 1


# def seg_validate(sem_net, val_data, device):
#     print('validating...')
#     sem_net.eval()
#     prediction_list, label_list = [], []
#     for item in val_data:
#         image = item[0].to(device)
#         label = item[1].to(device)
#         outputs = sem_net(image)
#         outputs = torch.nn.Upsample(size=(256, 512), mode='bilinear', align_corners=True)(outputs)
#
#         predictions = outputs.max(1)[1].squeeze_(1).cpu().numpy()
#         label_list.append(label.cpu().numpy())
#         prediction_list.append(predictions)
#     label_list = np.concatenate(label_list)
#     prediction_list = np.concatenate(prediction_list)
#     acc, acc_cls, mean_iu, fwavacc = evaluate(prediction_list, label_list, 19)
#     return mean_iu


def seg_main(args):
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
    if args.dataset == 'cityscapes_translation':
        source_dataset = CityscapesTranslation('datasets/source_dataset', data_folder='translation',
                                               transforms=train_transform)
    elif args.dataset == 'cityscapes':
        source_dataset = Cityscapes('datasets/source_dataset', transforms=train_transform)

    elif args.dataset == 'freiburg_ir':
        source_dataset = Freiburg('datasets/freiburg', split='train', domain='IR', transforms=train_transform,
                                  with_label=True)
    elif args.dataset == 'freiburg_rgb':
        source_dataset = Freiburg('datasets/freiburg', split='train', domain='RGB', transforms=train_transform,
                                  with_label=True)
    elif args.dataset == 'freiburg_translation':
        source_dataset = Freiburg('datasets/freiburg', split='train', domain='RGB', transforms=train_transform,
                                  with_label=True, segmentation_mode=True, translation_name=args.translation_name)
    else:
        raise ValueError('dataset does not exist.')
    # Creating data indices for training and validation splits:
    dataset_size = len(source_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
                                  drop_last=True, sampler=train_sampler)
    val_dataloader = DataLoader(source_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=2, pin_memory=True,
                                drop_last=True, sampler=valid_sampler)
    if args.net_mode == 'one_channel':
        net = thermal_semantic_segmentation_models.deeplabv2_resnet101_thermal(num_classes=args.num_classes,
                                                                               pretrained_backbone=False).to(device)
    elif args.net_mode == 'three_channels':
        net = semantic_segmentation_models.deeplabv2_resnet101(num_classes=args.num_classes,
                                                                               pretrained_backbone=False).to(device)
    else:
        raise ValueError('net mode does not exist.')
    restart_epoch = 0
    best_score = 0
    lowest_val_loss = 1000
    if args.load_model:
        load_checkpoint = torch.load(os.path.join(MODEL_ROOT_PATH, args.checkpoint_name))
        restart_epoch = load_checkpoint['epoch'] + 1
        print('loading trained model. start from epoch {}.'.format(restart_epoch))
        net.load_state_dict(load_checkpoint['sem_net_state_dict'])
        # if 'best_score' in load_checkpoint:
        #     best_score = load_checkpoint['best_score']
        if 'lowest_val_loss' in load_checkpoint:
            lowest_val_loss = load_checkpoint['lowest_val_loss']

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_index, reduction='mean')
    loss_dict = {'train_loss': [], 'epoch_counter_ratio': []}
    for epoch in range(restart_epoch, restart_epoch+args.epochs):
        print("--------START TRAINING [EPOCH: {}]--------".format(epoch))
        seg_train(net, train_dataloader, loss_function, optimizer, device, visualizer, epoch, loss_dict)
        # torch.save({
        #     'epoch': epoch,
        #     'sem_net_state_dict': net.state_dict(),
        #     'best_score': best_score,
        # }, os.path.join(MODEL_ROOT_PATH, args.checkpoint_name))
        mean_iu, val_loss = seg_validate(args, net, val_dataloader, loss_function, device, visualizer)
        scheduler.step(val_loss)

        if val_loss < lowest_val_loss:
            print('val loss reduced from {} to {}! Saving...'.format(lowest_val_loss, val_loss))
            print('mean iou score: '+str(mean_iu))
            lowest_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'sem_net_state_dict': net.state_dict(),
                'val_loss': lowest_val_loss,
            }, os.path.join(MODEL_ROOT_PATH, args.checkpoint_name))
        else:
            print('Model not improved.')


if __name__ == '__main__':
    args_ = seg_parse().parse_args()
    seg_main(args_)