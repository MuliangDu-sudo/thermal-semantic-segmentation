from models import thermal_semantic_segmentation_models, semantic_segmentation_models, Deeplab
import torch
from utils import transforms as T
from torchvision import transforms as TT
import os
from PIL import ImageFile
from utils import AverageMeter, freiburg_prediction_visualize, freiburg_palette
import time
from data import CityscapesTranslation, Cityscapes, FreiburgTest, Freiburg, FreiburgT2S, FreiburgTranslation
from torch.utils.data import DataLoader
import visdom
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from utils.eval_tools import evaluate
from options import evaluation_parse
from PIL import ImageFile, Image
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_ROOT_PATH = './checkpoints/semantic_segmentation'


def seg_validate(args, sem_net, val_data, loss_func, device, vis=None, num_classes=13):
    print('validating...')
    val_loss = AverageMeter('val_loss', ':3.4f')
    sem_net.eval()
    prediction_list, label_list = [], []
    print(len(val_data))
    # random_id = np.random.choice(len(val_data), args.num_samples_show)
    # i = 0
    for i, item in enumerate(tqdm(val_data)):
        image = item['image'].to(device)
        label = item['label'].to(device)
        # if all([args.baseline, args.target_domain == 'Grayscale', args.source_domain == 'RGB']) \
        #         or all([args.baseline, args.target_domain == 'Thermal', args.source_domain == 'RGB']):
        #     image = image.expand(-1, 3, -1, -1)
        outputs = sem_net(image)
        outputs = torch.nn.Upsample(size=(256, 512), mode='bilinear', align_corners=True)(outputs['out'])
        loss = loss_func(outputs, label)

        predictions = outputs.max(1)[1].squeeze_(1).cpu().numpy()
        label_list.append(label.cpu().numpy())
        prediction_list.append(predictions)
        val_loss.update(loss.item(), image.size(0))
        # if i in random_id:
        #     vis.images(image[0], win='image [{}]'.format(i), padding=2,
        #                opts=dict(title='image [{}]'.format(i), caption='image [{}]'.format(i)))
        #     vis.images(np.uint8(label[0].cpu().numpy()), win='label [{}]'.format(i), padding=2,
        #                opts=dict(title='label [{}]'.format(i), caption='label [{}]'.format(i)))
        #     vis.images(np.uint8(predictions[0]), win='prediction [{}]'.format(i), padding=2,
        #                opts=dict(title='prediction [{}]'.format(i), caption='prediction [{}]'.format(i)))
        # i += 1
        if args.visualize_prediction is not None:
            save_path_root = os.path.join(args.root, 'predictions/{}'.format(args.new_checkpoint_name.replace('.pth', '')))
            if args.baseline:
                save_path_root = 'baseline_predictions/apply_{}_image_on_{}_domain_model'.format(args.target_domain, args.source_domain,)
            if args.generator_type == 't2s':
                save_path_root = 'predictions/t2s/{}'.format(args.checkpoint_name.replace('.pth', ''))
            if not os.path.exists(save_path_root):
                os.makedirs(save_path_root)
            if args.visualize_prediction == 'save_all' and i % 1 == 0:
                new_mask = freiburg_prediction_visualize(predictions[0], freiburg_palette())
                label = freiburg_prediction_visualize(label[0].squeeze_(1).cpu().numpy(), freiburg_palette())
                image = TT.ToPILImage()(image[0])
                new_mask.save(os.path.join(save_path_root, str(i)+'_prediction.png'))
                image.save(os.path.join(save_path_root, str(i)+'_image.png'))
                label.save(os.path.join(save_path_root, str(i) + '_groundtruth.png'))
            elif args.visualize_prediction == 'save_one' and i == 0:
                new_mask = freiburg_prediction_visualize(predictions[0], freiburg_palette())
                label = freiburg_prediction_visualize(label[0].squeeze_(1).cpu().numpy(), freiburg_palette())
                image = TT.ToPILImage()(image[0])
                new_mask.save(os.path.join(save_path_root, str(i)+'_prediction_{}.png'.format(args.iter_counter)))
                image.save(os.path.join(save_path_root, str(i)+'_image.png'))
                label.save(os.path.join(save_path_root, str(i) + '_groundtruth.png'))

    label_list = np.concatenate(label_list)
    prediction_list = np.concatenate(prediction_list)
    acc, acc_cls, mean_iu, fwavacc, cls_iu = evaluate(prediction_list, label_list, num_classes)
    return mean_iu, val_loss.avg, cls_iu


def seg_evaluation(args):
    print('evaluating...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualizer = visdom.Visdom(env='thermal semantic segmentation')

    train_transform = T.Compose([
        #T.RandomResizedCrop(size=(256, 512), ratio=(1.5, 8 / 3.), scale=(0.5, 1.)),
        T.Resize((512, 256)),
        # T.RandomHorizontalFlip(),
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

    elif args.dataset == 'freiburg_translation':
        source_dataset = FreiburgTranslation(root='datasets/freiburg/translations/',
                                             folder='test_'+args.checkpoint_name.replace('_segmentation.pth', '')+'/',
                                             transforms=train_transform)
    elif args.dataset == 'freiburg_rgb':
        source_dataset = FreiburgTest('datasets/freiburg', split='test', domain='RGB', transforms=train_transform,
                                      with_label=True, grayscale=args.grayscale)
    elif args.dataset == 'freiburg_ir':
        source_dataset = FreiburgTest(args=args, root='datasets/freiburg', split='test', domain='IR', transforms=train_transform,
                                  with_label=True)
    elif args.dataset == 'freiburg_t2s':
        source_dataset = FreiburgT2S(folder=args.t2s_folder, transforms=train_transform)
    else:
        raise ValueError('dataset does not exist.')
    # Creating data indices for training and validation splits:
    if args.data_split:
        dataset_size = len(source_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        valid_sampler = SubsetRandomSampler(val_indices)
        val_dataloader = DataLoader(source_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=2,
                                    pin_memory=True,
                                    drop_last=True, sampler=valid_sampler)
    else:
        val_dataloader = DataLoader(source_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=2,
                                    pin_memory=True,
                                    drop_last=True)
    if args.net_mode == 'one_channel':
        # net = thermal_semantic_segmentation_models.deeplabv2_resnet101_thermal(num_classes=args.num_classes,
        #                                                                        pretrained_backbone=False).to(device)
        net = Deeplab(torch.nn.BatchNorm2d, num_classes=13, num_channels=1, freeze_bn=False, get_feat=True).to(device)
    elif args.net_mode == 'three_channels':
        net = semantic_segmentation_models.deeplabv2_resnet101(num_classes=args.num_classes,
                                                                               pretrained_backbone=False).to(device)
    else:
        raise ValueError('net_mode not exist.')
    load_checkpoint = torch.load(os.path.join(MODEL_ROOT_PATH, args.checkpoint_name))
    net.load_state_dict(load_checkpoint['sem_net_state_dict'])

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=13, reduction='mean')

    mean_iu, avg_loss, class_iou = seg_validate(args, net, val_dataloader, loss_function, device, visualizer)
    print('checkpoint name: '+args.checkpoint_name)
    print('mean iou score: [{}]. val_loss: [{}]'.format(mean_iu, avg_loss))
    for k, v in class_iou.items():
        fmt_str = 'target set class {}: {}'.format(k, v)
        print(fmt_str)

if __name__ == '__main__':
    args_ = evaluation_parse().parse_args()
    seg_evaluation(args_)
