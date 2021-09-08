import torch
from models import generators
from main import device, MODEL_ROOT_PATH
from data import Cityscapes, Freiburg, FreiburgTest
from torch.utils.data import DataLoader
import os
from torchvision import transforms as T
from utils import transforms as TT
from train import Denormalize
from options import translation_parse
from tqdm import tqdm
from PIL import Image

def translate(args):

    source_translate_transform = T.Compose([
        T.Resize((256, 512)),
        T.ToTensor(),
        T.Normalize(args.normalize, args.normalize)
    ])

    source_reverse_transform = T.Compose([
        Denormalize(args.denormalize, args.denormalize),
        T.Resize(args.save_image_size),
        T.ToPILImage()
    ])

    if args.dataset == 'Cityscapes':
        translate_datasets = Cityscapes('datasets/source_dataset', transforms=source_translate_transform, train_mode=False)
    elif args.dataset == 'freiburg_rgb':
        translate_datasets = Freiburg('datasets/freiburg', split='train', domain='RGB', grayscale=False, transforms=source_translate_transform,
                                      with_label=False, translation_mode=True,
                                      translation_name=args.checkpoint_name.replace('.pth', '')+'_'+args.translation_name_suffix)
    elif args.dataset == 'freiburg_ir':
        translate_datasets = Freiburg('datasets/freiburg', split='train', domain='IR', grayscale=False, transforms=source_translate_transform,
                                      with_label=False, translation_mode=True, translation_name=args.checkpoint_name.replace('.pth', '')+'_2rgb')
    elif args.dataset == 'freiburg_test':
        translate_datasets = FreiburgTest('datasets/freiburg', split='test', domain='RGB', transforms=source_translate_transform,
                                      with_label=True, transform_label=False, grayscale=args.grayscale)
    elif args.dataset == 'freiburg_test_t2s':
        translate_datasets = FreiburgTest('datasets/freiburg', split='test', domain='IR', transforms=source_translate_transform,
                                      with_label=True, transform_label=False)
    else:
        raise ValueError('dataset does not exist.')
    translate_dataloader = DataLoader(translate_datasets, batch_size=args.batch_size,
                                      shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    net_g = generators.unet_256(ngf=64, input_nc=args.input_nc, output_nc=args.output_nc).to(device)
    load_checkpoint = torch.load(os.path.join(MODEL_ROOT_PATH, args.checkpoint_name))
    net_g.load_state_dict(load_checkpoint['net_g_{}_state_dict'.format(args.generator_type)])
    net_g.eval()
    print('start translating.')
    if args.dataset == 'freiburg_test':
        save_root_path = 'datasets/freiburg/translations/test_' + args.checkpoint_name.replace('.pth', '')
        for i, [images, labels] in enumerate(translate_dataloader):
            images = images.to(device)
            translations = net_g(images).squeeze(dim=0)
            translations = source_reverse_transform(translations)
            labels = T.ToPILImage()(labels.squeeze_(1))
            if not os.path.exists(save_root_path):
                os.makedirs(save_root_path)
            translations.save(os.path.join(save_root_path, str(i) + '_translation.jpg'))
            labels.save(os.path.join(save_root_path, str(i) + '_groundtruth.png'))

            if i % 100 == 0:
                print('translation: [{}/{}]'.format(i, len(translate_dataloader)))

    elif args.dataset == 'freiburg_test_t2s':
        save_root_path = 'datasets/freiburg/translations/t2s/test_' + args.checkpoint_name.replace('.pth', '')
        for i, [images, labels] in enumerate(translate_dataloader):
            images = images.to(device)
            translations = net_g(images).squeeze(dim=0)
            translations = source_reverse_transform(translations)
            labels = T.ToPILImage()(labels.squeeze_(1))
            if not os.path.exists(save_root_path):
                os.makedirs(save_root_path)
            translations.save(os.path.join(save_root_path, str(i) + '_translation.jpg'))
            labels.save(os.path.join(save_root_path, str(i) + '_groundtruth.png'))

            if i % 100 == 0:
                print('translation: [{}/{}]'.format(i, len(translate_dataloader)))
    else:
        for i, [images, image_names] in enumerate(translate_dataloader):
            images = images.to(device)
            image_name = image_names[0]
            translations = net_g(images).squeeze(dim=0)
            translations = source_reverse_transform(translations)
            path_split = image_name.split("/")[:-1]     # to extract the path to save translation image
            image_save_path = "/".join(path_split)
            image_save_path = os.path.join(args.root, image_save_path)
            if not os.path.exists(image_save_path):
                os.makedirs(image_save_path)
            translations.save(os.path.join(args.root, image_name))

            if i % 100 == 0:
                print('translation: [{}/{}]'.format(i, len(translate_dataloader)))


def convert_freiburg(args):
    source_translate_transform = T.Compose([
        T.Resize(size=(256, 512)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    source_reverse_transform = T.Compose([
        Denormalize((0.5,), (0.5, )),
        T.Resize(size=args.save_image_size),
        T.ToPILImage()
    ])

    datasets = Freiburg('datasets/freiburg', split='train', domain='IR', grayscale=False,
                                  transforms=source_translate_transform,
                                  with_label=False, translation_mode=True,
                                  translation_name='convert')

    rgb_datasets = Freiburg('datasets/freiburg', split='train', domain='RGB', grayscale=False,
                                  transforms=source_translate_transform,
                                  with_label=False, translation_mode=True,
                                  translation_name='convert')

    dataloader = DataLoader(datasets, batch_size=args.batch_size,
                                      shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    rgb_dataloader = DataLoader(rgb_datasets, batch_size=args.batch_size,
                                      shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    for i, [images, image_names] in enumerate(dataloader):
        images = images.to(device)[0]
        image_name = image_names[0]
        images = source_reverse_transform(images)
        path_split = image_name.split("/")[:-1]     # to extract the path to save translation image
        image_save_path = "/".join(path_split)
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        images.save(os.path.join(image_name))

        if i % 100 == 0:
            print('translation: [{}/{}]'.format(i, len(dataloader)))


if __name__ == '__main__':
    trans_args = translation_parse().parse_args()
    translate(trans_args)
    # convert_freiburg(trans_args)

