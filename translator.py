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
    source_translate_transform = TT.Compose([
        TT.Resize((256, 512)),
        TT.ToTensor(),
        TT.Normalize(args.normalize, args.normalize)
    ])

    source_reverse_transform = T.Compose([
        Denormalize(args.denormalize, args.denormalize),
        T.Resize(args.save_image_size),
        T.ToPILImage()
    ])

    if args.dataset == 'Cityscapes':
        translate_datasets = Cityscapes('datasets/source_dataset', transforms=source_translate_transform,
                                        train_mode=False)
    elif args.dataset == 'freiburg_rgb':
        translate_datasets = Freiburg('datasets/freiburg', split='train', domain='RGB', grayscale=False,
                                      transforms=source_translate_transform,
                                      with_label=False, translation_mode=True,
                                      translation_name=args.checkpoint_name.replace('.pth',
                                                                                    '') + '_' + args.translation_name_suffix)
    elif args.dataset == 'freiburg_ir':
        translate_datasets = Freiburg('datasets/freiburg', split='train', domain='IR', grayscale=False,
                                      transforms=source_translate_transform,
                                      with_label=False, translation_mode=True,
                                      translation_name=args.checkpoint_name.replace('.pth', '') + '_2rgb')
    elif args.dataset == 'freiburg_test':
        translate_datasets = FreiburgTest(args=args, root='datasets/freiburg', split='test', domain='RGB',
                                          transforms=source_translate_transform,
                                          with_label=True)
    elif args.dataset == 'freiburg_test_t2s':
        translate_datasets = FreiburgTest('datasets/freiburg', split='test', domain='IR',
                                          transforms=source_translate_transform,
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
        for i, data_i in enumerate(translate_dataloader):
            images = data_i['image'].to(device)
            translations = net_g(images).squeeze(dim=0)
            translations = source_reverse_transform(translations)
            # labels = T.ToPILImage()(data_i['label'].squeeze_(1))
            if not os.path.exists(save_root_path):
                os.makedirs(save_root_path)
