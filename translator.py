import torch
from models import generators
from main import device, MODEL_ROOT_PATH
from data import Cityscapes, Freiburg
from torch.utils.data import DataLoader
import os
from torchvision import transforms as T
from train import Denormalize
from options import translation_parse
from tqdm import tqdm


def translate(args):

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
    if args.dataset == 'Cityscapes':
        translate_datasets = Cityscapes('datasets/source_dataset', transforms=source_translate_transform, train_mode=False)
    elif args.dataset == 'freiburg_rgb':
        translate_datasets = Freiburg('datasets/freiburg', split='train', domain='RGB', grayscale=True, transforms=source_translate_transform,
                                      with_label=False, translation_mode=True, translation_name=args.checkpoint_name.replace('.pth', ''))
    else:
        raise ValueError('dataset does not exist.')
    translate_dataloader = DataLoader(translate_datasets, batch_size=args.batch_size,
                                      shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    net_g_s2t = generators.unet_256(ngf=64, input_nc=args.input_nc, output_nc=args.output_nc).to(device)
    load_checkpoint = torch.load(os.path.join(MODEL_ROOT_PATH, args.checkpoint_name))
    net_g_s2t.load_state_dict(load_checkpoint['net_g_s2t_state_dict'])
    net_g_s2t.eval()
    print('start translating.')
    for i, [images, image_names] in enumerate(translate_dataloader):
        images = images.to(device)
        image_name = image_names[0]
        translations = net_g_s2t(images).squeeze(dim=0)
        translations = source_reverse_transform(translations)
        path_split = image_name.split("/")[:-1]     # to extract the path to save translation image
        image_save_path = "/".join(path_split)
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        translations.save(os.path.join(image_name))

        if i % 100 == 0:
            print('translation: [{}/{}]'.format(i, len(translate_dataloader)))


if __name__ == '__main__':
    trans_args = translation_parse().parse_args()
    translate(trans_args)

