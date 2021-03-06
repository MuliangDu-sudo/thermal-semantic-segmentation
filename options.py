import argparse


def train_parse():
    parser = argparse.ArgumentParser(description='train options')
    parser.add_argument('-batch_size', type=int, default=4, help='batch size for training.')
    parser.add_argument('-sem_loss', default=False, help='use semantic consistency loss.')
    parser.add_argument('-load_model', default=False, help='train with pretrained model.')
    parser.add_argument('-checkpoint_name', type=str, default='freiburg_rgb2ir.pth', help='the name of trained model.')
    parser.add_argument('-new_checkpoint_name', type=str, default='cityscapes_rgb2freiburg_ir.pth',
                        help='the name of trained model. If don\'t want to create new checkpoint, use same name as checkpoint_name.')
    parser.add_argument('-source_dataset', type=str, default='kitti', help='which dataset as source.')
    parser.add_argument('-target_dataset', type=str, default='freiburg_ir', help='which dataset as target.')
    parser.add_argument('-lr', type=float, default=0.0001, help='learning rate.')
    parser.add_argument('-num_epoch', type=int, default=65, help='number of training epoch.')
    parser.add_argument('-grayscale', type=bool, default=False, help='convert image to grayscale.')
    parser.add_argument('-s2t_input_nc', type=int, default=3, help='number of s2t generator input channel.')
    parser.add_argument('-t2s_input_nc', type=int, default=1, help='number of t2s generator input channel.')
    parser.add_argument('-normalize', type=tuple, default=(0.5, 0.5, 0.5), help='normalization in source data transform. '
                                                                        'single value for one channel, triple for three.')
    parser.add_argument('-with_contour', type=bool, default=False, help='includes contour loss')
    parser.add_argument('-canny_thermal_threshold', type=float, default=1, help='canny edge detector threshold for thermal')
    parser.add_argument('-canny_rgb_threshold', type=float, default=2.5, help='canny edge detector threshold for rgb')
    return parser


def translation_parse():
    parser = argparse.ArgumentParser(description='translation options')
    # parser.add_argument('--root', type=str, default='/data/data_bank/muliang_gp/Prototypical',
    #                     help='pseudo label update thred')
    parser.add_argument('--root', type=str, default='',
                                             help='pseudo label update thred')
    parser.add_argument('-dataset', type=str, default='Cityscapes', help='dataset to be translated.')
    parser.add_argument('-batch_size', type=int, default=1, help='batch size for translation.')
    parser.add_argument('-checkpoint_name', type=str, default='cityscapes_rgb2freiburg_ir.pth',
                        help='the name of trained model.')
    parser.add_argument('-input_nc', type=int, default=3, help='number of generator input channel.')
    parser.add_argument('-output_nc', type=int, default=1, help='number of generator output channel.')
    parser.add_argument('-save_image_size', type=tuple, default=(1024, 2048), help='images are save with this size.')
    parser.add_argument('-normalize', type=tuple, default=(0.5, 0.5, 0.5),
                        help='normalization of input data.')
    parser.add_argument('-denormalize', type=tuple, default=(0.5, ),
                        help='denormalization of output data.')
    parser.add_argument('-grayscale', type=bool, default=False, help='convert image to grayscale.')
    parser.add_argument('-generator_type', type=str, default='s2t', help='the type of generator.')
    parser.add_argument('-translation_name_suffix', type=str, default='cityscapes', help='suffix of translation name.'
                                                                                         'no suffix is ''.')
    return parser


def seg_parse():
    parser = argparse.ArgumentParser(description='segmentation options')
    parser.add_argument('--root', type=str, default='/data/data_bank/muliang_gp/Prototypical',
                        help='pseudo label update thred')
    parser.add_argument('-load_model', default=True, help='train with pretrained model.')
    parser.add_argument('-epochs', default=50, help='number of epochs to train.')
    parser.add_argument('-batch_size', type=int, default=8, help='batch size.')
    parser.add_argument('-val_batch_size', type=int, default=8, help='validation batch size.')
    parser.add_argument('-checkpoint_name', type=str, default='256_cityscapes_rgb2freiburg_ir_segmentation.pth',
                        help='the name of trained model.')
    parser.add_argument('-new_checkpoint_name', type=str, default='256_cityscapes_rgb2freiburg_ir_segmentation.pth',
                        help='the name of trained model. If don\'t want to create new checkpoint, use same name as checkpoint_name.')
    parser.add_argument('-num_samples_show', type=int, default=3, help='number of samples to show in visdom.')
    parser.add_argument('-net_mode', type=str, default='one_channel', help='select input channel number of the net (1 or 3).')
    parser.add_argument('-dataset', type=str, default='cityscapes_translation', help='select the dataset.')
    parser.add_argument('-num_classes', type=int, default=13, help='number of classes.')
    parser.add_argument('-lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('-data_split', type=bool, default=False, help='whether to split dataset.')
    parser.add_argument('-translation_name', type=str, default='freiburg_rgb2ir_130epochs', help='the folder name of translations.')
    parser.add_argument('-visualize_prediction', type=bool, default=False, help='whether to save visualized prediction.')
    parser.add_argument('-ignore_index', type=int, default=12, help='ignore index. cityscapes is 255; freiburg is 12.')
    parser.add_argument('-generator_type', type=str, default='s2t', help='the type of generator.')
    parser.add_argument('-t2s_folder', type=str, default='test_cityscapes_rgb2freiburg_ir/')
    parser.add_argument('-baseline', type=bool, default=False,
                        help='whether this is an evaluation of baseline approach.')
    parser.add_argument('-source_domain', type=str, default='Thermal', help='source domain for baseline evaluation.')
    parser.add_argument('-target_domain', type=str, default='Grayscale', help='target domain for baseline evaluation.')
    parser.add_argument('-with_feat', type=bool, default=False, help='whether to extract feature')
    parser.add_argument('-logdir', type=str, default='./logs')
    return parser


def evaluation_parse():
    parser = argparse.ArgumentParser(description='segmentation options')
    parser.add_argument('--root', default='')
    parser.add_argument('-val_batch_size', type=int, default=1, help='validation batch size.')
    parser.add_argument('-checkpoint_name', type=str, default='256_freiburg_rgb2ir_segmentation.pth',
                        help='the name of trained model.')
    parser.add_argument('-num_samples_show', type=int, default=3, help='number of samples to show in visdom.')
    parser.add_argument('-net_mode', type=str, default='one_channel', help='select input channel number of the net (1 or 3).')
    parser.add_argument('-dataset', type=str, default='freiburg_ir', help='select the dataset.')
    parser.add_argument('-grayscale', type=bool, default=False, help='convert image to grayscale.')
    parser.add_argument('-num_classes', type=int, default=13, help='number of classes.')
    parser.add_argument('-data_split', type=bool, default=False, help='whether to split dataset.')
    parser.add_argument('-translation_name', type=str, default='cityscapes_rgb2freiburg_ir', help='the folder name of translations.')
    parser.add_argument('-visualize_prediction', default=None, help='whether to save visualized prediction.')
    parser.add_argument('-ignore_index', type=int, default=12, help='ignore index. cityscapes is 255; freiburg is 12.')
    parser.add_argument('-generator_type', type=str, default='s2t', help='the type of generator.')
    parser.add_argument('-t2s_folder', type=str, default='test_cityscapes_rgb2freiburg_ir/')
    parser.add_argument('-baseline', type=bool, default=False, help='whether this is an evaluation of baseline approach.')
    parser.add_argument('-source_domain', type=str, default='Thermal', help='source domain for baseline evaluation.')
    parser.add_argument('-target_domain', type=str, default='Grayscale', help='target domain for baseline evaluation.')
    return parser

def calc_proto_parse():
    parser = argparse.ArgumentParser(description='prototype computation options.')
    parser.add_argument('-normalize', type=tuple, default=(0.5, ), help='normalization of input data.')
    parser.add_argument('-net_mode', type=str, default='one_channel',
                        help='select input channel number of the net (1 or 3).')
    parser.add_argument('-dataset', type=str, default='freiburg_ir', help='which dataset as target.')
    parser.add_argument('-num_classes', type=int, default=13, help='number of classes.')
    parser.add_argument('-root', type=str, default='', help='path for saving prototypes.')
    parser.add_argument('-epochs', default=4, help='number of epochs to train.')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('-checkpoint_name', type=str, default='freiburg_rgb2ir_cityscapes_segmentation.pth',
                        help='the name of trained model.')
    parser.add_argument('-with_feat', type=bool, default=True, help='whether to extract feature')
    return parser

def pseudo_generation_parse():
    parser = argparse.ArgumentParser(description='pseudo label generation options.')
    return parser