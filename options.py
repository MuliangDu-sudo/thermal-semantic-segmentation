import argparse


def train_parse():
    parser = argparse.ArgumentParser(description='train options')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size for training.')
    parser.add_argument('-sem_loss', default=False, help='use semantic consistency loss.')
    parser.add_argument('-load_model', default=False, help='train with pretrained model.')
    parser.add_argument('-checkpoint_name', type=str, default='freiburg_rgb2ir_2.pth', help='the name of trained model.')
    parser.add_argument('-source_dataset', type=str, default='freiburg_rgb', help='which dataset as source.')
    parser.add_argument('-target_dataset', type=str, default='freiburg_ir', help='which dataset as target.')
    parser.add_argument('-lr', type=float, default=0.0001, help='learning rate.')
    parser.add_argument('-num_epoch', type=int, default=100, help='number of training epoch.')
    parser.add_argument('-grayscale', type=bool, default=False, help='convert image to grayscale.')
    parser.add_argument('-s2t_input_nc', type=int, default=3, help='number of s2t generator input channel.')
    parser.add_argument('-t2s_input_nc', type=int, default=1, help='number of t2s generator input channel.')
    parser.add_argument('-normalize', type=tuple, default=(0.5, 0.5, 0.5), help='normalization in source data transform. '
                                                                        'single value for one channel, triple for three.')
    parser.add_argument('-with_contour', type=bool, default=False, help='includes contour loss')

    return parser


def translation_parse():
    parser = argparse.ArgumentParser(description='translation options')
    parser.add_argument('-dataset', type=str, default='freiburg_rgb', help='dataset to be translated.')
    parser.add_argument('-batch_size', type=int, default=1, help='batch size for translation.')
    parser.add_argument('-checkpoint_name', type=str, default='freiburg_rgb2ir_2.pth',
                        help='the name of trained model.')
    parser.add_argument('-input_nc', type=int, default=3, help='number of generator input channel.')
    parser.add_argument('-output_nc', type=int, default=1, help='number of generator output channel.')
    parser.add_argument('-save_image_size', type=tuple, default=(320, 700), help='images are save with this size.')
    return parser


def seg_parse():
    parser = argparse.ArgumentParser(description='segmentation options')
    parser.add_argument('-load_model', default=False, help='train with pretrained model.')
    parser.add_argument('-epochs', default=50, help='number of epochs to train.')
    parser.add_argument('-batch_size', type=int, default=16, help='batch size.')
    parser.add_argument('-val_batch_size', type=int, default=1, help='validation batch size.')
    parser.add_argument('-checkpoint_name', type=str, default='freiburg_rgb2ir_2_segmentation.pth',
                        help='the name of trained model.')
    parser.add_argument('-num_samples_show', type=int, default=3, help='number of samples to show in visdom.')
    parser.add_argument('-net_mode', type=str, default='one_channel', help='select input channel number of the net (1 or 3).')
    parser.add_argument('-dataset', type=str, default='freiburg_translation', help='select the dataset.')
    parser.add_argument('-num_classes', type=int, default=13, help='number of classes.')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-data_split', type=bool, default=False, help='whether to split dataset.')
    parser.add_argument('-translation_name', type=str, default='freiburg_rgb2ir_2', help='whether to split dataset.')
    parser.add_argument('-visualize_prediction', type=bool, default=True, help='whether to save visualized prediction.')
    return parser
