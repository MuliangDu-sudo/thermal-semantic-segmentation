import argparse


def train_parse():
    parser = argparse.ArgumentParser(description='train options')
    parser.add_argument('-batch_size', type=int, default=1, help='batch size for training.')
    parser.add_argument('-sem_loss', default=False, help='use semantic consistency loss.')
    parser.add_argument('-load_model', default=False, help='train with pretrained model.')
    parser.add_argument('-checkpoint_name', type=str, default='with_normalization.pth', help='the name of trained model.')
    return parser


def translation_parse():
    parser = argparse.ArgumentParser(description='translation options')
    parser.add_argument('-batch_size', type=int, default=1, help='batch size for translation.')
    parser.add_argument('-checkpoint_name', type=str, default='without_sem.pth',
                        help='the name of trained model.')
    return parser


def seg_parse():
    parser = argparse.ArgumentParser(description='segmentation options')
    parser.add_argument('-load_model', default=False, help='train with pretrained model.')
    parser.add_argument('-epochs', default=30, help='number of epochs to train.')
    parser.add_argument('-batch_size', type=int, default=16, help='batch size for translation.')
    parser.add_argument('-checkpoint_name', type=str, default='freiburg_ir_semantic_segmentation_cropped_label.pth',
                        help='the name of trained model.')
    parser.add_argument('-num_samples_show', type=int, default=3, help='number of samples to show in visdom.')
    parser.add_argument('-net_mode', type=str, default='one_channel', help='select input channel number of the net (1 or 3).')
    parser.add_argument('-dataset', type=str, default='freiburg_ir', help='select the dataset.')
    parser.add_argument('-num_classes', type=int, default=13, help='number of classes.')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    return parser
