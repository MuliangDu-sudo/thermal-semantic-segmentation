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
    parser.add_argument('-load_model', default=True, help='train with pretrained model.')
    parser.add_argument('-epochs', default=10, help='number of epochs to train.')
    parser.add_argument('-batch_size', type=int, default=1, help='batch size for translation.')
    parser.add_argument('-checkpoint_name', type=str, default='semantic_segmentation.pth',
                        help='the name of trained model.')
    return parser
