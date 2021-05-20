import argparse


def train_parse():
    parser = argparse.ArgumentParser(description='train options')
    parser.add_argument('-batch_size', type=int, default=1, help='batch size for training.')
    parser.add_argument('-sem_loss', default=False, help='use semantic consistency loss.')
    parser.add_argument('-load_model', default=True, help='train with pretrained model.')
    parser.add_argument('-checkpoint_name', type=str, default='without_normalization.pth', help='the name of trained model.')
    return parser

