import numpy as np


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def evaluate(predictions, gts, num_classes):
    print_class_list = {
        'freiburg': [
            "IoU road,parking",
            "IoU ground,sidewalk",
            "IoU building,",
            'IoU curb',
            'IoU fence',
            'IoU pole,traffic light,traffic sign',
            'IoU vegetation',
            'IoU terrain',
            'IoU sky',
            'IoU person,rider',
            'IoU car,truck,bus,train',
            'IoU motorcycle,bicycle'],
        'cityscapes': [
            "IoU road",
            "IoU sidewalk",
            "IoU building,",
            'IoU wall',
            'IoU fence',
            'IoU pole',
            'IoU traffic light',
            'IoU traffic sign',
            'IoU vegetation',
            'IoU terrain',
            'IoU sky',
            'IoU person',
            'IoU rider',
            'IoU car',
            'IoU truck',
            'IoU bus',
            'IoU train',
            'IoU motorcycle',
            'IoU bicycle']
        }

    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    if num_classes == 13:
        mean_iu = np.nanmean(iu[:11])
        cls_iu = dict(zip(print_class_list['freiburg'], iu))
    elif num_classes == 19:
        mean_iu = np.nanmean(iu)
        cls_iu = dict(zip(print_class_list['cityscapes'], iu))
    else:
        raise ValueError('invalid dataset for evaluation.')
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, cls_iu
