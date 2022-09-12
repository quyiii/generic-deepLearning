import random
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import torchvision.transforms as transforms

class BaseDataset(Dataset, ABC):
    """
    This is an abstract base class for all datasets

    To create a subclass, you must implement these following functions
    __int__
    __len__
    __getitem__
    """

    def __init__(self, cfg):
        self.cfg = cfg


def get_params(cfg, size):
    w, h = size
    new_h = h
    new_w= w
    if cfg.PROCESS.RESIZE and cfg.PROCESS.CROP:
        new_w = cfg.INPUT.SIZE[0]
        new_h = cfg.INPUT.SIZE[1]
    
    x = random.randint(0, np.maximum(0, new_w - cfg.PROCESS.CROP_SIZE[0]))
    y = random.randint(0, np.maximum(0, new_h - cfg.PROCESS.CROP_SIZE[1]))

    flip = random.random() > cfg.PROCESS.FLIP_P
    return {'crop_pos': (x, y), 'flip': flip}
 
def get_transform(cfg, params=None, method=transforms.InterpolationMode.BICUBIC):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if cfg.PROCESS.RESIZE:
        transform_list.append(transforms.Resize(cfg.INPUT.SIZE, method))
    if cfg.PROCESS.CROP:
        if params is None:
            transform_list.append(transforms.RandomCrop(cfg.PROCESS.CROP_SIZE))
        else:
            transform_list.append(transforms.Lambda(Lambda img: __crop(img, params['crop_pos'], cfg.PROCESS.CROP_SIZE)))
    if cfg.PROCESS.FLIP:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        else:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    if cfg.TOTENSOR:
        transform_list.append(transforms.ToTensor())
        if cfg.GRAYSCALE:
            transform_list.append(transforms.Normalize(cfg.INPUT.GRAY_MEAN, cfg.INPUT.GRAY_STD))
        else:
            transform_list.append(transforms.Normalize(cfg.INPUT.MEAN, cfg.INPUT.STD))
    return transforms.Compose(transform_list)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th =size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1+tw, y1+tw))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
