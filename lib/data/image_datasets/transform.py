import os
import random
from PIL import Image
import torchvision.transforms as transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image(file_name):
    return any(file_name.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths(dir, max_size=float('inf')):
    image_paths = []
    assert os.path.isdir(dir)
    # os.walk 遍历并输出指定目录下的所有子目录以及文件 返回三元组(root dirs files)
    # root 当前正在遍历的这个文件夹的地址
    # dirs 一个list 该文件夹下所有目录的名字 deep=1
    # files 一个list 该文件夹下所有文件的名字 deep=1
    # 遍历顺序在默认情况下 从输入的目录开始
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image(fname):
                path = os.path.join(root, fname)
                image_paths.append(path)
    return image_paths[:min(len(image_paths), max_size)]

def get_image(path):
    # W * H
    return Image.open(path).convert('RGB')

def get_params(cfg, size):
    w, h = size
    new_h = h
    new_w= w
    if cfg.PROCESS.RESIZE and cfg.PROCESS.CROP:
        new_w = cfg.INPUT.SIZE[0]
        new_h = cfg.INPUT.SIZE[1]
    
    x = random.randint(0, max(0, new_w - cfg.PROCESS.CROP_SIZE[0]))
    y = random.randint(0, max(0, new_h - cfg.PROCESS.CROP_SIZE[1]))

    flip = random.random() > cfg.PROCESS.FLIP_P
    return {'crop_pos': (x, y), 'flip': flip}

def tensorToPIL(img_tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(img_tensor)

def get_transform(cfg, params=None, grayscale=False,
                    method=transforms.InterpolationMode.BICUBIC):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if cfg.PROCESS.RESIZE:
        transform_list.append(transforms.Resize(cfg.INPUT.SIZE, method))
    if cfg.PROCESS.CROP:
        if params is None:
            transform_list.append(transforms.RandomCrop(cfg.PROCESS.CROP_SIZE))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], cfg.PROCESS.CROP_SIZE)))
    if cfg.PROCESS.FLIP:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        else:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    if cfg.PROCESS.TOTENSOR:
        transform_list.append(transforms.ToTensor())
        if grayscale:
            transform_list.append(transforms.Normalize(cfg.INPUT.GRAY_MEAN, cfg.INPUT.GRAY_STD))
        else:
            transform_list.append(transforms.Normalize(cfg.INPUT.MEAN, cfg.INPUT.STD))
    return transforms.Compose(transform_list)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw, th =size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1+tw, y1+tw))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
