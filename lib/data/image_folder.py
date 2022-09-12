import torch.utils.data as data
import os
from PIL import Image

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
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image(fname):
                path = os.path.join(root, fname)
                image_paths.append(path)
    return image_paths[:min(len(image_paths), max_size)]

def get_image(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):
    # loader means the tool load img by path
    def __init__(self, root, transform=None, return_paths=False, loader=get_image):
        img_paths = get_image_paths(root)
        if len(img_paths) == 0:
            raise(RuntimeError("Found 0 images in {}".format(root)))
        self.root = root
        self.img_paths = img_paths
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.image_paths)