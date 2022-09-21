from torch.utils.data import Dataset
from .transform import get_image, get_image_paths

class ImageFolder(Dataset):
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
        path = self.img_paths[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.img_paths)