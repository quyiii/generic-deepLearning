import os
from lib.data import BaseDataset
from .transform import get_params, get_transform, get_image_paths
from PIL import Image

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.
    
    paired image is a image consists of left: ground_truth_img and right: input_img
    """

    def __init__(self, cfg):
        super(AlignedDataset, self).__init__(cfg)
        self.direction = cfg.INPUT.DIRECTION
        self.phase = "train" if cfg.TRAIN.IS_TRAIN else "test"
        self.dir_AB = os.path.join(cfg.DATASET.ROOT_DIR, self.phase)
        self.AB_paths = sorted(get_image_paths(self.dir_AB, float(self.DATASET.MAX_SIZE)))
        assert(cfg.INPUT.SIZE >= cfg.PROCESS.CROP_SIZE)
        self.input_nc = cfg.INPUT.CHANNEL if self.direction else cfg.OUTPUT.CHANNEL
        self.output_nc = cfg.OUTPUT.CHANNEL if self.direction else cfg.INPUT.CHANNEL

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w/2)
        # A is right one(input)
        # crop (x0, y0, x1, y1)
        # w = x1 - x0  h = y1 - y0  so x:(x0, x1] same as y
        A = AB.crop((w2, 0, w, h))
        # B is left one(ground truth) 
        B = AB.crop((0, 0, w2, h))
        transform_params = get_params(self.cfg, A.size)
        A_transform = get_transform(self.cfg, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.cfg, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        return {'A':A, 'B':B, 'AB_path': AB_path} if self.direction else {'A':B, 'B':A, 'AB_path': AB_path}

    def __init__(self):
        return len(self.AB_paths)