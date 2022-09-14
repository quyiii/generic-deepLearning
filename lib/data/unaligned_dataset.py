import os
import random
from .image_folder import get_image_paths, get_image, get_transform
from .base_dataset import BaseDataset

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets 
    """

    def __init__(self, cfg):
        super(UnalignedDataset, self).__init__(cfg)
        self.phase = "train" if cfg.TRAIN.IS_TRAIN else "test"
        self.dir_A = os.path.join(cfg.DATASET.ROOT_DIR, self.phase+"A")
        self.dir_B = os.path.join(cfg.DATASET.ROOT_DIR, self.phase+"B")
        if cfg.INPUT.TYPE.lower() == 'image':
            self.A_paths = sorted(get_image_paths(slef.dir_A, float(cfg.DATASET.MAX_SIZE)))
            self.B_paths = sorted(get_image_paths(slef.dir_B, float(cfg.DATASET.MAX_SIZE)))
            self.A_size = len(self.A_paths)
            self.B_size = len(self.B_paths)
            # AtoB
            input_nc = self.cfg.INPUT.CHANNEL if cfg.DATASET.DIRECTION else self.cfg.OUTPUT.CHANNEL
            output_nc = self.cfg.OUTPUT.CHANNEL if cfg.DATASET.DIRECTION else self.cfg.INPUT.CHANNEL
            self.transform_A = get_transform(self.cfg, grayscale=(input_nc == 1))
            self.transform_B = get_transform(self.cfg, grayscale=(output_nc == 1))
        else:
            raise NotImplementError("unaligned dataset don't support data type {}".format(cfg.INPUT.TYPE))
        
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[random.randint(0, self.B_size-1)]
        A = self.transform_A(get_image(A_path))
        B = self.transform_B(get_image(B_path))
        return {'A': A, 'B':B, 'A_path': A_path, 'B_path': B_path}
    
    def __len__(self):
        return max(self.A_size, self.B_size)
        