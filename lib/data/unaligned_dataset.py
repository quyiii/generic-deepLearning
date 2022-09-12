import os
from data.image_folder import get_image_paths 
from data.base_dataset import BaseDataset

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
            self.A_paths = sorted(get_image_paths(slef.dir_A, cfg.DATASET.MAX_SIZE))
            self.B_paths = sorted(get_image_paths(slef.dir_B, cfg.DATASET.MAX_SIZE))
            self.A_size = len(self.A_paths)
            self.B_size = len(self.B_paths)
            # AtoB
            input_nc = self.cfg.INPUT.CHANNEL if cfg.DATASET.DIRECTION else self.cfg.OUTPUT.CHANNEL
            output_nc = self.cfg.OUTPUT.CHANNEL if cfg.DATASET.DIRECTION else self.cfg.INPUT.CHANNEL
            self.transform_A = get_trasn
        else:
            raise NotImplementError("unaligned dataset don't support data type {}".format(cfg.INPUT.TYPE))
        