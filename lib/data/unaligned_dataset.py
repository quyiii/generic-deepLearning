import os
from data.base_dataset import BaseDataset

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets 
    """

    def __init__(self, args):
        super(UnalignedDataset, self).__init__(args)
        self.dir_A = os.path.join(args.DATASET.ROOT_DIR, args.)