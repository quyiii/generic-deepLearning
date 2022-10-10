import torch
import h5py
import json
import os

from .transform import get_transform
from lib.data import BaseDataset

class CaptionDataset(BaseDataset):
    def __init__(self, cfg, split='train'):
        super(CaptionDataset, self).__init__(cfg)
        self.split = split.upper()
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        imgs_path = os.path.join(self.cfg.DATASET.ROOTDIR, self.split + '_IMAGES.hdf5')
        self.h = h5py.File(imgs_path, 'r')
        self.imgs = self.h['images']
        self.cpi = self.h.attrs['captions_per_image']

        caps_path = os.path.join(self.cfg.DATASET.ROOTDIR, self.split + '_CAPTIONS.json')
        with open(caps_path, 'r') as j:
            self.captions = json.load(j)
        
        caplens_path = os.path.join(self.cfg.DATASET.ROOTDIR, self.split + '_CAPLENS.json')
        with open(caplens_path, 'r') as j:
            self.caplens = json.load(j)

        self.dataset_size = len(self.captions) 
        
        self.transform = get_transform(cfg, params=None, grayscale=(self.INPUT.CHANNEL==1))
        