import torch
import h5py
import json
import os

from lib.data import BaseDataset

class CaptionDataset(BaseDataset):
    def __init__(self, cfg, split='train'):
        super(CaptionDataset, self).__init__(cfg)
        self.split = split.upper()
        
