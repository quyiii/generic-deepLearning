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

        self.captions_len = len(self.captions) 
        
        self.transform = get_transform(cfg, params=None, grayscale=(self.INPUT.CHANNEL==1))

    def __getitem__(self, index):
        img = torch.FloatTensor(self.imgs[index // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)
        
        caption = torch.LongTensor(self.captions[index])
        caplen = torch.LongTensor([self.caplens[index]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # return all the captions of one img
            all_captions = torch.LongTensor(self.captions[((index // self.cpi) * self.cpi):(((index // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions
    
    def __len__(self):
        return self.captions_len