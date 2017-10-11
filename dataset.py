import json
import h5py

import numpy as np
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset

from config import cfg

class VQADataset(Dataset):
    def  __init__(self, split):
        self.codebook = json.load(open('{}/data.json'.format(cfg.DATA_DIR)))

        data = h5py.File('{}/data.h5'.format(cfg.DATA_DIR))['/{}'.format(split)]
        self.img_pos = data['img_pos'].value
        self.que = data['que'].value
        self.que_id = data['que_id'].value
        if 'ans' in data:
            self.ans = data['ans'].value

        if split == 'train':
            fea_fname = '{}/train2014_36_feature.npy'.format(cfg.DATA_DIR)
            self.img_feas = open_memmap(fea_fname, dtype='float32')
            obj_fname = '{}/train2014_36_class-fea.npy'.format(cfg.DATA_DIR)
            self.obj_feas = open_memmap(obj_fname, dtype='float32')
        else:
            fea_fname = '{}/val2014_36_feature.npy'.format(cfg.DATA_DIR)
            self.img_feas = open_memmap(fea_fname, dtype='float32')
            obj_fname = '{}/val2014_36_class-fea.npy'.format(cfg.DATA_DIR)
            self.obj_feas = open_memmap(obj_fname, dtype='float32')

    def __getitem__(self, idx):
        ip = self.img_pos[idx]
        img = np.array(self.img_feas[ip])
        obj = np.array(self.obj_feas[ip])
        que_id = self.que_id[idx]
        que = self.que[idx]
        if hasattr(self, 'ans'):
            ans = self.ans[idx]
            return img, que_id, que, obj, ans
        else:
            return img, que_id, que, obj

    def __len__(self):
        return self.que.shape[0]

    @property
    def num_words(self):
        return len(self.codebook['itow'])

    @property
    def num_ans(self):
        return len(self.codebook['itoa'])

