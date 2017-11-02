import json
import h5py

import numpy as np
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset

from config import cfg, get_feature_path

class VQADataset(Dataset):
    def  __init__(self, split, needT=False):
        self.codebook = json.load(open('{}/data.json'.format(cfg.DATA_DIR)))

        data = h5py.File('{}/data.h5'.format(cfg.DATA_DIR))['/{}'.format(split)]
        self.img_pos = data['img_pos'].value
        self.que = data['que'].value
        self.que_id = data['que_id'].value
        if 'ans' in data:
            self.ans = data['ans'].value

        if split == 'train':
            fea_fname = get_feature_path('train2014', 'feature')
            self.img_feas = open_memmap(fea_fname, dtype='float32')
            if needT:
                obj_fname = get_feature_path('train2014', 'class-fea')
                self.obj_feas = np.load(obj_fname)
        else:
            fea_fname = get_feature_path('val2014', 'feature')
            self.img_feas = open_memmap(fea_fname, dtype='float32')
            if needT:
                obj_fname = get_feature_path('val2014', 'class-fea')
                self.obj_feas = np.load(obj_fname)

        self.needT = needT

    def __getitem__(self, idx):
        ip = self.img_pos[idx]
        img = np.array(self.img_feas[ip])
        if self.needT:
            obj = np.array(self.obj_feas[ip])
        que_id = self.que_id[idx]
        que = self.que[idx]
        if hasattr(self, 'ans'):
            ans = self.ans[idx]
            if cfg.SOFT_LOSS:
                ans = ans.astype('float32')
            if self.needT:
                return que_id, img, que, obj, ans
            else:
                return que_id, img, que, ans
        else:
            if self.needT:
                return que_id, img, que, obj
            else:
                return que_id, img, que

    def __len__(self):
        return self.que.shape[0]

    @property
    def num_words(self):
        return len(self.codebook['itow'])

    @property
    def num_ans(self):
        return len(self.codebook['itoa'])

