import json
import h5py

import numpy as np
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset

from config import cfg, get_feature_path

class VQADataset(Dataset):
    def  __init__(self, split, model_group_name):
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
            if model_group_name == 'avg_label':
                obj_fname = get_feature_path('train2014', 'class-fea')
                self.obj_feas = np.load(obj_fname)
            elif model_group_name == 'onehot_label':
                obj_fname = get_feature_path('train2014', 'class')
                self.obj_feas = np.load(obj_fname)
        else:
            fea_fname = get_feature_path('val2014', 'feature')
            self.img_feas = open_memmap(fea_fname, dtype='float32')
            if model_group_name == 'avg_label':
                obj_fname = get_feature_path('val2014', 'class-fea')
                self.obj_feas = np.load(obj_fname)
            elif model_group_name == 'onehot_label':
                obj_fname = get_feature_path('val2014', 'class')
                self.obj_feas = np.load(obj_fname)

        if model_group_name == 'onehot_label':
            with open('data/objects_vocab.txt') as f:
                self.objects_vocab = f.read().splitlines()
            self.objects_vocab = ['__no_objects__'] + self.objects_vocab

    def __getitem__(self, idx):
        item = []
        ip = self.img_pos[idx]
        item.append(self.que_id[idx])
        item.append(np.array(self.img_feas[ip]))
        item.append(self.que[idx])

        if hasattr(self, 'obj_feas'):
            item.append(np.array(self.obj_feas[ip]))

        if hasattr(self, 'ans'):
            item.append(self.ans[idx])

        return item

    def __len__(self):
        return self.que.shape[0]

    @property
    def num_words(self):
        return len(self.codebook['itow'])

    @property
    def num_ans(self):
        return len(self.codebook['itoa'])

