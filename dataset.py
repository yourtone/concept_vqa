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
            if cfg.SOFT_LOSS:
                self.ans = self.ans.astype(np.float32)

        # load image features
        self.splits = cfg[split.upper()].SPLITS
        self.img_feas = []
        for data_split in self.splits:
            fea_fname = get_feature_path(data_split, 'feature')
            if cfg.LOAD_ALL_DATA:
                img_fea = np.load(fea_fname)
            else:
                img_fea = open_memmap(fea_fname, dtype='float32')
            self.img_feas.append(img_fea)
        self.img_cnts = list(map(len, self.img_feas))

        self.model_group_name = None
        self.reload_obj(model_group_name)


    def _split_pos(self, abs_ip):
        ip = 0
        sub_ip = abs_ip
        for cnt in self.img_cnts:
            if sub_ip >= cnt:
                ip += 1
                sub_ip -= cnt
            else:
                break
        return ip, sub_ip


    def __getitem__(self, idx):
        item = []
        abs_ip = self.img_pos[idx]
        ip, sub_ip = self._split_pos(abs_ip)
        item.append(self.que_id[idx])
        item.append(np.array(self.img_feas[ip][sub_ip]))
        item.append(self.que[idx])

        if hasattr(self, 'obj_feas'):
            item.append(np.array(self.obj_feas[abs_ip]))

        if hasattr(self, 'ans'):
            item.append(self.ans[idx])

        return item


    def __len__(self):
        return self.que.shape[0]


    def reload_obj(self, model_group_name):
        if model_group_name == self.model_group_name:
            return

        if hasattr(self, 'obj_feas'):
            del self.obj_feas
        if hasattr(self, 'objects_vocab'):
            del self.objects_vocab

        # load object features
        obj_fea_name = None
        if model_group_name == 'avg_label':
            obj_fea_name = 'class-fea'
        elif model_group_name == 'onehot_label':
            obj_fea_name = 'class'
        elif model_group_name == 'prob_label':
            obj_fea_name = 'class-prob'

        if obj_fea_name:
            self.obj_feas = []
            for data_split in self.splits:
                obj_fname = get_feature_path(data_split, obj_fea_name)
                self.obj_feas.append(np.load(obj_fname))
            if len(self.obj_feas) > 0:
                self.obj_feas = np.vstack(self.obj_feas)

        # load object labels
        if model_group_name in ('onehot_label', 'prob_label'):
            with open('data/objects_vocab.txt') as f:
                self.objects_vocab = f.read().splitlines()
            self.objects_vocab = ['__no_objects__'] + self.objects_vocab

        self.model_group_name = model_group_name


    @property
    def num_words(self):
        return len(self.codebook['itow'])


    @property
    def num_ans(self):
        return len(self.codebook['itoa'])

