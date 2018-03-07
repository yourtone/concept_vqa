import os
import sys
import argparse
import json
from operator import itemgetter
from importlib import import_module
from collections import Counter

import torch
import progressbar
import numpy as np
from torch.autograd import Variable

import get_data
from config import cfg, cfg_from_file, cfg_from_list, get_emb_size
from dataset import VQADataset


parser = argparse.ArgumentParser()
parser.add_argument('model_dir')
parser.add_argument('--bs', '--batch_size', default=128, type=int,
                    help='batch size for predicting')
parser.add_argument('--gpu_id', default=0, type=int, metavar='N',
                    help='index of the gpu')
parser.add_argument('--cfg', dest='cfg_file', default=None, type=str,
                    help='optional config file')
parser.add_argument('--set', dest='set_cfgs', default=None,
                    nargs=argparse.REMAINDER, help='set config keys')


def main():
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # select device
    torch.cuda.set_device(args.gpu_id)
    print('[Info] use gpu: {}'.format(torch.cuda.current_device()))

    # get parameters
    sys.path.insert(0, args.model_dir)
    from params import params
    assert len(params) > 1

    last_cfg = params[0][-1]
    last_cfg()
    get_data.main()
    dataset = VQADataset('test', params[0][1])
    itoa = dataset.codebook['itoa']

    vote_buff = [{} for i in range(len(dataset))]
    conf_buff = np.zeros((len(dataset), len(itoa)))
    sm_conf_buff = np.zeros((len(dataset), len(itoa)))
    l2_conf_buff = np.zeros((len(dataset), len(itoa)))
    que_ids = dataset.que_id
    for fpath, mgrp, mname, acc, cfg_func, in params:
        # data
        if cfg_func != last_cfg:
            cfg_func()
            get_data.main()
            last_cfg = cfg_func
            dataset = VQADataset('test', mgrp)
            itoa = dataset.codebook['itoa']

        dataset.reload_obj(mgrp)
        dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.bs, shuffle=False,
                num_workers=2, pin_memory=True)

        # model
        model_group = import_module('models.' + mgrp)
        model = getattr(model_group, mname)
                num_words=dataset.num_words,
                num_ans=dataset.num_ans,
                emb_size=get_emb_size())
        cp_file = os.path.join(args.model_dir, fpath)
        checkpoint = torch.load(cp_file, map_location=lambda s, l: s.cuda(0))
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        model.eval()

        # predict
        bar = progressbar.ProgressBar()
        start = 0
        # sample: (que_id, img, que, [obj])
        for sample in bar(dataloader):
            sample_var = [Variable(d).cuda() for d in list(sample)[1:]]
            score = model(*sample_var)
            sm_score = torch.nn.functional.softmax(score)
            l2_score = torch.nn.functional.normalize(score)

            bs = score.size(0)
            conf_buff[start:start+bs] += score.data.cpu().numpy()
            sm_conf_buff[start:start+bs] += sm_score.data.cpu().numpy()
            l2_conf_buff[start:start+bs] += l2_score.data.cpu().numpy()

            _, ans_ids = torch.max(score.data, dim=1)
            for i, ans_id in enumerate(ans_ids):
                ans = itoa[ans_id]
                ans_score = acc + vote_buff[start + i].get(ans, 0)
                vote_buff[start + i][ans] = ans_score

            start += bs

    # origin score ensemble
    _, ans_ids = torch.max(torch.from_numpy(conf_buff), dim=1)
    conf_result = []
    for que_id, ans_id in zip(que_ids, ans_ids):
        conf_result.append({'question_id': int(que_id), 'answer': itoa[ans_id]})
    fname = os.path.join(args.model_dir, 'result-conf.json')
    json.dump(conf_result, open(fname, 'w'))

    # softmax score ensemble
    _, ans_ids = torch.max(torch.from_numpy(sm_conf_buff), dim=1)
    conf_result = []
    for que_id, ans_id in zip(que_ids, ans_ids):
        conf_result.append({'question_id': int(que_id), 'answer': itoa[ans_id]})
    fname = os.path.join(args.model_dir, 'result-sm-conf.json')
    json.dump(conf_result, open(fname, 'w'))

    # l2 normalized score ensemble
    _, ans_ids = torch.max(torch.from_numpy(l2_conf_buff), dim=1)
    conf_result = []
    for que_id, ans_id in zip(que_ids, ans_ids):
        conf_result.append({'question_id': int(que_id), 'answer': itoa[ans_id]})
    fname = os.path.join(args.model_dir, 'result-l2-conf.json')
    json.dump(conf_result, open(fname, 'w'))

    # voting ensemble
    vote_buff = [Counter(v).most_common(1)[0][0] for v in vote_buff]
    vote_result = []
    for que_id, ans in zip(que_ids, vote_buff):
        vote_result.append({'question_id': int(que_id), 'answer': ans})
    fname = os.path.join(args.model_dir, 'result-vote.json')
    json.dump(vote_result, open(fname, 'w'))


if __name__ == '__main__':
    main()
