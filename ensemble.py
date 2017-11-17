import os
import argparse
import json
from operator import itemgetter
from importlib import import_module
from collections import Counter

import torch
import progressbar
import numpy as np
from torch.autograd import Variable

from config import cfg
from dataset import VQADataset


parser = argparse.ArgumentParser()
parser.add_argument('model_dir')


def main():
    args = parser.parse_args()
    files = os.listdir(args.model_dir)
    model_files = [f for f in files if f.endswith('.pth.tar')]
    model_info = []
    for cp_file in model_files:
        file_name = cp_file.rsplit('.', 2)[0]
        model_group_name, model_name, acc_text, *_ = file_name.split('-')
        model_acc = int(acc_text)
        cp_path = os.path.join(args.model_dir, cp_file)
        model_info.append((model_group_name, model_name, model_acc, cp_path))

    model_info = sorted(model_info, key=itemgetter(0))
    assert len(model_info) > 1

    dataset = VQADataset('test', model_info[0][0])
    itoa = dataset.codebook['itoa']

    # infer embedding size
    emb_size = 300
    if cfg.WORD_EMBEDDINGS:
        emb_names = cfg.WORD_EMBEDDINGS.split('+')
        emb_size = 0
        for emb_name in emb_names:
            emb_path = '{}/word-embedding/{}'.format(cfg.DATA_DIR, emb_name)
            with open(emb_path) as f:
                line = f.readline()
            emb_size += len(line.split()) - 1

    vote_buff = [{} for i in range(len(dataset))]
    conf_buff = np.zeros((len(dataset), len(itoa)))
    que_ids = dataset.que_id
    for model_group_name, model_name, model_acc, cp_file in model_info:
        # data
        if dataset.model_group_name != model_group_name:
            dataset.reload_obj(model_group_name)
        dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                num_workers=2, pin_memory=True)

        # model
        model_group = import_module('models.' + model_group_name)
        model = getattr(model_group, model_name)(
                num_words=dataset.num_words,
                num_ans=dataset.num_ans,
                emb_size=emb_size)
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

            bs = score.size(0)
            conf_buff[start:start+bs] += score.data.cpu().numpy()
            _, ans_ids = torch.max(score.data, dim=1)
            for i, ans_id in enumerate(ans_ids):
                ans = itoa[ans_id]
                ans_score = model_acc + vote_buff[start + i].get(ans, 0)
                vote_buff[start + i][ans] = ans_score

            start += bs

    _, ans_ids = torch.max(torch.from_numpy(conf_buff), dim=1)
    conf_result = []
    for que_id, ans_id in zip(que_ids, ans_ids):
        conf_result.append({'question_id': int(que_id), 'answer': itoa[ans_id]})

    vote_buff = [Counter(v).most_common(1)[0][0] for v in vote_buff]
    vote_result = []
    for que_id, ans in zip(que_ids, vote_buff):
        vote_result.append({'question_id': int(que_id), 'answer': ans})

    fname = os.path.join(args.model_dir, 'result-conf.json')
    json.dump(conf_result, open(fname, 'w'))
    fname = os.path.join(args.model_dir, 'result-vote.json')
    json.dump(vote_result, open(fname, 'w'))


if __name__ == '__main__':
    main()
