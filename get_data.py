import json
import argparse
from collections import Counter
from operator import itemgetter

import h5py
import numpy as np

from config import cfg, get_feature_path


def main():
    # load data
    trn_data = []
    for split_name in cfg.TRAIN.SPLITS:
        fname = '{}/raw-{}.json'.format(cfg.DATA_DIR, split_name)
        print('[Load] {}'.format(fname))
        trn_data.extend(json.load(open(fname)))
    tst_data = []
    for split_name in cfg.TEST.SPLITS:
        fname = '{}/raw-{}.json'.format(cfg.DATA_DIR, split_name)
        print('[Load] {}'.format(fname))
        tst_data.extend(json.load(open(fname)))

    # determine answer candidates
    ans_freq = Counter()
    for pair in trn_data:
        ans_freq.update(dict(pair['answers']))
    itoa = [a for a, c in ans_freq.most_common() if c > cfg.MIN_ANS_FREQ]
    atoi = {a:i for i, a in enumerate(itoa)}
    if cfg.DEBUG:
        print('[Debug] top answer')
        print(' '.join(itoa[:10]))

    # filter training sample
    trn_data_reduce = [d for d in trn_data if d['answers'][0][0] in atoi]
    print('[Info] reduce {} training sample'.format(
        len(trn_data)-len(trn_data_reduce)))
    trn_data = trn_data_reduce

    # determine vocabulary
    word_freq = Counter()
    for pair in trn_data:
        word_freq.update(pair['question'])
    word_freq = word_freq.most_common()
    itow = [w for w, c in word_freq if c > cfg.MIN_WORD_FREQ]
    print('[Info] Reserved words count: {}/{}({:%})'
            .format(len(itow), len(word_freq), len(itow)/len(word_freq)))
    assert('<UNK>' not in itow)
    assert('<PAD>' not in itow)
    itow = ['<PAD>'] + itow + ['<UNK>']
    wtoi = {w: i for i, w in enumerate(itow)}
    if cfg.DEBUG:
        print('[Debug] top word')
        print(' '.join(itow[:10]))
        print('[Debug] last word')
        print(' '.join(itow[-10:]))

    # index image feature
    trn_img_ids = map(itemgetter('image_id'), trn_data)
    trn_img_pos = get_img_pos(cfg.TRAIN.SPLITS, trn_img_ids)

    tst_img_ids = map(itemgetter('image_id'), tst_data)
    tst_img_pos = get_img_pos(cfg.TEST.SPLITS, tst_img_ids)

    # encode question
    trn_que, trn_que_id = encode_que(trn_data, wtoi)
    trn_ans = encode_ans(trn_data, atoi)
    tst_que, tst_que_id = encode_que(tst_data, wtoi)

    # Save
    codebook = {}
    codebook['itoa'] = itoa
    codebook['itow'] = itow
    json_fname = '{}/data.json'.format(cfg.DATA_DIR)
    print('[Store] {}'.format(json_fname))
    json.dump(codebook, open(json_fname, 'w'))

    h5_fname = '{}/data.h5'.format(cfg.DATA_DIR)
    print('[Store] {}'.format(h5_fname))
    with h5py.File(h5_fname, 'w') as f:
        group = f.create_group('train')
        group.create_dataset('que_id', dtype='int64', data=trn_que_id)
        group.create_dataset('img_pos', dtype='int64', data=trn_img_pos)
        group.create_dataset('que', dtype='int64', data=trn_que)
        group.create_dataset('ans', dtype='int64', data=trn_ans)

        group = f.create_group('test')
        group.create_dataset('que_id', dtype='int64', data=tst_que_id)
        group.create_dataset('img_pos', dtype='int64', data=tst_img_pos)
        group.create_dataset('que', dtype='int64', data=tst_que)


def get_img_pos(splits, data_ids):
    fea_ids = []
    for split in splits:
        fea_ids.append(np.load(get_feature_path(split, 'id')))
    if len(fea_ids) > 0:
        fea_ids = np.hstack(fea_ids)
    id_to_pos = {img_id: i for i, img_id in enumerate(fea_ids)}
    data_poses = [id_to_pos[img_id] for img_id in data_ids]
    return np.array(data_poses, dtype='int64')


def encode_que(data, wtoi):
    N = len(data)
    question_id = np.zeros((N,), dtype='int64')
    que = np.zeros((N, cfg.MAX_QUESTION_LEN), dtype='int64')

    unk_cnt = 0
    trun_cnt = 0
    unk_idx = wtoi.get('<UNK>')
    for i,sample in enumerate(data):
        question_id[i] = sample['question_id']

        words = sample['question']
        nword = len(words)
        if nword > cfg.MAX_QUESTION_LEN:
            trun_cnt += 1
            nword = cfg.MAX_QUESTION_LEN
            words = words[:nword]
        for j, w in enumerate(words):
            que[i][cfg.MAX_QUESTION_LEN-nword+j] = wtoi.get(w, unk_idx)
            unk_cnt += (0 if w in wtoi else 1)

    print('[Info] Truncated question count: {}'.format(trun_cnt))
    print('[Info] Unknown word count: {}'.format(unk_cnt))
    return que, question_id


def encode_ans(data, atoi):
    ans = np.zeros((len(data),), dtype='int64')
    for i, answers in enumerate(map(itemgetter('answers'), data)):
        ans[i] = atoi.get(answers[0][0])
    return ans


if __name__ == '__main__':
    main()
