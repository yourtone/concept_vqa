import json
import argparse
from collections import Counter
from operator import itemgetter

import h5py
import numpy as np


DEBUG = True


parser = argparse.ArgumentParser(
	description='Prepare data for training and testing')

parser.add_argument('--nodebug', action='store_true',
                    help='suppress dubug printing')
parser.add_argument('--train-split', action='append', default=['train2014'],
		    metavar='SPLIT', help='name of split used in training')
parser.add_argument('--test-split', action='append', default=['val2014'],
		    metavar='SPLIT', help='name of split used in testing')
parser.add_argument('--ans-min-freq', default=16, type=int, metavar='N',
                    help='mininum frequency of answer candidates')
parser.add_argument('--word-min-freq', default=1, type=int, metavar='N',
                    help='mininum frequency of words')
parser.add_argument('--que-max-len', default=14, type=int, metavar='N',
                    help='maxinum length of questions')


def main():
    global parser
    args = parser.parse_args()
    args_str = json.dumps(vars(args), indent=2)
    print('[Info] arguments:')
    print(args_str)
    with open('data/args_get_data', 'w') as f:
        f.write(args_str)

    if args.nodebug:
        global DEBUG
        DEBUG = False

    # load data
    trn_data = []
    for split_name in args.train_split:
        fname = 'data/raw-{}.json'.format(split_name)
        print('[Load] {}'.format(fname))
        trn_data.extend(json.load(open(fname)))
    tst_data = []
    for split_name in args.test_split:
        fname = 'data/raw-{}.json'.format(split_name)
        print('[Load] {}'.format(fname))
        tst_data.extend(json.load(open(fname)))

    # determine answer candidates
    ans_freq = Counter()
    for pair in trn_data:
        ans_freq.update(dict(pair['answers']))
    itoa = [a for a, c in ans_freq.most_common() if c > args.ans_min_freq]
    atoi = {a:i for i, a in enumerate(itoa)}
    if DEBUG:
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
    itow = [w for w, c in word_freq if c > args.word_min_freq]
    print('[Info] Reserved words count: {}/{}({:%})'
            .format(len(itow), len(word_freq), len(itow)/len(word_freq)))
    assert('<UNK>' not in itow)
    assert('<PAD>' not in itow)
    itow = ['<PAD>'] + itow + ['<UNK>']
    wtoi = {w: i for i, w in enumerate(itow)}
    if DEBUG:
        print('[Debug] top word')
        print(' '.join(itow[:10]))
        print('[Debug] last word')
        print(' '.join(itow[-10:]))

    # index image feature
    img_ids = np.load('data/train2014_36_id.npy')
    # with open('data/train2014_36_id.npy') as f:
    #     img_ids = map(int, f.readlines())
    id_to_pos = {img_id: i for i, img_id in enumerate(img_ids)}
    trn_img_pos = [id_to_pos[img_id] for img_id in
                    map(itemgetter('image_id'), trn_data)]
    trn_img_pos = np.array(trn_img_pos, dtype='int64')

    img_ids = np.load('data/val2014_36_id.npy')
    id_to_pos = {img_id: i for i, img_id in enumerate(img_ids)}
    tst_img_pos = [id_to_pos[img_id] for img_id in
                    map(itemgetter('image_id'), tst_data)]
    tst_img_pos = np.array(tst_img_pos, dtype='int64')

    # encode question
    trn_que, trn_que_id = encode_que(trn_data, args.que_max_len, wtoi)
    trn_ans = encode_ans(trn_data, atoi)
    tst_que, tst_que_id = encode_que(tst_data, args.que_max_len, wtoi)

    # Save
    codebook = {}
    codebook['itoa'] = itoa
    codebook['itow'] = itow
    codebook['train'] = args.train_split
    codebook['test'] = args.test_split
    json_fname = 'data/data.json'
    print('[Store] {}'.format(json_fname))
    json.dump(codebook, open(json_fname, 'w'))

    h5_fname = 'data/data.h5'
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


def encode_que(data, max_length, wtoi):
    N = len(data)
    question_id = np.zeros((N,), dtype='int64')
    que = np.zeros((N, max_length), dtype='int64')

    unk_cnt = 0
    trun_cnt = 0
    unk_idx = wtoi.get('<UNK>')
    for i,sample in enumerate(data):
        question_id[i] = sample['question_id']

        words = sample['question']
        nword = len(words)
        if nword > max_length:
            trun_cnt += 1
            nword = max_length
            words = words[:nword]
        for j, w in enumerate(words):
            que[i][max_length-nword+j] = wtoi.get(w, unk_idx)
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
