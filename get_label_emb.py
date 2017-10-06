import argparse

import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--fname', metavar='PATH',
                    default='data/objects_vocab.txt',
                    help='path to label file')
parser.add_argument('--emb-file',
                    default='data/word-embedding/glove.6B.300d.txt',
                    help='path to embedding file')


def get_emb(emb_file):
    with open(emb_file) as f:
        raw = f.read().splitlines()
    word_vec = [l.split(' ', 1) for l in raw]
    vocab, vecs_txt = zip(*word_vec)
    vecs = np.fromstring(' '.join(vecs_txt), dtype='float32', sep=' ')
    vecs = vecs.reshape(-1, 300)
    emb_dict = dict(zip(vocab, vecs))
    return emb_dict, np.mean(vecs), np.std(vecs)


def fill_emb(labels, emb_dict, mean, std):
    emb_size = next(iter(emb_dict.values())).shape[0]
    result = np.random.randn(len(labels), emb_size).astype('float32')
    result = (result + mean) * std
    fill_cnt = 0
    for i, line in enumerate(labels):
        synonyms = line.split(',')
        act_num = []
        for label in synonyms:
            words = label.split()
            act = 0
            for word in words:
                if word in emb_dict:
                    act += 1
            act_num.append(act)
        act_idx = max(range(len(act_num)), key=lambda x: act_num[x])
        if act_num[act_idx] > 0:
            emb = np.zeros((emb_size,), dtype='float32')
            for word in synonyms[act_idx]:
                if word in emb_dict:
                    emb += emb_dict[word]
            emb /= act_num[act_idx]
            result[i] = emb
            fill_cnt += 1
    print('fill embedding: {}/{}'.format(fill_cnt, len(labels)))
    return result


def avg_emb(probs, label_emb):
    probs_t = torch.from_numpy(probs)
    label_emb_t = torch.from_numpy(label_emb)
    return torch.matmul(probs_t, label_emb_t).numpy()


def main():
    args = parser.parse_args()
    print('load glove embedding...')
    emb_dict, mean, std = get_emb(args.emb_file)

    print('fill label embedding...')
    with open(args.fname) as f:
        labels = ['__no_object__'] + f.read().splitlines()
    label_emb = fill_emb(labels, emb_dict, mean, std)

    print('average train2014 class embedding...')
    train_probs = np.load('data/n_train2014_36_class-prob.npy')
    train_fea = avg_emb(train_probs, label_emb)
    np.save('data/train2014_36_class-fea.npy', train_fea)

    print('average val2014 class embedding...')
    val_probs = np.load('data/n_val2014_36_class-prob.npy')
    val_fea = avg_emb(val_probs, label_emb)
    np.save('data/val2014_36_class-fea.npy', val_fea)


if __name__ == '__main__':
    main()

