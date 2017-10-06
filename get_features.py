import argparse
import base64
import csv
import sys


import numpy as np
import progressbar


csv.field_size_limit(sys.maxsize)


parser = argparse.ArgumentParser(description='Extract feature from .tsv')
parser.add_argument('fname', metavar='PATH', help='path to .tsv file')
parser.add_argument('--split', default='train2014')


def decode(data, dtype):
    data_bytes = base64.decodebytes(data.encode('ascii'))
    return np.frombuffer(data_bytes, dtype=dtype)


def read_tsv(fname, split):
    FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes',
                  'cls_prob', 'attr_prob', 'cls_idx', 'attr_idx', 'features']
    img_ids = []
    boxes = []
    cls_probs = []
    attr_probs = []
    cls_ids = []
    attr_ids = []
    features = []
    with open(fname) as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        bar = progressbar.ProgressBar()
        for item in bar(reader):
            img_ids.append(int(item['image_id']))
            boxes.append(decode(item['boxes'], 'float32').reshape(36, -1))
            cls_probs.append(decode(item['cls_prob'], 'float32').reshape(36, -1))
            attr_probs.append(decode(item['attr_prob'], 'float32').reshape(36, -1))
            cls_ids.append(decode(item['cls_idx'], 'int64'))
            attr_ids.append(decode(item['attr_idx'], 'int64'))
            features.append(decode(item['features'], 'float32').reshape(36, -1))

    return (np.array(img_ids, dtype='int64'),
            np.array(boxes, dtype='float32'),
            np.array(cls_probs, dtype='float32'),
            np.array(attr_probs, dtype='float32'),
            np.array(cls_ids, dtype='int64'),
            np.array(attr_ids, dtype='int64'),
            np.array(features, dtype='float32'))


if __name__ == '__main__':
    args = parser.parse_args()
    field_names = ['id', 'box', 'class-prob', 'attribute-prob',
                   'class', 'attribute', 'feature']
    results = read_tsv(args.fname, args.split)
    for name, data in zip(field_names, results):
        np.save('data/n_{}_36_{}.npy'.format(args.split, name), data)

