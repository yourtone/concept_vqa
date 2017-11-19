import json
from operator import itemgetter

import nltk
import progressbar

from config import cfg


def main():
    include_ids = set()
    for split in cfg.TRAIN.SPLITS:
        if split == 'vg':
            continue
        samples = json.load(open('{}/raw-{}.json'.format(cfg.DATA_DIR, split)))
        include_ids.update(map(itemgetter('image_id'), samples))
    print('[Info] train image count: {}'.format(len(include_ids)))

    vg_img_info = json.load(open('{}/image_data.json'.format(cfg.VG_DIR)))
    vg_coco_img = []
    for img_info in vg_img_info:
        if img_info['coco_id'] is not None:
            vg_coco_img.append(img_info)
    print('[Info] COCO image in vg: {}'.format(len(vg_coco_img)))

    vg_to_coco = {}
    for img_info in vg_coco_img:
        if img_info['coco_id'] in include_ids:
            vg_to_coco[img_info['image_id']] = img_info['coco_id']
    print('[Info] reserve image in training split: {}/{}'
            .format(len(vg_to_coco), len(vg_coco_img)))

    vg_data = json.load(open('{}/question_answers.json'.format(cfg.VG_DIR)))
    vg_qas = []
    for sample_per_image in vg_data:
        if sample_per_image['id'] in vg_to_coco:
            vg_qas.extend(sample_per_image['qas'])
    total = sum(map(len, map(itemgetter('qas'), vg_data)))
    print('[Info] reserve qa pairs: {}/{}'.format(len(vg_qas), total))

    qas = []
    bar = progressbar.ProgressBar()
    for sample in bar(vg_qas):
        new_sample = {}
        new_sample['image_id'] = vg_to_coco[sample['image_id']]
        new_sample['question_id'] = sample['qa_id']

        answer = sample['answer'].lower().strip()
        if answer[-1] == '.':
            answer = answer[:-1]
        new_sample['answers'] = [[answer, 1]]

        question = sample['question'].lower().strip()
        if question[-1] == '?':
            question = question[:-1]
        new_sample['question'] = nltk.word_tokenize(question)

        qas.append(new_sample)

    json.dump(qas, open('{}/raw-vg.json'.format(cfg.DATA_DIR), 'w'))


if __name__ == '__main__':
    main()

