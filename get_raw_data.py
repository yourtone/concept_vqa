import os
import re
import json
import random
import argparse
from collections import Counter
from operator import itemgetter

import nltk
import progressbar

from config import cfg


def main():
    for split_name in ('train2014', 'val2014', 'test-dev2015', 'test2015'):
        pairs = load_vqa_data(split_name)
        fname = '{}/raw-{}.json'.format(cfg.DATA_DIR, split_name)
        print('[Store] {}'.format(fname))
        json.dump(pairs, open(fname, 'w'))

    if 'vg' in cfg.TRAIN.SPLITS:
        pairs = load_vg_data()
        fname = '{}/raw-vg.json'.format(cfg.DATA_DIR)
        print('[Store] {}'.format(fname))
        json.dump(pairs, open(fname, 'w'))


def load_vqa_data(split_name):
    qfname = '{}/Questions/v2_OpenEnded_mscoco_{}_questions.json'.format(
            cfg.VQA_DIR, split_name)
    afname = '{}/Annotations/v2_mscoco_{}_annotations.json'.format(
            cfg.VQA_DIR, split_name)

    print('[Load] load question data from "{}"'.format(qfname))
    vqa_pairs = json.load(open(qfname))['questions']

    print('[Info] tokenizing')
    bar = progressbar.ProgressBar()
    for pair in bar(vqa_pairs):
        qtext = pair['question'].lower().strip()
        if qtext[-1] == '?':
            qtext = qtext[:-1]
        pair['question'] = nltk.word_tokenize(qtext)
    if cfg.DEBUG:
        print('[Debug] question after tokenizing')
        questions = map(itemgetter('question'), vqa_pairs)
        sample_ques = random.sample(list(questions), k=5)
        print('\n'.join([' '.join(q) for q in sample_ques]))

    if os.path.exists(afname):
        print('[Load] load annotation data from "{}"'.format(afname))
        anns = json.load(open(afname))['annotations']
        qid_anns = {a['question_id']: a for a in anns}
        bar = progressbar.ProgressBar()
        for q in bar(vqa_pairs):
            answers = qid_anns.get(q['question_id']).get('answers')
            for item in answers:
                item['answer'] = norm_answer(item['answer'])

            ans_text = set(map(itemgetter('answer'), answers))
            ans_score = []
            for at in ans_text:
                accs = []
                for gt in answers:
                    other_gt = [a for a in answers if a != gt]
                    matched_gt = [a for a in other_gt if a['answer'] == at]
                    accs.append(min(1, len(matched_gt) / 3))
                ans_score.append((at, sum(accs)/len(accs)))
            ans_score = sorted(ans_score, key=itemgetter(1), reverse=True)
            q['answers'] = ans_score
        if cfg.DEBUG:
            print('[Debug] one vqa pair')
            print(random.choice(vqa_pairs))

    return vqa_pairs


def load_vg_data():
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
        answer = norm_answer(answer)
        new_sample['answers'] = [[answer, 1]]

        question = sample['question'].lower().strip()
        if question[-1] == '?':
            question = question[:-1]
        new_sample['question'] = nltk.word_tokenize(question)

        qas.append(new_sample)

    return qas


def norm_answer(ans):
    ans = ans.replace('\n', ' ').replace('\t', ' ').strip()
    ans = process_punct(ans)
    ans = process_digit_article(ans)
    return ans


# borrow from vqaEval.py
m_contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've",
    "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't",
    "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
    "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
    "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock",
    "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've",
    "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's",
    "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've",
    "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "twas": "'twas",
    "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
    "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've"}
m_manual_map = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
                'four': '4', 'five': '5', 'six': '6', 'seven': '7',
                'eight': '8', 'nine': '9', 'ten': '10'}
m_articles = ['a', 'an', 'the']
m_period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
m_comma_strip = re.compile("(\d)(\,)(\d)")
m_punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+',
           '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']


def process_punct(in_text):
    out_text = in_text
    for p in m_punct:
        if (p + ' ' in in_text or ' ' + p in in_text
                or re.search(m_comma_strip, in_text) != None):
            out_text = out_text.replace(p, '')
        else:
            out_text = out_text.replace(p, ' ')
    out_text = m_period_strip.sub("", out_text, re.UNICODE)
    return out_text


def process_digit_article(in_text):
    out_text = []
    for word in in_text.lower().split():
        if word not in m_articles:
            word = m_manual_map.setdefault(word, word)
            word = m_contractions.setdefault(word, word)
            out_text.append(word)
    return ' '.join(out_text)


if __name__ == '__main__':
    main()

