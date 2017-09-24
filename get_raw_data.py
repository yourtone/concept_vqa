import os
import json
import random
import argparse
from collections import Counter
from operator import itemgetter

import nltk
import progressbar


DEBUG = True


parser = argparse.ArgumentParser(
	description='Prepare data for training and testing')

parser.add_argument('--nodebug', action='store_true',
                    help='suppress dubug printing')
parser.add_argument('--data-dir', default='vqa-tools',
		    help='directory of vqa data')


def main():
    global parser
    args = parser.parse_args()

    if args.nodebug:
        global DEBUG
        DEBUG = False

    for split_name in ('train2014', 'val2014', 'test-dev2015', 'test2015'):
        pairs = merge_vqa_pair(args.data_dir, split_name)
        fname = 'data/raw-{}.json'.format(split_name)
        print('[Store] {}'.format(fname))
        json.dump(pairs, open(fname, 'w'))


def merge_vqa_pair(data_dir, split_name):
    qfname = '{}/Questions/v2_OpenEnded_mscoco_{}_questions.json'.format(
            data_dir, split_name)
    afname = '{}/Annotations/v2_mscoco_{}_annotations.json'.format(
            data_dir, split_name)

    print('[Load] load question data from "{}"'.format(qfname))
    vqa_pairs = json.load(open(qfname))['questions']

    print('[Info] tokenizing')
    bar = progressbar.ProgressBar()
    for pair in bar(vqa_pairs):
        qtext = pair['question'].lower().strip()
        if qtext[-1] == '?':
            qtext = qtext[:-1]
        pair['question'] = nltk.word_tokenize(qtext)
    if DEBUG:
        print('[Debug] question after tokenizing')
        questions = map(itemgetter('question'), vqa_pairs)
        sample_ques = random.sample(list(questions), k=5)
        print('\n'.join([' '.join(q) for q in sample_ques]))

    if os.path.exists(afname):
        print('[Load] load annotation data from "{}"'.format(afname))
        anns = json.load(open(afname))['annotations']
        qid_anns = {a['question_id']: a for a in anns}
        for q in vqa_pairs:
            answers = qid_anns.get(q['question_id']).get('answers')
            ans_text = map(itemgetter('answer'), answers)
            ans_freq = Counter(ans_text).most_common()
            ans_score = [(a, min(c/3, 1)) for a, c in ans_freq]
            q['answers'] = ans_score
        if DEBUG:
            print('[Debug] one vqa pair')
            print(random.choice(vqa_pairs))

    return vqa_pairs


if __name__ == '__main__':
    main()

