import os
import re
import h5py
import json
import random
import argparse
import pickle
from importlib import import_module
from operator import itemgetter

import torch
import torch.nn.functional as F
import numpy as np
import progressbar
import visdom
from scipy.misc import imread, imsave
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate

from predict import format_result
from eval_tools import get_eval
from dataset import VQADataset
from config import cfg, get_feature_path, get_emb_size
from config import cfg_from_file, cfg_from_list



parser = argparse.ArgumentParser()
parser.add_argument('model_dir')
parser.add_argument('--split', default='val2014')
parser.add_argument('--file-ptn', default=None)
parser.add_argument('--cfg', dest='cfg_file', default=None, type=str,
                    help='optional config file')
parser.add_argument('--set', dest='set_cfgs', default=None,
                    nargs=argparse.REMAINDER, help='set config keys')


def main():
    global args
    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    files = os.listdir(args.model_dir)
    cp_files = [f for f in files if f.endswith('.pth.tar')]
    if args.file_ptn is not None:
        ptn = re.compile(args.file_ptn)
        cp_files = [f for f in files if ptn.search(f)]

    model_info = []
    for cp_file in cp_files:
        file_name = cp_file.rsplit('.', 2)[0]
        model_group_name, model_name, _ = file_name.split('-')
        cp_path = os.path.join(args.model_dir, cp_file)
        model_info.append((model_group_name, model_name, cp_path))

    att_query = AttQuery(model_info, args.split, args.model_dir)

    visualizer = Visualizer()
    while True:
        att_query.print_query()
        command = input('input command: ')
        if ' ' in command:
            opt, arg = command.split(' ', 1)
        else:
            opt, arg = command, None

        if opt in ('q', 'question'):
            att_query.set_question(arg)
        elif opt in ('a', 'answer'):
            att_query.set_answer(arg)
        elif opt in ('c', 'condition'):
            att_query.set_condition(arg)
        elif opt in ('x', 'clean'):
            att_query.set_question(None)
            att_query.set_answer(None)
            att_query.set_condition(None)
        elif opt in ('s', 'save'):
            if arg is None:
                inds = range(att_query.get_res_cnt())
            else:
                inds = map(int, arg.split(','))
            att_query.save(inds)
        elif opt in ('r', 'run'):
            if arg is None:
                arg = 1
            result = att_query.run(int(arg))
            visualizer.visualize(result, att_query.model_info)
        elif opt in ('e', 'exit'):
            break
        else:
            print('wrong command !')


class AttQuery(object):
    def __init__(self, model_info, split, save_dir):
        assert len(model_info) > 0
        assert len(cfg.TEST.SPLITS) == 1 and cfg.TEST.SPLITS[0] == split

        model_info = sorted(model_info, key=itemgetter(0))

        self._split = split
        self.model_info = model_info
        self.save_dir = save_dir

        # load model
        self._pred_ans = []
        self._scores = []
        self._att_weights = []
        dataset = VQADataset('test', model_info[0][0])
        emb_size = get_emb_size()
        for model_group_name, model_name, cp_file in model_info:
            cache_file = cp_file + '.cache'
            if os.path.isfile(cache_file):
                print("load from cache: '{}".format(cache_file))
                cache = pickle.load(open(cache_file, 'rb'))
                self._pred_ans.append(cache['pred_ans'])
                self._scores.append(cache['scores'])
                self._att_weights.append(cache['att_weights'])
                continue

            # dataset
            if dataset.model_group_name != model_group_name:
                dataset.reload_obj(model_group_name)
            dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=cfg.BATCH_SIZE,
                    shuffle=False, num_workers=2, pin_memory=True)
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

            # predicting
            itoa = dataloader.dataset.codebook['itoa']
            batch_att_weight = []
            pred_ans = []
            bar = progressbar.ProgressBar()
            print('predicting answers...')
            # sample: (que_id, img, que, [obj])
            for sample in bar(dataloader):
                # setting hook
                att_weight_buff = torch.FloatTensor(len(sample[0]), 36)
                def get_weight(self, input, output):
                    att_weight_buff.copy_(output.data.view_as(att_weight_buff))
                hook = model.att_net.register_forward_hook(get_weight)

                # forward
                sample_var = [Variable(d).cuda() for d in list(sample)[1:]]
                score = model(*sample_var)
                att_weight = F.softmax(Variable(att_weight_buff)).data.numpy()
                batch_att_weight.append(att_weight)
                pred_ans.extend(format_result(sample[0], score, itoa))

                hook.remove()
            att_weights = np.vstack(batch_att_weight)

            # evaluation
            print('evaluting results...')
            if split in ('train2014', 'val2014'):
                vqa_eval = get_eval(pred_ans, split)
                scores = []
                for i in range(len(dataset)):
                    qid = int(dataset[i][0])
                    score = vqa_eval.evalQA.get(qid)
                    scores.append(score)
            else:
                scores = None

            self._pred_ans.append(pred_ans)
            self._scores.append(scores)
            self._att_weights.append(att_weights)

            # save cache
            cache = {}
            cache['pred_ans'] = pred_ans
            cache['scores'] = scores
            cache['att_weights'] = att_weights
            pickle.dump(cache, open(cache_file, 'wb'))

        print('done.')

        # load data
        print('load raw data...')
        split_fname = '{}/raw-{}.json'.format(cfg.DATA_DIR, split)
        self._data = json.load(open(split_fname))
        print('load boxes...')
        self._boxes = self._load_box()

        # query key
        self._question = None
        self._answer = None
        self._condition = None

        # query result
        self._r_question = None
        self._r_answer = None
        self._r_condition = None

        # dirty flag
        self._d_question = True
        self._d_answer = True
        self._d_condition = True

        self.last_results = None


    def get_res_cnt(self):
        if self.last_results is None:
            return 0
        return len(self.last_results)


    def print_query(self):
        for i, info in enumerate(self.model_info):
            print('model{}: {}'.format(i, info[-1].rsplit('/', 1)[-1]))
        print('Current Query Key')
        print('question: ' + str(self._question))
        print('answer: ' + str(self._answer))
        print('condition: ' + str(self._condition))
        print('')


    def _load_box(self):
        data = h5py.File('{}/data.h5'.format(cfg.DATA_DIR))['/test']
        img_pos = data['img_pos'].value
        box_dict_fname = get_feature_path(self._split, 'box')
        return np.load(box_dict_fname)[img_pos]


    def set_question(self, regex):
        self._question = regex
        self._d_question = True


    def set_answer(self, regex):
        self._answer = regex
        self._d_answer = True


    def set_condition(self, exp):
        self._condition = exp
        self._d_condition = True


    def run(self, num):
        mask = self._get_mask()
        if not np.where(mask):
            return []
        inds = list(np.where(mask)[0])

        if len(inds) > num:
            inds = random.sample(inds, num)

        data = [self._data[i] for i in inds]
        boxes = [self._boxes[i] for i in inds]
        pred_ans = [[a[i] for i in inds] for a in self._pred_ans]
        scores = [[s[i] for i in inds] for s in self._scores]
        att_weights = [[w[i] for i in inds] for w in self._att_weights]

        results = []

        # image, question and answer
        split = self._split
        if split == 'test-dev2015':
            split = 'test2015'
        for d in data:
            r = {}
            img_path = '{}/Images/mscoco/{}/COCO_{}_{:012}.jpg'.format(
                    cfg.VQA_DIR, split, split, d['image_id'])
            img = imread(img_path)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
            r['image'] = img
            r['image_path'] = img_path
            r['question'] = ' '.join(d['question']) + ' ?'
            if 'answers' in d:
                answers = ['{}\t\t{:.2f}'.format(a, s*100) for a, s in d['answers']]
                r['gt_answer'] = '\n'.join(answers)
            results.append(r)

        # mask image
        for weights in att_weights:
            for r, box, weight in zip(results, boxes, weights):
                masked_img = self._mask_img(r['image'], box, weight)
                if 'masked_image' not in r:
                    r['masked_image'] = [masked_img]
                else:
                    r['masked_image'].append(masked_img)

        # predicted answer
        for r, ans in zip(results, zip(*pred_ans)):
            r['pred_answer'] = map(itemgetter('answer'), ans)

        # scores of predicted answer
        for r, score in zip(results, zip(*scores)):
            r['score'] = score

        self.last_results = results

        return results


    def save(self, inds):
        if self.last_results is None:
            return
        for ind in inds:
            if ind >= len(self.last_results):
                continue
            result = self.last_results[ind]
            img_name = result['image_path'].rsplit('/', 1)[-1]
            fname = os.path.join(self.save_dir, img_name)
            imsave(fname, result['image'])

            for i, img in enumerate(result['masked_image']):
                fname = os.path.join(
                        self.save_dir, 'model{}_{}'.format(i, img_name))
                imsave(fname, img)


    def _mask_img(self, img, boxes, att_weights):
        height, width = img.shape[:2]
        mask = np.zeros((height, width))
        for box, weight in zip(boxes, att_weights):
            x1, y1, x2, y2 = map(int, box)
            #region = mask[x1:x2+1, y1:y2+1]
            #region[region < weight] = weight
            mask[x1:x2+1, y1:y2+1] += weight

        mask = (mask - mask.min() + 0.1) / (mask.max() - mask.min() + 0.1)
        return img * mask[:, :, np.newaxis]


    def _get_mask(self):
        self._query_question()
        self._query_answer()
        self._query_condition()
        return self._r_question & self._r_answer & self._r_condition


    def _query_question(self):
        if not self._d_question:
            return
        if self._question is None:
            self._r_question = np.ones(len(self._data), dtype=bool)
        else:
            ptn = re.compile(self._question)
            questions = [' '.join(s['question']) for s in self._data]
            mask = [bool(ptn.search(q)) for q in questions]
            self._r_question = np.array(mask, dtype=bool)
        self._d_question = False


    def _query_answer(self):
        if not self._d_answer:
            return
        if self._answer is None or 'answers' not in self._data[0]:
            self._r_answer = np.ones(len(self._data), dtype=bool)
        else:
            ptn = re.compile(self._answer)
            answerss = map(itemgetter('answers'), self._data)
            answers = [' '.join(map(itemgetter(0), a)) for a in answerss]
            mask = [bool(ptn.search(a)) for a in answers]
            self._r_answer = np.array(mask, dtype=bool)
        self._d_answer = False


    def _query_condition(self):
        if not self._d_condition:
            return
        if self._condition is None or self._scores[0] is None:
            self._r_condition = np.ones(len(self._data), dtype=bool)
        else:
            exps = [self._condition.format(*scores)
                    for scores in zip(*self._scores)]
            mask = [eval(exp) for exp in exps]
            self._r_condition = np.array(mask, dtype=bool)
        self._d_condition = False


class Visualizer(object):
    def __init__(self):
        self.vis = visdom.Visdom(env='main')
        self.vis.close()
        self.last_sample_cnt = 0

    @staticmethod
    def get_title(text):
        return '<div style="font-size:18px;font-weight:bold;">{}</div>'.format(text)

    @staticmethod
    def get_paragraph(text):
        return '<div style="font-size:14px;">{}</div>'.format(text)


    @staticmethod
    def get_scale_size(img, maxsize=400):
        height, width = img.shape[:2]
        if height > width:
            ratio = 400 / height
        else:
            ratio = 400 / width
        return height * ratio, width * ratio


    def visualize(self, results, model_info):
        for i, result in enumerate(results):
            text = (self.get_title('[IMAGE PATH]')
                    + self.get_paragraph(result['image_path']))
            text += (self.get_title('[QUESTION]')
                     + self.get_paragraph(result['question']))
            text += (self.get_title('[ANSWER]')
                     + self.get_paragraph(result['gt_answer']))

            text += self.get_title('[PREDICTED ANSWER]')
            for j, a in enumerate(result['pred_answer']):
                text += self.get_paragraph('model{}({}/{}): {}'
                        .format(j, model_info[j][0], model_info[j][1], a))

            text += self.get_title('[SCORE]')
            for j, s in enumerate(result['score']):
                text += self.get_paragraph('model{}({}/{}): {:.2f}'
                        .format(j, model_info[j][0], model_info[j][1], s))
            self.vis.text(text, win='text_{}'.format(i))

            height, width = self.get_scale_size(result['image'])
            self.vis.image(result['image'].transpose(2, 0, 1),
                           win='image_origin_{}'.format(i),
                           opts={'title': 'Origin Image: ' + result['image_path'],
                                 'height': height,
                                 'width': width})
            for j, m in enumerate(result['masked_image']):
                caption = 'Model{}({}/{}) Attetion'.format(
                        j, model_info[j][0], model_info[j][1])
                self.vis.image(m.transpose(2, 0, 1),
                               win='image_model{}_{}'.format(j, i),
                               opts={'title': caption,
                                     'height': height,
                                     'width': width})
        if self.last_sample_cnt > len(results):
            for i in range(len(results), self.last_sample_cnt):
                self.vis.close(win='text_{}'.format(i))
                self.vis.close(win='image_origin_{}'.format(i))
                for j in range(len(model_info)):
                    self.vis.close(win='image_model{}_{}'.format(j, i))
        self.last_sample_cnt = len(results)


if __name__ == '__main__':
    main()

