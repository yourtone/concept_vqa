import argparse
import os
import json
from importlib import import_module

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import progressbar
import numpy as np
from torch.autograd import Variable

from dataset import VQADataset
from config import cfg, get_emb_size
from eval_tools import get_eval


parser = argparse.ArgumentParser(description='Train VQA model')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--model', '-m', default='ocr_label/MFHModelc',
                    help='name of the model')
parser.add_argument('--bs', '--batch_size', default=64, type=int,
                    help='batch size for predicting')
parser.add_argument('--gpu_id', default=0, type=int, metavar='N',
                    help='index of the gpu')


def main():
    global args
    args = parser.parse_args()
    args_str = json.dumps(vars(args), indent=2)
    print('[Info] called with: ' + args_str)

    # checkpoint directory
    cfg.LOG_DIR = os.path.join(cfg.LOG_DIR, 'ensemble')

    # select device
    torch.cuda.set_device(args.gpu_id)
    print('[Info] use gpu: {}'.format(torch.cuda.current_device()))

    # data
    print('[Info] init dataset')
    model_group_name, model_name = args.model.split('/')
    val_set = VQADataset('test', model_group_name)
    val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.bs, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    print('[Info] sample count: {}'.format(len(val_set)))

    # model
    print('[Info] construct model')
    model_group = import_module('models.' + model_group_name)
    model = getattr(model_group, model_name)(
            num_words=val_set.num_words,
            num_ans=val_set.num_ans,
            emb_size=get_emb_size())
    model.cuda()
    cudnn.benchmark = True
    print('[Info] model name: ' + args.model)

    itoa_emb = np.load('{}/image-feature/bottomup/itoa_emb.npy'
        .format(cfg.DATA_DIR)) # (8205, 300)
    print(itoa_emb.shape)
    itoa_emb = Variable(torch.from_numpy(itoa_emb)).cuda() # (8205, 300)
    itoa_emb = itoa_emb.unsqueeze(0) # (1, 8205, 300)

    qid_ocr_file = os.path.join(cfg.DATA_DIR, 'qid_ocr_{}.json'.format(cfg.TEST.SPLITS[0]))
    qid_ocr = json.load(open(qid_ocr_file, 'r'))
    itoa = val_loader.dataset.codebook['itoa']
    print(len(itoa))

    fnames = [name for name in os.listdir(cfg.LOG_DIR) if name.endswith('.pth.tar')]
    print(len(fnames))
    print(fnames)

    conf_buff = np.zeros((len(val_set), len(itoa)+50))
    results = []
    for name in fnames:
        # predict
        model.eval()
        cp_file = os.path.join(cfg.LOG_DIR, name)
        print("[Info] loading checkpoint '{}'".format(cp_file))
        checkpoint = torch.load(cp_file)
        model.load_state_dict(checkpoint['state_dict'])

        bar = progressbar.ProgressBar()
        start = 0
        # sample: (que_id, img, que, [obj])
        for sample in bar(val_loader):
            sample_var = [Variable(d).cuda() for d in list(sample)[1:]]
            fuse_emb = model(*sample_var) # (64, 300)
            ans_emb = torch.cat((itoa_emb.expand(sample_var[-1].data.size(0),-1,-1), sample_var[-1]), 1) # (64, 8205+50, 300)
            score = torch.bmm(ans_emb, fuse_emb.unsqueeze(2)).squeeze() # (64, 8205+50)
            bs = score.size(0)
            conf_buff[start:start+bs] += score.data.cpu().numpy()
            start += bs

    results.extend(format_result(val_set.que_id, torch.from_numpy(conf_buff), itoa, qid_ocr))
    result_file = os.path.join(cfg.LOG_DIR, 'result-conf.json')
    json.dump(results, open(result_file, 'w'))

    # VQA eval tools
    do_test = (len(cfg.TEST.SPLITS) == 1 and cfg.TEST.SPLITS[0] in ('train', 'val'))
    if do_test:
        vqa_eval = get_eval(results, cfg.TEST.SPLITS[0])
        print(vqa_eval.accuracy['overall'])
        acc_file = os.path.join(cfg.LOG_DIR, 'accuracy-model-best.json')
        json.dump(vqa_eval.accuracy, open(acc_file, 'w'))


def format_result(que_ids, scores, itoa, qid_ocr):
    ans_len = len(itoa)
    result = []
    for i, que_id in enumerate(que_ids):
        ocr_list = qid_ocr[str(que_id)]
        _, ans_id = torch.max(scores[i][:ans_len+len(ocr_list)], dim=0)
        if ans_id[0] < ans_len:
            result.append({'question_id': int(que_id), 'answer': itoa[ans_id[0]]})
        else:
            result.append({'question_id': int(que_id), 'answer': ocr_list[ans_id[0]-ans_len]})
    return result


if __name__ == '__main__':
    main()