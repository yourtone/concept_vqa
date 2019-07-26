import argparse
import os
import json
from importlib import import_module

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import progressbar
from torch.autograd import Variable

from dataset import VQADataset
from config import cfg, cfg_from_file, cfg_from_list, get_emb_size


parser = argparse.ArgumentParser(description='Train VQA model')
parser.add_argument('checkpoint', metavar='DIR',
                    help='directory of checkpoints')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--start-epoch', default=15, type=int, metavar='N',
                    help='epoch evaluation starting at')
parser.add_argument('--end-epoch', default=45, type=int, metavar='N',
                    help='epoch evaluation ending at')
parser.add_argument('--epoch-freq', default=5, type=int, metavar='N',
                    help='number of epochs to skip for every evaluation')
parser.add_argument('--model', '-m', default='ocr_label/MFHModel',
                    help='name of the model')
parser.add_argument('--bs', '--batch_size', default=128, type=int,
                    help='batch size for predicting')
parser.add_argument('--gpu_id', default=0, type=int, metavar='N',
                    help='index of the gpu')


def main():
    global args
    args = parser.parse_args()
    args_str = json.dumps(vars(args), indent=2)
    print('[Info] called with: ' + args_str)

    # checkpoint directory
    cfg.LOG_DIR= os.path.join(cfg.LOG_DIR, args.checkpoint)

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
    print('sample count: {}'.format(len(val_set)))

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

    # predict
    fnames = [(i, 'checkpoint-{:03}.pth.tar'.format(i)) for i in range(
            args.start_epoch, args.end_epoch, args.epoch_freq)]
    cp_files = [(i, os.path.join(cfg.LOG_DIR, fname)) for i, fname in fnames]
    for epoch, cp_file in cp_files:
        if os.path.isfile(cp_file):
            print("[Info] loading checkpoint '{}'".format(cp_file))
            checkpoint = torch.load(cp_file)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("[Info] no checkpoint found at '{}'".format(cp_file))
            continue

        results = predict(val_loader, model)
        result_file = os.path.join(cfg.LOG_DIR,
                                   'result-{:03}.json'.format(epoch))
        json.dump(results, open(result_file, 'w'))


def predict(val_loader, model):
    model.eval()
    itoa = val_loader.dataset.codebook['itoa']

    results = []
    bar = progressbar.ProgressBar()
    # sample: (que_id, img, que, [obj])
    for sample in bar(val_loader):
        sample_var = [Variable(d).cuda() for d in list(sample)[1:]]
        score = model(*sample_var)
        results.extend(format_result(sample[0], score, itoa))

    return results


def predict_train(train_loader, model):
    model.eval()
    itoa = train_loader.dataset.codebook['itoa']

    results = []
    bar = progressbar.ProgressBar()
    # sample: (que_id, img, que, [obj], ans)
    for sample in bar(train_loader):
        sample_var = [Variable(d).cuda() for d in list(sample)[1:-1]]
        score = model(*sample_var)
        results.extend(format_result(sample[0], score, itoa))

    return results


def format_result(que_ids, scores, itoa):
    _, ans_ids = torch.max(scores.data, dim=1)

    result = []
    for que_id, ans_id in zip(que_ids, ans_ids):
        result.append({'question_id': que_id,
                       'answer': itoa[ans_id]})
    return result


if __name__ == '__main__':
    main()

