import argparse
import sys
import os
import shutil
import time
import logging
import datetime
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import progressbar
import numpy as np
from visdom import Visdom

from dataset import VQADataset
from models import Baseline, ConceptAttModel, MergeAttModel, AllAttModel
from models import SplitAttModel
from eval_tools import get_eval


parser = argparse.ArgumentParser(description='Train VQA model')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--log-dir', default='log', metavar='DIR',
                    help='directory of log files')


torch.manual_seed(42)
torch.cuda.manual_seed(42)

logger = logging.getLogger('vqa')
logger.setLevel(logging.DEBUG)


def main():
    global args
    args = parser.parse_args()
    args_str = json.dumps(vars(args), indent=2)
    print(args_str)
    with open('data/args_train', 'w') as f:
        f.write(args_str)

    # use timestamp as log subdirectory
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    args.log_dir = os.path.join(args.log_dir, timestamp)
    os.mkdir(args.log_dir)
    shutil.copy('data/args_get_data', args.log_dir)
    shutil.copy('data/args_train', args.log_dir)

    # init ploter
    ploter = Ploter(timestamp)

    # setting log handlers
    fh = logging.FileHandler(os.path.join(args.log_dir, 'log'))
    fh.setLevel(logging.DEBUG)
    fhc = logging.FileHandler('current.log')
    fhc.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)

    fmt = '[%(asctime)-15s] %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt)

    fh.setFormatter(formatter)
    fhc.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(fhc)
    logger.addHandler(sh)

    # data
    logger.debug('[Info] init dataset')
    trn_set = VQADataset('data', 'train')
    val_set = VQADataset('data', 'test')

    train_loader = torch.utils.data.DataLoader(
            trn_set, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # model
    logger.debug('[Info] construct model, criterion and optimizer')
    model = MergeAttModel(num_words=trn_set.num_words,
                            num_ans=trn_set.num_ans,
                            num_objs=trn_set.num_objs)

    # initialize word embedding with pretrained
    emb = model.we.weight.data.numpy()
    words = trn_set.codebook['itow']
    with open('data/word-embedding/glove.6B.300d.txt') as f:
        word_vec_txt = [l.strip().split(' ', 1) for l in f.readlines()]
    vocab, vecs_txt = zip(*word_vec_txt)
    # fromstring faster than loadtxt
    vecs = np.fromstring(' '.join(vecs_txt), dtype='float32', sep=' ')
    vecs = vecs.reshape(-1, 300)
    word_vec = dict(zip(vocab, vecs))
    assert '<PAD>' not in word_vec
    for i, w in enumerate(words):
        if w in word_vec:
            emb[i] = word_vec[w]
    model.we.weight = nn.Parameter(torch.from_numpy(emb))

    # initialize object embedding with pretrained
    obj_emb = model.obj_net[0].weight.data.numpy()
    for i, line in enumerate(trn_set.objects_vocab):
        synonyms = line.split(',')
        act_num = []
        for label in synonyms:
            words = label.split()
            act = 0
            for word in words:
                if word in word_vec:
                    act += 1
            act_num.append(act)
        act_idx = max(range(len(act_num)), key=lambda x: act_num[x])
        vec = np.zeros((300,), dtype='float32')
        if act_num[act_idx] > 0:
            for word in synonyms[act_idx]:
                if word in word_vec:
                    vec += word_vec[word]
            vec /= act_num[act_idx]
            obj_emb[i] = vec
    model.obj_net[0].weight = nn.Parameter(torch.from_numpy(obj_emb))

    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), args.lr,
                                    weight_decay=args.weight_decay)
    cudnn.benchmark = True


    # train
    logger.debug('[Info] start training...')
    best_acc = 0
    best_epoch = -1
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        loss = train(train_loader, model, criterion, optimizer, epoch)
        acc = validate(val_loader, model, criterion, epoch)

        ploter.append(epoch, loss, 'train-loss')
        ploter.append(epoch, acc, 'val-acc')

        if acc > best_acc:
            is_best = True
            best_acc = acc
            best_epoch = epoch

        logger.debug('Evaluate Result:\t'
                     'Acc  {0}\t'
                     'Best {1} ({2})'.format(acc, best_acc, best_epoch))

        # save checkpoint
        cp_fname = 'checkpoint-{:03}.pth.tar'.format(epoch)
        cp_path = os.path.join(args.log_dir, cp_fname)
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()
            }
        torch.save(state, cp_path)
        if is_best:
            best_path = os.path.join(args.log_dir, 'model-best.pth.tar')
            shutil.copyfile(cp_path, best_path)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, (img, que_id, que, obj, ans) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img_var = torch.autograd.Variable(img).cuda()
        que_var = torch.autograd.Variable(que).cuda()
        obj_var = torch.autograd.Variable(obj).cuda()
        ans_var = torch.autograd.Variable(ans).cuda()

        score = model(img_var, que_var, obj_var)
        loss = criterion(score, ans_var)

        losses.update(loss.data[0], img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.debug(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                 epoch, i, len(train_loader), batch_time=batch_time,
                 data_time=data_time, loss=losses))
    return losses.avg


def validate(val_loader, model, criterion, epoch):
    model.eval()
    itoa = val_loader.dataset.codebook['itoa']

    results = []
    end = time.time()
    bar = progressbar.ProgressBar()
    for img, que_id, que, obj in bar(val_loader):
        img_var = torch.autograd.Variable(img).cuda()
        que_var = torch.autograd.Variable(que).cuda()
        obj_var = torch.autograd.Variable(obj).cuda()

        score = model(img_var, que_var, obj_var)

        results.extend(format_result(que_id, score, itoa))

    vqa_eval = get_eval(results, 'val2014')

    # save result and accuracy
    result_file = os.path.join(args.log_dir,
                               'result-{:03}.json'.format(epoch))
    json.dump(results, open(result_file, 'w'))
    acc_file = os.path.join(args.log_dir,
                            'accuracy-{:03}.json'.format(epoch))
    json.dump(vqa_eval.accuracy, open(acc_file, 'w'))

    return vqa_eval.accuracy['overall']


def format_result(que_ids, scores, itoa):
    _, ans_ids = torch.max(scores.data, dim=1)

    result = []
    for que_id, ans_id in zip(que_ids, ans_ids):
        result.append({'question_id': que_id,
                       'answer': itoa[ans_id]})
    return result


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    pass


class Ploter(object):
    def __init__(self, env_name):
        self.viz = Visdom(env=env_name)
        self.win = None

    def append(self, x, y, name):
        if self.win is not None:
            self.viz.updateTrace(
                    X=np.array([x]),
                    Y=np.array([y]),
                    win=self.win,
                    name=name)
        else:
            self.win = self.viz.line(
                    X=np.array([x]),
                    Y=np.array([y]),
                    opts={'legend': [name]})


if __name__ == '__main__':
    main()

