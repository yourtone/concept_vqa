'''
train_resume_ft_atype.py
2019/03/08 20:08 @yourtone
e.g.
python train_resume_ft_atype.py --model prob_label/DiffAttModel --epochs 80 --ft_epoch 30 --atype_id 0

Training pattern:
if ft_epoch =0       -> train_subset by atype
if ft_epoch >epochs  -> no ft
if 0<ft_epoch<epochs -> train_all_ft_atype

Resume train:
if resume =True  -> resume train from previous best model
if resume =False -> normal train

Predict subset:
if pred_subset =True  -> validate on subset by atype
if pred_subset =False -> validate on all val set
'''
import argparse
import sys
import os
import shutil
import time
import logging
import datetime
import json
import math
from importlib import import_module

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import progressbar
import numpy as np
from torch.autograd import Variable
from visdom import Visdom

from dataset import VQADataset
from eval_tools import get_eval, get_eval_subset
from config import cfg
from predict import predict


parser = argparse.ArgumentParser(description='Train VQA model')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=-1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-freq', default=100, type=int, metavar='N',
                    help='frequency of saving checkpoint')
parser.add_argument('--model', '-m', default='prob_label/DiffAttModel',
                    help='name of the model')
parser.add_argument('--gpu_id', default=0, type=int, metavar='N',
                    help='index of the gpu')
parser.add_argument('--bs', '--batch-size', default=64, type=int, metavar='N',
                    help='batch size')
parser.add_argument('--lr', default=1e-3, type=float, metavar='FLOAT',
                    help='initial learning rate')
parser.add_argument('--lr-decay-start', default=4, type=int, metavar='N',
                    help='epoch number starting decay learning rate')
parser.add_argument('--lr-decay-factor', default=0.5, type=float, metavar='FLOAT',
                    help='learning rate decay factor for every 10 epochs')
parser.add_argument('--lr-decay-freq', default=4, type=int, metavar='N',
                    help='frequency of learning rate decaying')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='FLOAT', help='weight decay')
parser.add_argument('--atype_id', default=0, type=int, metavar='N',
                    help='index of the clusters')
# make ft_epoch large for no ft, make ft_epoch 0 for train_subset
parser.add_argument('--ft_epoch', default=30, type=int, metavar='N',
                    help='at which epoch to finetune using train_subset')
parser.add_argument('-r', '--resume', action='store_true', 
                    help='resume from checkpoint')
parser.add_argument('--ts', '--timestamp', default='20190126214414',
                    help='resume from which timestamp')
parser.add_argument('--pred_subset', action='store_true', 
                    help='validate on subset by atype')

if cfg.USE_RANDOM_SEED:
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)

logger = logging.getLogger('vqa')
logger.setLevel(logging.DEBUG)


def load_data(ver='v2', split_name='val2014', RES_DIR='result'):
    if ver == 'v1':
        qIdfname = '{}/{}/OpenEnded_mscoco_{}_questions_id.npy'.format(RES_DIR, ver, split_name)
        #qFeafname = '{}/{}/OpenEnded_mscoco_{}_questions_fea.npy'.format(RES_DIR, ver, split_name)
        #qId2qTypeId_fname = '{}/{}/mscoco_{}_annotations_qid2qtid.json'.format(RES_DIR, ver, split_name)
        qId2aTypeId_fname = '{}/{}/mscoco_{}_annotations_qid2atid.json'.format(RES_DIR, ver, split_name)
    elif ver == 'v2':
        qIdfname = '{}/{}/v2_OpenEnded_mscoco_{}_questions_id.npy'.format(RES_DIR, ver, split_name)
        #qFeafname = '{}/{}/v2_OpenEnded_mscoco_{}_questions_fea.npy'.format(RES_DIR, ver, split_name)
        #qId2qTypeId_fname = '{}/{}/v2_mscoco_{}_annotations_qid2qtid.json'.format(RES_DIR, ver, split_name)
        qId2aTypeId_fname = '{}/{}/v2_mscoco_{}_annotations_qid2atid.json'.format(RES_DIR, ver, split_name)
    logger.debug('[Load] {}'.format(qIdfname))
    queIds = np.load(qIdfname)
    #logger.debug('[Load] {}'.format(qFeafname))
    #queFea = np.load(qFeafname)
    #logger.debug('[Load] {}'.format(qId2qTypeId_fname))
    #with open(qId2qTypeId_fname, 'r') as f:
        #qid2qtid = json.load(f)
    logger.debug('[Load] {}'.format(qId2aTypeId_fname))
    with open(qId2aTypeId_fname, 'r') as f:
        qid2atid = json.load(f)
    #qTypeIds = np.zeros(len(queIds), dtype=int)
    aTypeIds = np.zeros(len(queIds), dtype=int)
    for i,qid in enumerate(queIds):
        #qTypeIds[i] = qid2qtid[str(int(qid))]
        aTypeIds[i] = qid2atid[str(int(qid))]
    return queIds, aTypeIds # queIds, queFea, qTypeIds, aTypeIds

def select_subset(VQAset, sel=None):
    if sel is None:
        return VQAset
    else:
        #sel = [qid in quesIds for qid in VQAset.que_id]
        sel = np.array(sel)
        VQAset.img_pos = VQAset.img_pos[sel]
        VQAset.que = VQAset.que[sel]
        VQAset.que_id = VQAset.que_id[sel]
        if 'train2014' in VQAset.splits:
            VQAset.ans = VQAset.ans[sel]
        return VQAset

def gen_dataloader(args, data_set, is_train=True):
    data_loader = torch.utils.data.DataLoader(
            data_set, batch_size=args.bs, shuffle=is_train,
            num_workers=args.workers, pin_memory=True)
    return data_loader

def main():
    global args
    args = parser.parse_args()
    args_str = json.dumps(vars(args), indent=2)

    # use timestamp as log subdirectory
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    cfg.LOG_DIR = os.path.join(cfg.LOG_DIR, timestamp)
    os.mkdir(cfg.LOG_DIR)
    json.dump(cfg, open(cfg.LOG_DIR + '/config.json', 'w'), indent=2)
    model_group_name, model_name = args.model.split('/')
    shutil.copy('models/' + model_group_name + '.py', cfg.LOG_DIR)

    # init ploter
    ploter = Ploter(timestamp)

    # setting log handlers
    fh = logging.FileHandler(os.path.join(cfg.LOG_DIR, 'log'))
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
    logger.debug('[Info] called with: ' + args_str)

    logger.debug('[Info] timestamp: ' + timestamp)
    logger.debug('[Info] CPU random seed: {}'.format(torch.initial_seed()))
    logger.debug('[Info] GPU random seed: {}'.format(torch.cuda.initial_seed()))

    # select device
    torch.cuda.set_device(args.gpu_id)
    logger.debug('[Info] use gpu: {}'.format(torch.cuda.current_device()))

    # display some information
    train_pattern = '[Info] Training pattern: {}\n'\
        '\t[train_subset by atype, no finetune, train_all_ft_atype]'
    if args.ft_epoch == 0:
        logger.debug(train_pattern.format('train_subset by atype'))
    elif args.ft_epoch > args.epochs:
        logger.debug(train_pattern.format('no finetune'))
    else: #0<ft_epoch<epochs
        logger.debug(train_pattern.format('train_all_ft_atype'))
    resume_train = '[Info] Resume train: {}'
    if args.resume:
        logger.debug(resume_train.format('resume train from previous best model'))
    else:
        logger.debug(resume_train.format('normal train'))
    pred_sub = '[Info] Predict subset: {}'
    if args.pred_subset:
        logger.debug(pred_sub.format('validate on subset by atype'))
    else:
        logger.debug(pred_sub.format('validate on all val set'))

    # load data
    logger.debug('[Info] init dataset')
    do_test = (len(cfg.TEST.SPLITS) == 1 and cfg.TEST.SPLITS[0] in ('train2014', 'val2014'))
    trn_set = VQADataset('train', model_group_name)
    train_loader = gen_dataloader(args, trn_set, is_train=True)
    if do_test:
        val_set = VQADataset('test', model_group_name)
        val_loader = gen_dataloader(args, val_set, is_train=False)

    # model
    emb_size = 300
    if cfg.WORD_EMBEDDINGS:
        word_vec = merge_embeddings(cfg.WORD_EMBEDDINGS)
        aword = next(iter(word_vec))
        emb_size = len(word_vec[aword])
        logger.debug('[Info] embedding size: {}'.format(emb_size))

    logger.debug('[Info] construct model, criterion and optimizer')
    model_group = import_module('models.' + model_group_name)
    model = getattr(model_group, model_name)(
            num_words=trn_set.num_words,
            num_ans=trn_set.num_ans,
            emb_size=emb_size)
    logger.debug('[Info] model name: ' + args.model)
    total_param = 0
    for param in model.parameters():
        total_param += param.nelement()
    logger.debug('[Info] total parameters: {}M'.format(math.ceil(total_param / 2**20)))

    # initialize word embedding with pretrained
    if cfg.WORD_EMBEDDINGS:
        emb = model.we.weight.data.numpy()
        words = trn_set.codebook['itow']
        assert '<PAD>' not in word_vec
        fill_cnt = 0
        for i, w in enumerate(words):
            if w in word_vec:
                emb[i] = word_vec[w]
                fill_cnt += 1
        logger.debug('[debug] word embedding filling count: {}/{}'
                .format(fill_cnt, len(words)))
        model.we.weight = nn.Parameter(torch.from_numpy(emb))
        if model_group_name in ('onehot_label', 'prob_label'):
            # initialize object embedding with pretrained
            obj_emb = model.obj_net[0].weight.data.numpy()
            if model_group_name == 'prob_label':
                obj_emb = obj_emb.T
            fill_cnt = 0
            for i, line in enumerate(trn_set.objects_vocab):
                avail, vec = get_class_embedding(line, word_vec, emb_size)
                if avail:
                    obj_emb[i] = vec
                    fill_cnt += 1
            logger.debug('[debug] class embedding filling count: {}/{}'
                    .format(fill_cnt, len(trn_set.objects_vocab)))
            if model_group_name == 'prob_label':
                obj_emb = obj_emb.T
            model.obj_net[0].weight = nn.Parameter(torch.from_numpy(obj_emb))

    model.cuda()

    if cfg.SOFT_LOSS:
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    logger.debug('[Info] criterion name: ' + criterion.__class__.__name__)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.wd)
    cudnn.benchmark = True

    # resume
    if args.resume:
        ckpt = torch.load(os.path.join(cfg.LOG_DIR.split('/')[0],
                            args.timestamp, 'model-best.pth.tar'))
        best_acc = ckpt['best_acc']
        start_epoch = best_epoch = ckpt['best_epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        ckpt = None
        best_acc = 0
        best_epoch = -1
        start_epoch = args.start_epoch # -1

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_freq, 
    #                        gamma=args.lr_decay_factor, last_epoch=start_epoch)

    # train
    logger.debug('[Info] start training...')
    for epoch in range(start_epoch+1, args.epochs):
        is_best = False
        lr = adjust_learning_rate(optimizer, epoch)
        ploter.append(epoch, lr, 'lr')

        if epoch == args.ft_epoch:
            logger.debug('[Info] finetune using atype_id {} data, at epoch {}'
                .format(args.atype_id, args.ft_epoch))
            # load atypes
            RES_DIR = '/home/lyt/code/bert-as-service-test/result'
            queIds, aTypeIds = load_data(split_name='train2014', RES_DIR=RES_DIR)
            assert queIds.tolist() == trn_set.que_id.tolist()
            # select specified data for training
            sel = aTypeIds == args.atype_id
            trn_quesIds = queIds[sel].tolist()
            logger.debug('[Info] #Train set before/after selecting {}/{}'
                .format(queIds.shape[0], len(trn_quesIds)))
            trn_set = select_subset(trn_set, sel)
            # set train loader
            train_loader = gen_dataloader(args, trn_set, is_train=True)
            # validation set
            if do_test and args.pred_subset:
                # load atypes
                queIds, aTypeIds = load_data(split_name='val2014', RES_DIR=RES_DIR)
                assert queIds.tolist() == val_set.que_id.tolist()
                # select specified data for training
                sel = aTypeIds == args.atype_id
                val_quesIds = queIds[sel].tolist()
                logger.debug('[Info] #Val set before/after selecting {}/{}'
                    .format(queIds.shape[0], len(val_quesIds)))
                val_set = select_subset(val_set, sel)
                # set val loader
                val_loader = gen_dataloader(args, val_set, is_train=False)


        loss = train(train_loader, model, criterion, optimizer, epoch)
        ploter.append(epoch, loss, 'train-loss')

        if do_test:
            if args.pred_subset:
                acc = validate(val_loader, model, criterion, epoch, quesIds=val_quesIds)
            else:
                acc = validate(val_loader, model, criterion, epoch)
            ploter.append(epoch, acc, 'val-acc')
            if acc > best_acc:
                is_best = True
                best_acc = acc
                best_epoch = epoch
            logger.debug('Evaluate Result:\tAcc  {0}\tBest {1} ({2})'
                .format(acc, best_acc, best_epoch))

        # save checkpoint
        state = {
            'epoch': epoch,
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        if epoch % args.save_freq == 0:
            cp_fname = 'checkpoint-{:03}.pth.tar'.format(epoch)
            cp_path = os.path.join(cfg.LOG_DIR, cp_fname)
            torch.save(state, cp_path)
        if is_best:
            best_path = os.path.join(cfg.LOG_DIR, 'model-best.pth.tar')
            torch.save(state, best_path)


def get_class_embedding(class_name, word_vec, emb_size):
    synonyms = class_name.split(',')
    act_num = []
    act_ratio = []
    for label in synonyms:
        words = label.split()
        act = sum([1 for word in words if word in word_vec])
        act_num.append(act)
        act_ratio.append(act / len(words))
    act_idx = max(range(len(act_num)), key=lambda x: act_ratio[x])
    vec = np.zeros((emb_size,), dtype='float32')
    pretrained_avail = act_num[act_idx] > 0
    if pretrained_avail:
        for word in synonyms[act_idx].split():
            if word in word_vec:
                vec += word_vec[word]
        vec /= act_num[act_idx]
    return pretrained_avail, vec


def merge_embeddings(embedding_names):
    names = embedding_names.split('+')
    vocabs = []
    vecs = []
    for name in names:
        vocab, vec = load_embeddings(name)
        vocabs.append(vocab)
        vecs.append(vec)

    final_vocab = set(vocabs[0])
    for vocab in vocabs[1:]:
        final_vocab &= set(vocab)
    final_vocab = list(final_vocab)

    final_vec = []
    for vocab, vec in zip(vocabs, vecs):
        w2i = dict(zip(vocab, range(len(vocab))))
        inds = np.array([w2i[w] for w in final_vocab])
        final_vec.append(vec[inds])
    final_vec = np.hstack(final_vec)

    return dict(zip(final_vocab, final_vec))


def load_embeddings(name):
    emb_path = '{}/word-embedding/{}'.format(cfg.DATA_DIR, name)
    logger.debug('[Load] ' + emb_path)
    with open(emb_path) as f:
        word_vec_txt = [l.rstrip().split(' ', 1) for l in f.readlines()]
    vocab, vecs_txt = zip(*word_vec_txt)
    # infer vector dimention
    vec_size = len(vecs_txt[0].split())
    # fromstring faster than loadtxt
    vecs = np.fromstring(' '.join(vecs_txt), dtype='float32', sep=' ')
    vecs = vecs.reshape(-1, vec_size)
    return vocab, vecs


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    # sample: (que_id, img, que, [obj], ans)
    for i, sample in enumerate(train_loader):
        data_time.update(time.time() - end)

        sample_var = [Variable(d).cuda() for d in list(sample)[1:]]

        score = model(*sample_var[:-1])
        loss = criterion(score, sample_var[-1])

        losses.update(loss.data[0], sample[0].size(0))

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


def validate(val_loader, model, criterion, epoch, quesIds=None):
    results = predict(val_loader, model)
    if quesIds is None:
        vqa_eval = get_eval(results, cfg.TEST.SPLITS[0])
    else:
        vqa_eval = get_eval_subset(results, cfg.TEST.SPLITS[0], quesIds)

    # save result and accuracy
    result_file = os.path.join(cfg.LOG_DIR,
                               'result-{:03}.json'.format(epoch))
    json.dump(results, open(result_file, 'w'))
    acc_file = os.path.join(cfg.LOG_DIR,
                            'accuracy-{:03}.json'.format(epoch))
    json.dump(vqa_eval.accuracy, open(acc_file, 'w'))

    return vqa_eval.accuracy['overall']


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
    exponent = max(0, (epoch - args.lr_decay_start) // args.lr_decay_freq + 1)
    lr = args.lr * (args.lr_decay_factor ** exponent)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class Ploter(object):
    def __init__(self, env_name):
        self.viz = Visdom(env=env_name)
        self.win = None

    def append(self, x, y, name):
        if self.win is not None:
            self.viz.line(
                    X=np.array([x]),
                    Y=np.array([y]),
                    win=self.win,
                    name=name,
                    update='append')
        else:
            self.win = self.viz.line(
                    X=np.array([x]),
                    Y=np.array([y]),
                    opts={'legend': [name]})


if __name__ == '__main__':
    main()

