import argparse
import sys
import os
import time
import logging
import json
import math
import pickle
import copy
from importlib import import_module

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np

from dataset import VQADataset
from eval_tools import get_eval, get_eval_subset
from config import cfg, cfg_from_file, cfg_from_list, get_emb_size
from predict import predict

from sklearn.cluster import KMeans


parser = argparse.ArgumentParser(description='Train VQA model')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--model', '-m', default='normal/V2V',
                    help='name of the model')
parser.add_argument('--timestamp', '-t', default='20190129201205',
                    help='timestamp')
parser.add_argument('--gpu_id', default=0, type=int, metavar='N',
                    help='index of the gpu')
parser.add_argument('--bs', '--batch-size', default=64, type=int, metavar='N',
                    help='batch size')
parser.add_argument('--cfg', dest='cfg_file', default=None, type=str,
                    help='optional config file')
parser.add_argument('--set', dest='set_cfgs', default=None,
                    nargs=argparse.REMAINDER, help='set config keys')
#parser.add_argument('--clus_id', default=0, type=int, metavar='N',
#                    help='index of the clusters')
parser.add_argument('--cluster_alg', default='kmeans',
                    help='cluster algorithm (kmeans, gmm, specclu, birch, aggclu)')
parser.add_argument('--n_clusters', default=4, type=int, metavar='N',
                    help='number of clusters to find (default: 4)')

if cfg.USE_RANDOM_SEED:
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)

logger = logging.getLogger('vqa')
logger.setLevel(logging.DEBUG)


def load_data(ver='v2', split_name='val2014', RES_DIR='result'):
    if ver == 'v1':
        qIdfname = '{}/{}/OpenEnded_mscoco_{}_questions_id.npy'.format(RES_DIR, ver, split_name)
        qFeafname = '{}/{}/OpenEnded_mscoco_{}_questions_fea.npy'.format(RES_DIR, ver, split_name)
        qId2qTypeId_fname = '{}/{}/mscoco_{}_annotations_qid2qtid.json'.format(RES_DIR, ver, split_name)
        #qId2aTypeId_fname = '{}/{}/mscoco_{}_annotations_qid2atid.json'.format(RES_DIR, ver, split_name)
    elif ver == 'v2':
        qIdfname = '{}/{}/v2_OpenEnded_mscoco_{}_questions_id.npy'.format(RES_DIR, ver, split_name)
        qFeafname = '{}/{}/v2_OpenEnded_mscoco_{}_questions_fea.npy'.format(RES_DIR, ver, split_name)
        qId2qTypeId_fname = '{}/{}/v2_mscoco_{}_annotations_qid2qtid.json'.format(RES_DIR, ver, split_name)
        #qId2aTypeId_fname = '{}/{}/v2_mscoco_{}_annotations_qid2atid.json'.format(RES_DIR, ver, split_name)

    logger.debug('[Load] {}'.format(qIdfname))
    queIds = np.load(qIdfname)
    logger.debug('[Load] {}'.format(qFeafname))
    queFea = np.load(qFeafname)
    logger.debug('[Load] {}'.format(qId2qTypeId_fname))
    with open(qId2qTypeId_fname, 'r') as f:
        qid2qtid = json.load(f)
    qTypeIds = np.zeros(len(queIds), dtype=int)
    for i,qid in enumerate(queIds):
        qTypeIds[i] = qid2qtid[str(int(qid))]

    return queIds, queFea, qTypeIds

def clustering(X, clu_num, clu_alg='kmeans', savefname=None):
    # load existing clustering object
    if savefname and os.path.exists(savefname):
        with open(savefname, 'rb') as f:
            clustering=pickle.load(f)
        newLables = clustering.predict(X)
    else:
        if clu_alg == 'kmeans':
            clustering = KMeans(n_clusters=clu_num, random_state=0)
        else:
            logger.error('[ERROR] wrong cluster_alg: '+clu_alg)
        logger.debug('[Info] clustering {} centroids using: {} ...'.format(clu_num, clu_alg))
        newLables = clustering.fit_predict(X)
    return newLables

def select_subset(VQAset, sel=None):
    if sel is None:
        return VQAset
    else:
        #sel = [qid in quesIds for qid in VQAset.que_id]
        #sel = np.array(sel)
        newVQAset = copy.deepcopy(VQAset)
        newVQAset.img_pos = VQAset.img_pos[sel]
        newVQAset.que = VQAset.que[sel]
        newVQAset.que_id = VQAset.que_id[sel]
        if 'train2014' in VQAset.splits:
            newVQAset.ans = VQAset.ans[sel]
        return newVQAset

def main():
    global args
    args = parser.parse_args()
    args_str = json.dumps(vars(args), indent=2)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # use timestamp as log subdirectory
    timestamp = args.timestamp
    cfg.LOG_DIR= os.path.join(cfg.LOG_DIR, timestamp)
    model_group_name, model_name = args.model.split('/')

    # setting log handlers
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.debug('[Info] called with: ' + args_str)
    logger.debug('[Info] timestamp: ' + timestamp)

    # select device
    torch.cuda.set_device(args.gpu_id)
    logger.debug('[Info] use gpu: {}'.format(torch.cuda.current_device()))

    # data
    assert(len(cfg.TEST.SPLITS) == 1 and cfg.TEST.SPLITS[0] in ('val2014'))
    logger.debug('[Info] init dataset')
    val_set = VQADataset('test', model_group_name)
    RES_DIR = '/home/lyt/code/bert-as-service-test/result'
    queIds, queFea, _ = load_data(split_name='val2014', RES_DIR=RES_DIR)
    assert queIds.tolist() == val_set.que_id.tolist()
    logger.debug('[Info] Clustering using {}, {} clusters'
        .format(args.cluster_alg, args.n_clusters))
    clusfilename = '{}/{}/{}_{}_n{}.pkl'.format(RES_DIR, 'v2', 'train2014', 
                        args.cluster_alg, args.n_clusters)
    logger.debug('[Info] cluster file: {}'.format(clusfilename))
    val_qTypeLabels = clustering(queFea, clu_num=args.n_clusters, 
                        clu_alg=args.cluster_alg, savefname=clusfilename)

    # model
    logger.debug('[Info] construct model')
    model_group = import_module('models.' + model_group_name)
    model = getattr(model_group, model_name)(
            num_words=val_set.num_words,
            num_ans=val_set.num_ans,
            emb_size=get_emb_size())
    logger.debug('[Info] model name: ' + args.model)
    total_param = 0
    for param in model.parameters():
        total_param += param.nelement()
    logger.debug('[Info] total parameters: {}M'
            .format(math.ceil(total_param / 2**20)))
    model.cuda()
    cudnn.benchmark = True

    # load best model, predict
    logger.debug('[Info] load model ...')
    best_path = os.path.join(cfg.LOG_DIR, 'model-best.pth.tar')
    #if os.path.isfile(best_path):
    assert os.path.isfile(best_path)
    logger.debug("[Info] loading checkpoint '{}'".format(best_path))
    cp_state = torch.load(best_path)
    best_acc = cp_state['best_acc']
    logger.debug('[Info] best model with best acc {}'.format(best_acc))
    model.load_state_dict(cp_state['state_dict'])
    #else:
    #    logger.debug("[Info] no checkpoint found at '{}'".format(best_path))

    for i in range(args.n_clusters):
        logger.debug('[Info] choose cluster ID: {}'.format(i))
        #sel = val_qTypeLabels == args.clus_id
        sel = val_qTypeLabels == i
        val_quesIds = queIds[sel].tolist()
        logger.debug('[Info] #Val set before/after clustering and choosing {}/{}'
            .format(queIds.shape[0], len(val_quesIds)))
        val_set_sub = select_subset(val_set, sel)
        val_loader = torch.utils.data.DataLoader(
                val_set_sub, batch_size=args.bs, shuffle=False,
                num_workers=args.workers, pin_memory=True)
        logger.debug('sample count: {}'.format(len(val_set_sub)))
        acc = validate(val_loader, model, None, None, quesIds=val_quesIds)
        logger.debug('Evaluate Result:\tAcc  {0}'.format(acc))


def validate(val_loader, model, criterion, epoch, quesIds=None):
    results = predict(val_loader, model)
    if quesIds is None:
        vqa_eval = get_eval(results, cfg.TEST.SPLITS[0])
    else:
        vqa_eval = get_eval_subset(results, cfg.TEST.SPLITS[0], quesIds)

    # save result and accuracy
    #result_file = os.path.join(cfg.LOG_DIR, 'result-{:03}.json'.format(epoch))
    #json.dump(results, open(result_file, 'w'))
    #acc_file = os.path.join(cfg.LOG_DIR, 'accuracy-{:03}.json'.format(epoch))
    #json.dump(vqa_eval.accuracy, open(acc_file, 'w'))
    print(vqa_eval.accuracy['overall'])
    print(vqa_eval.accuracy['perAnswerType'])
    return vqa_eval.accuracy['overall']


if __name__ == '__main__':
    main()

