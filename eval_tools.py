import sys
from config import cfg
sys.path.insert(0, cfg.VQA_DIR + '/PythonHelperTools/vqaTools')
sys.path.insert(0, cfg.VQA_DIR + '/PythonEvaluationTools')
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval

import json
import os
import tempfile
from operator import itemgetter

que_fname_ptn = '{}/Questions/v2_OpenEnded_mscoco_{}_questions.json'
ann_fname_ptn = '{}/Annotations/v2_mscoco_{}_annotations.json'

trn_que_fname = que_fname_ptn.format(cfg.VQA_DIR, 'train2014')
trn_ann_fname = ann_fname_ptn.format(cfg.VQA_DIR, 'train2014')
trn_vqa = VQA(trn_ann_fname, trn_que_fname)

val_que_fname = que_fname_ptn.format(cfg.VQA_DIR, 'val2014')
val_ann_fname = ann_fname_ptn.format(cfg.VQA_DIR, 'val2014')
val_vqa = VQA(val_ann_fname, val_que_fname)


def get_eval(result, split):
    if split == 'train2014':
        vqa = trn_vqa
        que_fname = trn_que_fname
    elif split == 'val2014':
        vqa = val_vqa
        que_fname = val_que_fname
    else:
        raise ValueError('split must be "train2014" or "val2014"')

    res_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    json.dump(result, res_file)
    res_file.close()

    vqa_res = vqa.loadRes(res_file.name, que_fname)
    vqa_eval = VQAEval(vqa, vqa_res, n=2)

    vqa_eval.evaluate()

    os.unlink(res_file.name)
    return vqa_eval

