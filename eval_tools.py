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


class VQATool(object):
    _instances = {}

    def __init__(self):
        self.que_fptn = '{}/Questions/Questions_TextVQA_{}.json'
        self.ann_fptn = '{}/Annotations/Annotations_TextVQA_{}.json'


    def get_que_path(self, vqa_dir, split):
        return self.que_fptn.format(vqa_dir, split)


    def get_ann_path(self, vqa_dir, split):
        return self.ann_fptn.format(vqa_dir, split)


    def get_vqa(self, vqa_dir, split):
        if split not in self._instances:
            que_fname = self.get_que_path(vqa_dir, split)
            ann_fname = self.get_ann_path(vqa_dir, split)
            self._instances[split] = VQA(ann_fname, que_fname)
        return self._instances[split]


def get_eval(result, split):
    if split not in ('train', 'val'):
        raise ValueError('split must be "train" or "val"')
    vqa_tool = VQATool()
    vqa = vqa_tool.get_vqa(cfg.VQA_DIR, split)
    que_fname = vqa_tool.get_que_path(cfg.VQA_DIR, split)

    res_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    json.dump(result, res_file)
    res_file.close()

    vqa_res = vqa.loadRes(res_file.name, que_fname)
    vqa_eval = VQAEval(vqa, vqa_res, n=2)

    vqa_eval.evaluate()

    os.unlink(res_file.name)
    return vqa_eval

def get_eval_subset(result, split, quesIds):
    if split not in ('train', 'val'):
        raise ValueError('split must be "train" or "val"')
    vqa_tool = VQATool()
    vqa = vqa_tool.get_vqa(cfg.VQA_DIR, split)
    que_fname = vqa_tool.get_que_path(cfg.VQA_DIR, split)

    res_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    json.dump(result, res_file)
    res_file.close()

    vqa_res = vqa.loadResNoAssert(res_file.name, que_fname)
    vqa_eval = VQAEval(vqa, vqa_res, n=2)

    vqa_eval.evaluate(quesIds=quesIds)

    os.unlink(res_file.name)
    return vqa_eval

