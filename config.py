# config system imitating Faster RCNN

import os
import os.path as osp
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# print debug information
__C.DEBUG = True

# Path to data
__C.DATA_DIR = 'data'

# Path to vqa tools
__C.VQA_DIR = 'vqa-tools'

# Path to log files
__C.LOG_DIR = 'log'

# Splits of VQA to use during training
__C.TRAIN_SPLITS = ('train2014',)

# Splits of VQA to use during testing
__C.TEST_SPLITS = ('val2014',)

# Minimun frequency of the answer which can be choosed as a candidate
__C.MIN_ANS_FREQ = 16

# Minimun frequency of the word which can be included in the vocaburay
__C.MIN_WORD_FREQ = 1

# Maximun length of the question which the model take as input
# the question longer than that will be truncated
__C.MAX_QUESTION_LEN = 14

# Random seed
__C.USE_RANDOM_SEED = True
__C.SEED = 42

# minibatch size (number of question)
__C.BATCH_SIZE = 512

# learning rate
__C.LEARNING_RATE = 3e-4

# weight decay
__C.WEIGHT_DECAY = 0


##############################################################################
# Copy from RCNN


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value

