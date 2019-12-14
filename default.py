# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0
_C.LOG_CONFIG = "logging.conf"
_C.WORLD_SIZE = 8
_C.SEED = 42

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'cls_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CN(new_allowed=True)

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# DATASET 
_C.DATASET = CN()
_C.DATASET.DATASET = 'moments'
_C.DATASET.DATA_FORMAT ='avi'
_C.DATASET.ROOT = '/datasets/Moments_in_Time_Small'
_C.DATASET.TEST_SET ='val'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.N_CLASSES = 39

# TRAIN
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE_PER_GPU = 4
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 100
_C.TRAIN.RESUME = True
_C.TRAIN.LR_FACTOR = 0.5
_C.TRAIN.LR_STEP_SIZE = 15
_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.LR = 0.03
_C.TRAIN.CLIP_LEN = 32
_C.TRAIN.RESIZE_H_W = (128, 170)
_C.TRAIN.CROP_SIZE = 128
_C.TRAIN.MODEL_DIR = "model"


# TEST
_C.TEST = CN()
_C.TEST.MODEL_FILE = ''
_C.TEST.MODEL_PATH = ''
_C.TEST.BATCH_SIZE_PER_GPU =8
_C.TEST.NUM_WORKERS = 4
_C.TEST.CLIP_LEN = 32
_C.TEST.RESIZE_H_W = (128, 170)
_C.TEST.CROP_SIZE = 128

# DEBUG
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False


def update_config(cfg, options=None, config_file=None):
    cfg.defrost()

    if config_file:
        cfg.merge_from_file(config_file)

    if options:
        cfg.merge_from_list(options)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)