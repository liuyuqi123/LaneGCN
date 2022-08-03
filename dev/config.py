"""
Just the config file.

Modified based on the original config dict.

"""

import numpy as np
import os
import sys
from fractions import gcd
from numbers import Number

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from data import ArgoDataset, collate_fn
from utils import gpu, to_long,  Optimizer, StepLR

from layers import Conv1d, Res1d, Linear, LinearRes, Null
from numpy import float64, ndarray
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


file_path = os.path.abspath(__file__)
root_path = os.path.dirname(file_path)
model_name = os.path.basename(file_path).split(".")[0]


### config ###
config = dict()

"""Train"""
config["display_iters"] = 205942
config["val_iters"] = 205942 * 2
config["save_freq"] = 1.0
config["epoch"] = 0
config["horovod"] = True
config["opt"] = "adam"
config["num_epochs"] = 36
config["lr"] = [1e-3, 1e-4]
config["lr_epochs"] = [32]
config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])

# TODO use related path
config['save_dir'] = '/home/liuyuqi/PycharmProjects/LaneGCN/dev/model/36.000.ckpt'


if "save_dir" not in config:
    config["save_dir"] = os.path.join(
        root_path, "results", model_name
    )

if not os.path.isabs(config["save_dir"]):
    config["save_dir"] = os.path.join(root_path, "results", config["save_dir"])

config["batch_size"] = 32
config["val_batch_size"] = 32
config["workers"] = 0
config["val_workers"] = config["workers"]


"""Dataset"""
# Raw Dataset
config["train_split"] = os.path.join(
    root_path, "dataset/train/data"
)
# config["val_split"] = os.path.join(root_path, "dataset/val/data")

config["val_split"] = '/home/liuyuqi/PycharmProjects/LaneGCN_datasets/dataset/val/data'

config["test_split"] = os.path.join(root_path, "dataset/test_obs/data")

# ==================================================
# modifications
config['sample_split'] = os.path.join(root_path, "dataset/forecasting_sample/data")

# Preprocessed Dataset
# todo check the usage: whether use preprocessed, this option is available when the preprocess data is prepared
# config["preprocess"] = False  # True

# the val preprocess is used in debug
config["preprocess"] = True  #


config["preprocess_train"] = os.path.join(
    root_path, "dataset", "preprocess", "train_crs_dist6_angle90.p"
)

# config["preprocess_val"] = os.path.join(
#     root_path, "dataset", "preprocess", "val_crs_dist6_angle90.p"
# )

# debug the preprocess data
# as the file
config["preprocess_val"] = '/home/liuyuqi/PycharmProjects/LaneGCN/dev/dataset/preprocess/val_crs_dist6_angle90.p'

# as the path
# config["preprocess_val"] = '/home/liuyuqi/PycharmProjects/LaneGCN/dev/dataset/preprocess/'




config['preprocess_test'] = os.path.join(root_path, "dataset", 'preprocess', 'test_test.p')

# ==================================================
# todo check the file format
# this is the sample dataset
config["preprocess_sample"] = os.path.join(
    root_path, "dataset", "preprocess", "sample.p"
)


"""Model"""
config["rot_aug"] = False
config["pred_range"] = [-100.0, 100.0, -100.0, 100.0]
config["num_scales"] = 6
config["n_actor"] = 128
config["n_map"] = 128
config["actor2map_dist"] = 7.0
config["map2actor_dist"] = 6.0
config["actor2actor_dist"] = 100.0
config["pred_size"] = 30
config["pred_step"] = 1
config["num_preds"] = config["pred_size"] // config["pred_step"]
config["num_mods"] = 6
config["cls_coef"] = 1.0
config["reg_coef"] = 1.0
config["mgn"] = 0.2
config["cls_th"] = 2.0
config["cls_ignore"] = 0.2
### end of config ###
