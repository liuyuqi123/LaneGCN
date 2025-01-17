"""
original intro:
Preprocess the data(csv), build graph from the HDMAP and saved as pkl

modifications:
 - check the preprocess of HDMap info

"""

import argparse
import os
import pickle
import random
import sys
import time
from importlib import import_module

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

# # original line
# from data import ArgoDataset as Dataset, from_numpy, ref_copy, collate_fn

from dev.argoverse_dataset import ArgoDataset as Dataset
from data import from_numpy, ref_copy, collate_fn

from utils import Logger, load_pretrain, gpu

os.umask(0)


root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


parser = argparse.ArgumentParser(
    description="Data preprocess for argo forcasting dataset"
)
parser.add_argument(
    "-m", "--model", default="lanegcn", type=str, metavar="MODEL", help="model name"
)


# ==========  original main method  ==========
# def main():
#     # Import all settings for experiment.
#     args = parser.parse_args()
#     model = import_module(args.model)
#     print(args.model)
#     config, *_ = model.get_model()
#
#     config["preprocess"] = False  # we use raw data to generate preprocess data
#     config["val_workers"] = 32
#     config["workers"] = 32
#     config['cross_dist'] = 6
#     config['cross_angle'] = 0.5 * np.pi
#
#     os.makedirs(os.path.dirname(config['preprocess_train']),exist_ok=True)
#
#     # # original line
#     # val(config)
#     # test(config)
#     # train(config)


def main():
    """
    Debug the methods.
    """

    # debug with the sample data.
    run_debug()


def run_debug():
    """
    Debug and tune the graph building.
    :return:
    """

    # import the config dict from file
    # use this as the only entrance of config file for dev version code
    from config import config

    # we use raw data to generate preprocess data
    config["preprocess"] = False  # False if preprocess data files exist
    config["val_workers"] = 32
    config["workers"] = 0  # 32
    config['cross_dist'] = 6
    config['cross_angle'] = 0.5 * np.pi

    os.makedirs(os.path.dirname(config['preprocess_sample']), exist_ok=True)

    # # original line
    # val(config)
    # test(config)
    # train(config)

    # debug the methods
    debug_sample(config)


def debug_sample(config):
    """
    Debug the laneGCN methods using sample dataset.

    :param config:
    :return:
    """
    # Data loader for training set
    dataset = Dataset(
        config["sample_split"],
        config,
        train=False
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=False,  # False for debug and test
        collate_fn=collate_fn,
        pin_memory=True,  # todo check this param transform path
        drop_last=False,
    )

    # todo how to determine the loop range
    #  check the range number
    weired_range = int(100)
    stores = [None for x in range(weired_range)]  # list to store result
    t = time.time()

    # todo check if the tqdm issue???
    # for i, data in enumerate(tqdm(train_loader)):

    # don't use the tqdm
    for i, data in enumerate(train_loader):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "idx",
                "city",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "gt_preds",
                "has_preds",
                "graph",
            ]:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    dataset = PreprocessDataset(stores, config, train=True)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    # the tag sample refers to the sample dataset
    modify(config, data_loader, config["preprocess_sample"])


def train(config):
    # Data loader for training set
    dataset = Dataset(config["train_split"], config, train=True)
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    stores = [None for x in range(205942)]
    t = time.time()
    for i, data in enumerate(tqdm(train_loader)):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "idx",
                "city",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "gt_preds",
                "has_preds",
                "graph",
            ]:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()



    dataset = PreprocessDataset(stores, config, train=True)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    modify(config, data_loader,config["preprocess_train"])


def val(config):
    # Data loader for validation set
    dataset = Dataset(config["val_split"], config, train=False)
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    stores = [None for x in range(39472)]

    t = time.time()
    for i, data in enumerate(tqdm(val_loader)):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "idx",
                "city",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "gt_preds",
                "has_preds",
                "graph",
            ]:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    dataset = PreprocessDataset(stores, config, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    modify(config, data_loader,config["preprocess_val"])


def test(config):
    dataset = Dataset(config["test_split"], config, train=False)
    test_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    stores = [None for x in range(78143)]

    t = time.time()
    for i, data in enumerate(tqdm(test_loader)):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "idx",
                "city",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "graph",
            ]:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    dataset = PreprocessDataset(stores, config, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    modify(config, data_loader, config["preprocess_test"])


def to_numpy(data):
    """Recursively transform torch.Tensor to numpy.ndarray.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_numpy(x) for x in data]
    if torch.is_tensor(data):
        data = data.numpy()
    return data


def to_int16(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_int16(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_int16(x) for x in data]
    if isinstance(data, np.ndarray) and data.dtype == np.int64:
        data = data.astype(np.int16)
    return data


def modify(config, data_loader, save):
    """

    :param config:
    :param data_loader:
    :param save: file path, pickle will create the folder if not exists
    :return:
    """
    t = time.time()
    store = data_loader.dataset.split
    for i, data in enumerate(data_loader):
        data = [dict(x) for x in data]

        out = []
        for j in range(len(data)):
            out.append(preprocess(to_long(gpu(data[j])), config['cross_dist']))

        for j, graph in enumerate(out):
            idx = graph['idx']
            store[idx]['graph']['left'] = graph['left']
            store[idx]['graph']['right'] = graph['right']

        if (i + 1) % 100 == 0:
            print((i + 1) * config['batch_size'], time.time() - t)
            t = time.time()

    f = open(os.path.join(root_path, 'preprocess', save), 'wb')
    pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


class PreprocessDataset():
    def __init__(self, split, config, train=True):
        self.split = split
        self.config = config
        self.train = train

    def __getitem__(self, idx):
        from data import from_numpy, ref_copy

        data = self.split[idx]
        graph = dict()
        for key in ['lane_idcs', 'ctrs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs', 'feats']:
            graph[key] = ref_copy(data['graph'][key])
        graph['idx'] = idx
        return graph

    def __len__(self):
        return len(self.split)


def preprocess(graph, cross_dist, cross_angle=None):
    """
    This is a key method to preprocess the lane data.

    :param graph:
    :param cross_dist:
    :param cross_angle:
    :return:
    """
    left, right = dict(), dict()

    lane_idcs = graph['lane_idcs']
    num_nodes = len(lane_idcs)
    num_lanes = lane_idcs[-1].item() + 1

    dist = graph['ctrs'].unsqueeze(1) - graph['ctrs'].unsqueeze(0)
    dist = torch.sqrt((dist ** 2).sum(2))
    hi = torch.arange(num_nodes).long().to(dist.device).view(-1, 1).repeat(1, num_nodes).view(-1)
    wi = torch.arange(num_nodes).long().to(dist.device).view(1, -1).repeat(num_nodes, 1).view(-1)

    # todo row_idcs refers to???
    row_idcs = torch.arange(num_nodes).long().to(dist.device)

    # this block is not used
    if cross_angle is not None:
        f1 = graph['feats'][hi]
        f2 = graph['ctrs'][wi] - graph['ctrs'][hi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = t2 - t1
        m = dt > 2 * np.pi
        dt[m] = dt[m] - 2 * np.pi
        m = dt < -2 * np.pi
        dt[m] = dt[m] + 2 * np.pi
        mask = torch.logical_and(dt > 0, dt < config['cross_angle'])
        left_mask = mask.logical_not()
        mask = torch.logical_and(dt < 0, dt > -config['cross_angle'])
        right_mask = mask.logical_not()

    pre = graph['pre_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    pre[graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1]] = 1
    suc = graph['suc_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    suc[graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1]] = 1

    # ==================================================
    # this block till the end is how the left and right connections processed

    pairs = graph['left_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        left_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        left_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            left_dist[hi[left_mask], wi[left_mask]] = 1e6

        min_dist, min_idcs = left_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        left['u'] = ui.cpu().numpy().astype(np.int16)
        left['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        left['u'] = np.zeros(0, np.int16)
        left['v'] = np.zeros(0, np.int16)

    pairs = graph['right_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        right_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        right_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            right_dist[hi[right_mask], wi[right_mask]] = 1e6

        min_dist, min_idcs = right_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        right['u'] = ui.cpu().numpy().astype(np.int16)
        right['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        right['u'] = np.zeros(0, np.int16)
        right['v'] = np.zeros(0, np.int16)

    out = dict()
    out['left'] = left
    out['right'] = right
    out['idx'] = graph['idx']
    return out



def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data


def worker_init_fn(pid):
    np_seed = hvd.rank() * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


if __name__ == "__main__":
    main()
