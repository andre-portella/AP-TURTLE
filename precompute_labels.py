import argparse
import os

import numpy as np
from torch.utils.data import ConcatDataset, Subset

from dataset_preparation.data_utils import get_datasets
from utils import seed_everything


def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--root_dir', type=str, default="data")
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(args)


def get_labels(dataset):

    if isinstance(dataset, ConcatDataset):
        labels = []
        for d in dataset.datasets:
            labels.extend(get_labels(d))
            print(labels)
        return labels

    if isinstance(dataset, Subset):
        base_labels = get_labels(dataset.dataset)
        return [base_labels[i] for i in dataset.indices]

    if hasattr(dataset, "targets"):
        return dataset.targets

    if hasattr(dataset, "labels"):
        return dataset.labels

    if hasattr(dataset, "_samples"):
        return [s[1] for s in dataset._samples]

    return [dataset[i][1] for i in range(len(dataset))]


def run(args=None):
    args = _parse_args(args)
    seed_everything(args.seed)

    train_ds, val_ds = get_datasets(args.dataset, None, args.root_dir)

    labels_train = get_labels(train_ds)
    labels_val = get_labels(val_ds)

    print(f"Num train: {len(labels_train)}")
    print(f"Num val:   {len(labels_val)}")
    print(f"Num classes: {len(np.unique(labels_train))}")

    out_dir = os.path.join(args.root_dir, "labels")
    os.makedirs(out_dir, exist_ok=True)

    np.save(f"{out_dir}/{args.dataset}_train.npy", labels_train)
    np.save(f"{out_dir}/{args.dataset}_val.npy", labels_val)


if __name__ == "__main__":
    run()
