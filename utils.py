# -*- coding: utf-8 -*-

import torch
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.datasets import MNIST

from tqdm import tqdm
from tools import rotate_digits
from functools import reduce


def make_noise(shape, type="Gaussian"):
    """
    Generate random noise.
    Parameters
    ----------
    shape: List or tuple indicating the shape of the noise
    type: str, "Gaussian" or "Uniform", default: "Gaussian".
    Returns
    -------
    noise tensor
    """

    if type == "Gaussian":
        noise = Variable(torch.randn(shape))
    elif type == "Uniform":
        noise = Variable(torch.randn(shape).uniform_(-1, 1))
    else:
        raise Exception("ERROR: Noise type {} not supported".format(type))
    return noise


def prep_loader(args, x, y):
    ds = DomainDataset(x, y)
    ldr = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    return ldr

def prep_continual_nested_loader(args, xs, ys):
    datasets = []
    for i in range(5):
        ds = DomainDataset(xs[i], ys[i])
        datasets.append(ds)
    ds_merged = ConcatDataset(datasets)
    ldr = DataLoader(
        ds_merged,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return ldr


def dataset_preparation(args, n_domains, step_size, output_type, train):
    # Prepares a list of dataloaders

    # Output types: classic, continual, nested-continual
    # classic: Outputs a list of dataloaders with one dataloader for each domain that contains all tasks
    # continual: Outputs a list of dataloaders with one dataloader for each task. Five consecutive dataloaders make up one domain
    # nested-continual: Outputs a nested list of dataloaders. List contains lists pertaining to each domain. Nested lists comprises of dataloaders of each task

    dataset = MNIST(
        root="data",
        train=bool(train),
        download=False,
    )

    data_x, data_y = rotate_digits(
        dataset=dataset,
        n_domains=n_domains,
        step_size=step_size,
    )

    x_keys = list(data_x.keys())

    dataloaders = []

    for d in tqdm(range(n_domains)):
        x1, x2, x3, x4, x5 = [], [], [], [], []
        y1, y2, y3, y4, y5 = [], [], [], [], []
        domain_x = data_x[x_keys[d]]
        y = data_y
        for i, x in enumerate(domain_x):
            if y[i] == 0 or y[i] == 1:
                x1.append(x)
                y1.append(y[i] % 2)
            elif y[i] == 2 or y[i] == 3:
                x2.append(x)
                y2.append(y[i] % 2)
            elif y[i] == 4 or y[i] == 5:
                x3.append(x)
                y3.append(y[i] % 2)
            elif y[i] == 6 or y[i] == 7:
                x4.append(x)
                y4.append(y[i] % 2)
            else:
                x5.append(x)
                y5.append(y[i] % 2)

        xs = [x1, x2, x3, x4, x5]
        ys = [y1, y2, y3, y4, y5]

        if output_type == 'classic':
            xs = reduce(lambda a,b:a+b, xs)
            ys = reduce(lambda a,b:a+b, ys)
            loaders = prep_loader(args, xs, ys)

        elif output_type == 'continual-nested':
            loaders = prep_continual_nested_loader(args, xs, ys)

        else:
            loaders = [prep_loader(args, xs[i], ys[i]) for i in range(5)]

        if output_type == 'continual-nested' or output_type == 'classic':
            dataloaders.append(loaders)
        else: # output_type == 'continual'
            dataloaders += loaders

    return dataloaders


class DomainDataset(Dataset):
    """Customized dataset for each domain"""

    def __init__(self, X, Y):
        self.X = X  # set data
        self.Y = Y  # set lables

    def __len__(self):
        return len(self.X)  # return length

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]  # return list of batch data [data, labels]
