# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import os
import logging
import time
import datetime
from tqdm import tqdm
import argparse

# Import model
from model import RNN

# Import functions
from utils import dataset_preparation, make_noise
from tools import line_plot

parser = argparse.ArgumentParser(description="DomainGen_Graph")

parser.add_argument(
    "--name",
    required=True,
    type=str,
    help="enter name of the experiment",
)

parser.add_argument(
    "--dataset",
    default="MNIST",
    type=str,
    help="dataset choice",
)

parser.add_argument(
    "--experiment",
    default="classic",
    type=str,
    help="choose between classic, continual and continual-nested",
)

parser.add_argument(
    "--n_domains",
    default=15,  # Includes the number of test domains
    type=int,
    help="the number of domains",
)

parser.add_argument(
    "--eval_split",
    default=5,
    type=int,
    help="the number of test domains",
)

parser.add_argument(
    "--step_size",
    default=15,
    type=float,
    help="the increment angle for rotation",
)

parser.add_argument(
    "--train",
    default=1,
    type=int,
    help="to use train or test MNIST dataset",
)

# Hyper-parameters
parser.add_argument(
    "--noise_dim",
    default=16,
    type=float,
    help="the dimension of the LSTM input noise.",
)
parser.add_argument(
    "--num_rnn_layer",
    default=1,
    type=float,
    help="number of RNN hierarchical layers.",
)
parser.add_argument(
    "--latent_dim",
    default=32,
    type=float,
    help="latent dim of RNN variables.",
)
parser.add_argument(
    "--hidden_dim",
    default=16,
    type=float,
    help="latent dime of RNN variables.",
)
parser.add_argument(
    "--noise_type",
    choices=["Gaussian", "Uniform"],
    default="Gaussian",
    help="noise type to feed into generator.",
)

parser.add_argument(
    "--num_workers",
    default=0,
    type=int,
    help="the number of threads for loading data.",
)
parser.add_argument(
    "--epoches",
    default=10,
    type=int,
    help="the number of epoches for each task.",
)
parser.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help="the number of epoches for each task.",
)
parser.add_argument(
    "--learning_rate",
    default=1e-3,
    type=float,
    help="lr for each single task.",
)

parser.add_argument(
    "--is_test",
    default=True,
    type=bool,
    help="if this is a testing period.",
)

args = parser.parse_args()

output_directory = f"output/{args.name}"
model_directory = "models-{}".format(args.dataset)

if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
if not os.path.isdir(model_directory):
    os.makedirs(model_directory)

# setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_file = f"output/{args.name}/log_{args.name}.log"
open(log_file, "a").close()

# create logger
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# add to log file
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

def log(str):
    logger.info(str)

log("Is GPU available? {}".format(torch.cuda.is_available()))
# print('Is GPU available? {}'.format(torch.cuda.is_available()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    dataloader, optimizer, rnn_unit, args, task_id=0, input_E=None, input_hidden=None
):
    E = input_E
    hidden = input_hidden
    log("Start Training on Domain {}...".format(task_id))
    for epoch in range(args.epoches):
        accs = []
        with tqdm(dataloader, unit="batch") as tepoch:
            for X, Y in tepoch:
                tepoch.set_description("Task_ID: {} Epoch {}".format(task_id, epoch))

                X, Y = X.float().to(device), Y.float().to(device)
                initial_noise = make_noise((1, args.noise_dim), args.noise_type).to(
                    device
                )

                #  Training on Single Domain
                rnn_unit.train()
                optimizer.zero_grad()
                E, hidden, pred = rnn_unit(X, initial_noise, E, hidden)
                E = E.detach()
                hidden = tuple([i.detach() for i in hidden])
                loss = F.binary_cross_entropy(pred.squeeze(-1), Y)

                prediction = torch.as_tensor((pred.detach() - 0.5) > 0).float()
                accuracy = (
                    prediction.squeeze(-1) == Y
                ).float().sum() / prediction.shape[0]
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                accs.append(accuracy.item())
                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy.item())

            # log("Task_ID: {}\tEpoch: {}\tAverage Training Accuracy: {}".format(task_id, epoch, np.mean(accs)))
    return E, hidden, rnn_unit


def evaluation(dataloader, rnn_unit, args, input_E, input_hidden):
    rnn_unit.eval()
    E = input_E
    hidden = input_hidden
    test_accs = []
    with tqdm(dataloader, unit="batch") as tepoch:
        for X, Y in tepoch:
            X, Y = X.float().to(device), Y.float().to(device)
            initial_noise = make_noise((1, args.noise_dim), args.noise_type).to(device)
            with torch.no_grad():
                _, _, pred = rnn_unit(X, initial_noise, E, hidden)
                loss = F.binary_cross_entropy(pred.squeeze(-1), Y)

                prediction = torch.as_tensor((pred.detach() - 0.5) > 0).float()
                accuracy = (
                    prediction.squeeze(-1) == Y
                ).float().sum() / prediction.shape[0]
                test_accs.append(accuracy.item())
                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy.item())
    log("Average Testing Accuracy is {}".format(np.mean(test_accs)))
    return np.mean(test_accs)


def main(args):
    log(f'Experiment {args.name} setup:')
    for key, value in vars(args).items():
        log(f"{key}: {value}")

    log("use {} data".format(args.dataset))
    log("-" * 40)

    # Defining dataloaders for each domain
    dataloaders = dataset_preparation(
        args=args,
        n_domains=args.n_domains,
        step_size=args.step_size,
        output_type=args.experiment,
        train=args.train,
    )

    rnn_unit = RNN(data_size=28 * 28, device=device, args=args).to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(rnn_unit.parameters(), lr=args.learning_rate)

    starting_time = time.time()

    # Training
    Es, hiddens = [None], [None]
    for task_id, dataloader in enumerate(dataloaders[: -int(args.eval_split)]):
        E, hidden, rnn_unit = train(
            dataloader, optimizer, rnn_unit, args, task_id, Es[-1], hiddens[-1]
        )
        Es.append(E)
        hiddens.append(hidden)
        print("========== Finished Domain #{} ==========".format(task_id))

    ending_time = time.time()

    print("Training time:", ending_time - starting_time)

    # Testing
    if args.experiment == 'classic' or args.experiment == 'continual-nested':
        test_accuracies = []
        for task_id, dataloader in enumerate(dataloaders[-int(args.eval_split) :]):
            log(f'Testing domain {task_id}')
            acc = evaluation(dataloader, rnn_unit, args, Es[-1], hiddens[-1])
            test_accuracies.append(acc)
        line_plot(args, test_accuracies, args.name)
    elif args.experiment == 'continual':
        test_accuracies = {'0': [], '1': [], '2': [], '3': [], '4': []}
        for task_id, dataloader in enumerate(dataloaders[-int(args.eval_split) :]):
            log(f'Testing task {task_id % 5}')
            acc = evaluation(dataloader, rnn_unit, args, Es[-1], hiddens[-1])
            test_accuracies[f'{task_id % 5}'].append(acc)
        for key, value in test_accuracies.items():
            log(f'Test accuracies for task {key} : {value}')
            line_plot(args, value, f'{args.name} Task:{key}')
    else:
        pass

if __name__ == "__main__":
    print("Start Training...")

    # Initialize the time
    timestr = time.strftime("%Y%m%d-%H%M%S")

    main(args)
