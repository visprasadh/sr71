from utils import dataset_preparation
from collections import namedtuple

import argparse

# Arg = namedtuple("Arg", "num_workers batch_size")
# args = Arg(num_workers=0, batch_size=16)

# dataloaders = dataset_preparation(
#     args=args,
#     n_domains=10,
#     step_size=15,
#     output_type="classic",
#     train=False,
# )

# print(len(dataloaders))
# # print(len(dataloaders[0]))
# [print(dataloaders[0])]



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

args = parser.parse_args()

print(vars(args))

# for key, value in vars(args).items():
#     print(f'{key}: {value}')

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('dark')
# a = list(range(5))
# b = [21, 25, 56, 32 ,32]

# data = [a, b]

# plt.rcParams['figure.figsize'] = (10, 8)
# sns.lineplot(data = b)
# plt.grid()

# plt.title(args.name)
# plt.xticks(a)
# plt.xlabel('Test task')
# plt.ylabel('Accuracy')
# plt.savefig('fig.jpg')

angles = [15, 30, 45, 60]
acc = [0.964, 0.697, 0.721, 0.621]
plt.plot(angles, acc)
plt.grid()
sns.despine(offset=10, trim=True)
plt.title('Impact of increment angles on accuracy')
plt.xticks(angles)
plt.xlabel('Increment angle (in degrees)')
plt.ylabel('Accuracy')
plt.savefig('angles.jpg')