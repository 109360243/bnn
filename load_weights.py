from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
#from conv import Conv2d
#from linear import Linear
torch.manual_seed(0)
import numpy as np
from torch.nn import Module

# import brevitas.nn as qnn
from brevitas.nn import QuantLinear, QuantHardTanh, QuantIdentity
from brevitas.nn import QuantSigmoid
from brevitas.nn import QuantTanh
from brevitas.quant import Int8Bias as BiasQuant

from brevitas.core.quant.binary import BinaryQuant
from brevitas.core.scaling import ConstScaling

from common import *
weights = torch.load("C:/Users/alexzhong/bnn/mnist_bnn_mlp.pt")
# print(weights.keys())
# print((weights['fc1.weight']).shape)

# weight_test = weights['fc1.weight'].cpu().numpy()
# print(type(weight_test))
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                    transform=transform)
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                    help='disables macOS GPU training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()




torch.manual_seed(args.seed)


device = torch.device("cpu")

train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}
if use_cuda:
    cuda_kwargs = {'num_workers': 6,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

torch.manual_seed(args.seed)


device = torch.device("cpu")

test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

for data in test_loader:
    data, target = data.to(device), target.to(device)
    print(torch.flatten(data,start_dim=1))
