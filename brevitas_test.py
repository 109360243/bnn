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


def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 防止溢位
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
Weights = torch.load('C:/Users/alexzhong/bnn/mnist_bnn_mlp.pt',map_location=torch.device('cpu'))
W1 = Weights['fc1.weight']
W1 = W1.cpu().numpy()
W1 = W1.T
W2 = Weights['fc2.weight']
W2 = W2.cpu().numpy()
W2 = W2.T
W3 = Weights['fc3.weight']
W3 = W3.cpu().numpy()
W3 = W3.T
W4 = Weights['fc4.weight']
W4 = W4.cpu().numpy()
W4 = W4.T
b1 = Weights['bn1.bias']
b1 = b1.cpu().numpy()
b2= Weights['bn2.bias']
b2 = b2.cpu().numpy()
b3 = Weights['bn3.bias']
b3 = b3.cpu().numpy()
b4= Weights['bn4.bias']
b4 = b4.cpu().numpy()




def forward_propagation(X, W1, b1, W2, b2,W3,b3,W4,b4):

    X = torch.flatten(X,start_dim=1).numpy() 


    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)

    A4 = np.dot(A3, W4) + b4
    A4 = softmax(A4)

    return A4






def test( device, test_loader):
    
    test_loss = 0
    correct = 0
    corret_sum = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = forward_propagation(X = data,W1 = W1, W2 = W2, W3 = W3,W4=W4, b1 = b1, b2 = b2, b3 = b3,b4=b4)
            
            predictions = np.argmax(output, axis=1)  # 取每行最大的索引作為預測類別
            labels = target.cpu().numpy()
            print("predictions:",predictions)  
            print("target:",labels)# 將 one-hot 編碼轉為類別標籤
            #這邊會生成一段布林陣列，mean(布
            correct = np.sum(predictions == labels)

# 输出正确预测的数量
            print(f"Correct predictions: {correct}")
            corret_sum = corret_sum+correct
            # correct += predictions.eq(target.view_as(predictions)).sum().item()
            

    print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(
         corret_sum, len(test_loader.dataset),
        100. * corret_sum / len(test_loader.dataset)))
    


def main():
    # Training settings
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

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
   
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    print("training using "+str(device))
   
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    


    
    
    for epoch in range(1, args.epochs + 1):
        
        test( device, test_loader)
        



if __name__ == '__main__':
    main()
