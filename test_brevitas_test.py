import torch
import numpy as np
from torchvision import datasets, transforms

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Load model weights
Weights = torch.load('C:/Users/alexzhong/bnn/mnist_bnn_mlp.pt', map_location=torch.device('cpu'))
W1 = Weights['fc1.weight'].cpu().numpy().T
W2 = Weights['fc2.weight'].cpu().numpy().T
W3 = Weights['fc3.weight'].cpu().numpy().T
W4 = Weights['fc4.weight'].cpu().numpy().T
b1 = Weights['bn1.bias'].cpu().numpy()
b2 = Weights['bn2.bias'].cpu().numpy()
b3 = Weights['bn3.bias'].cpu().numpy()
b4 = Weights['bn4.bias'].cpu().numpy()

def forward_propagation(X, W1, b1, W2, b2, W3, b3, W4, b4):
    X = torch.flatten(X, start_dim=1).numpy()  # Flatten input and convert to numpy

    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)

    A4 = np.dot(A3, W4) + b4
    A4 = softmax(A4)

    return A4

def test(device, test_loader):
    test_loss = 0
    correct_sum = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = forward_propagation(X=data, W1=W1, W2=W2, W3=W3, W4=W4, b1=b1, b2=b2, b3=b3, b4=b4)

            predictions = np.argmax(output, axis=1)
            labels = target.cpu().numpy()
            correct = np.sum(predictions == labels)

            print(f"Predictions: {predictions}")
            print(f"Target: {labels}")
            print(f"Correct predictions in batch: {correct}")
            correct_sum += correct

    print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(
        correct_sum, len(test_loader.dataset),
        100. * correct_sum / len(test_loader.dataset)))

def main():
    batch_size = 128
    test_batch_size = 1000
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=test_batch_size)

    device = torch.device("cpu")
    test(device, test_loader)

if __name__ == '__main__':
    main()
