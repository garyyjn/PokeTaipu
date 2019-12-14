# Source: https://github.com/pytorch/examples/blob/master/mnist/main.py

# TODO: Modify parts with data loader

from __future__ import print_function
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import loader
import random


class Net(nn.Module):
    def __init__(self): 
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_data, train_target, optimizer, epoch):
    model.train() # Sets the module in training mode
    # TODO: Add the last half batch
    batch_num = int(len(train_data))

    for batch_idx in range(batch_num):
        start_idx = batch_idx * args.batch_size
        end_idx = (batch_idx + 1) * args.batch_size
        data = train_data[start_idx:end_idx]
        target = train_target[start_idx:end_idx]

        # Below is original code
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_data),
                100. * batch_idx / len(train_data), loss.item()))


def test(args, model, device, test_data, test_target):
    model.eval() # Sets the module in evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in (test_data, test_target):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_data)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data),
        100. * correct / len(test_data)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Conv Net Using Pytorch To Classify Pokemon')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_data, test_data = data()

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_data, optimizer, epoch)
        test(args, model, device, test_data)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "poketaipu_cnn.pt")

def data():
    # Load data
    image, name, type = loader.load_image_name_type()

    # Convert data. Make it good for training
    image_tensor = torch.from_numpy(image).float()
    type_tensor = torch.from_numpy(type).float()

    train_data = []
    test_data = []

    # First add everything to train_data
    for i in range(len(torch.from_numpy(image).float())):
        data = []
        data.append(image_tensor[i])
        data.append(type_tensor[i])
        train_data.append(data)
        print(data)
        input()

    # Then shuffle and move 20% to test_data
    random.shuffle(train_data)
    for i in range(int(len(torch.from_numpy(image).float()) / 5)):
        test_data.append(train_data[0])
        train_data.pop(0)

    return train_data, test_data

def permutation(image, labels):
    new_image = np.ones(image.shape, dtype=np.int)
    new_label = np.ones(labels.shape, dtype=np.int)
    sample_num = new_image.shape[0]
    perm_index = list(range(sample_num))
    random.shuffle(perm_index)
    for i in range(sample_num):
        new_image[i] = image[perm_index[i]]
        new_label[i] = labels[perm_index[i]]
    return new_image, new_label

if __name__ == '__main__':
    main()