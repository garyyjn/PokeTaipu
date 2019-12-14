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
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 28, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(36288, 128) # 128 => 256?
        self.fc2 = nn.Linear(128, 18)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 3)
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
    batch_num = int(len(train_data) / args.batch_size)

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
                100. * batch_idx / batch_num, loss.item()))


def test(args, model, device, test_data, test_target):
    model.eval() # Sets the module in evaluation mode
    test_loss = 0
    correct = 0

    batch_num = int(len(test_data) / args.test_batch_size)

    with torch.no_grad():
        for batch_idx in range(batch_num):
            start_idx = batch_idx * args.test_batch_size
            end_idx = (batch_idx + 1) * args.test_batch_size
            data = test_data[start_idx:end_idx]
            target = test_target[start_idx:end_idx]

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            '''
            print(pred)
            print(target)
            print(pred.eq(target.view_as(pred)))
            input()
            '''
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
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
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
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_data, train_target, test_data, test_target = data()

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_data, train_target, optimizer, epoch)
        test(args, model, device, test_data, test_target)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "poketaipu_cnn.pt")

def data():
    # Load data
    image, name, type = loader.load_image_name_type()
    # Shuffle data
    image, type = permutation(image, type)
    type = np.argmax(type, axis = 1)
    train_sample_num = int(image.shape[0] * 0.8)
    train_data = image[:train_sample_num]
    train_target = type[:train_sample_num]
    test_data = image[train_sample_num:]
    test_target = type[train_sample_num:]
    train_data = torch.from_numpy(train_data).float()
    test_data = torch.from_numpy(test_data).float()
    train_target = torch.from_numpy(train_target).long()
    test_target = torch.from_numpy(test_target).long()

    return train_data, train_target, test_data, test_target

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