import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models.resnet import resnet18
import numpy as np
from data_loader import *
import math
import random
batch_sie = 64
lr = 0.01
momentum = 0.1
weight_decay = 0
train_perc = .8

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

model = resnet18(pretrained=False, num_classes = 18)
image, name, labels = load_image_name_type()



labels = np.argmax(labels, axis=1)
image,labels = permutation(image,labels)
sample_num = image.shape[0]
train_sample_num = math.floor(sample_num*train_perc)
train_image = image[:train_sample_num]
train_labels = labels[:train_sample_num]
test_image = image[train_sample_num:]
test_labels = labels[train_sample_num:]
train_image = torch.from_numpy(train_image).float()
test_image = torch.from_numpy(test_image).float()
train_labels = torch.from_numpy(train_labels).long()
test_labels = torch.from_numpy(test_labels).long()
optimizer = torch.optim.SGD(model.parameters(), lr,
                            momentum = momentum,
                            weight_decay = weight_decay)



def train_batch(data, label):
    model.train()
    output = model(data)
    loss = criterion(output, label)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_epoch(data, label, batch_size = 16):
    sample_num = data.shape[0]
    batch_num = math.floor(sample_num/batch_size)
    for i in range(batch_num):
        start_idx = i*batch_size
        end_idx = (i+1)*batch_size
        batch_data = data[start_idx:end_idx]
        batch_label = label[start_idx:end_idx]
        train_batch(batch_data, batch_label)
    print(validate(test_image,test_labels))

def validate(data, label, batch_size = 16):
    cor = 0
    total = 0
    sample_num = data.shape[0]
    batch_num = math.floor(sample_num / batch_size)
    model.eval()
    for i in range(batch_num):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_data = data[start_idx:end_idx]
        batch_label = label[start_idx:end_idx]
        yhat = model.forward(batch_data)
        batch_cor, t = top1accu(yhat,batch_label)
        cor += batch_cor
        total += batch_size
    return cor, total

def top1accu(output, target):
    total = 0
    cor = 0
    for i in range(output.shape[0]):
        if np.argmax(output[0].detach().numpy())==target[0].numpy():
            cor+=1
        total += 1
    return cor,total

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

for i in range(10):
    train_epoch(train_image, train_labels)