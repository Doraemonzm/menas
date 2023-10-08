'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
from torch.autograd import Variable
import sys
import torchvision
import torchvision.transforms as transforms
import copy
import os
import argparse
import numpy as np

from torchprofile import profile_macs

from My_Mobilenet import MobileNetV3
from utils import *
from auto_aug.archive import fa_reduced_cifar10, autoaug_paper_cifar10
from auto_aug.augmentations import Augmentation
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch-size', type=int, default=96, help='batch size')
parser.add_argument('--img-size', type=int, default=224,
                    help='input resolution (192 -> 256)')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--autoaug', default='autoaug_cifar10', type=str)


parser.add_argument('--checkpoint', default=None, type=str)

# Random Erasing
parser.add_argument('--random_erase', action='store_true', default=False, help='use cutout')
parser.add_argument('--p', default=0, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')

parser.add_argument('--soft_label', action='store_true', default=False, help='use soft label cross entropy loss')


# mixup
parser.add_argument('--mixup', action='store_true', default=False, help='use mixup')

parser.add_argument('--clip', action='store_true', default=False, help='use grad clip')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')



# cutmix
parser.add_argument('--cutmix', action='store_true', default=False, help='use cutmix')
parser.add_argument('--beta', default=1.0, type=float, help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0.5, type=float, help='cutmix probability')

args = parser.parse_args()
for k, v in sorted(vars(args).items()):
    print(k, ' = ', v)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img



def _data_transforms(args):

    norm_mean = [0.49139968, 0.48215827, 0.44653124]
    norm_std = [0.24703233, 0.24348505, 0.26158768]



    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

   

    if args.autoaug:
        print('augmentation: %s' % args.autoaug)
        if args.autoaug == 'fa_reduced_cifar10':
            train_transform.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
        elif args.autoaug == 'autoaug_cifar10':
            train_transform.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))


    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    if args.random_erase:
        train_transform.transforms.append(RandomErasing(probability = args.p, sh = args.sh, r1 = args.r1, ))



    valid_transform = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=3),  # BICUBIC interpolation
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    return train_transform, valid_transform







# Data
print('==> Preparing data..')


train_transform, valid_transform = _data_transforms(args)


trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=valid_transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=200, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

config={'ks': [5, 3, 5, 3, 7, 7, 3, 5, 7, 5, 7, 3, 3, 5, 7, 3, 7, 7, 5, 3, 5, 7, 7, 5, 3], 'e': [1, 2, 2, 3, 2, 4, 5, 1, 4, 6, 5, 5, 4, 1, 6, 1, 5, 4, 2, 5, 1, 3, 4, 2, 2], 'd': [6, 2, 5, 6, 6], 's': [1, 2, 2, 2, 2], 'width-mul': [1.25, 2.0, 1.25, 2.0, 1.5], 'se': [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]}

net = MobileNetV3(config, num_classes=100)

# calculate #Paramaters and #FLOPS
inputs = torch.randn(1, 3, args.img_size, args.img_size)
flops = profile_macs(copy.deepcopy(net), inputs) / 1e6
params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
net_name = "net_flops@{:.0f}".format(flops)
print('#params {:.2f}M, #flops {:.0f}M'.format(params, flops))



print('==> Initializing from checkpoint..')

init = torch.load(args.checkpoint, map_location='cpu')['state_dict']
new_state_dict = net.state_dict()
for k, v in init.items():
    # strip `module.` prefix
    name = k[7:] if k.startswith('module') else k
    if name in new_state_dict.keys() and v.size()== new_state_dict[name].size():
        new_state_dict[name] = v
net.load_state_dict(new_state_dict)


#
#
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net = net.to(device)


if args.soft_label:
    criterion=LabelSmoothSoftmaxCE()
else:
    criterion = nn.CrossEntropyLoss()


optimizer=torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = F.interpolate(inputs, size=args.img_size, mode='bicubic', align_corners=False)
        inputs, targets = inputs.to(device), targets.to(device)

        if args.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            train_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
            optimizer.zero_grad()
            loss.backward()
            if args.clip:
                nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += inputs.size(0)
            correct += predicted.eq(targets).sum().item()


        elif args.cutmix:
            r = np.random.rand(1)
            if args.beta > 0 and r < args.cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(inputs.size()[0]).cuda()
                target_a = targets
                target_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                # compute output
                output = net(inputs)
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            else:
                # compute output
                output = net(inputs)
                loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            if args.clip:
                nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()



            train_loss += loss.item()
            _, predicted = output.max(1)
            total += inputs.size(0)
            correct += predicted.eq(targets).sum().item()
        
        else:

            optimizer.zero_grad()
            outputs = net(inputs)
        
            loss = criterion(outputs, targets)
            loss.backward()
            if args.clip:
                nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()



            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print('Train : | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    print('Epoch Train Loss: %.3f | Epoch Train Acc: %.3f%% (%d/%d)' % (
        train_loss / len(trainloader), 100. * correct / total, correct, total))




def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)


            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print('Test : | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        print('Epoch Test Loss: %.3f | Epoch Test Acc: %.3f%% (%d/%d)' % (
            test_loss / len(testloader), 100. * correct / total, correct, total))


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
print('best acc is:',best_acc)

