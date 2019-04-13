import math, shutil, os, time, argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from mnistModel import MnistModel
from mnistSvmModel import MnistSvmModel

def str2bool(v):
    if v.lower() in ('y', 'yes', 'true', 't', '1'):
        return True
    if v.lower() in ('n', 'no', 'false', 'f', '0'):
        return False
    else :
        return argparse.ArgumentTypeError('Bool type input required')

parser = argparse.ArgumentParser(description = 'Mnist model with only linear layer')
parser.add_argument('--test_mode', type = str2bool, nargs = '?', const = True, default = 'False', help = 'set true to start quick test')
parser.add_argument('--reset', type = str2bool, nargs = '?', const = True, default = 'False', help = 'set true to reset the weights')
parser.add_argument('--epochs', type = int,default = 100, help = 'the training epochs')
parser.add_argument('--model', type = str, default = 'ann', help = 'choose a model between ann and svm')
args = parser.parse_args()

doLoad = not args.reset
doTest = args.test_mode

workers = 8
epochs = args.epochs
batch_size = 100

base_lr = 0.0001
lr = base_lr

best_prec1 = 1e20
prec1 = 0

count = 0
count_test = 0

def main():
    global args, best_prec1, prec1
    if args.model.lower() == 'ann':
        model = MnistModel()
        critirion = nn.NLLLoss()
    elif args.model.lower() == 'svm':
        model = MnistSvmModel()
        critirion = nn.MSELoss()
    else:
        return ('invalid input of the choice of svm and ann, please type --model [CHOICE of THE MODEL] to initialize a model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    epoch = 0

    if doLoad:
        saved = load_checkpoint()
        if saved:
            print('loading checkpoint for epoch %05d with loss %.5f'%(saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            try:
                model.module.load_state_dict(state)
            except:
                model.load_state_dict(state)
            epoch = saved['epoch']
            best_prec1 = saved['best_prec1']
        else:
            print('cannot load checkpoint')
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    dataTrain = datasets.MNIST(root = './data',
                               transform = transform,
                               train = True,
                               download = True)

    dataVal = datasets.MNIST(root = './data',
                               transform = transform,
                               train = False,
                               )
    train_loader = torch.utils.data.DataLoader(dataTrain,
                                               batch_size = batch_size,
                                               shuffle = True,
                                               num_workers = workers,
                                               pin_memory = True)

    val_loader = torch.utils.data.DataLoader(dataVal,
                                             batch_size = batch_size,
                                             shuffle = False,
                                             num_workers = workers,
                                             pin_memory = True)


    optimizer = torch.optim.Adam(model.parameters(),
                                 lr,)
    for epoch in range(0, epoch):
        adjust_learning_rate(optimizer, epoch)
    for epoch in range(epoch, epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, device, critirion, optimizer, epoch)
        prec1 = validate(val_loader, model, device,  critirion, epoch)
        is_best = prec1<best_prec1
        best_prec1 = min(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch+ 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)

def train(train_loader, model, device, critirion, optimizer, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()

    for i, (image, image_label) in enumerate(train_loader):
        data_time.update(time.time() - end)

        image = image.to(device)
        if args.model == "svm":
            image_label = image_label.to(device).float()
        else:
            image_label = image_label.to(device)
        output = model(image)
        loss = critirion(output, image_label)

        losses.update(loss.data.item(), image.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        count = count +1
        print('Epoch(train): [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f}({data_time.avg:3f})\t'
              'Loss {loss.val:.4f}({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), batch_time = batch_time, data_time = data_time, loss = losses)
              )

def validate(val_loader, model, device, critirion, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    end = time.time()

    for i,( image, image_label) in enumerate(val_loader):
        data_time.update(time.time() - end)

        image = image.to(device)
        if args.model == 'ann':
            image_label = image_label.to(device)
        else:
            image_label = image_label.to(device).float()
        with torch.no_grad():
            output = model(image)

        if args.model.lower() =='ann':
            pred = output.argmax(dim = 1, keepdim = True)
        else:
            pred = output
        loss = critirion(output, image_label)

        losses.update(loss.data.item(), image.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch(val): [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f}({loss.avg:.4f})\t'
              .format(epoch, i, len(val_loader), batch_time = batch_time, loss = losses)
              )
        return losses.avg

CHECKPOINTS_PATH = '.'

def load_checkpoint(filename = 'checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state

def save_checkpoint(state, is_best, filename = 'checkpoint.pth.tar'):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)
    bestFileName = os.path.join(CHECKPOINTS_PATH, 'best_'+ filename)
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFileName)

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n = 1):
        self.val = val
        self.sum += val*n
        self.count +=n
        self.avg = self.sum/ self.count

def adjust_learning_rate(optimizer, epoch):
    lr = base_lr * (0.1 ** (epoch//30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
    print('DONE')
