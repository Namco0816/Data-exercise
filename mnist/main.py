import math, shutil, os, time, argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

loss_list = []
acc_list = []

#split = range(10)
#train_sampler = SubsetRandomSampler(split)
def main():
    global args, best_prec1, prec1, loss_list, acc_list
    if args.model.lower() == 'ann':
        model = MnistModel()
    elif args.model.lower() == 'svm':
        model = MnistSvmModel()
    else:
        return ('invalid input of the choice of svm and ann, please type --model [CHOICE of THE MODEL] to initialize a model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    critirion = nn.NLLLoss()
    epoch = 0

    if doLoad:
        if args.model.lower() =='ann':
            saved = load_checkpoint(model_name = 'ann')
        else:
            saved = load_checkpoint(model_name = 'svm')

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
                                               pin_memory = True,
                                               sampler = None)

    val_loader = torch.utils.data.DataLoader(dataVal,
                                             batch_size = batch_size,
                                             shuffle = False,
                                             num_workers = workers,
                                             pin_memory = True)


    optimizer = torch.optim.Adam(model.parameters(),
                                 lr,)

    if doTest:
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        tar_num = torch.zeros((1,10))
        acc_num = torch.zeros((1,10))
        pred_num = torch.zeros((1,10))
        with torch.no_grad():
            for data, target in val_loader:
                data,target = data.to(device), target.to(device)
                output = model(data)
                test_loss+= F.nll_loss(output, target, reduction = 'sum').item()
                pred = output.argmax(dim =1, keepdim = True)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)

                correct += pred.eq(target.view_as(pred)).sum().item()
                pre_mask = torch.zeros(output.size()).scatter_(1, predicted.cpu().view(-1,1),1.)
                tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1,1),1.)

                acc_mask = pre_mask*tar_mask

                acc_num += acc_mask.sum(0)
                pred_num += pre_mask.sum(0)
                tar_num += tar_mask.sum(0)

        test_loss /= len(val_loader.dataset)

        recall = acc_num/tar_num
        precision = acc_num/pred_num

        print('Recall:', recall, '\t', 'Precision:', precision,"\t")
        print('Test set: Avg loss:{:.4f}, Acc Rate:{}/{}'.format(test_loss, correct, len(val_loader.dataset)))
        return
    for epoch in range(0, epoch):
        adjust_learning_rate(optimizer, epoch)
    for epoch in range(epoch, epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, device, critirion, optimizer, epoch)
        prec1 = validate(val_loader, model, device,  critirion, epoch)
        is_best = prec1<best_prec1
        best_prec1 = min(prec1, best_prec1)
        if args.model.lower() == 'ann':
            save_checkpoint({
            'epoch': epoch+ 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            }, is_best,model_name = 'ann')
        else:
            save_checkpoint({
            'epoch': epoch+ 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            }, is_best,model_name = 'svm')

    loss_monitor(loss_list, acc_list, args.epochs, args.model)
def train(train_loader, model, device, critirion, optimizer, epoch):
    global count, loss_list
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()

    for i, (image, image_label) in enumerate(train_loader):
        data_time.update(time.time() - end)

        image = image.to(device)
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

    loss_list.append(losses.avg)
def validate(val_loader, model, device, critirion, epoch):
    global count, acc_list
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    correct = 0
    model.eval()
    end = time.time()

    for i,( image, image_label) in enumerate(val_loader):
        data_time.update(time.time() - end)

        image = image.to(device)
        image_label = image_label.to(device)
        with torch.no_grad():
            output = model(image)

        pred = output.argmax(dim = 1, keepdim = True)
        correct += pred.eq(image_label.view_as(pred)).sum().item()

        loss = critirion(output, image_label)

        losses.update(loss.data.item(), image.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch(val): [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f}({loss.avg:.4f})\t'
              .format(epoch, i, len(val_loader), batch_time = batch_time, loss = losses)
              )
    acc_rate = correct/ len(val_loader.dataset)
    acc_list.append(acc_rate)
    return losses.avg

CHECKPOINTS_PATH = '.'

def load_checkpoint(model_name,filename = 'checkpoint.pth.tar'):
    map_location = None
    filename = model_name+filename
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    if not os.path.isfile(filename):
        return None
    if torch.cuda.is_available():
        map_loacation = None
    else:
        map_location = 'cpu'
    state = torch.load(filename, map_location = map_location)
    return state

def save_checkpoint(state, is_best,model_name, filename ='checkpoint.pth.tar'):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)
    filename = model_name+filename
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

def loss_monitor(loss_list,acc_list, epoch, model_name):

    epoch_list = range(0,epoch)
    plt.subplot(2,1,1)
    plt.plot(epoch_list, loss_list, 'o-')
    plt.title('NLL Loss /Accuracy vs Training Epoch')
    plt.ylabel('NLL Loss')

    plt.subplot(2,1,2)
    plt.plot(epoch_list, acc_list, '.-')
    plt.ylabel('Accuracy')

    plt.savefig(model_name+'_loss_monitor.jpg')


if __name__ == '__main__':
    main()
    print('DONE')
