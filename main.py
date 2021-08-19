import time
import json

import torch
import torch.nn as nn

from utils import AverageMeter, ProgressMeter, save_checkpoint
from data import get_dataloader

def train(_run, args, net):
    # train data
    train_loader, val_loader = get_dataloader(data_dir=args.data_dir, batch_size=args.batch_size, split='train', val_split=args.val_split)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    best_acc = 0.0
    for epoch in range(1, args.num_epochs+1):
        print('\nEpoch {}/{}'.format(epoch, args.num_epochs))
        train_epoch(_run, args, net, train_loader, optimizer, criterion)
        print('Validation')
        acc = evaluate(_run, args, net, val_loader, criterion)
        print('acc', acc)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        
        save_checkpoint({
            'epoch': epoch,
            'model': net.state_dict(),
            'args': args,
        }, is_best=is_best, filename='checkpoint_{}.pt'.format(epoch), save_dir=args.output_dir)
    
    return best_acc


def evaluate(_run, args, net, dataloader, criterion):
    net.eval()
    batch_loss = AverageMeter(name='loss', fmt=':.4e')
    batch_time = AverageMeter(name='b_t', fmt=':5.2f')
    data_time = AverageMeter(name='d_t', fmt=':5.2f')
    progress = ProgressMeter(len(dataloader), [batch_time, data_time, batch_loss], prefix='Validation')

    correct_cnt = 0
    total_cnt = 0
    end = time.time()
    for batch_idx, (images, labels) in enumerate(dataloader):
        data_time.update(time.time()-end)
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        loss = criterion(outputs, labels)
        _, pred_label = torch.max(outputs.data, 1)
        total_cnt += images.data.size()[0]
        correct_cnt += (pred_label == labels.data).sum()

        # Record batch time and loss 
        batch_loss.update(loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_step == 0:
            progress.display(batch_idx)
    return correct_cnt.item() * 1.0 / total_cnt


def test(_run, args, net):
    net.eval()

    dataloader = get_dataloader(data_dir=args.data_dir, batch_size=args.batch_size, split='test')

    batch_time = AverageMeter(name='b_t', fmt=':5.2f')
    data_time = AverageMeter(name='d_t', fmt=':5.2f')
    progress = ProgressMeter(len(dataloader), [batch_time, data_time], prefix='Testing')

    correct_cnt = 0
    total_cnt = 0
    end = time.time()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            data_time.update(time.time()-end)
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, pred_label = torch.max(outputs.data, 1)
            total_cnt += images.data.size()[0]
            correct_cnt += (pred_label == labels.data).sum()

            # Record batch time and loss 
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_step == 0:
                progress.display(batch_idx)

    return correct_cnt.item() * 1.0 / total_cnt

def train_epoch(_run, args, net, dataloader, optimizer, criterion):
    ''' One epoch of training'''
    net.train()
    batch_loss = AverageMeter(name='loss', fmt=':.4e')
    batch_time = AverageMeter(name='b_t', fmt=':5.2f')
    data_time = AverageMeter(name='d_t', fmt=':5.2f')
    progress = ProgressMeter(len(dataloader), [batch_time, data_time, batch_loss], prefix='Training')

    lst_train_error = []
    end = time.time()
    for batch_idx, (images, labels) in enumerate(dataloader):
        data_time.update(time.time()-end)

        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Record batch time and loss 
        batch_loss.update(loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_step == 0:
            progress.display(batch_idx)
            _run.log_scalar("training.loss", loss.item(), args.log_step)

    return loss.item()