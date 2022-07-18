# -*- coding: utf-8 -*-

import argparse, re, os, glob, datetime, time, random

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

# My own lib
from dataset import CancerDetection
import mylog
import nets

# Params
def str2list(isFloat, s, split='_'):
    l = s.split('_')
    if isFloat:
        l = [float(x) for x in l]
    else:
        l = [int(x) for x in l]
    return l

def str2bool(s):
    assert s in ['True', 'False']
    if s == 'True':
        return True
    else:
        return False
parser = argparse.ArgumentParser(description='')
parser.add_argument('--isTrain', action='store_true')
parser.add_argument('--net', default='MLP_CE', type=str)
parser.add_argument('--isCNN', action='store_true')
parser.add_argument('--isRandom', action='store_true')

args, _ = parser.parse_known_args()
if args.isTrain:
    parser.add_argument('--exp_name', default='.')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--milestones', default='100_150', type=str)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--reweight', default=2, type=float)
    # dataset
    parser.add_argument('--dir_root', default='./data')
    parser.add_argument('--task', default='task_2')
    parser.add_argument('--filename', default='.')
    parser.add_argument('--save_dir', default='./experiments')
    args = parser.parse_args()
    vars(args)['milestones'] = str2list(isFloat=False, s=args.milestones)
else:
    parser.add_argument('--dir_root', default='./data')
    parser.add_argument('--task', default='task_2')
    parser.add_argument('--filename', default='.')
    parser.add_argument('--model_dir', default='./experiments/.')
    parser.add_argument('--isVal', action='store_true')
    args = parser.parse_args()

# vars(args)['isTrain'] = str2bool(args.isTrain)
def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


if __name__ == '__main__':
    # Get objects
    obj_net = getattr(nets, args.net)

    if args.isTrain:
        if not args.isRandom:
            print('Fix random seed ............ ')
            random.seed(0)
            np.random.seed(seed=0)
            torch.manual_seed(0)
        else:
            print('Randomly set seeds ..............')

        # add log
        model_save_dir = os.path.join(args.save_dir, args.exp_name)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        logger = mylog.get_logger(os.path.join(model_save_dir, 'train_result.txt'))
        for arg in vars(args):
            logger.info('{}: {}'.format(arg, getattr(args, arg)))
        
        # model setting
        print('===> Building model')
        model = obj_net()
        criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda(torch.tensor([args.reweight,1]))
        logger.info(model)
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2)  # learning rates

        # dataset
        DLoader_tr = DataLoader(
            dataset=CancerDetection(args, mode='train'), 
            num_workers=8, drop_last=True, batch_size=args.batch_size, shuffle=True)          
        DLoader_val = DataLoader(
            dataset=CancerDetection(args, mode='validation'), 
            num_workers=1, drop_last=True, batch_size=1, shuffle=False)

        # trainning
        log_best_epoch = {'epoch':0, 'acc':0}
        log_epoch = []
        log_losses = []
        log_accu_train = []
        log_accu_val = []

        for epoch in range(0, args.epoch):
            scheduler.step(epoch)  # step to the learning rate in this epcoh
            epoch_loss = 0
            # training phase
            start_time = time.time()
            model.train()
            for n_count, batch_tr in enumerate(DLoader_tr):
                optimizer.zero_grad()
                pred = model(batch_tr[0].to(torch.float32).cuda())
                # import matplotlib.pyplot as plt; import pdb; pdb.set_trace()
                loss = criterion(pred, batch_tr[1].to(torch.int64).view(-1).cuda())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            torch.save(model, os.path.join(model_save_dir, 'model_latest.pth'))

            # evaluation phase
            model.eval()
                # training set validation
            with torch.no_grad():
                n_acc = []
                i = 0
                for _, batch_eval in enumerate(DLoader_tr):
                    i +=1
                    label = batch_eval[1].cpu().view(-1).numpy()
                    pred = model(batch_eval[0].to(torch.float32).cuda()).cpu().numpy()
                    pred = np.argmax(pred, axis=1)
                    n_acc.append(pred==label)
                acc_avg_tr = np.mean(n_acc)
                num_tr = i*args.batch_size

                # validation
            with torch.no_grad():
                n_acc = []
                num_val = 0
                for _, batch_eval in enumerate(DLoader_val):
                    num_val +=1
                    label = batch_eval[1].cpu().view(-1).item()
                    pred = model(batch_eval[0].to(torch.float32).cuda()).cpu().numpy()
                    pred = np.argmax(pred, axis=1)
                    n_acc.append(pred==label)
                acc_avg_val = np.mean(n_acc)

            # logging
            elapsed_time = time.time() - start_time
            log_epoch += [epoch]
            log_losses += [epoch_loss/n_count]
            log_accu_train += [acc_avg_tr]
            log_accu_val += [acc_avg_val]
            if acc_avg_val > log_best_epoch['acc']:
                log_best_epoch['acc'] = acc_avg_val
                log_best_epoch['epoch'] = epoch
                torch.save(model, os.path.join(model_save_dir, 'model_best.pth'))

            logger.info('EPOCH = {:03d}/{:03d}, [time] = {:.2f}s, [Acc {}-tr]:{:.3f}, [avrg loss] = {:.7f}. [Accu {}-Val]:{:.3f}. Best @ {:03d}, with accu {:.3f}.'\
                        .format(epoch, args.epoch, elapsed_time, num_tr, acc_avg_tr, epoch_loss/n_count, num_val, acc_avg_val, log_best_epoch['epoch'], log_best_epoch['acc']))

            # plot figure
            if epoch % 10 ==0:
                # fig1
                log_fig1 = os.path.join(args.save_dir, args.exp_name, 'fig1.pdf')
                fig_data = {'x':log_epoch, 'y':log_losses, 'color':'b', 'title':'training loss', 'label':'loss'}
                mylog.log_figure(log_fig1, fig_data)
                # fig2
                log_fig2 = os.path.join(args.save_dir, args.exp_name, 'fig2.pdf')
                fig_data = {'n_lines':2, 'title':'training accuracy',
                            'x1':log_epoch, 'y1':log_accu_train, 'label1':'tr', 'color1':'b',
                            'x2':log_epoch, 'y2':log_accu_val, 'label2':'val', 'color2':'r'}
                mylog.log_figure_multilines(log_fig2, fig_data)
    else:
        print('===> Testing')
        # add log
        logger = mylog.get_logger(os.path.join('./experiments/results', 'test_result'+'.txt'))
        for arg in vars(args):
            logger.info('{}: {}'.format(arg, getattr(args, arg)))
        # model setting
        model = torch.load(args.model_dir)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        # dataset
        if args.isVal:
            Dataset_test = CancerDetection(args, mode='validation')
        else:
            Dataset_test = CancerDetection(args, mode='test')
        DLoader_test = DataLoader(dataset=Dataset_test, num_workers=1, drop_last=True, batch_size=1, shuffle=False)

        # test set evaluation
        n_acc = []
        result_label = []
        result_pred = []
        with torch.no_grad():
            i = 0
            for _, batch_test in enumerate(DLoader_test):
                i +=1
                label = batch_test[1].cpu().view(-1).item(); result_label.append(label)
                pred = model(batch_test[0].to(torch.float32).cuda()).cpu().numpy(); result_pred.append(pred)
                pred = np.argmax(pred, axis=1)
                n_acc.append(pred==label)


        logger.info('[Accu of {}-TestImgs]:{:.3f}. \n\n'.format(i, np.mean(n_acc)))
        file2save = args.task + '_label_pred_' + args.filename
        results_dict = {'label':result_label, 'pred':result_pred}
        sio.savemat(os.path.join('./experiments/results',file2save), results_dict)
        






