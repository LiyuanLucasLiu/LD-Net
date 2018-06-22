from __future__ import print_function
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import codecs
import pickle
import math

from model_word_ada.LM import LM
from model_word_ada.basic import BasicRNN
from model_word_ada.ddnet import DDRNN
from model_word_ada.ldnet import LDRNN
from model_word_ada.densenet import DenseRNN
from model_word_ada.dataset import LargeDataset, EvalDataset
from model_word_ada.adaptive import AdaptiveSoftmax
import model_word_ada.utils as utils

from tensorboardX import SummaryWriter

import argparse
import json
import os
import sys
import itertools
import functools

def evaluate(data_loader, lm_model, limited = 76800):
    print('evaluating')
    lm_model.eval()
    lm_model.init_hidden()
    total_loss = 0
    total_len = 0
    for word_t, label_t in data_loader:
        label_t = label_t.view(-1)
        tmp_len = label_t.size(0)
        total_loss += tmp_len * lm_model(word_t, label_t).item()
        total_len += tmp_len

        if limited >=0 and total_len > limited:
            break

    ppl = math.exp(total_loss / total_len)
    print('PPL: ' + str(ppl))

    return ppl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='one_0')
    parser.add_argument('--dataset_folder', default='./data/one_billion/')
    parser.add_argument('--load_checkpoint', default='')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sequence_length', type=int, default=20)
    parser.add_argument('--hid_dim', type=int, default=2048)
    parser.add_argument('--word_dim', type=int, default=300)
    parser.add_argument('--label_dim', type=int, default=-1)
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--droprate', type=float, default=0.1)
    parser.add_argument('--add_relu', action='store_true')
    parser.add_argument('--layer_drop', type=float, default=0.5)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--clip', type=float, default=5)
    parser.add_argument('--update', choices=['Adam', 'Adagrad', 'Adadelta'], default='Adam', help='adam is the best')
    parser.add_argument('--rnn_layer', choices=['Basic', 'DDNet', 'DenseNet', 'LDNet'], default='Basic')
    parser.add_argument('--rnn_unit', choices=['gru', 'lstm', 'rnn', 'bnlstm'], default='lstm')
    parser.add_argument('--lr', type=float, default=-1)
    parser.add_argument('--cut_off', nargs='+', default=[4000,40000,200000])
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--check_interval', type=int, default=4000)
    parser.add_argument('--patience', type=float, default=10)
    parser.add_argument('--checkpath', default='checkpoint/')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")

    print('loading dataset')
    dataset = pickle.load(open(args.dataset_folder + 'test.pk', 'rb'))
    w_map, test_data, range_idx = dataset['w_map'], dataset['test_data'], dataset['range']

    cut_off = args.cut_off + [len(w_map) + 1]

    train_loader = LargeDataset(args.dataset_folder, range_idx, args.batch_size, args.sequence_length)
    test_loader = EvalDataset(test_data, args.batch_size)

    print('building model')

    rnn_map = {'Basic': BasicRNN, 'DDNet': DDRNN, 'DenseNet': DenseRNN, 'LDNet': functools.partial(LDRNN, layer_drop = args.layer_drop)}
    rnn_layer = rnn_map[args.rnn_layer](args.layer_num, args.rnn_unit, args.word_dim, args.hid_dim, args.droprate)

    if args.label_dim > 0:
        soft_max = AdaptiveSoftmax(args.label_dim, cut_off)
    else:
        soft_max = AdaptiveSoftmax(rnn_layer.output_dim, cut_off)

    lm_model = LM(rnn_layer, soft_max, len(w_map), args.word_dim, args.droprate, label_dim = args.label_dim, add_relu=args.add_relu)
    lm_model.rand_ini()

    optim_map = {'Adam' : optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta}
    if args.lr > 0:
        optimizer=optim_map[args.update](lm_model.parameters(), lr=args.lr)
    else:
        optimizer=optim_map[args.update](lm_model.parameters())

    if args.load_checkpoint:
        if os.path.isfile(args.load_checkpoint):
            print("loading checkpoint: '{}'".format(args.load_checkpoint))
            checkpoint_file = torch.load(args.load_checkpoint, map_location=lambda storage, loc: storage)
            model_file = checkpoint_file['lm_model']
            lm_model.load_state_dict(model_file, False)
        else:
            print("no checkpoint found at: '{}'".format(args.load_checkpoint))

    lm_model.to(device)

    writer = SummaryWriter(log_dir='./runs_1b/'+args.log_dir)
    name_list = ['batch_loss', 'train_ppl', 'test_ppl']
    bloss, tr_ppl, te_ppl = [args.log_dir+'/'+tup for tup in name_list]

    batch_index = 0
    patience = 0
    epoch_loss = 0
    best_train_ppl = float('inf')
    cur_lr = args.lr

    try:
        for indexs in range(args.epoch):

            lm_model.train()

            for word_t, label_t in train_loader.get_tqdm(device):

                if 1 == train_loader.cur_idx:
                    lm_model.init_hidden()

                label_t = label_t.view(-1)

                lm_model.zero_grad()
                loss = lm_model(word_t, label_t)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lm_model.parameters(), args.clip)
                optimizer.step()

                batch_index += 1 

                if 0 == batch_index % args.interval:
                    s_loss = utils.to_scalar(loss)
                    writer.add_scalar(bloss, s_loss, batch_index)
                
                epoch_loss += utils.to_scalar(loss)
                if 0 == batch_index % args.check_interval:
                    epoch_ppl = math.exp(epoch_loss / args.check_interval)
                    writer.add_scalar(tr_ppl, epoch_ppl, batch_index)
                    if epoch_loss < best_train_ppl:
                        best_train_ppl = epoch_loss
                        patience = 0
                    else:
                        patience += 1
                    epoch_loss = 0

                if patience > args.patience and cur_lr > 0:
                    patience = 0
                    best_train_ppl = float('inf')
                    cur_lr *= 0.1
                    print('adjust_learning_rate...')
                    utils.adjust_learning_rate(optimizer, cur_lr)

            test_ppl = evaluate(test_loader.get_tqdm(device), lm_model)
            writer.add_scalar(te_ppl, test_ppl, indexs)
        
            torch.save({'lm_model': lm_model.state_dict(), 'opt':optimizer.state_dict()}, args.checkpath+args.log_dir+'.model')

    except KeyboardInterrupt:

        print('Exiting from training early')
        test_ppl = evaluate(test_loader.get_tqdm(device), lm_model)
        writer.add_scalar(te_ppl, test_ppl, indexs)
        torch.save({'lm_model': lm_model.state_dict(), 'opt':optimizer.state_dict()}, args.checkpath+args.log_dir+'.model')

    writer.close()