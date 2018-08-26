from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
import pickle
import math

from model_word_ada.LM import LM
from model_word_ada.basic import BasicRNN
from model_word_ada.densenet import DenseRNN
from model_word_ada.ldnet import LDRNN

from model_seq.crf import CRFLoss, CRFDecode
from model_seq.dataset import SeqDataset
from model_seq.evaluator import eval_wc
from model_seq.seqlabel import SeqLabel, Vanilla_SeqLabel
from model_seq.seqlm import BasicSeqLM
from model_seq.sparse_lm import SparseSeqLM
import model_seq.utils as utils

import tbwrapper.wrapper as wrapper

import argparse
import json
import os
import sys
import itertools
import functools

from ipdb import set_trace

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', type=str, default="auto")
    parser.add_argument('--cp_root', default='./checkpoint')
    parser.add_argument('--checkpoint_name', default='pner')
    parser.add_argument('--git_tracking', action='store_true')

    parser.add_argument('--corpus', default='../DDCLM/data/ner_dataset.pk')
    parser.add_argument('--load_seq', default='../DDCLM/cp/ner/nld4.model')

    parser.add_argument('--lm_hid_dim', type=int, default=300)
    parser.add_argument('--lm_word_dim', type=int, default=300)
    parser.add_argument('--lm_label_dim', type=int, default=1600)
    parser.add_argument('--lm_layer_num', type=int, default=10)
    parser.add_argument('--lm_droprate', type=float, default=0.5)
    parser.add_argument('--lm_rnn_layer', choices=['Basic', 'DenseNet', 'LDNet'], default='LDNet')
    parser.add_argument('--lm_rnn_unit', choices=['gru', 'lstm', 'rnn', 'bnlstm'], default='lstm')

    parser.add_argument('--seq_c_dim', type=int, default=30)
    parser.add_argument('--seq_c_hid', type=int, default=150)
    parser.add_argument('--seq_c_layer', type=int, default=1)
    parser.add_argument('--seq_w_dim', type=int, default=100)
    parser.add_argument('--seq_w_hid', type=int, default=300)
    parser.add_argument('--seq_w_layer', type=int, default=1)
    parser.add_argument('--seq_droprate', type=float, default=0.5)
    parser.add_argument('--seq_rnn_unit', choices=['gru', 'lstm', 'rnn'], default='lstm')
    parser.add_argument('--seq_model', choices=['vanilla', 'lm-aug'], default='lm-aug')
    parser.add_argument('--seq_lambda0', type=float, default=0.05)
    parser.add_argument('--seq_lambda1', type=float, default=3)
    parser.add_argument('--fix_rate', action='store_true')

    parser.add_argument('--update', choices=['Adam', 'Adagrad', 'Adadelta', 'SGD'], default='SGD')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--clip', type=float, default=5)
    parser.add_argument('--lr', type=float, default=0.015)
    parser.add_argument('--lr_decay', type=float, default=0.05)
    args = parser.parse_args()

    tbw = wrapper(os.path.join(args.cp_root, args.checkpoint_name), args.checkpoint_name, enable_git_track=args.git_tracking)
    tbw.set_level('info')
    logger = tbw.get_logger()

    gpu_index = tbw.auto_device() if 'auto' == args.gpu else int(args.gpu)
    device = torch.device("cuda:" + str(gpu_index) if gpu_index >= 0 else "cpu")

    logger.info('Loading data from {}.'.format(args.corpus))

    dataset = pickle.load(open(args.corpus, 'rb'))
    name_list = ['flm_map', 'blm_map', 'gw_map', 'c_map', 'y_map', 'emb_array', 'train_data', 'test_data', 'dev_data']
    flm_map, blm_map, gw_map, c_map, y_map, emb_array, train_data, test_data, dev_data = [dataset[tup] for tup in name_list ]

    logger.info('Building language models and seuqence labeling models.')

    rnn_map = {'Basic': BasicRNN, 'DenseNet': DenseRNN, 'LDNet': functools.partial(LDRNN, layer_drop = 0)}
    flm_rnn_layer = rnn_map[args.lm_rnn_layer](args.lm_layer_num, args.lm_rnn_unit, args.lm_word_dim, args.lm_hid_dim, args.lm_droprate)
    blm_rnn_layer = rnn_map[args.lm_rnn_layer](args.lm_layer_num, args.lm_rnn_unit, args.lm_word_dim, args.lm_hid_dim, args.lm_droprate)
    flm_model = LM(flm_rnn_layer, None, len(flm_map), args.lm_word_dim, args.lm_droprate, label_dim = args.lm_label_dim)
    blm_model = LM(blm_rnn_layer, None, len(blm_map), args.lm_word_dim, args.lm_droprate, label_dim = args.lm_label_dim)
    flm_model_seq = SparseSeqLM(flm_model, False, args.lm_droprate, args.fix_rate)
    blm_model_seq = SparseSeqLM(blm_model, True, args.lm_droprate, args.fix_rate)
    SL_map = {'vanilla':Vanilla_SeqLabel, 'lm-aug': SeqLabel}
    seq_model = SL_map[args.seq_model](flm_model_seq, blm_model_seq, len(c_map), args.seq_c_dim, args.seq_c_hid, args.seq_c_layer, len(gw_map), args.seq_w_dim, args.seq_w_hid, args.seq_w_layer, len(y_map), args.seq_droprate, unit=args.seq_rnn_unit)

    logger.info('Loading pre-trained models from {}.'.format(args.load_seq))

    seq_file = torch.load(args.load_seq, map_location=lambda storage, loc: storage)['seq_model']
    seq_model.load_state_dict(seq_file)
    seq_model.to(device)
    crit = CRFLoss(y_map)
    decoder = CRFDecode(y_map)
    evaluator = eval_wc(decoder, 'f1')

    logger.info('Constructing dataset.')

    train_dataset, test_dataset, dev_dataset = [SeqDataset(tup_data, flm_map['\n'], blm_map['\n'], gw_map['<\n>'], c_map[' '], c_map['\n'], y_map['<s>'], y_map['<eof>'], len(y_map), args.batch_size) for tup_data in [train_data, test_data, dev_data]]

    logger.info('Constructing optimizer.')

    param_dict = filter(lambda t: t.requires_grad, seq_model.parameters())
    optim_map = {'Adam' : optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta, 'SGD': functools.partial(optim.SGD, momentum=0.9)}
    if args.lr > 0:
        optimizer=optim_map[args.update](param_dict, lr=args.lr)
    else:
        optimizer=optim_map[args.update](param_dict)

    logger.info('Saving configues.')
    tbw.save_configue(args)

    logger.info('Setting up training environ.')
    best_f1 = float('-inf')
    current_lr = args.lr
    patience_count = 0
    batch_index = 0
    normalizer = 0
    tot_loss = 0

    logger.info('Start training...')
    for indexs in range(args.epoch):

        iterator = train_dataset.get_tqdm(device)

        seq_model.train()
        for f_c, f_p, b_c, b_p, flm_w, blm_w, blm_ind, f_w, f_y, f_y_m, _ in iterator:

            seq_model.zero_grad()
            output = seq_model(f_c, f_p, b_c, b_p, flm_w, blm_w, blm_ind, f_w)
            loss = crit(output, f_y, f_y_m)

            tot_loss += utils.to_scalar(loss)
            normalizer += 1

            if args.seq_lambda0 > 0:
                f_reg0, f_reg1, f_reg3 = flm_model_seq.regularizer(args.seq_lambda1)
                b_reg0, b_reg1, b_reg3 = blm_model_seq.regularizer(args.seq_lambda1)

                loss += args.seq_lambda0 * (f_reg3 + b_reg3)

                if (f_reg0 + b_reg0 > args.seq_lambda1):
                    loss += args.seq_lambda0 * (f_reg1 + b_reg1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(seq_model.parameters(), args.clip)
            optimizer.step()

            if 0 == (batch_index + 1) % 100:
                tbw.add_loss_vs_batch({'training_loss': tot_loss / (normalizer + 1e-9)}, batch_index, add_log = False)
                tot_loss = 0
                normalizer = 0
        
            batch_index += 1

        current_lr = args.lr / (1 + (indexs + 1) * args.lr_decay)
        utils.adjust_learning_rate(optimizer, current_lr)

        dev_f1, dev_pre, dev_rec, dev_acc = evaluator.calc_score(seq_model, dev_dataset.get_tqdm(device))
        nonezero_count = (flm_model_seq.rnn.weight_list.data > 0).int().cpu().sum() + (blm_model_seq.rnn.weight_list.data > 0).cpu().int().sum()

        tbw.add_loss_vs_batch({'dev_f1': dev_f1, 'none_zero_count': nonezero_count}, batch_index, add_log = True)
        tbw.add_loss_vs_batch({'dev_pre': dev_pre, 'dev_rec': dev_rec}, batch_index, add_log = False)

        tbw.info('Saveing model...')
        tbw.save_checkpoint(model = seq_model, is_best = (nonezero_count <= args.seq_lambda1 and dev_f1 > best_f1))

        if nonezero_count <= args.seq_lambda1 and dev_f1 > best_f1:
            nonezero_count = nonezero_count

            test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(seq_model, test_dataset.get_tqdm(device))
            best_f1, best_dev_pre, best_dev_rec, best_dev_acc = dev_f1, dev_pre, dev_rec, dev_acc

            tbw.add_loss_vs_batch({'tot_loss': tot_loss/(normalizer+1e-9), 'test_f1': test_f1}, batch_index, add_log = True)
            tbw.add_loss_vs_batch({'test_pre': test_pre, 'test_rec': test_rec}, batch_index, add_log = False)

            patience_count = 0

        elif dev_f1 > best_f1:
            test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(seq_model, test_dataset.get_tqdm(device))

            tbw.add_loss_vs_batch({'tot_loss': tot_loss/(normalizer+1e-9), 'test_f1': test_f1}, batch_index, add_log = True)
            tbw.add_loss_vs_batch({'test_pre': test_pre, 'test_rec': test_rec}, batch_index, add_log = False)

        else:
            patience_count += 1
            if patience_count >= args.patience:
                break

    tbw.close()

    tbw.add_loss_vs_batch({'best_test_f1': test_f1, 'best_test_pre': test_pre, 'best_test_rec': test_rec}, 0, add_log = True, add_writer = False)
    tbw.add_loss_vs_batch({'best_dev_f1': best_f1, 'best_dev_pre': best_dev_pre, 'best_dev_rec': best_dev_rec}, 0, add_log = True, add_writer = False)

    if args.pruned_output:

        logger.info('Loading best_performing_model.')
        seq_param = wrapper.restore_best_checkpoint(tbw.path)['model']
        seq_model.load_state_dict(seq_param)
        seq_model.cuda()

        logger.info('Test before deleting layers.')
        test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(seq_model, test_dataset.get_tqdm(device))
        dev_f1, dev_pre, dev_rec, dev_acc = evaluator.calc_score(seq_model, dev_dataset.get_tqdm(device))

        tbw.add_loss_vs_batch({'best_test_f1': test_f1, 'best_dev_f1': dev_f1}, 1, add_log = True, add_writer = False)

        logger.info('Deleting layers.')
        seq_model.cpu()
        seq_model.prune_dense_rnn()
        seq_model.cuda()

        logger.info('Resulting models display.')
        print(seq_model)

        logger.info('Test after deleting layers.')
        test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(seq_model, test_dataset.get_tqdm(device))
        dev_f1, dev_pre, dev_rec, dev_acc = evaluator.calc_score(seq_model, dev_dataset.get_tqdm(device))

        tbw.add_loss_vs_batch({'best_test_f1': test_f1, 'best_dev_f1': dev_f1}, 2, add_log = True, add_writer = False)

        seq_model.cpu()
        tbw.info('Saveing model...')
        tbw.save_checkpoint(model = seq_model, is_best = True)
