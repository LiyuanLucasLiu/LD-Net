"""
.. module:: elmo
    :synopsis: deep contextualized representation
 
.. moduleauthor:: Liyuan Liu
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import model_seq.utils as utils
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

class EBUnit(nn.Module):
    def __init__(self, ori_unit, droprate, fix_rate):
        super(EBUnit, self).__init__()

        self.layer = ori_unit.layer

        self.droprate = droprate

        self.output_dim = ori_unit.output_dim

    def forward(self, x):

        out, _ = self.layer(x)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        return out

class ERNN(nn.Module):
    def __init__(self, ori_drnn, droprate, fix_rate):
        super(ERNN, self).__init__()

        self.layer_list = [EBUnit(ori_unit, droprate, fix_rate) for ori_unit in ori_drnn.layer._modules.values()]

        self.gamma = nn.Parameter(torch.FloatTensor([1.0]))
        self.weight_list = nn.Parameter(torch.FloatTensor([0.0] * len(self.layer_list)))

        self.layer = nn.ModuleList(self.layer_list)

        for param in self.layer.parameters():
            param.requires_grad = False

        self.output_dim = self.layer_list[-1].output_dim

    def regularizer(self, lambda1):
        srd_weight = self.weight_list - (1.0 / len(self.layer_list))
        return (srd_weight ** 2).sum()

    def forward(self, x):
        out = 0
        nw = self.gamma * F.softmax(self.weight_list, dim=0)
        for ind in range(len(self.layer_list)):
            x = self.layer[ind](x)
            out += x * nw[ind]
        return out

class ElmoLM(nn.Module):

    def __init__(self, ori_lm, backward, droprate, fix_rate):
        super(ElmoLM, self).__init__()

        self.rnn = ERNN(ori_lm.rnn, droprate, fix_rate)

        self.w_num = ori_lm.w_num
        self.w_dim = ori_lm.w_dim
        self.word_embed = ori_lm.word_embed
        self.word_embed.weight.requires_grad = False

        self.output_dim = ori_lm.rnn_output

        self.backward = backward

    def init_hidden(self):
        return

    def regularizer(self, lambda1):
        return self.rnn.regularizer(lambda1)

    def prox(self, lambda0, lambda1):
        return 0.0

    def forward(self, w_in, ind=None):
        w_emb = self.word_embed(w_in)
        
        out = self.rnn(w_emb)

        if self.backward:
            out_size = out.size()
            out = out.view(out_size[0] * out_size[1], out_size[2]).index_select(0, ind).contiguous().view(out_size)

        return out