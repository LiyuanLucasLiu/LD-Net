"""
.. module:: seqlm
    :synopsis: language model for sequence labeling
 
.. moduleauthor:: Liyuan Liu
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import model_seq.utils as utils

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicSeqLM(nn.Module):

    def __init__(self, ori_lm, backward, droprate, fix_rate):
        super(BasicSeqLM, self).__init__()

        self.rnn = ori_lm.rnn

        for param in self.rnn.parameters():
            param.requires_grad = False

        self.w_num = ori_lm.w_num
        self.w_dim = ori_lm.w_dim
        self.word_embed = ori_lm.word_embed
        self.word_embed.weight.requires_grad = False

        self.output_dim = ori_lm.rnn_output

        self.backward = backward

    def init_hidden(self):
        self.rnn.init_hidden()
    
    def regularizer(self):
        return self.rnn.regularizer()

    def forward(self, w_in, ind=None):
        w_emb = self.word_embed(w_in)
        
        out = self.rnn(w_emb)

        if self.backward:
            out_size = out.size()
            out = out.view(out_size[0] * out_size[1], out_size[2]).index_select(0, ind).contiguous().view(out_size)

        return out