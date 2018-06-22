"""
.. module:: utils
    :synopsis: utils
 
.. moduleauthor:: Liyuan Liu
"""
import numpy as np
import torch
import json

import torch
import torch.nn as nn
import torch.nn.init

from torch.autograd import Variable

def sparse_clip_norm(parameters, max_norm):
    parameters = list(filter(lambda x: x.grad, parameters))
    max_norm = float(max_norm)
    total_norm = 0
    for p in parameters:
        if is_sparse(p.grad):
            grad = p.grad.data.coalesce()
            param_norm = grad._values().norm()
        else:
            param_norm = p.grad.data.norm()
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if is_sparse(p.grad):
                p.grad.data._values().mul_(clip_coef)
            else:
                p.grad.data.mul_(clip_coef)
    return total_norm

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor:
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def to_scalar(var):
    return var.view(-1).item()

def init_embedding(input_embedding):
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)

def init_linear(input_linear):
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_lstm(input_lstm):
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
    
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1