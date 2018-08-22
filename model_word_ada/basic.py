"""
.. module:: basic
    :synopsis: basic rnn
 
.. moduleauthor:: Liyuan Liu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_word_ada.utils as utils

class BasicUnit(nn.Module):
    """
    The basic recurrent unit for the vanilla stacked RNNs.

    Parameters
    ----------
    unit : ``torch.nn.Module``, required.
        The type of rnn unit.
    input_dim : ``float``, required.
        The input dimension fo the unit.
    hid_dim : ``float``, required.
        The hidden dimension fo the unit.
    droprate : ``float``, required.
        The dropout ratrio.
    """
    def __init__(self, unit, input_dim, hid_dim, droprate):
        super(BasicUnit, self).__init__()

        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

        self.batch_norm = (unit == 'bnlstm')

        self.layer = rnnunit_map[unit](input_dim, hid_dim, 1)
        self.droprate = droprate

        self.output_dim = hid_dim

        self.init_hidden()

    def init_hidden(self):
        """
        Initialize hidden states.
        """
        self.hidden_state = None

    def rand_ini(self):
        """
        Random Initialization.
        """
        if not self.batch_norm:
            utils.init_lstm(self.layer)

    def forward(self, x):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        output: ``torch.FloatTensor``.   
            The output of RNNs.
        """
        out, new_hidden = self.layer(x, self.hidden_state)

        self.hidden_state = utils.repackage_hidden(new_hidden)
        
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        return out

class BasicRNN(nn.Module):
    """
    The multi-layer recurrent networks for the vanilla stacked RNNs.

    Parameters
    ----------
    layer_num: ``float``, required.
        The number of layers. 
    unit : ``torch.nn.Module``, required.
        The type of rnn unit.
    input_dim : ``float``, required.
        The input dimension fo the unit.
    hid_dim : ``float``, required.
        The hidden dimension fo the unit.
    droprate : ``float``, required.
        The dropout ratrio.
    """
    def __init__(self, layer_num, unit, emb_dim, hid_dim, droprate):
        super(BasicRNN, self).__init__()

        layer_list = [BasicUnit(unit, emb_dim, hid_dim, droprate)] + [BasicUnit(unit, hid_dim, hid_dim, droprate) for i in range(layer_num - 1)]
        self.layer = nn.Sequential(*layer_list)
        self.output_dim = layer_list[-1].output_dim

        self.init_hidden()

    def init_hidden(self):
        """
        Initialize hidden states.
        """
        for tup in self.layer.children():
            tup.init_hidden()

    def rand_ini(self):
        """
        Random Initialization.
        """
        for tup in self.layer.children():
            tup.rand_ini()

    def forward(self, x):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The output of RNNs.
        """
        return self.layer(x)