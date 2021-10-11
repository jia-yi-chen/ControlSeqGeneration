#!/usr/bin/env python


import numpy as np
import string
import sys
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import os
from preprocess.preprocess_cornell import get_id2line,get_data

from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import math

matplotlib_is_available = True
try:
  from matplotlib import pyplot as plt
except ImportError:
  print("Will skip plotting; matplotlib is not available.")
  matplotlib_is_available = False


CUDA=True






class AttributeEncoder(nn.Module):
    def __init__(self, emotion_num, embedding_dim, hidden_size, out_size, dropout=0):
        super(AttributeEncoder, self).__init__()


        self.emotion_num = emotion_num
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.out_size = out_size # will as part to be feeded to hidden

        self.embedding = nn.Embedding(emotion_num, embedding_dim)
        self.emb2hidden = nn.Linear(embedding_dim, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, out_size)

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        # #   because our input size is a word embedding with number of features == hidden_size
        # self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_emotion):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_emotion)
        embedded = embedded.view(1, -1, self.embedding_dim)
        hidden = self.emb2hidden(embedded)
        out = self.hidden2out(hidden)
        # # Pack padded batch of sequences for RNN module
        # packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        # outputs, hidden = self.gru(packed, hidden)
        # # Unpack padding
        # outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # # Sum bidirectional GRU outputs
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return out