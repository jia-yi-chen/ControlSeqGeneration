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
from preprocess_cornell import get_id2line,get_data

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



class Seq2seqAttGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len,batch_size):
        super(Seq2seqAttGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        self.max_dec_steps = params.max_tgt_len + 1 if max_dec_steps is None else max_dec_steps
        self.enc_attn = params.enc_attn
        self.enc_attn_cover = params.enc_attn_cover
        self.dec_attn = params.dec_attn
        self.pointer = params.pointer
        self.cover_loss = params.cover_loss
        self.cover_func = params.cover_func
        enc_total_size = params.hidden_size * 2 if params.enc_bidi else params.hidden_size
        if params.dec_hidden_size:
            dec_hidden_size = params.dec_hidden_size
            self.enc_dec_adapter = nn.Linear(enc_total_size, dec_hidden_size)
        else:
            dec_hidden_size = enc_total_size
            self.enc_dec_adapter = None

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=vocab.PAD,
                                      _weight=embedding_weights)
        self.encoder = EncoderRNN(self.embed_size, params.hidden_size, params.enc_bidi,
                                  rnn_drop=params.enc_rnn_dropout)
        self.decoder = DecoderRNN(self.vocab_size, self.embed_size, dec_hidden_size,
                                  enc_attn=params.enc_attn, dec_attn=params.dec_attn,
                                  pointer=params.pointer, out_embed_size=params.out_embed_size,
                                  tied_embedding=self.embedding if params.tie_embed else None,
                                  in_drop=params.dec_in_dropout, rnn_drop=params.dec_rnn_dropout,
                                  out_drop=params.dec_out_dropout, enc_hidden_size=enc_total_size)

        # self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.gru = nn.GRU(embedding_dim, hidden_dim)
        # self.gru2out = nn.Linear(hidden_dim, vocab_size)


    def forward(self, input, hidden):
                                            # batch_size
        emb = self.embeddings(input)                              # batch_size x embedding_dim
        emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
        out, hidden = self.gru(emb, hidden)                     # 1 x batch_size x hidden_dim (out)
        out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        out = F.log_softmax(out, dim=1)# log(P(yt|y1:t-1)) < 0
        return out, hidden

    def init_hidden_feed_z(self, batch_size=1):
        if CUDA:
            h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda()
        else:
            h=autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
        # z feed??
        return h

    def pretrain_MLE_Loss(self, inputs, targets):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """

        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inputs.size()
        inputs = inputs.permute(1, 0)           # seq_len x batch_size
        targets = targets.permute(1, 0)     # seq_len x batch_size
        h = self.init_hidden_feed_z(batch_size)

        loss = 0
        for i in range(seq_len):
            input=inputs[i]# the row (line)
            out, h = self.forward(input, h)# use real data (supervised) (not last out) as next step's input
            loss += loss_fn(out, targets[i])

        return loss # per batch


    def advers_REINFORCE_Loss(self, inputs, targets, rewards):# data s0=Y_1:T (initial state = generated sequences= MC results)
        """
            G is a policy network
            Pseudo-loss that gives corresponding policy gradients for adversarial training of Generator
        """
        # inputs: generated x's inputs
        # targets: generated x's outputs
        # rewards: generated x's D(G(Z))， applied to each token of the corresponding sentence)



        reward=rewards-torch.mean(rewards)# avg pooling?
        # reward[reward < 0] = 0 # ????????

        num_input, seq_len = inputs.size()

        inputs = inputs.permute(1, 0)           # seq_len x batch_size
        targets = targets.permute(1, 0)     # seq_len x batch_size
        h = self.init_hidden_feed_z(self.batch_size)

        batch_loss = 0.0
        for position in range(seq_len):

            input=inputs[position]# 1 x bsz  Y_t-1
            out, h = self.forward(input, h) #  Y_1:t-1 (inputs/state) -> P(yt "each action" | Y_1:t-1)

            for sentence in range(num_input):
                action = targets.data[position][sentence]
                # a token (action) selected by MC search (previous generation), but now is Yt

                Q=reward[sentence].item()# a q for taking action by MC search, but now is Y's q
                batch_loss += -Q*out[sentence][action]
                # sum_up
                # min R*(1-log(Pg(y_t|Y_1:t-1)))   :  R=0-1   Pg->1  ;   R<0    Pg->0
                # out[sentence][g_x_realhot] = log(Pg(action=y_t|Y_1:t-1))    hope it

        return batch_loss/num_input # per batch


    def generate_one_sentence(self,vocabulary):
        result= torch.zeros(self.batch_size, self.max_seq_len).type(torch.LongTensor)

        inputs=torch.zeros(self.batch_size, self.max_seq_len) # a batch of random
        if CUDA:
            inputs = Variable(inputs).type(torch.LongTensor).cuda()
            result=result.cuda()
        else:
            inputs = Variable(inputs).type(torch.LongTensor)
        inputs = inputs.permute(1, 0)

        h = self.init_hidden_feed_z(self.batch_size)

        inp=inputs[0]

        # check three generated sentences
        a_sentence = []
        b_sentence = []
        c_sentence = []
        for i in range(self.max_seq_len):
            out, h = self.forward(inp, h)
            out = torch.multinomial(torch.exp(out), 1)
            inp = out.view(-1)

            result[:, i] = out.view(-1).data
            # for l in range(batch_size):
            word=vocabulary[int(result[5, i])]
            a_sentence.append(word)

            word = vocabulary[int(result[16, i])]
            b_sentence.append(word)

            word = vocabulary[int(result[25, i])]
            c_sentence.append(word)

        a_sentence="".join([" " + i for i in a_sentence]).strip()
        b_sentence = "".join([" " + i for i in b_sentence]).strip()
        c_sentence = "".join([" " + i for i in c_sentence]).strip()
        print(a_sentence)
        print(b_sentence)
        print(c_sentence)

        return result


    def preparing_samples_from_fixed_generator(self, num_sentences):

        Samples = []
        Y= []

        # generate x can be only done in batch size
        for i in range(math.ceil(num_sentences/self.batch_size)):

            Samples_i = torch.zeros(self.batch_size, self.max_seq_len).type(torch.LongTensor)
            Y_i = torch.zeros(self.batch_size) # fake = 0

            inputs = torch.zeros(self.batch_size, self.max_seq_len)  # a batch of random
            inputs = Variable(inputs).type(torch.LongTensor)
            inputs = inputs.permute(1, 0)

            h = self.init_hidden_feed_z(self.batch_size)
            if CUDA:
                Samples_i = Samples_i.cuda()
                inputs = inputs.cuda()

            inp = inputs[0]
            for i in range(self.max_seq_len):
                out, h = self.forward(inp, h)
                out = torch.multinomial(torch.exp(out), 1)
                inp = out.view(-1)
                Samples_i[:, i] = out.view(-1).data

            Samples.append(Samples_i)
            Y.append(Y_i)

        return torch.cat(Samples,0), torch.cat(Y) # num_sentences x max_seq_len



    def preparing_samples_from_G_for_foolD(self, num_sentences):

        Samples = []
        # Y= []

        # generate x can be only done in batch size
        for i in range(math.ceil(num_sentences/self.batch_size)):

            Samples_i = torch.zeros(self.batch_size, self.max_seq_len).type(torch.LongTensor)
            # Y_i = torch.ones(self.batch_size) # to_fool = 0

            inputs = torch.zeros(self.batch_size, self.max_seq_len)  # a batch of random
            inputs = Variable(inputs).type(torch.LongTensor)
            inputs = inputs.permute(1, 0)

            h = self.init_hidden_feed_z(self.batch_size)
            if CUDA:
                Samples_i = Samples_i.cuda()
                inputs = inputs.cuda()

            inp = inputs[0]
            for i in range(self.max_seq_len):
                out, h = self.forward(inp, h)
                out = torch.multinomial(torch.exp(out), 1)# monte carlo ??
                inp = out.view(-1)
                Samples_i[:, i] = out.view(-1).data

            Samples.append(Samples_i)
            # Y.append(Y_i)

        # return torch.cat(Samples,0), torch.cat(Y) # num_sentences x max_seq_len
        return torch.cat(Samples, 0)



    def evaluate(self, real_data):
        # Turn on evaluation mode which disables dropout.
        self.eval()

        total_loss=0.0
        with torch.no_grad():
            for i in range(0, self.num_samples, self.batch_size):


                samples_i=real_data[i:i + self.batch_size]


                inputs = torch.zeros(self.batch_size, self.max_seq_len)
                inputs= samples_i[:, :self.max_seq_len ]# batch_size x max_seq_len

                targets = torch.zeros(self.batch_size, self.max_seq_len)
                targets= samples_i[:, 1:self.max_seq_len+1]# batch_size x max_seq_len


                loss = self.pretrain_MLE_Loss(inputs, targets)/self.max_seq_len


                total_loss += loss.data.item()
                # hiddens = repackage_hidden(hiddens)

        perplexity = np.exp(total_loss/(self.num_samples/self.batch_size))
        return perplexity





class EncoderRNN(nn.Module):

    def __init__(self, embed_size, hidden_size, bidi=True, *, rnn_drop: float=0):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidi else 1
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=bidi, dropout=rnn_drop)

    def forward(self, embedded, hidden, input_lengths=None):
        """
        :param embedded: (src seq len, batch size, embed size)
        :param hidden: (num directions, batch size, encoder hidden size)
        :param input_lengths: list containing the non-padded length of each sequence in this batch;
                              if set, we use `PackedSequence` to skip the PAD inputs and leave the
                              corresponding encoder states as zeros
        :return: (src seq len, batch size, hidden size * num directions = decoder hidden size)

        Perform multi-step encoding.
        """
        if input_lengths is not None:
          embedded = pack_padded_sequence(embedded, input_lengths)

        output, hidden = self.gru(embedded, hidden)

        if input_lengths is not None:
          output, _ = pad_packed_sequence(output)

        if self.num_directions > 1:
              # hidden: (num directions, batch, hidden) => (1, batch, hidden * 2)
              batch_size = hidden.size(1)
              hidden = hidden.transpose(0, 1).contiguous().view(1, batch_size,
                                                                self.hidden_size * self.num_directions)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=DEVICE)


class DecoderRNN(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, *, enc_attn=True, dec_attn=True,
               enc_attn_cover=True, pointer=True, tied_embedding=None, out_embed_size=None,
               in_drop: float=0, rnn_drop: float=0, out_drop: float=0, enc_hidden_size=None):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.combined_size = self.hidden_size
        self.enc_attn = enc_attn
        self.dec_attn = dec_attn
        self.enc_attn_cover = enc_attn_cover
        self.pointer = pointer
        self.out_embed_size = out_embed_size
        if tied_embedding is not None and self.out_embed_size and embed_size != self.out_embed_size:
            print("Warning: Output embedding size %d is overriden by its tied embedding size %d."
                % (self.out_embed_size, embed_size))
            self.out_embed_size = embed_size

        self.in_drop = nn.Dropout(in_drop) if in_drop > 0 else None
        self.gru = nn.GRU(embed_size, self.hidden_size, dropout=rnn_drop)

        if enc_attn:
            if not enc_hidden_size: enc_hidden_size = self.hidden_size
            self.enc_bilinear = nn.Bilinear(self.hidden_size, enc_hidden_size, 1)
            self.combined_size += enc_hidden_size
            if enc_attn_cover:
                self.cover_weight = nn.Parameter(torch.rand(1))

        if dec_attn:
            self.dec_bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
            self.combined_size += self.hidden_size

        self.out_drop = nn.Dropout(out_drop) if out_drop > 0 else None
        if pointer:
            self.ptr = nn.Linear(self.combined_size, 1)

        if tied_embedding is not None and embed_size != self.combined_size:
            # use pre_out layer if combined size is different from embedding size
            self.out_embed_size = embed_size

        if self.out_embed_size:  # use pre_out layer
            self.pre_out = nn.Linear(self.combined_size, self.out_embed_size)
            size_before_output = self.out_embed_size
        else:  # don't use pre_out layer
            size_before_output = self.combined_size

        self.out = nn.Linear(size_before_output, vocab_size)
        if tied_embedding is not None:
            self.out.weight = tied_embedding.weight

    def forward(self, embedded, hidden, encoder_states=None, decoder_states=None, coverage_vector=None, *,
              encoder_word_idx=None, ext_vocab_size: int=None, log_prob: bool=True):
        """
        :param embedded: (batch size, embed size)
        :param hidden: (1, batch size, decoder hidden size)
        :param encoder_states: (src seq len, batch size, hidden size), for attention mechanism
        :param decoder_states: (past dec steps, batch size, hidden size), for attention mechanism
        :param encoder_word_idx: (src seq len, batch size), for pointer network
        :param ext_vocab_size: the dynamic vocab size, determined by the max num of OOV words contained
                               in any src seq in this batch, for pointer network
        :param log_prob: return log probability instead of probability
        :return: tuple of four things:
                 1. word prob or log word prob, (batch size, dynamic vocab size);
                 2. RNN hidden state after this step, (1, batch size, decoder hidden size);
                 3. attention weights over encoder states, (batch size, src seq len);
                 4. prob of copying by pointing as opposed to generating, (batch size, 1)

        Perform single-step decoding.
        """
        batch_size = embedded.size(0)
        combined = torch.zeros(batch_size, self.combined_size, device=CUDA)

        if self.in_drop: embedded = self.in_drop(embedded)

        output, hidden = self.gru(embedded.unsqueeze(0), hidden)  # unsqueeze and squeeze are necessary
        combined[:, :self.hidden_size] = output.squeeze(0)        # as RNN expects a 3D tensor (step=1)
        offset = self.hidden_size
        enc_attn, prob_ptr = None, None  # for visualization

        if self.enc_attn or self.pointer:
            # energy and attention: (num encoder states, batch size, 1)
            num_enc_steps = encoder_states.size(0)
            enc_total_size = encoder_states.size(2)
            enc_energy = self.enc_bilinear(hidden.expand(num_enc_steps, batch_size, -1).contiguous(),
                                         encoder_states)
            if self.enc_attn_cover and coverage_vector is not None:
                enc_energy += self.cover_weight * torch.log(coverage_vector.transpose(0, 1).unsqueeze(2) + eps)
            # transpose => (batch size, num encoder states, 1)
            enc_attn = F.softmax(enc_energy, dim=0).transpose(0, 1)
            if self.enc_attn:
                # context: (batch size, encoder hidden size, 1)
                enc_context = torch.bmm(encoder_states.permute(1, 2, 0), enc_attn)
                combined[:, offset:offset+enc_total_size] = enc_context.squeeze(2)
                offset += enc_total_size
            enc_attn = enc_attn.squeeze(2)

        if self.dec_attn:
            if decoder_states is not None and len(decoder_states) > 0:
                dec_energy = self.dec_bilinear(hidden.expand_as(decoder_states).contiguous(),
                                               decoder_states)
                dec_attn = F.softmax(dec_energy, dim=0).transpose(0, 1)
                dec_context = torch.bmm(decoder_states.permute(1, 2, 0), dec_attn)
                combined[:, offset:offset + self.hidden_size] = dec_context.squeeze(2)
            offset += self.hidden_size

        if self.out_drop: combined = self.out_drop(combined)

        # generator
        if self.out_embed_size:
            out_embed = self.pre_out(combined)
        else:
            out_embed = combined
        logits = self.out(out_embed)  # (batch size, vocab size)

        # pointer
        if self.pointer:
            output = torch.zeros(batch_size, ext_vocab_size, device=CUDA)
            # distribute probabilities between generator and pointer
            prob_ptr = F.sigmoid(self.ptr(combined))  # (batch size, 1)
            prob_gen = 1 - prob_ptr
            # add generator probabilities to output
            gen_output = F.softmax(logits, dim=1)  # can't use log_softmax due to adding probabilities
            output[:, :self.vocab_size] = prob_gen * gen_output
            # add pointer probabilities to output
            ptr_output = enc_attn
            output.scatter_add_(1, encoder_word_idx.transpose(0, 1), prob_ptr * ptr_output)
            if log_prob: output = torch.log(output + eps)
        else:
            if log_prob: output = F.log_softmax(logits, dim=1)
            else: output = F.softmax(logits, dim=1)

        return output, hidden, enc_attn, prob_ptr

