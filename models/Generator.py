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
device = torch.device("cuda" if CUDA else "cpu")



num_samples=200000 # for all time (pre-training and adversarial training)
voc_size = 15000
max_seq_len = 20




# para for G
batch_size=64




class Generator(nn.Module):
    def __init__(self, emotion_num, emotion_z_size, embedding_dim, hidden_dim, vocab_size, max_seq_len,batch_size):
        super(Generator, self).__init__()

        self.emotion_z_size = emotion_z_size
        self.hidden_dim = hidden_dim
        self.latent_z_size = hidden_dim - emotion_z_size

        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size



        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)


    # def forward(self, input,  last_hidden):
    #
    #
    #     emb = self.embeddings(input)                              # batch_size x embedding_dim
    #     emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
    #     out, hidden = self.gru(emb, last_hidden)                     # 1 x batch_size x hidden_dim (out)
    #     out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
    #     out = F.log_softmax(out, dim=1)# log(P(yt|y1:t-1)) < 0
    #     return out, hidden
    def forward(self, input, emotion_hidden, last_hidden):

        # last_hidden
        last_hidden[:,:,:self.emotion_z_size] += emotion_hidden  # for each t's hidden, add the emotion hidden， to the first 32 dims

        emb = self.embeddings(input)                              # batch_size x embedding_dim
        emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
        out, hidden = self.gru(emb, last_hidden)                     # 1 x batch_size x hidden_dim (out)
        out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        out = F.log_softmax(out, dim=1)# log(P(yt|y1:t-1)) < 0
        return out, hidden

    def init_hidden_feed_z(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)).to(device)
        # z feed??
        return h

    def pretrain_MLE_Loss(self, inputs, targets, emotion_hidden, h):

        batch_size, seq_len = inputs.size()
        inputs = inputs.permute(1, 0)           # seq_len x batch_size
        targets = targets.permute(1, 0)     # seq_len x batch_size


        loss = 0
        for i in range(seq_len):
            input=inputs[i]# the row (line)
            out, h = self.forward(input, emotion_hidden, h) # use real data (supervised) (not last out) as next step's input
            # loss += loss_fn(out, targets[i])
            loss += F.nll_loss(out, targets[i]) # The negative log likelihood loss (logsoftmax---[sequence_length, voc_size])

        return loss.to(device), h # per batch

    def advers_REINFORCE_Loss(self, inputs, targets, emotion_context, h,rewards):# data s0=Y_1:T (initial state = generated sequences= MC results)
        """
            G is a policy network
            Pseudo-loss that gives corresponding policy gradients for adversarial training of Generator
        """
        # inputs: generated x's inputs
        # targets: generated x's outputs
        # rewards: generated x's D(G(Z))， applied to each token of the corresponding sentence)



        reward=rewards-torch.mean(rewards)# avg pooling?
        posQ_num=0.0
        # reward[reward < 0] = 0 # ????????

        num_input, seq_len = inputs.size()

        inputs = inputs.permute(1, 0)           # seq_len x batch_size
        targets = targets.permute(1, 0)     # seq_len x batch_size


        batch_loss = 0.0
        for position in range(seq_len):

            input=inputs[position]# 1 x bsz  Y_t-1
            out, h = self.forward(input, emotion_context, h) #  Y_1:t-1 (inputs/state) -> P(yt "each action" | Y_1:t-1)

            for sentence in range(num_input):
                action = targets.data[position][sentence].unsqueeze(0)
                outi=out[sentence].unsqueeze(0)
                # a token (action) selected by MC search (previous generation), but now is Yt

                Q=reward[sentence].item()# a q for taking action by MC search, but now is Y's q
                # only encourage Q > 0
                if Q>0:
                    batch_loss += Q*F.cross_entropy(outi, action)
                    posQ_num +=1
                # encourage both Q<0 and Q>0
                # batch_loss += Q * F.cross_entropy(outi, action)# min (R-r)log(Pg(y_t|Y_1:t-1)))   :  R=0-1   Pg->1  ;   R<0    Pg->0

        return batch_loss/posQ_num, h # per batch


    def generate_one_sentence_emo(self,Encoder, vocabulary, all_emotion_types, emotion, Generation_MODE):
        eint="".join(str(all_emotion_types[emotion])+": ")
        result= torch.zeros(batch_size, self.max_seq_len).type(torch.LongTensor).to(device)
        inputs = torch.zeros(batch_size, max_seq_len).type(torch.LongTensor).to(device) # a batch of random
        required_emotions = torch.zeros(batch_size)

        inputs = inputs.permute(1, 0)

        h = self.init_hidden_feed_z(batch_size)

        inp=inputs[0]
        for i in range(self.batch_size):
            required_emotions[i]=emotion
        required_emotions =  Variable(required_emotions).type(torch.LongTensor).to(device)


        # generated sentences
        a_sentence = []
        if Generation_MODE == "random_sampling":
            b_sentence = []
            c_sentence = []
        emotion_hidden = Encoder(required_emotions)
        # h=emotion_hidden
        for i in range(max_seq_len):
            out, h = self.forward(inp, emotion_hidden, h)#out, h = self.forward(inp,  h)
            if Generation_MODE=="top1":
                _, out = out.topk(1, dim=1)
            elif Generation_MODE=="random_sampling":
                out = torch.multinomial(torch.exp(out), 1)
            inp = out.view(-1)
            result[:, i] = out.view(-1).data
            word=vocabulary[int(result[5, i])]
            a_sentence.append(word)
            if Generation_MODE == "random_sampling":
                word = vocabulary[int(result[16, i])]
                b_sentence.append(word)
                word = vocabulary[int(result[32, i])]
                c_sentence.append(word)

        a_sentence="".join([" " + i for i in a_sentence]).strip()
        print(eint + a_sentence)
        if Generation_MODE == "random_sampling":
            b_sentence = "".join([" " + i for i in b_sentence]).strip()
            print(eint + b_sentence)
            c_sentence = "".join([" " + i for i in c_sentence]).strip()
            print(eint + c_sentence)



    def generate_one_sentence(self,vocabulary):
        result= torch.zeros(batch_size, self.max_seq_len).type(torch.LongTensor)

        inputs=torch.zeros(batch_size, max_seq_len) # a batch of random
        if CUDA:
            inputs = Variable(inputs).type(torch.LongTensor).cuda()
            result=result.cuda()
        else:
            inputs = Variable(inputs).type(torch.LongTensor)
        inputs = inputs.permute(1, 0)

        h = self.init_hidden_feed_z(batch_size)

        inp=inputs[0]

        # check three generated sentences
        a_sentence = []
        b_sentence = []
        c_sentence = []
        for i in range(max_seq_len):
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


    def preparing_samples_from_fixed_generator(self, num_sentences, E, emotion_num):
        num_sentences=self.batch_size*(num_sentences//self.batch_size)

        attributes = torch.LongTensor(num_sentences).random_(0, emotion_num)
        Samples = []
        Y= []

        # generate x can be only done in batch size
        for i in range(0, num_sentences, self.batch_size):
            Samples_i = torch.zeros(self.batch_size, self.max_seq_len).type(torch.LongTensor).to(device)
            attributes_i=Variable(attributes[i:i+self.batch_size]).to(device)

            inputs = torch.zeros(self.batch_size, max_seq_len)  # a batch of random
            inputs = Variable(inputs).type(torch.LongTensor).to(device)
            inputs = inputs.permute(1, 0)

            h = self.init_hidden_feed_z(self.batch_size)

            emotion_hidden = E(attributes_i)
            inp = inputs[0]
            for i in range(max_seq_len):
                out, h = self.forward(inp, emotion_hidden, h)
                out = torch.multinomial(torch.exp(out), 1)
                inp = out.view(-1)
                Samples_i[:, i] = out.view(-1).data

            Y_i = torch.zeros(self.batch_size).type(torch.LongTensor).to(device)
            for i in range(self.batch_size):
                Y_i[i] = attributes_i[i]+emotion_num # index： 13, 14, ...25

            Samples.append(Samples_i)
            Y.append(Y_i)

        return torch.cat(Samples,0), torch.cat(Y) # num_sentences x max_seq_len



    def evaluate(self, Encoder, real_data, real_data_emotioninputs):
        num_samples, _ = real_data.size()

        Encoder.eval()
        self.eval()

        total_loss=0.0
        b=0.0
        hidden = self.init_hidden_feed_z(batch_size)
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):

                samples_i=real_data[i:i + batch_size]
                samples_i_emo = real_data_emotioninputs[i:i + batch_size]

                bsn,_=samples_i.size()
                if bsn==batch_size:


                    inputs= samples_i[:, :max_seq_len ] # batch_size x max_seq_len
                    emotions = samples_i_emo # batch_size
                    targets= samples_i[:, 1:max_seq_len+1] # batch_size x max_seq_len

                    emotion_context = Encoder(emotions).to(device)
                    loss, hidden = self.pretrain_MLE_Loss(inputs, targets,emotion_context,hidden)


                    total_loss += loss.data.item()
                    b+=1
                    # hiddens = repackage_hidden(hiddens)

        perplexity = np.exp(total_loss/(max_seq_len*b))
        return perplexity

    def preparing_new_training_samples_for_G(self, num_fakedata, E, emotion_num, CUDA):
        # 1) Generate num_fakedata samples from the fixed G
        # 2) Sampling num_realdata samples from realdataset (already get truncated)

        # X, Y = G.preparing_samples_from_G_for_foolD(num_fakedata)
        X, Y_foolD = self.preparing_samples_from_G_for_foolD(num_fakedata, E, emotion_num)

        # shuffle
        perm = torch.randperm(X.size()[0])
        Y_foolD = Y_foolD[perm]
        X = X[perm]

        return X, Y_foolD
        # return X




    def preparing_samples_from_G_for_foolD(self, num_sentences, E, emotion_num):
        num_sentences = self.batch_size * (num_sentences // self.batch_size)

        attributes = torch.LongTensor(num_sentences).random_(0, emotion_num)
        Samples = []
        Y = []

        # generate x can be only done in batch size
        for i in range(0, num_sentences, self.batch_size):
            Samples_i = torch.zeros(self.batch_size, self.max_seq_len).type(torch.LongTensor).to(device)
            Y_i = Variable(attributes[i:i + self.batch_size]).to(device)  # index： 0, 14, ...12

            inputs = torch.zeros(self.batch_size, max_seq_len)  # a batch of random
            inputs = Variable(inputs).type(torch.LongTensor).to(device)
            inputs = inputs.permute(1, 0)

            h = self.init_hidden_feed_z(self.batch_size)

            emotion_hidden = E(Y_i)
            inp = inputs[0]
            for i in range(max_seq_len):
                out, h = self.forward(inp, emotion_hidden, h)
                out = torch.multinomial(torch.exp(out), 1)
                inp = out.view(-1)
                Samples_i[:, i] = out.view(-1).data



            Samples.append(Samples_i)
            Y.append(Y_i)

        # return torch.cat(Samples,0), torch.cat(Y) # num_sentences x max_seq_len
        return torch.cat(Samples,0), torch.cat(Y)
