#!/usr/bin/env python


import numpy as np
import string
import sys
import torch
import time
import torch.nn as nn

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


SAVE_MODEL_GAN_D='Adver_trained_D_model.hdf5'



# para for D
D_batch_size = 64



from sampling_nclass import preparing_new_training_samples_for_D


class DiscriminatorCNN(nn.Module):
    def __init__(self, embedding_dim, class_num, vocab_size, num_filters, MODE, window_sizes=(3, 4, 5)):
        super(DiscriminatorCNN, self).__init__()

        self.emb_dim = embedding_dim # every filter size: 3/4/5 x 32
        self.num_classes= class_num
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.MODE=MODE
        if MODE=="2*emo_num":
            self.num_classes = class_num*2
            # [real-emo1, real-emo2...real-emoN, fake-emo1, fake-emo2, ...., fake-emoN]
        elif MODE=="emo_num+1":
            self.num_classes = class_num+1
            # [real/fake, emo1, emo2, ..., emoN]


        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, [window_size, embedding_dim], padding=(window_size - 1, 0)) for window_size in window_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(window_sizes), self.num_classes)


    def forward(self, input):

        ## TODO :  truncate input by [PAD]

        x = self.embedding(input)           # [B, T, E]

        # Apply a convolution + max pool layer for each window size
        x = torch.unsqueeze(x, 1)       # [B, C, T, E] Add a channel dim.
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))        # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 1)            # [B, F*window]

        x = self.dropout(x)
        # FC
        logits = self.fc(x)             # [B, class]

        # Prediction
        # [B, class]
        # classes = torch.max(probs, 1)[1]# [B]
        # probs = torch.sigmoid(logits).view(-1)

        # return probs, classes
        return logits


    def loss(self,outputs, targets):
        # outputs:  [batch_size x num_class]
        # target:   batch_size

        return F.cross_entropy(outputs, targets)#have log_softmax

    def evaluate(self):
        self.eval()


    def train_D(self, d_optimizer, real_data_samples, real_data_samples_label, d_steps, G, E, emotion_num, MODE="emo_num+1",TRAIN_D_ACC_MINIMUM=0.90):  # fix G to generate g_x  ; need real data
        if MODE=="2*emo_num":

            # Define dev data
            num_dev_realdata=200
            num_dev_fakedata=200
            Dev_X , Dev_Target = preparing_new_training_samples_for_D(real_data_samples, real_data_samples_label, num_dev_realdata, G, E, num_dev_fakedata, emotion_num, device) # Variables & already in cuda()
            ## TODO
            # truncate [PAD]  , should batchsize=1

            # Define training data
            num_trn_realdata=10000
            num_trn_fakedata=10000
            Trn_X, Trn_Target = preparing_new_training_samples_for_D(real_data_samples, real_data_samples_label, num_trn_realdata, G, E, num_trn_fakedata, emotion_num, device)  # Variables & already in cuda()
            num_trn = Trn_X.shape[0]

            ## TODO
            # truncate [PAD]  , should batchsize=1

            step=0
            accuracy_trn = 0.0
            while accuracy_trn < TRAIN_D_ACC_MINIMUM:
            # for step in range(d_steps):
                step = step+1
                # shuffle the 5000 trn_data
                perm = torch.randperm(Trn_Target.size()[0])
                Trn_Target = Trn_Target[perm]
                Trn_X = Trn_X[perm]

                # train one time (go through all batches)
                # train mode
                self.train()
                total_loss=0.0
                total_acc_num=0
                for i in range(0, num_trn, D_batch_size):
                    inputs=Trn_X[i:i + D_batch_size]              # batch_size  x max_seq_len
                    targets = Trn_Target[i:i + D_batch_size]      # batch_size  x 1

                    d_optimizer.zero_grad()

                    out = self.forward(inputs) # batch_size  (1 value) not*2
                    lossD = self.loss(out, targets)  # zeros = fake
                    lossD.backward()
                    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

                    total_loss  += lossD.item()

                    prediction=torch.max(out, 1)[1]
                    prediction=prediction.view(targets.size()).data
                    total_acc_num  += (prediction == targets.data).sum().item()


                # eval mode
                total_loss = total_loss/(math.ceil(num_trn/float(D_batch_size)))
                accuracy_trn= total_acc_num/(float(num_trn))
                print('       Training D  ---  step {:3d} | cur_avg_loss {:5.2f} | prediction accuracy {:8.2f}'.format(
                               step, total_loss, accuracy_trn))
                # D.evaluate()
                self.eval()

        torch.save(self.state_dict(), SAVE_MODEL_GAN_D)# save model ????????
        # return D

    def rewards(self, X, Y):

        out = self.forward(X)  # # the larger, the better

        # type 1
        prediction = torch.max(out, 1)[1]
        prediction = prediction.view(Y.size()).data
        accuracy_top1 = (prediction == Y.data).sum().item()/float(Y.size()[0])

        # type 2
        accuracy_ce=torch.zeros(Y.size())
        for i in range(Y.size()[0]):
            outi = out[i].unsqueeze(0)# 1 x 2*num_emo
            Yi=Y[i].unsqueeze(0)# 1
            accuracy_ce[i] = math.exp(-F.cross_entropy(outi, Yi).item())# the lower, the better (0 is the best, +inf is the worst)

        return accuracy_top1, accuracy_ce
