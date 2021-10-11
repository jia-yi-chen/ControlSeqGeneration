"""
author: Jiayi Chen
time: 12/28/2019
"""

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

num_samples=150000 # for all time (pre-training and adversarial training)
voc_size = 15000
max_seq_len = 20


pretrain=False
Pretrain_EPOCHS = 10
PreTraining_trn_num=150000
REAL_DATA="text.txt"
SAVE_MODEL='Pretrained_model_longertext_hiddim256_bsz64.hdf5'
# SAVE_MODEL='Pretrained_model.hdf5'


Advs_TrainGAN = True
Advtrain_EPOCHS=30
SAVE_MODEL_GAN_G='Adver_trained_G_model.hdf5'
SAVE_MODEL_GAN_D='Adver_trained_D_model.hdf5'




# para for G
G_embedding_dim = 64
G_hidden_dim = 256
batch_size=64
g_steps = 20# no use
TRAIN_G_LOSS_MINIMUM= 0.5

# para for G
D_embedding_dim = 64
D_batch_size = 64
D_num_filter = 50
D_kernel_sizes = (3,4,5)
d_steps = 20
TRAIN_D_ACC_MINIMUM= 0.95

from sampling import sampling_real_data
from sampling import preparing_new_training_samples_for_D



class Generator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len,batch_size):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size


        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)


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
        h = self.init_hidden_feed_z(batch_size)

        batch_loss = 0.0
        for position in range(seq_len):

            input=inputs[position]# 1 x bsz  Y_t-1
            out, h = self.forward(input, h) #  Y_1:t-1 (inputs/state) -> P(yt "each action" | Y_1:t-1)

            for sentence in range(num_input):
                action = targets.data[position][sentence]
                # a token (action) selected by MC search (previous generation), but now is Yt

                Q=reward[sentence].item()# a q for taking action by MC search, but now is Y's q
                if Q>0:
                    batch_loss += -Q*out[sentence][action]
                # elif Q<0:
                #     batch_loss += Q * out[sentence][action]

                # sum_up
                # min R*(1-log(Pg(y_t|Y_1:t-1)))   :  R=0-1   Pg->1  ;   R<0    Pg->0
                # out[sentence][g_x_realhot] = log(Pg(action=y_t|Y_1:t-1))    hope it

        return batch_loss/num_input # per batch


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


    def preparing_samples_from_fixed_generator(self, num_sentences):

        Samples = []
        Y= []

        # generate x can be only done in batch size
        for i in range(math.ceil(num_sentences/self.batch_size)):

            Samples_i = torch.zeros(self.batch_size, self.max_seq_len).type(torch.LongTensor)
            Y_i = torch.zeros(self.batch_size) # fake = 0

            inputs = torch.zeros(self.batch_size, max_seq_len)  # a batch of random
            inputs = Variable(inputs).type(torch.LongTensor)
            inputs = inputs.permute(1, 0)

            h = self.init_hidden_feed_z(self.batch_size)
            if CUDA:
                Samples_i = Samples_i.cuda()
                inputs = inputs.cuda()

            inp = inputs[0]
            for i in range(max_seq_len):
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

            inputs = torch.zeros(self.batch_size, max_seq_len)  # a batch of random
            inputs = Variable(inputs).type(torch.LongTensor)
            inputs = inputs.permute(1, 0)

            h = self.init_hidden_feed_z(self.batch_size)
            if CUDA:
                Samples_i = Samples_i.cuda()
                inputs = inputs.cuda()

            inp = inputs[0]
            for i in range(max_seq_len):
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
            for i in range(0, num_samples, batch_size):


                samples_i=real_data[i:i + batch_size]


                inputs = torch.zeros(batch_size, max_seq_len)
                inputs= samples_i[:, :max_seq_len ]# batch_size x max_seq_len

                targets = torch.zeros(batch_size, max_seq_len)
                targets= samples_i[:, 1:max_seq_len+1]# batch_size x max_seq_len


                loss = self.pretrain_MLE_Loss(inputs, targets)/self.max_seq_len


                total_loss += loss.data.item()
                # hiddens = repackage_hidden(hiddens)

        perplexity = np.exp(total_loss/(num_samples/batch_size))
        return perplexity

    def preparing_new_training_samples_for_G(self, num_fakedata, CUDA):
        # 1) Generate num_fakedata samples from the fixed G
        # 2) Sampling num_realdata samples from realdataset (already get truncated)

        # X, Y = G.preparing_samples_from_G_for_foolD(num_fakedata)
        X = self.preparing_samples_from_G_for_foolD(num_fakedata)

        # shuffle
        perm = torch.randperm(X.size()[0])
        # Y = Y[perm]
        X = X[perm]

        # return X, Y
        return X

    def train_G(self, g_optimizer, g_steps, D, real_data, Voc):  # fix D, use D to get rewards


        # Define dev data
        num_dev_samples=200
        Dev_X  = self.preparing_new_training_samples_for_G(num_dev_samples,  CUDA=CUDA) # Variables & already in cuda()
        ## TODO : truncate [PAD]  , should batchsize=1

        #
        # # TODO : Generated some sentences HERE ??
        num_trn_samples=5000

        Trn_X = self.preparing_new_training_samples_for_G(num_trn_samples, CUDA=CUDA)  # Variables & already in cuda()
        Rewards = D.forward(Trn_X).data.view(-1) # reward should not be the variable
        avg_rewards = torch.mean(Rewards)
        num_trn_samples = Rewards.shape[0]
        print("       >_< generate new g_x .....(average of rewards:{:5.2f})".format(avg_rewards.item()))

        MC_time=0
        # for step in range(1):
        while avg_rewards < 0.9:
            MC_time = MC_time+1



            print("       ^_^ Train policy G based on current MC searched actions/sentences .....")
            # G.train()
            total_loss = 1000.0
            cur_loss = total_loss / num_trn_samples
            epoch=0
            while cur_loss> TRAIN_G_LOSS_MINIMUM or epoch<g_steps:
                epoch = epoch+1
            # for epoch in range(5):

                # # shuffle the 5000 trn_data
                perm = torch.randperm(Trn_X.size()[0])
                Trn_X = Trn_X[perm]
                Rewards = Rewards[perm]
                self.train()
                for i in range(0, num_trn_samples, self.batch_size):

                    ## TODO :  truncate Trn_X by [PAD]  , should batchsize=1

                    samples_i = Trn_X[i:i + self.batch_size]  # num_samples  x max_seq_len+1
                    targets = samples_i
                    inputs = torch.zeros(self.batch_size, self.max_seq_len)
                    inputs[:, 0] = 0
                    inputs[:, 1:] = samples_i[:, :self.max_seq_len - 1]

                    Reward_i=Rewards[i:i + self.batch_size]
                    # Rewards = D.forward(samples_i) # Variable


                    if CUDA:
                        inputs = Variable(inputs).type(torch.LongTensor).cuda()
                        targets = Variable(targets).type(torch.LongTensor).cuda()
                        # Reward_i = Variable(Reward_i).cuda()
                        Reward_i=Reward_i.cuda()
                    else:
                        inputs = Variable(inputs).type(torch.LongTensor)
                        targets = Variable(targets).type(torch.LongTensor)
                        # Reward_i = Variable(Reward_i)


                    g_optimizer.zero_grad()

                    lossG = self.advers_REINFORCE_Loss(inputs, targets, Reward_i)  # zeros = fake

                    lossG.backward()
                    g_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

                    total_loss  += lossG.item()


                cur_loss = total_loss / num_trn_samples
                print('       Training G  ---  the {:3d}th MC action   |  epoch {:3d} | cur_avg_loss {:5.2f} '.format(MC_time, epoch, cur_loss)) # not perplexity
                total_loss = 0.

                self.eval()
                # eval mode of G:
                # perplexity_trn = G.evaluate(real_data)
                # print("Perplexity is : {}".format(perplexity_trn))
                # print(" ")
                print("E.g. 3 generated random sequences:")
                g_x = self.generate_one_sentence(Voc)
                print(" ")


            ###################### Generate new sentences / New MC search ######################################
            Trn_X = self.preparing_new_training_samples_for_G( num_trn_samples, CUDA=CUDA)  # Variables & already in cuda()
            Rewards = D.forward(Trn_X).data.view(-1) # reward should not be the variable
            avg_rewards = torch.mean(Rewards)
            num_trn_samples = Rewards.shape[0]
            print("       >_< generate new g_x .....(average of rewards:{:5.2f})".format(avg_rewards.item()))

        torch.save(self.state_dict(), SAVE_MODEL_GAN_G)  # save model ????????
        # return G



class DiscriminatorCNN(nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_filters, window_sizes=(3, 4, 5)):
        super(DiscriminatorCNN, self).__init__()

        self.emb_dim = embedding_dim # every filter size: 3/4/5 x 32
        self.num_classes=2
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, [window_size, embedding_dim], padding=(window_size - 1, 0)) for window_size in window_sizes
        ])

        # self.fc = nn.Linear(num_filters * len(window_sizes), self.num_classes)
        self.fc = nn.Linear(num_filters * len(window_sizes), 1)


    def forward(self, input):

        ## TODO :  truncate input by [PAD]

        x = self.embedding(input)           # [B, T, E]

        # Apply a convolution + max pool layer for each window size
        x = torch.unsqueeze(x, 1)       # [B, C, T, E] Add a channel dim.
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))        # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)            # [B, F, window]

        # FC
        x = x.view(x.size(0), -1)       # [B, F * window]
        logits = self.fc(x)             # [B, class]

        # Prediction
        # probs = F.softmax(logits)       # [B, class]
        # classes = torch.max(probs, 1)[1]# [B]
        probs = torch.sigmoid(logits).view(-1)

        # return probs, classes
        return probs


    def loss(self,outputs, targets):
        # outputs:  [batch_size x num_class]
        # target:   batch_size

        loss_bce = nn.BCELoss()
        # loss_bce=nn.CrossEntropyLoss()
        return loss_bce(outputs, targets)

    def evaluate(self):
        self.eval()


    def train_D(self, d_optimizer, real_data_samples, d_steps, G):  # fix G to generate g_x  ; need real data

        # Define dev data
        num_dev_realdata=200
        num_dev_fakedata=200
        Dev_X , Dev_Target = preparing_new_training_samples_for_D(real_data_samples, num_dev_realdata, G, num_dev_fakedata,  CUDA=CUDA) # Variables & already in cuda()
        ## TODO
        # truncate [PAD]  , should batchsize=1

        # Define training data
        num_trn_realdata=5000
        num_trn_fakedata=5000
        Trn_X, Trn_Target = preparing_new_training_samples_for_D(real_data_samples, num_trn_realdata, G, num_trn_fakedata, CUDA=CUDA)  # Variables & already in cuda()
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
                total_acc_num  += torch.sum((out > 0.5) == (targets > 0.5)).data.item()


            # eval mode
            total_loss = total_loss/(math.ceil(num_trn/float(D_batch_size)))
            accuracy_trn= total_acc_num/(float(num_trn))
            print('       Training D  ---  step {:3d} | cur_avg_loss {:5.2f} | prediction accuracy {:8.2f}'.format(
                           step, total_loss, accuracy_trn))
            # D.evaluate()
            self.eval()

        torch.save(self.state_dict(), SAVE_MODEL_GAN_D)# save model ????????
        # return D




def train():


    G = Generator(embedding_dim=G_embedding_dim,
                  hidden_dim=G_hidden_dim,
                  vocab_size=voc_size,
                  max_seq_len=max_seq_len,
                  batch_size=batch_size)
    D = DiscriminatorCNN(embedding_dim=D_embedding_dim,
                  vocab_size=voc_size,
                  num_filters=D_num_filter,
                  window_sizes=D_kernel_sizes)


    print('Prepare real data...')
    real_data_samples, Vocabulary = sampling_real_data(REAL_DATA, num_samples) # PreTraining_trn_num = 50000
    print(" ")


    if CUDA:
        real_data_samples = real_data_samples.cuda()
        G = G.cuda()
        D = D.cuda()

    print("generate three random sequences before training:")
    g_x = G.generate_one_sentence(Vocabulary)
    print(" ")


    print("Pre-training Generator..............")
    if pretrain:
        print('Starting Generator MLE Training...')
        gen_pretrain_opt = optim.Adam(G.parameters(), lr=1e-2)
        for epoch in range(Pretrain_EPOCHS):
            # sys.stdout.flush()

            ## TODO
            # shuffle

            # train mode
            G.train() # !!!!!!! important
            total_loss = 0.0
            for i in range(0, PreTraining_trn_num, batch_size):

                samples_i=real_data_samples[i:i + batch_size]# num_samples  x max_seq_len+1

                inputs = torch.zeros(batch_size, max_seq_len)
                inputs= samples_i[:, :max_seq_len ]# batch_size x max_seq_len

                targets = torch.zeros(batch_size, max_seq_len)
                targets= samples_i[:, 1:max_seq_len+1]# batch_size x max_seq_len

                if CUDA:
                    inputs = Variable(inputs).type(torch.LongTensor).cuda()
                    targets = Variable(targets).type(torch.LongTensor).cuda()
                else:
                    inputs = Variable(inputs).type(torch.LongTensor)
                    targets = Variable(targets).type(torch.LongTensor)

                gen_pretrain_opt.zero_grad()

                loss = G.pretrain_MLE_Loss(inputs, targets)


                loss.backward()
                gen_pretrain_opt.step()

                total_loss += loss.data.item()

                if (i / batch_size) % np.ceil(
                        np.ceil(PreTraining_trn_num / float(batch_size)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    # sys.stdout.flush()

                if i % 400 == 0 and i > 0:
                    cur_loss = total_loss / 400
                    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                          ' cur_avg_loss {:5.2f} | current_perplexity {:8.2f}'.format(
                        epoch, i//batch_size, PreTraining_trn_num // batch_size, 1e-2,
                        cur_loss, math.exp(cur_loss)))
                    total_loss = 0

            # eval mode
            perplexity_trn = G.evaluate(real_data_samples)
            print("Perplexity is : {}".format(perplexity_trn))
            print(" ")

            # generation
            print("generate a random sequence after training:")
            g_x=G.generate_one_sentence(Vocabulary)
            print(" ")

        torch.save(G.state_dict(), SAVE_MODEL)# save model ????????
    else:
        G.load_state_dict(torch.load(SAVE_MODEL))
        if CUDA:
            G = G.cuda()
        print("generate a random sequence after loading model:")
        g_x = G.generate_one_sentence(Vocabulary)
        print(" ")
    print("End of Pre-training\n")



    print("Adversarial Training.........")
    if Advs_TrainGAN:

        # criterion = nn.BCELoss()
        d_learning_rate = 1e-2
        g_learning_rate = 1e-3
        sgd_momentum = 0.9
        d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=sgd_momentum)  # for adversarial training
        g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)

        for epoch in range(Advtrain_EPOCHS):
            print("-----Adversarial Training (Epoch {})-----".format(epoch))

            print("   Train Discriminator......")
            # D = train_D(D, d_optimizer, real_data_samples, d_steps, G ) # fix G to generate g_x  ; need real data
            D.train_D(d_optimizer, real_data_samples, d_steps, G ) # fix G to generate g_x  ; need real data

            print("   Train Generator......")
            # G = train_G(G, g_optimizer, g_steps, D, real_data_samples, Vocabulary) # fix D, use D to get rewards
            G.train_G(g_optimizer, g_steps, D, real_data_samples, Vocabulary)  # fix D, use D to get rewards


            print(" ")
            print(" ")
            print("After one step adversarial training:")
            g_x = G.generate_one_sentence(Vocabulary)
            print(" ")



        # if matplotlib_is_available:
        #     print("Plotting the generated distribution...")
        #     values = extract(g_fake_data)
        #     print(" Values: %s" % (str(values)))
        #     plt.hist(values, bins=50)
        #     plt.xlabel('Value')
        #     plt.ylabel('Count')
        #     plt.title('Histogram of Generated Distribution')
        #     plt.grid(True)
        #     plt.show()


train()