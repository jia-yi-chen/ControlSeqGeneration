"""
author: Jiayi Chen
time: 12/28/2019
"""

import logging
import logging.config

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s-%(levelname)s: %(message)s',
                    filename='data/logging.log',)
logger = logging.getLogger('logging_write')
logger.debug('This is debug message')
logger.info('This is info message')
logger.warning('This is warning message')

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
CUDA_LAUNCH_BLOCKING=1


stdout_backup = sys.stdout
log_file = open("logging.log", "w")
sys.stdout = log_file

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


pretrain=False
Pretrain_EPOCHS = 10
pretrain_lr=1e-3
PreTraining_trn_num=num_samples
REAL_DATA="data/emotion1/text_emotion.csv"
SAVE_preMODEL_E='Pretrained_encoder_emocontrol_voc15000.hdf5'
SAVE_preMODEL_G='Pretrained_model_emocontrol_voc15000.hdf5'
# SAVE_preMODEL_G='Pretrained_model_longertext_hiddim256_bsz64.hdf5'
# SAVE_preMODEL_G='Pretrained_model.hdf5'


Advs_TrainGAN = True
Advtrain_EPOCHS=30
SAVE_MODEL_GAN_E='Adver_trained_E_model.hdf5'
SAVE_MODEL_GAN_G='Adver_trained_G_model.hdf5'
SAVE_MODEL_GAN_D='Adver_trained_D_model.hdf5'
rewards_thres=[0.82, 0.80, 0.8, 0.7, 0.6]




# para for G
G_embedding_dim = 64
Emotion_hidden_dim = 64
G_hidden_dim = 256
batch_size=64
g_steps = 10# no use
TRAIN_G_LOSS_MINIMUM= 0.2 # should be 0~10， the acc_ce on D
g_learning_rate = 1e-3


# para for G
D_embedding_dim = 64
D_batch_size = 64
D_num_filter = 50
D_kernel_sizes = (3,4,5)
d_steps = 20
TRAIN_D_ACC_MINIMUM= [0.83, 0.82, 0.81, 0.80, 0.80]
d_learning_rate = 1e-2*2
sgd_momentum = 0.9

Generation_MODE = "top1"#""random_sampling"

from sampling_nclass import sampling_real_data
from sampling_nclass import preparing_new_training_samples_for_D
from AttributeEncoder import AttributeEncoder
from Generator import Generator
from ClassDiscriminator import DiscriminatorCNN


def train_G(G, E, emotion_num, e_optimizer, g_optimizer, g_steps, D, real_data, Voc, emotion_types, Reward_Thres):  # fix D, use D to get rewards

    # Define dev data
    num_dev_samples = 200
    Dev_X, Dev_Y_foolD = G.preparing_new_training_samples_for_G(num_dev_samples, E, emotion_num, CUDA=CUDA)  # Variables & already in cuda()

    num_trn_samples = G.batch_size*20000//G.batch_size
    Trn_X, Trn_Y_foolD = G.preparing_new_training_samples_for_G(num_trn_samples, E, emotion_num, CUDA=CUDA)  # Variables & already in cuda()
    Rewards1ACC, Rewards2CE = D.rewards(Trn_X, Trn_Y_foolD)
    print("       >_< Monte-Carlo searching new samples .....( rewards(Acc, higher is better):{:5.2f} in 0~1 ,  avg_rewards(Softmax, higher is better):{:5.2f} in 0~1)".format(Rewards1ACC, torch.mean(Rewards2CE).item()))

    MC_time = 0
    while Rewards1ACC < Reward_Thres:
        MC_time = MC_time + 1

        print("       Update G based on {}th Monte-Carlo searched samples .....".format(MC_time))

        total_loss = 1000000000.0
        cur_loss = total_loss / num_trn_samples
        epoch = 0
        while cur_loss > TRAIN_G_LOSS_MINIMUM or epoch < g_steps:
            epoch = epoch + 1

            print("       shuffle all data")
            perm = torch.randperm(Trn_X.size()[0])
            Trn_X = Trn_X[perm]
            Trn_Y_foolD = Trn_Y_foolD[perm]
            Rewards2CE = Rewards2CE[perm]

            E.train()
            G.train()
            total_loss=0.0
            b=0.0
            hidden = G.init_hidden_feed_z(batch_size)
            for i in range(0, num_trn_samples, G.batch_size):

                ## TODO :  truncate Trn_X by [PAD]  , should batchsize=1

                samples_i = Trn_X[i:i + G.batch_size]  # num_samples  x max_seq_len+1
                bsz, _ = samples_i.size()
                if bsz == G.batch_size:
                    targets = samples_i
                    inputs = torch.zeros(G.batch_size, G.max_seq_len)
                    inputs[:, 0] = 0
                    inputs[:, 1:] = samples_i[:, :G.max_seq_len - 1]
                    emotions = Trn_Y_foolD[i:i + G.batch_size]
                    Reward_i = Rewards2CE[i:i + G.batch_size]
                    # Rewards = D.forward(samples_i) # Variable

                    inputs = Variable(inputs).type(torch.LongTensor).to(device)
                    targets = Variable(targets).type(torch.LongTensor).to(device)
                    emotions = Variable(emotions).type(torch.LongTensor).to(device)
                    Reward_i = Reward_i.to(device)  # Reward_i = Variable(Reward_i).cuda()

                    hidden = repackage_hidden(hidden)
                    e_optimizer.zero_grad()
                    g_optimizer.zero_grad()

                    emotion_context=E(emotions)
                    lossG, hidden = G.advers_REINFORCE_Loss(inputs, targets, emotion_context, hidden, Reward_i)  # zeros = fake

                    lossG.backward()
                    e_optimizer.step()
                    g_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()


                    _ = torch.nn.utils.clip_grad_norm_(E.parameters(), 1)
                    _ = torch.nn.utils.clip_grad_norm_(G.parameters(), 1)

                    for p in G.parameters():
                        p.data.add_(-g_learning_rate, p.grad.data)
                    for p in E.parameters():
                        p.data.add_(-g_learning_rate*0.1, p.grad.data)

                    total_loss += lossG.item()
                    b+=1

                if i % 100 == 0 and i > 0:
                    cur_loss = total_loss / b
                    print('| the {:3d}th MC action | epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                          ' cur_avg_loss {:5.2f} '.format(
                        MC_time, epoch, i//batch_size, num_trn_samples // batch_size, g_learning_rate, cur_loss))
                    total_loss = 0.0
                    b=0.0


            E.eval()
            G.eval()
            # eval mode of G:
            # perplexity_trn = G.evaluate(real_data)
            # print("Perplexity is : {}".format(perplexity_trn))
            print("#### Test #### (Generate top1/random sequences training all data)".format(MC_time))
            for i in range(emotion_num):
                print("(1) top1:")
                G.generate_one_sentence_emo(E, Voc, emotion_types, i, Generation_MODE)  # [0, 1, 2, .....emotion_num-1] is one-hot for emotions
                print("(2) random_sampling:")
                G.generate_one_sentence_emo(E, Voc, emotion_types, i, "random_sampling")
            print(" ")

        ###################### Generate new sentences / New MC search ######################################
        Trn_X, Trn_Y_foolD = G.preparing_new_training_samples_for_G(num_trn_samples, E, emotion_num,CUDA=CUDA)  # Variables & already in cuda()
        Rewards1ACC, Rewards2CE = D.rewards(Trn_X, Trn_Y_foolD)
        print("       >_< Monte-Carlo searching new samples .....( rewards(Acc, higher is better):{:5.2f} in 0~1 ,  avg_rewards(Softmax, higher is better):{:5.2f} in 0~1)".format(Rewards1ACC, torch.mean(Rewards2CE).item()))

        torch.save(E.state_dict(), SAVE_MODEL_GAN_E)
        torch.save(G.state_dict(), SAVE_MODEL_GAN_G)  # save model ????????
    return E, G


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train():


    print('Prepare real data...')
    real_data_samples, real_data_samples_label, Vocabulary, emotion_num, all_emotion_types = sampling_real_data(REAL_DATA, num_samples, voc_size)
    print(" ")

    E = AttributeEncoder(emotion_num=emotion_num, embedding_dim=32, hidden_size=32,out_size=Emotion_hidden_dim)
    G = Generator(emotion_num=emotion_num,
                  emotion_z_size=Emotion_hidden_dim,
                  embedding_dim=G_embedding_dim,
                  hidden_dim=G_hidden_dim,
                  vocab_size=voc_size,
                  max_seq_len=max_seq_len,
                  batch_size=batch_size)
    D = DiscriminatorCNN(embedding_dim=D_embedding_dim,
                  vocab_size=voc_size,
                  class_num=emotion_num,
                  num_filters=D_num_filter,
                  window_sizes=D_kernel_sizes,
                  MODE="2*emo_num")

    if CUDA:
        real_data_samples = real_data_samples.cuda()
        real_data_samples_label=real_data_samples_label.cuda()
        E = E.cuda()
        G = G.cuda()
        D = D.cuda()


    for i in range(emotion_num):
        print("(1) top1:")
        G.generate_one_sentence_emo(E, Vocabulary, all_emotion_types, i, Generation_MODE)  # [0, 1, 2, .....emotion_num-1] is one-hot for emotions
        print("(2) random_sampling:")
        G.generate_one_sentence_emo(E, Vocabulary, all_emotion_types, i, "random_sampling")
    print(" ")

    gen_pretrain_opt = optim.Adam(G.parameters(), lr=pretrain_lr)
    emo_pretraon_opt = optim.Adam(E.parameters(), lr=pretrain_lr*0.1)
    print("Pre-training Generator..............")
    if pretrain:
        print('Starting Generator MLE Training...')

        for epoch in range(Pretrain_EPOCHS):
            sys.stdout.flush()

            # shuffle
            perm = torch.randperm(real_data_samples.size()[0])
            real_data_samples = real_data_samples[perm]
            real_data_samples_label = real_data_samples_label[perm]

            # train mode
            E.train()
            G.train() # !!!!!!! important
            total_loss = 0.0
            hidden = G.init_hidden_feed_z(batch_size)
            b=0
            for i in range(0, PreTraining_trn_num, batch_size):

                samples_i=real_data_samples[i:i + batch_size]# num_samples  x max_seq_len+1
                samples_i_emo = real_data_samples_label[i:i + batch_size]

                batch_sent_num, _ = samples_i.size()
                if batch_sent_num == batch_size:

                    inputs= samples_i[:, :max_seq_len ]# batch_size x max_seq_len

                    emotions= samples_i_emo# batch_size x max_seq_len

                    targets= samples_i[:, 1:max_seq_len+1]# batch_size x max_seq_len

                    inputs = Variable(inputs).type(torch.LongTensor).to(device)
                    targets = Variable(targets).type(torch.LongTensor).to(device)
                    emotions = Variable(emotions).type(torch.LongTensor).to(device)


                    hidden = repackage_hidden(hidden)
                    gen_pretrain_opt.zero_grad()
                    emo_pretraon_opt.zero_grad()


                    # Forward pass through encoder
                    emotion_context = E(emotions).to(device) # 1 x batch_size x emotion_hidden_dim
                    # Forward pass through decoder
                    loss, hidden = G.pretrain_MLE_Loss(inputs, targets, emotion_context, hidden)# include forward



                    loss.backward()
                    emo_pretraon_opt.step()
                    gen_pretrain_opt.step()

                    _ = torch.nn.utils.clip_grad_norm_(E.parameters(), 1)
                    _ = torch.nn.utils.clip_grad_norm_(G.parameters(), 1)



                    for p in G.parameters():
                        p.data.add_(-pretrain_lr, p.grad.data)
                    for p in E.parameters():
                        p.data.add_(-pretrain_lr, p.grad.data)


                    total_loss += loss.data.item()
                    b +=1


                    if i % 100 == 0 and i > 0:
                        cur_loss = total_loss / (b*max_seq_len)
                        print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                              ' cur_avg_loss {:5.2f} | current_perplexity {:8.2f}'.format(
                            epoch, i//batch_size, PreTraining_trn_num // batch_size, pretrain_lr,
                            cur_loss, math.exp(cur_loss)))
                        total_loss = 0
                        b=0

            # eval mode

            perplexity_trn = G.evaluate(E, real_data_samples, real_data_samples_label)
            print("Perplexity is : {}".format(perplexity_trn))
            print(" ")

            # generation
            print("#### Test #### (Generate random sequences after training)")
            for i in range(emotion_num):
                G.generate_one_sentence_emo(E, Vocabulary, all_emotion_types, i,Generation_MODE)  # [0, 1, 2, .....emotion_num-1] is one-hot for emotions
            print(" ")

            torch.save(E.state_dict(), SAVE_preMODEL_E)
            torch.save(G.state_dict(), SAVE_preMODEL_G)
    else:
        E.load_state_dict(torch.load(SAVE_preMODEL_E))
        G.load_state_dict(torch.load(SAVE_preMODEL_G))
        if CUDA:
            E = E.cuda()
            G = G.cuda()
        print("#### Test #### (Generate random sequences after loading model)")
        for i in range(emotion_num):
            print("(1) top1:")
            G.generate_one_sentence_emo(E, Vocabulary, all_emotion_types, i, Generation_MODE)  # [0, 1, 2, .....emotion_num-1] is one-hot for emotions
            print("(2) random_sampling:")
            G.generate_one_sentence_emo(E, Vocabulary, all_emotion_types, i, "random_sampling")
        print(" ")
    print("End of Pre-training\n")



    print("Adversarial Training.........")
    if Advs_TrainGAN:

        # criterion = nn.BCELoss()
        d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=sgd_momentum)  # for adversarial training
        g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)
        e_optimizer = optim.Adam(E.parameters(), lr=g_learning_rate*0.1)

        for epoch in range(Advtrain_EPOCHS):
            print("-----Adversarial Training (Epoch {})-----".format(epoch))

            print("   Train Discriminator......")
            D.train()
            E.eval()
            G.eval()
            # D = train_D(D, d_optimizer, real_data_samples, d_steps, G ) # fix G to generate g_x  ; need real data
            D.train_D(d_optimizer, real_data_samples, real_data_samples_label,  d_steps,
                      G, E , emotion_num, "2*emo_num",
                      TRAIN_D_ACC_MINIMUM[epoch]) # fix G to generate g_x  ; need real data

            print("   Train Generator......")
            D.eval()
            E.train()
            G.train()
            # G = train_G(G, g_optimizer, g_steps, D, real_data_samples, Vocabulary) # fix D, use D to get rewards
            E, G = train_G(G, E, emotion_num, e_optimizer, g_optimizer, g_steps,
                            D, real_data_samples,
                            Vocabulary, all_emotion_types,
                            rewards_thres[epoch])  # fix D, use D to get rewards


            print(" ")
            print(" ")
            print("After one step adversarial training:")
            # generation
            print("#### Test #### (Generate random sequences after training)")
            G.eval()
            E.eval()
            for i in range(emotion_num):
                print("(1) top1:")
                G.generate_one_sentence_emo(E, Vocabulary, all_emotion_types, i,Generation_MODE)  # [0, 1, 2, .....emotion_num-1] is one-hot for emotions
                print("(2) random_sampling:")
                G.generate_one_sentence_emo(E, Vocabulary, all_emotion_types, i, "random_sampling")
            print(" ")




train()


log_file.close()
sys.stdout = stdout_backup
