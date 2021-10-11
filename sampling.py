#!/usr/bin/env python



import torch
from preprocess.preprocess_cornell import get_id2line,get_data
from torch.autograd import Variable


batch_size=32
voc_size = 15000
max_seq_len = 20
start_word = 0

def sampling_real_data(corpus_name, num_samples):
    sentences, v_list, vocb = get_id2line(corpus_name, voc_size)
    corpus = get_data(sentences, v_list, vocb, max_seq_len)
    num_sentence=corpus.shape[0]
    if num_samples<num_sentence:
        print("Real data Corpus: {:3d}".format(num_samples))
        X = torch.zeros(num_samples, max_seq_len+1)
        randomlist=torch.LongTensor(num_samples).random_(0, num_sentence-1)
        for i in range(num_samples):
            X[i, :] =corpus[randomlist[i],:]
    else:
        print("Real data Corpus: {:3d}".format(num_sentence))
        X = torch.zeros(num_sentence, max_seq_len+1)
        randomlist=torch.LongTensor(num_sentence).random_(0, num_sentence-1)
        for i in range(num_sentence):
            X[i, :] =corpus[randomlist[i],:]

    return X.type(torch.LongTensor), vocb # X is [num_sentence x (1+max_seq_len)] including [START]



def preparing_from_all_real_data(all_real_data, num_sentence,CUDA):
    X = torch.zeros(num_sentence, max_seq_len)
    Y = torch.ones(num_sentence)
    num_all = all_real_data.shape[0]
    randomlist=torch.LongTensor(num_sentence).random_(0, num_all-1)
    for i in range(num_sentence):
        X[i, :] =all_real_data[randomlist[i],1:]
    X=X.type(torch.LongTensor)
    if CUDA:
        X=X.cuda()
    return X, Y  # X is [num_sentence x max_seq_len]




def preparing_new_training_samples_for_D(real_data_samples, num_realdata, G, num_fakedata, CUDA):
    # 1) Generate num_fakedata samples from the fixed G
    # 2) Sampling num_realdata samples from realdataset (already get truncated)

    realX, labels1 = preparing_from_all_real_data(real_data_samples, num_realdata,CUDA)
    fakeX, labels2 = G.preparing_samples_from_fixed_generator(num_fakedata)
    X = torch.cat((realX, fakeX), 0) # cat require both set are .cuda()
    Y = torch.cat((labels1, labels2))  # not LongTensor ??

    # shuffle
    perm = torch.randperm(Y.size()[0])
    Y = Y[perm]
    X = X[perm]

    # Turn to Variables for BP
    X = Variable(X)
    Y = Variable(Y)

    if CUDA:
        X = X.cuda()
        Y = Y.cuda()

    return X, Y



#
# def preparing_new_training_samples_for_G(G, num_fakedata, CUDA):
#     # 1) Generate num_fakedata samples from the fixed G
#     # 2) Sampling num_realdata samples from realdataset (already get truncated)
#
#
#     # X, Y = G.preparing_samples_from_G_for_foolD(num_fakedata)
#     X = G.preparing_samples_from_G_for_foolD(num_fakedata)
#
#     # shuffle
#     perm = torch.randperm(X.size()[0])
#     # Y = Y[perm]
#     X = X[perm]
#
#
#
#     # return X, Y
#     return X