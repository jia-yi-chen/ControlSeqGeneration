#!/usr/bin/env python



import torch
from preprocess.preprocess_emotiondata import get_id2line,get_data
from torch.autograd import Variable


batch_size=32
max_seq_len = 20
start_word = 0

def sampling_real_data(corpus_name, num_samples, voc_size):
    """
    Y  : from [1, 2, 3 , .....num_emo]  do not have 0
    """
    sentences, emotion_list, emotion_dict,  v_list, vocb = get_id2line(corpus_name, voc_size)
    corpus, emotions, emotion_num, all_emotion_types = get_data(sentences, emotion_list, emotion_dict, v_list, vocb, max_seq_len)
    num_sentence=corpus.shape[0]
    if num_samples<num_sentence:
        print("Real data Corpus: {:3d}".format(num_samples))
        X = torch.zeros(num_samples, max_seq_len+1)
        Y_label = torch.zeros(num_samples)
        Y_input = torch.zeros(num_samples)
        randomlist=torch.LongTensor(num_samples).random_(0, num_sentence-1)
        for i in range(num_samples):
            X[i, :] =corpus[randomlist[i],:]
            Y_label[i] = emotions[randomlist[i]]
            Y_input[i] = emotions[randomlist[i]]-1
    else:
        print("Real data Corpus: {:3d}".format(num_sentence))
        X = torch.zeros(num_sentence, max_seq_len+1)
        Y_label = torch.zeros(num_sentence)
        Y_input = torch.zeros(num_sentence)
        randomlist=torch.LongTensor(num_sentence).random_(0, num_sentence-1)
        for i in range(num_sentence):
            X[i, :] =corpus[randomlist[i],:]
            Y_label[i] = emotions[randomlist[i]]
            Y_input[i] = emotions[randomlist[i]] - 1

    return X.type(torch.LongTensor), Y_input.type(torch.LongTensor), vocb, emotion_num , all_emotion_types# X is [num_sentence x (1+max_seq_len)] including [START]



def preparing_from_all_real_data(all_real_data, all_real_data_Y, num_sentence,device):
    X = torch.zeros(num_sentence, max_seq_len)
    Y = torch.zeros(num_sentence)
    num_all = all_real_data.shape[0]
    randomlist=torch.LongTensor(num_sentence).random_(0, num_all)# uniform distribution over [from, to - 1]
    for i in range(num_sentence):
        X[i, :] =all_real_data[randomlist[i],1:]
        Y[i] = all_real_data_Y[randomlist[i]]

    X = X.type(torch.LongTensor).to(device)
    Y = Y.type(torch.LongTensor).to(device)

    return X, Y  # X is [num_sentence x max_seq_len]




def preparing_new_training_samples_for_D(real_data_samples, real_data_samples_Y, num_realdata, G, E, num_fakedata, emotion_num, device):

    realX, labels1 = preparing_from_all_real_data(real_data_samples, real_data_samples_Y, num_realdata,device)
    fakeX, labels2 = G.preparing_samples_from_fixed_generator(num_fakedata, E, emotion_num)
    X = torch.cat((realX, fakeX), 0) # cat require both set are .cuda()
    Y = torch.cat((labels1, labels2))  # not LongTensor ??

    # shuffle
    perm = torch.randperm(Y.size()[0])
    Y = Y[perm]
    X = X[perm]

    # Turn to Variables for BP
    X = Variable(X).to(device)
    Y = Variable(Y).to(device)


    return X, Y

