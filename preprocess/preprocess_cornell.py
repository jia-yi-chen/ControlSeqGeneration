#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import random
import os
import torch
import sys
import string
from nltk import word_tokenize
from nltk import FreqDist

import warnings
from collections import OrderedDict
from collections import defaultdict
from hashlib import md5
import json

import numpy as np
from six.moves import range
from six.moves import zip


''' 
    1. Read from 'movie-lines.txt'
    2. Create a dictionary with ( key = line_id, value = text )
'''



def get_id2line(path, voc_size):
    assert os.path.exists(path)
    with open(path, mode='r', encoding="utf-8") as f:
        for sentence in f:
            words = sentence.split()

    total_tokennum = 0
    sentences = []
    dictionary_idx = {}
    sequencelen_list = []
    all_tokens=[]
    lines = open(path,'r').read().split('\n')
    id2line = {}
    for i,line in enumerate(lines):
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
            sentence=_line[4]
            words=word_tokenize(sentence)
            if len(words)<6:# filter out two short sentences
                pass
            else:
                words=[word.lower() for word in words]# ['', '', ''] a sentence
                sentences.append(words)# [ [], ['i', 'love', 'you', '.'], [], [].........]
                sequencelen_list.append(len(words))
                total_tokennum += len(words)
                all_tokens.extend(words)
        print(i)
    fdist=FreqDist(all_tokens).most_common(voc_size-3)
    dictionary=[voc[0] for voc in fdist]
    vocabulary=['[START]']
    vocabulary.extend(dictionary)
    vocabulary.extend(['[PAD]','[UNK]'])
    V = len(vocabulary)


    dictionary_idx={}
    for i,token in enumerate(vocabulary):
        dictionary_idx[token]=i

    fileObject = open('vocabulary_saved.txt', 'w')
    for v in range(V):
        fileObject.write(vocabulary[v])
        fileObject.write(' ')
        fileObject.write(str(dictionary_idx[vocabulary[v]])) # index  begin from 1
        fileObject.write(' ')
        if vocabulary[v] not in ['[START]','[PAD]','[UNK]']:
            fileObject.write(str(fdist[v-1][1]))# frequency
        if v < V - 1:
            fileObject.write('\n')
    fileObject.close()

    return sentences,dictionary_idx, vocabulary

def get_data(sentences,dictionary_idx,vocabulary,max_len):

    hot_data=torch.zeros(len(sentences),1+max_len)
    for l, line in enumerate(sentences):
        print(l)
        for w in range(1+max_len):
            if w==0:
                hot_data[l, w] = dictionary_idx['[START]']  # START
            if w>0:

                if w > len(line):
                    hot_data[l, w] = dictionary_idx['[PAD]']
                else:
                    word = line[w - 1]
                    if word in vocabulary:
                        hot_data[l,w]=dictionary_idx[word]
                    else:
                        hot_data[l,w]=dictionary_idx['[UNK]']

    return hot_data


'''
    1. Read from 'movie_conversations.txt'
    2. Create a list of [list of line_id's]
'''


def get_conversations():
    conv_lines = open('movie_conversations.txt','r').read().split('\n')
    convs = []
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        convs.append(_line.split(','))
    return convs


'''
    1. Get each conversation
    2. Get each line from conversation
    3. Save each conversation to file
'''


def extract_conversations(convs, id2line, path=''):
    idx = 0
    for conv in convs:
        f_conv = open(path + str(idx) + '.txt', 'w')
        for line_id in conv:
            f_conv.write(id2line[line_id])
            f_conv.write('\n')
        f_conv.close()
        idx += 1


'''
    Get lists of all conversations as Questions and Answers
    1. [questions]
    2. [answers]
'''


def gather_dataset(convs, id2line):
    questions = [];
    answers = []

    for conv in convs:
        if len(conv) % 2 != 0:
            conv = conv[:-1]
        for i in range(len(conv)):
            if i % 2 == 0:
                questions.append(id2line[conv[i]])
            else:
                answers.append(id2line[conv[i]])

    return questions, answers


def prepare_seq2seq_files(questions, answers, path='', TESTSET_SIZE=30000):
    # open files
    train_enc = open(path + 'train.enc', 'w')
    train_dec = open(path + 'train.dec', 'w')
    test_enc = open(path + 'test.enc', 'w')
    test_dec = open(path + 'test.dec', 'w')

    # choose 30,000 (TESTSET_SIZE) items to put into testset
    test_ids = random.sample([i for i in range(len(questions))], TESTSET_SIZE)

    for i in range(len(questions)):
        if i in test_ids:
            test_enc.write(questions[i] + '\n')
            test_dec.write(answers[i] + '\n')
        else:
            train_enc.write(questions[i] + '\n')
            train_dec.write(answers[i] + '\n')
        if i % 10000 == 0:
            print("written %d lines' % (i)")

            # close files
    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()
