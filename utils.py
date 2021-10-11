
from keras.callbacks import LearningRateScheduler, Callback
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from random import shuffle

import numpy as np
import json
import h5py
import csv
import re

class save_model_weights(Callback):
    def __init__(self, textgenrnn, num_epochs, save_epochs):
        self.textgenrnn = textgenrnn
        self.weights_name = textgenrnn.config['name']
        self.num_epochs = num_epochs
        self.save_epochs = save_epochs

    def on_epoch_end(self, epoch, logs={}):
        if len(self.textgenrnn.model.inputs) > 1:
            self.textgenrnn.model = Model(inputs=self.model.input[0],
                                          outputs=self.model.output[1])
        if self.save_epochs > 0 and (epoch+1) % self.save_epochs == 0 and self.num_epochs != (epoch+1):
            print("Saving Model Weights — Epoch #{}".format(epoch+1))
            self.textgenrnn.model.save_weights(
                "{}_weights_epoch_{}.hdf5".format(self.weights_name, epoch+1))
        else:
            self.textgenrnn.model.save_weights(
                "{}_weights.hdf5".format(self.weights_name))