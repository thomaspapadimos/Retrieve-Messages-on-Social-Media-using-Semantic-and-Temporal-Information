DATA_PATH = '/home/pap/temporal_rnn/data/'



import tokenize
import keras
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences
from keras.backend import clear_session

import time
import os

import preprocessor as p
import numpy as np

from nltk.tokenize import TweetTokenizer
from keras.preprocessing.text import Tokenizer
from gensim.models.keyedvectors import KeyedVectors


texts = []
for file in ['trec-2011', 'trec-2012', 'trec-2013', 'trec-2014']:
        with open(DATA_PATH + file + '/a.toks') as f:
            texts.extend([line for line in f])
        with open(DATA_PATH + file + '/b.toks') as f:
            texts.extend([line for line in f])
print(texts[39781])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
vocab = tokenizer.word_index
print('Found %s unique tokens.' % len(vocab))
print(vocab['bbc'])