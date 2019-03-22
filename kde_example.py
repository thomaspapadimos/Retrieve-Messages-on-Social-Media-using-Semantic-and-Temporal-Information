import csv

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences
from keras.backend import clear_session


with open('id_timestamps.txt') as f:
    lines = f.read().splitlines()