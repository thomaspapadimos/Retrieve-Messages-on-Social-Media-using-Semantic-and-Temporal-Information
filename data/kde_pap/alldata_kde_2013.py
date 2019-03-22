import config
import models
import metrics

import csv

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences
from keras.backend import clear_session

from text_preprocessing import create_vocab, load_embeddings_from_file, get_sequences
from batch_generator import batch_gen

import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy import stats
from KDEpy import FFTKDE
from matplotlib.pyplot import plot
from numpy import array, linspace

import matplotlib.pyplot as plt
from scipy import stats
import statistics
from scipy.stats import gaussian_kde
from pylab import plot


query_sequences = []
train_document_sequences = []
train_labels = []

train_submission_ids = []
train_extra_features = []

train_dirs = ['trec-2013']

for file in train_dirs:
        with open(file + '/a.seq') as f:
            query_sequences.extend(
                [[0] if line is '\n' else list(map(int, line.replace('\n', '').split(','))) for line in f]) 
    
        with open(file + '/id_timestamps.txt') as f:
            lines = [line.split(' ') for line in f]
            tweets_timestamp=np.array([int(line[2]) for line in lines])
            topic_id = np.array([int(line[0]) for line in lines])-110
        with open(file + '/timestamps_of_topics.txt') as f:
           lines = [line.split(' ') for line in f]
           querys_timestamps = np.array([int(line[1]) for line in lines])


q=[[] for i in range(max(topic_id))]

for t in range(len(query_sequences)):
    
    q[topic_id[t]-1].append(tweets_timestamp[t])
    

#q_sub = np.full((len(q[0])), querys_timestamps[0], dtype=int)  - q[0]
#
#   
#xmax= max(q_sub)
#xmin = min(q_sub)
#
#kde = gaussian_kde(q_sub)
#f = kde.covariance_factor()
#bw = f * q_sub.std()
#
#x_grid = np.linspace(xmin,xmax,len(q_sub))
#kde_eval = kde.evaluate(x_grid)
#plot(x_grid, kde.evaluate(x_grid))
#
#f= open("guru99.txt","w+")
#for i in range(len(kde_eval)):    
#    f.writelines(str(kde_eval[i]) + '\n')
#f.close()    

q_sub=[[] for i in range(max(topic_id))]
kde_eval=[[] for i in range(max(topic_id))]

for j in range(max(topic_id)):
    
    q_sub[j]=(np.full((len(q[j])), querys_timestamps[j+110], dtype=int)) - q[j]
    xmax = max(q_sub[j])
    xmin = min(q_sub[j])
    kde = gaussian_kde(q_sub[j])
    f = kde.covariance_factor()
  
    x_grid = np.linspace(xmin,xmax,len(q_sub[j]))
    kde_eval[j] = kde.evaluate(x_grid)*10**6


f= open("%skde.txt" % train_dirs,"w+")  
for i in range(max(topic_id)):
    for j in range(len(kde_eval[i])):
        
        f.writelines(str(kde_eval[i][j]) + '\n')
f.close()    


