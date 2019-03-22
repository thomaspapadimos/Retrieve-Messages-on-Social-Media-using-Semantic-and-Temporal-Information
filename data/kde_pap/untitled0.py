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
import pandas as pd

query_sequences = []
train_document_sequences = []
train_labels = []

train_submission_ids = []
train_extra_features = []

train_dirs = ['trec-2011']

for file in train_dirs:
        with open(file + '/a.seq') as f:
            query_sequences.extend(
                [[0] if line is '\n' else list(map(int, line.replace('\n', '').split(','))) for line in f]) 
    
        with open(file + '/id_timestamps.txt') as f:
            lines = [line.split(' ') for line in f]
            tweets_timestamp=np.array([float(line[2]) for line in lines])
            topic_id = np.array([int(line[0]) for line in lines])
            
            
            docno = np.array([int(line[1]) for line in lines])
            
            
        with open(file + '/timestamps_of_topics.txt') as f:
           lines = [line.split(' ') for line in f]
           querys_timestamps = np.array([float(line[1]) for line in lines])


q=[[] for i in range(max(topic_id))]
docno_class = [[] for i in range(max(topic_id))]

for t in range(len(query_sequences)):
    
    q[topic_id[t]-1].append(float(tweets_timestamp[t]))
    docno_class[topic_id[t]-1].append(docno[t])

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
    
    q_sub[j]=(np.full((len(q[j])), querys_timestamps[j], dtype=int)) - q[j]
    xmax = max(q_sub[j])
    xmin = min(q_sub[j])
    kde = gaussian_kde(q_sub[j])
    f = kde.covariance_factor()
  
    x_grid = np.linspace(xmin,xmax,len(q_sub[j]))
    kde_eval[j] = kde.evaluate(x_grid)*10**6


#Read text tuples from trec_top_file of the form 
#     030  Q0  ZF08-175-870  0   4238   prise1 
#     qid iter   docno      rank  sim   run_id 
     
f= open("%skde.txt" % train_dirs,"w+")  
for i in range(max(topic_id)): 
    df = pd.DataFrame({'qid' : [i+1]*len(kde_eval[i]), 'docn' : docno_class[i], 'sim' : kde_eval[i]})
    df=df.sort_values(by=['sim'] , ascending=False)
    df=df.reset_index(drop=True)
    for j in range(len(kde_eval[i])):        
        f.writelines(str(df.qid[j]) + ' ' + 'Q0' + ' ' + str(df.docn[j]) + ' ' + str(df.index[j]+1) +  ' ' + str("%.6f" %df.sim[j]) +  ' ' + 'lucene4lm' + '\n')
f.close()    



#df = pd.DataFrame({'qid' : [0+51]*len(kde_eval[0]), 'docn' : docno_class[0], 'sim' : kde_eval[0]})
#df=df.sort_values(by=['sim'])
#df=df.reset_index(drop=True)