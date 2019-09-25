import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import statistics
import pandas as pd
from scipy.stats import entropy

query_sequences = []
train_document_sequences = []
train_labels = []

train_submission_ids = []
train_extra_features = []

train_dirs = ['trec-2011']

for file in train_dirs:
    if file == 'trec-2011':
        n = 0
    elif file == 'trec-2012':
        n = 50
    elif file == 'trec-2013':
        n = 110
    else:
        n = 170
    with open(file + '/id.txt') as f:
        test_submission_ids = [line.replace('\n', '').split(' ') for line in f]
        rank_feature = np.array([float(line[3]) for line in test_submission_ids])
    with open(file + '/id_timestamps.txt') as f:
        lines = [line.split(' ') for line in f]
        tweets_timestamp=np.array([float(line[2]) for line in lines])
        topic_id = np.array([int(line[0]) for line in lines]) - n
        docno = np.array([int(line[1]) for line in lines])
    with open(file + '/timestamps_of_topics.txt') as f:
        lines = [line.split(' ') for line in f]
        querys_timestamps = np.array([float(line[1]) for line in lines])
    with open(file + '/sim.txt') as f:
        test_labels = [[line.replace('\n', '')] for line in f]


sim = np.array(test_labels)
sim = sim.astype(np.int)
q=[[] for i in range(max(topic_id))]
ranks=[[] for i in range(max(topic_id))]
docno_class = [[] for i in range(max(topic_id))]
sim_class = [[] for i in range(max(topic_id))]
relevance = [[] for i in range(max(topic_id))]

for t in range(len(test_labels)):

    ranks[topic_id[t]-1].append(float(rank_feature[t]))
    q[topic_id[t]-1].append(float(tweets_timestamp[t]))
    docno_class[topic_id[t]-1].append(docno[t])
    sim_class[topic_id[t]-1].append(sim[t])
    relevance[topic_id[t]-1].append(float(sim[t]))
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

q_sub = [[] for i in range(max(topic_id))]
kde_eval = [[] for i in range(max(topic_id))]
entrop = [[] for i in range(max(topic_id))]
v = []
w = [[] for i in range(max(topic_id))]
for j in range(max(topic_id)):
    q_sub[j]=(np.full((len(q[j])), querys_timestamps[j+n], dtype=int)) - q[j]
    q_sub[j]= (q_sub[j]/(3600*24))[:, np.newaxis]
    xmax = max(q_sub[j])
    xmin = min(q_sub[j])
   # w[j] = relevance[j] / (np.full((len(q[j])), sum(relevance[j]), dtype=float))
    w[j] = relevance[j]
    #w[j] = ranks[j]

#    kde = gaussian_kde(q_sub[j])
#    f = kde.covariance_factor()
    x_grid = np.linspace(xmin,xmax,len(q_sub[j]))
#    kde_eval[j] = kde.evaluate(x_grid)

    for x in q_sub[j]:
        for y in x:
            v.append(y)

    h= (4*(statistics.stdev(v)**5)/3*len(q_sub[j]))**(-1/5)
    #kde = KernelDensity(kernel='gaussian',bandwidth=h ).fit(q_sub[j], w[j])
    kde = KernelDensity(kernel='gaussian').fit(q_sub[j], w[j])
    #kde = KernelDensity(kernel='gaussian',bandwidth=h).fit(q_sub[j])

    log_dens = kde.score_samples(x_grid)
    kde_eval[j] = np.exp(log_dens)
    entrop[j]=(np.full((len(q[j])), entropy(kde_eval[j]), dtype=float))


    fig = plt.figure()
    plt.plot(x_grid, kde_eval[j])
    plt.plot(q_sub[j], sim_class[j], 'ro')
    plt.savefig('images/%s_kde_plot_%d.png' %(file,(j+1+n)))
#    plt.show()
    plt.close(fig)


#Read text tuples from trec_top_file of the form
#     030  Q0  ZF08-175-870  0   4238   prise1
#     qid iter   docno      rank  sim   run_id

f= open("%skde.txt" % train_dirs,"w+")
for i in range(max(topic_id)):
    df = pd.DataFrame({'qid' : [i+1+n]*len(kde_eval[i]), 'docn' : docno_class[i], 'sim' : kde_eval[i]})
    #df=df.sort_values(by=['sim'] , ascending=False)
    #df=df.reset_index(drop=True)
    for j in range(len(kde_eval[i])):
        f.writelines(str(df.qid[j]) + ' ' + 'Q0' + ' ' + str(df.docn[j]) + ' ' + str(df.index[j]+1) +  ' ' + str("%f" %df.sim[j]) +  ' ' + 'lucene4lm' + '\n')
f.close()

f= open("%sEntropy.txt" % train_dirs,"w+")
for i in range(max(topic_id)):
    df = pd.DataFrame({'qid' : [i+1+n]*len(kde_eval[i]), 'docn' : docno_class[i], 'entropy' : entrop[i]})
    #df=df.sort_values(by=['sim'] , ascending=False)
    #df=df.reset_index(drop=True)
    for j in range(len(kde_eval[i])):
        f.writelines(str(df.qid[j]) + ' ' + 'Q0' + ' ' + str(df.docn[j]) + ' ' + str(df.index[j]+1) +  ' ' + str("%f" %df.entropy[j]) +  ' ' + 'lucene4lm' + '\n')
f.close()


#df = pd.DataFrame({'qid' : [0+51]*len(kde_eval[0]), 'docn' : docno_class[0], 'sim' : kde_eval[0]})
#df=df.sort_values(by=['sim'])
#df=df.reset_index(drop=True)