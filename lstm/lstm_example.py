# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np


import pandas as pd

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

for t in range(len(test_labels)):
    
    ranks[topic_id[t]-1].append(float(np.exp(-rank_feature[t])))
    q[topic_id[t]-1].append(float(tweets_timestamp[t]))
    docno_class[topic_id[t]-1].append(docno[t])
    sim_class[topic_id[t]-1].append(sim[t])


q_sub=[[] for i in range(max(topic_id))]
kde_eval=[[] for i in range(max(topic_id))]
v=[]
w=[[] for i in range(max(topic_id))]
for j in range(max(topic_id)):
    
    
    q_sub[j]=(np.full((len(q[j])), querys_timestamps[j+n], dtype=int)) - q[j]
    q_sub[j]= (q_sub[j]/(3600*24))[:, np.newaxis]


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
dataset=pd.DataFrame(q_sub[1])
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=3, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions

#----------------------------------------------------------------------------------------------------------
