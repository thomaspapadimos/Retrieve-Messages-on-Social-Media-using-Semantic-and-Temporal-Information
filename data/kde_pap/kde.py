#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:41:47 2018

@author: pap
"""
import numpy as np
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

tweets_time = open("id_timestamps.txt","r") 
lines = [line.split(' ') for line in tweets_time]
tweets=np.array([float(line[2]) for line in lines])
data=tweets[:, np.newaxis]

query_time = open("timestamps_of_topics.txt","r") 
lines = [line.split(' ') for line in query_time]
querys = np.array([int(line[1]) for line in lines])[:, np.newaxis]

sim = open("sim.txt","r") 
lines = [line.split('\n') for line in sim]
similarity = [int(line[0]) for line in lines]

xmax= max(data)
xmin = min(data)

h= ((4*(statistics.stdev(tweets))**5)/3*len(data))**(-1/5)
kde = KernelDensity(kernel='gaussian',bandwidth=7000).fit(data)
kernel = stats.gaussian_kde(tweets)

s = linspace(xmin,xmax,80)[:, np.newaxis]
e = np.exp(kde.score_samples(s))
plot(s, e)

