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
from scipy.stats import gaussian_kde
from pylab import plot

docum_time = open("id_timestamps.txt","r") 
lines = [line.split(' ') for line in docum_time]
docsTime=np.array([int(line[2]) for line in lines])


query_time = open("timestamps_of_topics.txt","r") 
lines = [line.split(' ') for line in query_time]
querys = np.array([int(line[1]) for line in lines])


data2=np.full((796), querys[0], dtype=int)
data3 = data2 - docsTime

sim = open("sim.txt","r") 
lines = [line.split('\n') for line in sim]
similarity = [int(line[0]) for line in lines]

xmax= max(data3)
xmin = min(data3)

kde = gaussian_kde(data3)
f = kde.covariance_factor()
bw = f * data3.std()

x_grid = np.linspace(xmin,xmax,796)
plot(x_grid, kde.evaluate(x_grid))