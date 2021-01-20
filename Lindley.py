'''
Created on 13 dec. 2018

@author: marko
'''

import time

from numpy import append
from numpy import cumsum
from numpy import mean
from numpy import minimum
from numpy import split
from numpy import zeros
from numpy import cov
from numpy.ma.core import arange, sqrt, reshape, var, where
from scipy import stats
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt

def simulateLindleyEfficient(lam, mu, n):
    arrDist = stats.expon(scale=1/lam)   # note that this is the MEAN!!!!
    servDist = stats.expon(scale=1/mu)   # note that this is the MEAN!!!!
    a = arrDist.rvs(n - 1) # interarrival times
    b = servDist.rvs(n - 1) # service times
    d = append([0], b - a)
    cumd = cumsum(d)
    w = cumd - minimum.accumulate(cumd)  # running minimum
    return w

lam = 0.8
mu = 1.0
n = 10000000
t1 = time.time()
w = simulateLindleyEfficient(lam, mu, n) 
t2 = time.time()
print("Simulation time: %f seconds" % (t2 - t1))
print(mean(w))

# Transient behaviour
n = 250
nrRuns = 1000

t1 = time.time()
sumW = zeros(n)

for _ in range(nrRuns) :
    sim = simulateLindleyEfficient(lam, mu, n)
    sumW += sim
meanW = sumW / nrRuns
t2 = time.time()
print("Simulation time: %f seconds" % (t2 - t1))

# theoretical steady-state mean waiting time
EW = lam/(mu*(mu-lam))

plt.figure()
plt.plot(arange(1,n+1), meanW, 'b')
plt.hlines(xmin=0, xmax=n, y=EW, color='red')



# 1. Regular Confidence interval for the mean waiting time
# M is the number of runs, 
# N is the number of customers per run, 
# k is the length of the warm-up interval (in this case, the number of customers to disregard)
def ciMultipleRuns(M, N, k):
    sumW = 0
    sumW2 = 0
    for _ in range(M) :
        sim = simulateLindleyEfficient(lam, mu, N)
        meanWrun = mean(sim[k:N])
        sumW += meanWrun
        sumW2 += meanWrun**2
    meanW = sumW / M
    varW = (sumW2 - sumW**2/M)/(M - 1)
    ci = meanW - 1.96*sqrt(varW/M), meanW + 1.96*sqrt(varW/M)
    return ci


print('Regular confidence intervals with warm-up period:')
t1 = time.time()
ci1 = ciMultipleRuns(100, 100000, 200)
t2 = time.time()
print("Simulation time: %f seconds" % (t2 - t1))
print(ci1)
print(mean(ci1))
t1 = time.time()
ci2 = ciMultipleRuns(10000, 1000, 200)
t2 = time.time()
print("Simulation time: %f seconds" % (t2 - t1))
print(ci2)
print(mean(ci2))

# The same function as above, but now using parallel computing. Runs are executed in parallel
# on multiple processor cores, speeding up the simulation!
#
# Regular Confidence interval for the mean waiting time
# M is the number of runs, 
# N is the number of customers per run, 
# k is the length of the warm-up interval (in this case, the number of customers to disregard)
def ciMultipleRunsParallel(M, N, k):
    sumW = 0
    sumW2 = 0
    
    numCores = multiprocessing.cpu_count()
     
    resultsAllWs = Parallel(n_jobs = numCores)(delayed(simulateLindleyEfficient)(lam, mu, N) for _ in range(M))
    # compute the mean and variance
    for i in range(M) :
        sim = resultsAllWs[i]
        meanWrun = mean(sim[k:N])
        sumW += meanWrun
        sumW2 += meanWrun**2
    meanW = sumW / M
    varW = (sumW2 - sumW**2/M)/(M - 1)
    ci = meanW - 1.96*sqrt(varW/M), meanW + 1.96*sqrt(varW/M)
    return ci



print('PARALLEL Regular confidence intervals with warm-up period:')
t1 = time.time()
ci1 = ciMultipleRunsParallel(100, 100000, 200)
t2 = time.time()
print("Simulation time: %f seconds" % (t2 - t1))
print(ci1)
print(mean(ci1))
t1 = time.time()
ci2 = ciMultipleRunsParallel(10000, 1000, 200)
t2 = time.time()
print("Simulation time: %f seconds" % (t2 - t1))
print(ci2)
print(mean(ci2))

# Note that it is much more efficient to do fewer, but longer runs
# For this reason, the method below is much faster.


# 2. Batch means

def ciBatchMeans(M, N, k):
    sim = simulateLindleyEfficient(lam, mu, M * N + k)
    
    # throw away the first k observations, and divide the rest into 
    # subruns of length N each
    run = sim[k:(M * N + k)]
    p = reshape(run, (M, N))
    sample = mean(p, axis=0)  # take row means
    meanW = mean(sample)
    varW = var(sample)
    ci = meanW - 1.96*sqrt(varW/M), meanW + 1.96*sqrt(varW/M)
    return ci


print('Batch Means:')
t1 = time.time()
ci3 = ciBatchMeans(100, 10000, 200)
t2 = time.time()
print("Simulation time: %f seconds" % (t2 - t1))
print(ci3)
print(mean(ci3))



# 3. Regenerative method
def ciRegenerative(N): 
    sim = simulateLindleyEfficient(lam, mu, N)
    # Now we're going to split the simulation results vector every time we encounter
    # an ampty system (i.e. a waiting time of zero)
    idx = where(sim == 0)[0]     # the positions of the zeros
    sa = split(sim, idx)         # split the list into sub-lists
    Yi = [sum(x) for x in sa]    # the sum of the waiting times in each sub-list
    Ni = [len(x) for x in sa]    # the number of waiting times in each sub-list
    M = len(Yi)                  # the number of sub-lists
    Yavg = mean(Yi)              # The average of the sums of the waiting times
    Navg = mean(Ni)              # the mean number of waiting times of the sub-lists
    Wavg = Yavg / Navg           # The overall mean waiting time
         
    cv = cov(Yi, Ni)[0,1]  # sample covariance is element at (0, 1) or (1, 0) of the covariance matrix
    sV2 = var(Yi) + Wavg**2 * var(Ni) - 2 * Wavg * cv
    print(sV2)
    ci = Wavg - 1.96 * sqrt(sV2/M)/Navg, Wavg + 1.96 * sqrt(sV2/M)/Navg
    return(ci)

print('Regenerative:')
t1 = time.time()
ci4 = ciRegenerative(100*10000)
t2 = time.time()
print("Simulation time: %f seconds" % (t2 - t1))
print(ci4)
print(mean(ci4))

plt.show()




