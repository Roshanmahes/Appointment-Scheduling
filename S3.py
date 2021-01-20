import numpy as np
from service import SCV_to_params, service_times, static_schedule
from os import path
import sys

def simulate_adaptive_S3(SCV, omega, n, B, Delta):
    
    B_start = np.array([np.inf] * n)
    B_end = [None] * n
    
    # first step
    B_start[0] = 0
    B_end[0] = B[0]
    t = np.cumsum(np.pad(static_schedule(1,SCV,omega,n-1,0,0)[0],(1,0)))
    
    # prepare second step
    time_old = 0
    time_new = Delta
    
    while True:
                
        i_range = np.where((time_old <= t) & (t < time_new) & (t > 0))[0]
        
        if len(i_range): # clients have arrived in interval [time_old,time_new]
        
            t_first = t[i_range]

            i_first = i_range[0]
            i_last = i_range[-1]
            
            for j in range(i_first,i_last+1):
                B_start[j] = max(t_first[j-i_first], B_end[j-1])
                B_end[j] = B_start[j] + B[j]
            
        else: # no new clients have arrived
            i_last = np.where(t < time_new)[0][-1]
        
        # policy
        if B_end[i_last] < time_new: # system is idle
            if n-i_last-2: # multiple clients to be scheduled
                t_last = time_new + np.cumsum(np.pad(static_schedule(1,SCV,omega,n-1-i_last-1,0,0)[0],(1,0)))
            else: # last client to be scheduled
                t_last = [time_new]

            t = np.concatenate((t[:(i_last+1)], t_last))
        
        else: # a client is in service, clients are waiting

            waiting = np.where(B_start[:(i_last+1)] > time_new)[0]

            k = len(waiting) + 1
            if k == 1: # no clients waiting
                u = time_new - B_start[i_last]
            else:      
                u = float(time_new - B_start[waiting[0] - 1])

            t_last = time_new + np.cumsum(static_schedule(1,SCV,omega,n-i_last-1,k-1,u)[0])
            t = np.concatenate((t[:(i_last+1)], t_last))

        # prepare next step
        time_old = time_new
        time_new += Delta
        
        if max(t) < time_new: # all clients are scheduled
            break
    
    # Lindley recursion
    I = [0] * n
    W = [0] * n

    for i in range(1,n):
        
        L = W[i-1] + B[i-1] - (t[i] - t[i-1])
        
        if L > 0:
            W[i] = L
        else:
            I[i] = -L

    cost = omega * sum(I) + (1 - omega) * sum(W)
    
    return I, W, cost


if __name__ == '__main__':

    SCV = float(sys.argv[1])
    omega = float(sys.argv[2])
    n = int(sys.argv[3])
    Delta = float(sys.argv[4])

    try: # number of runs
        N = int(sys.argv[5])
    except:
        N = 1000
    
    # determine file name
    experiment = 1
    file_name = f'S3_SCV_{SCV:.2f}_omega_{omega:.1f}_n_{n}_Delta_{Delta}_run_{experiment}.csv'

    while path.exists(file_name):
        experiment += 1
        file_name = f'S3_SCV_{SCV:.2f}_omega_{omega:.1f}_n_{n}_Delta_{Delta}_run_{experiment}.csv'

    with open(file_name, 'w') as f:
        
        header = [f'I{i}' for i in range(1,n+1)] + [f'W{i}' for i in range(1,n+1)] + ['cost']
        f.write(','.join(header) + '\n')
        
        # run experiment
        for i in range(N):

            B = service_times(n, SCV)
            I, W, cost = simulate_adaptive_S3(SCV, omega, n, B, Delta)
            print(f'Run {i+1}/{N}')
            
            line = ','.join([str(i) for i in I]) + ','.join([str(w) for w in W]) + ',' + str(cost)
            f.write(line + '\n')
