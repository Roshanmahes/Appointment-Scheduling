import numpy as np
from service import SCV_to_params, service_times, static_schedule
from os import path
import sys

def simulate_adaptive_S4(SCV, omega, n, B):
    
    B_start = np.zeros(n)
    B_end = np.zeros(n)
    t = [0] * n

    # first customer
    B_start[0] = t[0]
    B_end[0] = B_start[0] + B[0]

    for i in range(1,n):

        k = sum(B_end[:i] > t[i-1]) # clients waiting in system
        u = t[i-1] - B_start[i-k] # service time client in service
        
        t[i] = t[i-1] + static_schedule(1,SCV,omega,n-i,k-1,u)[0][0] # next arrival time
        
        # compute time service starts & ends
        B_start[i] = max(t[i], B_end[i-1])
        B_end[i] = B_start[i] + B[i]

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

    try: # number of runs
        N = int(sys.argv[4])
    except:
        N = 1000
    
    # determine file name
    experiment = 1
    file_name = f'S4_SCV_{SCV:.2f}_omega_{omega:.1f}_n_{n}_run_{experiment}.csv'

    while path.exists(file_name):
        experiment += 1
        file_name = f'S4_SCV_{SCV:.2f}_omega_{omega:.1f}_n_{n}_run_{experiment}.csv'

    with open(file_name, 'w') as f:
        
        header = [f'I{i}' for i in range(1,n+1)] + [f'W{i}' for i in range(1,n+1)] + ['cost']
        f.write(','.join(header) + '\n')
        
        # run experiment
        for i in range(N):

            B = service_times(n, SCV)
            I, W, cost = simulate_adaptive_S4(SCV, omega, n, B)
            print(f'Run {i+1}/{N}')
            
            line = ','.join([str(i) for i in I]) + ','.join([str(w) for w in W]) + ',' + str(cost)
            f.write(line + '\n')
