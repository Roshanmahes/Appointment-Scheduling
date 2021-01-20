import math
import numpy as np
import pandas as pd

from scipy.stats import poisson
from scipy.optimize import minimize, LinearConstraint # optimization
from scipy.linalg.blas import dgemm, dgemv # matrix multiplication
from scipy.linalg import inv # matrix inversion
from scipy.sparse.linalg import expm # matrix exponential

def dynamic_schedule(mean, SCV, omega, n, i, k, u):
    """
    Computes the table of optimal interarrival times tau_{i}(k,u).
    """
    
    Delta = 0.01
    m = int(u / Delta)

    # retrieve files
    file_tau = f'output/omega-{round(omega,1)}/tau-SCV-{round(float(SCV),2)}-omega-{round(omega,1)}-n-20-m-{m}'
    file_xi = f'output/omega-{round(omega,1)}/xi-SCV-{round(float(SCV),2)}-omega-{round(omega,1)}-n-20-m-{m}'
    file_tau = file_tau.replace('.', '_') + '.csv'
    file_xi = file_xi.replace('.', '_') + '.csv'

    # create schedule
    df = pd.read_csv(file_tau, index_col='i').fillna('') * mean

    df = df.iloc[1-n:,:]
    df = df.where(np.tril(np.ones(df.shape)).astype(np.bool)).dropna(axis=1, how='all')
    df.reset_index(level=0, inplace=True)
    df = df.astype(float).round(2).fillna('').astype(str)

    df.iloc[:,0] = [str(i) for i in range(1,n)]
    df.loc[n,:] = [r'\(i\) \(/\) \(k\)'] + [str(i) for i in range(1,n)]
    df.index = list(range(1,n)) + ['i / k']

    # get cost
    df_cost = pd.read_csv(file_xi, index_col='i').fillna('')
    cost = df_cost.iloc[20+i-1-n,k-1]
    
    return df, cost


def SCV_to_params(SCV, mean=1):
    """
    Applies the phase-type fit.
    """
    
    # weighted Erlang case
    if SCV <= 1:
        K = math.floor(1/SCV)
        p = ((K + 1) * SCV - math.sqrt((K + 1) * (1 - K * SCV))) / (SCV + 1)
        mu = (K + 1 - p) / mean
    
        return K, p, mu
    
    # hyperexponential case
    else:
        p = 0.5 * (1 + np.sqrt((SCV - 1) / (SCV + 1)))
        mu = 1 / mean
        mu1 = 2 * p * mu
        mu2 = 2 * (1 - p) * mu
        
        return p, mu1, mu2


def service_times(n, SCV, mean=1):
    """
    Generates service times according to 
    the fitted phase-type distribution.
    """
    
    B = [0] * n
    Unifs = np.random.uniform(size=n)
    
    # weighted Erlang case
    if SCV <= 1:
        K, p, mu = SCV_to_params(SCV, mean)
        
        for i in range(n):
            
            if Unifs[i] < p:
                B[i] = np.random.gamma(K, 1/mu)
            else:
                B[i] = np.random.gamma(K+1, 1/mu)
    
    # hyperexponential case
    else:
        p, mu1, mu2 = SCV_to_params(SCV, mean)
        
        for i in range(n):
            
            if Unifs[i] < p:
                B[i] = np.random.exponential(1 / mu1)
            else:
                B[i] = np.random.exponential(1 / mu2)
    
    return B


def find_Salpha(mean, SCV, u):
    """
    Returns the transition rate matrix, initial distribution
    and parameters of the phase-fitted service times given
    the mean, SCV, and the time that the client is in service at time 0.
    """
    
    # weighted Erlang case
    if SCV < 1:
        
        # parameters
        K = math.floor(1/SCV)
        p = ((K + 1) * SCV - math.sqrt((K + 1) * (1 - K * SCV))) / (SCV + 1)
        mu = (K + 1 - p) / mean
        
        # initial dist. client in service
        alpha_start = np.zeros((1,K+1))
        B_sf = poisson.cdf(K-1, mu*u) + (1 - p) * poisson.pmf(K,mu*u)
        for z in range(K+1):
            alpha_start[0,z] = poisson.pmf(z,mu*u) / B_sf
        alpha_start[0,K] *= (1 - p) 
        
        # initial dist. other clients
        alpha = np.zeros((1,K+1))
        alpha[0,0] = 1
        
        # transition rate matrix
        S = -mu * np.eye(K+1)
        
        for i in range(K-1):
            S[i,i+1] = mu
        
        S[K-1,K] = (1-p) * mu
            
    # hyperexponential case
    else:
        
        # parameters
        p = (1 + np.sqrt((SCV - 1) / (SCV + 1))) / 2
        mu1 = 2 * p / mean
        mu2 = 2 * (1 - p) / mean
        
        # initial dist. client in service
        alpha_start = np.zeros((1,2))
        B_sf = p * np.exp(-mu1 * u) + (1 - p) * np.exp(-mu2 * u)
        alpha_start[0,0] = p * np.exp(-mu1 * u) / B_sf
        alpha_start[0,1] = 1 - alpha_start[0,0]
        
        # initial dist. other clients
        alpha = np.zeros((1,2))
        alpha[0,0] = p
        alpha[0,1] = 1 - p
        
        # transition rate matrix
        S = np.zeros((2,2))
        S[0,0] = -mu1
        S[1,1] = -mu2
            
    return S, alpha_start, alpha


def create_Sn(S, alpha_start, alpha, N):
    """
    Creates the matrix Sn as given in Kuiper, Kemper, Mandjes, Sect. 3.2.
    """

    B = np.matrix(-sum(S.T)).T @ alpha
    m = S.shape[0]
    
    S_new = np.zeros(((N+1)*m, (N+1)*m))
    
    # compute S2
    S_new[0:m,0:m] = S
    S_new[m:2*m, m:2*m] = S
    S_new[0:m, m:2*m] = np.matrix(-sum(S.T)).T @ alpha_start
    
    # compute Si
    for i in range(1,N+1):
        S_new[i*m:((i+1)*m), i*m:(i+1)*m] = S
        S_new[(i-1)*m:i*m, i*m:(i+1)*m] = B
    
    return S_new


def Transient_EIEW(x, alpha_start, alpha, Sn, Sn_inv, omega, wis):
    """
    Evaluates the cost function given all parameters.
    In here, we used the FORTRAN dgem-functions 
    instead of @ for efficient matrix multiplication.
    """
    
    N = x.shape[0]
    m = alpha.shape[1]
    
    P_alpha_F = alpha_start
    cost = omega * np.sum(x)
    
    # cost of clients already entered (only waiting time)
    for i in range(1,wis+1):
        
        cost += (omega - 1) * np.sum(dgemm(1, P_alpha_F, Sn_inv[0:i*m,0:i*m]))
        
        F = 1 - np.sum(P_alpha_F)
        P_alpha_F = np.hstack((np.matrix(P_alpha_F), alpha * F))
    
    # cost of clients to be scheduled
    for i in range(wis+1,N+wis+1):
                
        exp_Si = expm(Sn[0:i*m,0:i*m] * x[i-wis-1])
        cost += float(dgemv(1, dgemm(1, P_alpha_F, Sn_inv[0:i*m,0:i*m]), np.sum(omega * np.eye(i*m) - exp_Si,1)))
        
        P = dgemm(1, P_alpha_F, exp_Si)
        F = 1 - np.sum(P)
        P_alpha_F = np.hstack((np.matrix(P), alpha * F))

    return cost


def Transient_IA(SCV, u, omega, N, x0=None, wis=0, tol=1e-4):
    """
    Computes the optimal schedule.
    wis = waiting in system.
    """
        
    # sojourn time distribution transition rate matrices
    S, alpha_start, alpha = find_Salpha(1, SCV, u)
    Sn = create_Sn(S, alpha_start, alpha, N)
    Sn_inv = inv(Sn)
    
    # minimization
    if not x0:
        x0 = np.array([1.5 + wis] + [1.5] * (N - wis - 1))
        
    Trans_EIEW = lambda x: Transient_EIEW(x, alpha_start, alpha, Sn, Sn_inv, omega, wis)
    lin_cons = LinearConstraint(np.eye(N - wis), 0, np.inf)
        
    optimization = minimize(Trans_EIEW, x0, constraints=lin_cons, method='SLSQP', tol=tol)
    x = optimization.x
    fval = optimization.fun
        
    return x, fval

def static_schedule(mean, SCV, omega, n, wis=0, u=0):
    
    N = n + wis
    tol = None if N < 15 else 1e-4
    u = u / mean

    # if not u and not wis:
        # N = N - 1
        # x, y = Transient_IA(SCV, u, omega, N, [], wis, tol)
        # x = np.pad(x, (1,0))
    # else:
    x, y = Transient_IA(SCV, u, omega, N, [], wis, tol)

    x = x * mean
    
    return x, y
