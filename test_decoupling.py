#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:53:39 2024

Test goodness of approximation decoupling updates of lineage state probs from
probability density of the tree evolving under the birth-death model.


@author: david
"""
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm

def approx(init_pE, init_pD, t, params):
    
    """
        Approximate MTBD probability densities
    """
    
    beta, nu, gamma, s = params
    
    # Define transition prob generator matrix Q
    q = init_pD.copy() # init state probs
    Q = gamma - np.diag(np.sum(gamma,axis=1)) # set diagonals to negative row sums
    
    q_traj = np.zeros((num_states,len(t)))
    q_traj[:,0] = q
    
    pD_traj = np.zeros((num_states,len(t)))
    pD_traj[:,0] = init_pD.copy()
    
    for i in range(len(t)-1):
        
        time = t[i] # starting time is edge's event time (closest to present, in bkwds time)
        time_step = t[i+1] - t[i]
        init_time = 0. # if not using iterative pEs
        
        """
            Compute pE for all states at this time all at once
        """
        
        #gbd_sum = np.sum(gamma,1) + beta + nu # I think this is wrong b/c we'd be accounting for migration twice
        gbd_sum = beta + nu # ignoring migration term
        cnst_c = np.sqrt(np.square(gbd_sum) - 4 * nu * (1 - s) * beta)
        cnst_x = (-gbd_sum - cnst_c) / 2
        cnst_y = (-gbd_sum + cnst_c) / 2
        
        pE_num = (cnst_y + beta * init_pE) * cnst_x * np.exp(-cnst_c * time) - cnst_y * (cnst_x + beta * init_pE) * np.exp(-cnst_c * init_time)
        pE_denom = (cnst_y + beta * init_pE) * np.exp(-cnst_c * time) - (cnst_x + beta * init_pE) * np.exp(-cnst_c * init_time)
        
        pE = (-1 / beta) * pE_num / pE_denom
        
        """
            Compute pD for all states at this time all at once
        """
        
        pD_denom = ((cnst_y + beta * pE) * np.exp(-cnst_c * time_step)) - (cnst_x + beta * pE)
        pD_intermed = (cnst_y - cnst_x) / pD_denom # see _calc_PEs for a safe division alternative
        pD = np.exp(-cnst_c * time_step) * np.square(pD_intermed)
        
        """
            Update and normalize state probs
        """
        trans_probs = expm(Q*time_step) # exponentiate time-scaled transition rate matrix to get transition probs
        q = np.matmul(q,trans_probs)
        q = q * pD / (np.sum(q * pD)) 
        q_traj[:,i+1] = q
        
        """
            Update cumulative pD probs
        """
        prev_pD = pD_traj[:,i]
        new_pD = (trans_probs @ prev_pD) * pD
        pD_traj[:,i+1] = new_pD
        
    return q_traj, pD_traj


def exact_deriv(p, t, beta, nu, gamma, s):
    
    pE = p[0:num_states]
    pD = p[num_states:2*num_states]
    
    """
        Compute pE derivs
    """
    
    # death without sampling
    d_pE_death_no_sampling = (1-s) * nu
    
    # no birth or migration
    d_pE_no_births = beta * pE
    
    # no migration
    d_pE_no_mig = np.sum(gamma,1) * pE

    # no deaths
    d_pE_no_deaths = nu * pE

    # birth of line in j -- both lineages produce no samples
    d_pE_birth_no_sample = beta * pE * pE

    # migration into i from j
    d_pE_mig = gamma @ pE # matrix vector multiplication

    d_pE = d_pE_death_no_sampling - d_pE_no_births - d_pE_no_mig - d_pE_no_deaths + d_pE_birth_no_sample +  d_pE_mig
    
    
    """
        Compute pD derivs
    """
    
    # no births
    d_pD_no_births = beta * pD
    
    # no migration
    d_pD_no_mig = np.sum(gamma,1) * pD

    # no deaths
    d_pD_no_deaths = nu * pD

    # birth of line in j -- line in j produces no samples
    d_pD_birth_no_sample = 2 * beta * pE * pD

    # migration into i from j
    d_pD_mig = gamma @ pD # matrix vector multiplication

    d_pD = -(d_pD_no_births + d_pD_no_mig + d_pD_no_deaths) + d_pD_birth_no_sample +  d_pD_mig

    d_p = np.hstack((d_pE,d_pD))

    return d_p

"Set birth-death params"
num_states = 4 

# If assuming no births between states
#beta = np.ones(num_states) * 0.25 # uniform birth rate vector
beta = np.array([0.25,0.4,0.2,0.1]) # non-uniform betas

nu = np.ones(num_states) * 0.1 # death rate vector

m = 0.01 # base migration rate
gamma = np.array([[0.,m,2*m,3*m],
                  [m,0.,m,m],
                  [2*m,m,0.,m],
                  [3*m,m,m,0.]])

s = np.ones(num_states) * 0.1 # sampling prob

# A grid of time points (in days)
t = np.linspace(0, 100, 51)

# Initial conditions for pE and pD
init_pE = np.ones(num_states)
init_pD = np.zeros(num_states)
init_pD[0] = 1.0 

"""
    Approximate prob densities piecewise using local decoupling
"""
params = (beta, nu, gamma, s)
approx_q_traj, approx_pD_traj = approx(init_pE, init_pD, t, params)

"""
    Numerically integrate exact MTBD prob densities
"""
init = np.hstack((init_pE,init_pD)) 
ret = spi.odeint(exact_deriv, init, t, args=(beta, nu, gamma, s))
exact_pE = ret.T[0:num_states]
exact_pD = ret.T[num_states:2*num_states]
exact_pD_sum = np.sum(exact_pD,0)

"""
    Plot pD prob densities backwards through time
"""
sns.set()
sns.set_context("talk")

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
#pD_traj = np.log(pD_traj)
ax.plot(t, approx_pD_traj[0], 'o-', mew=1, ms=8, mec='w', label="0")
ax.plot(t, approx_pD_traj[1], 'o-', mew=1, ms=8, mec='w', label="1")
ax.plot(t, approx_pD_traj[2], 'o-', mew=1, ms=8, mec='w', label="2")
ax.plot(t, approx_pD_traj[3], 'o-', mew=1, ms=8, mec='w', label="3")
ax.set_xlabel('Time')
ax.set_ylabel('Approx prob density')
ax.legend()
fig.tight_layout()
fig.savefig('mtbd_pD_approx.png', dpi=200)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
#pD = np.log(pD)
ax.plot(t, exact_pD[0], 'o-', mew=1, ms=8, mec='w', label="0")
ax.plot(t, exact_pD[1], 'o-', mew=1, ms=8, mec='w', label="1")
ax.plot(t, exact_pD[2], 'o-', mew=1, ms=8, mec='w', label="2")
ax.plot(t, exact_pD[3], 'o-', mew=1, ms=8, mec='w', label="3")
ax.set_xlabel('Time')
ax.set_ylabel('Exact prob density')
ax.legend()
fig.tight_layout()
fig.savefig('mtbd_pD_exact.png', dpi=200)

"""
    Plot lineage state probs (q_traj) backwards through time
"""
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(t, approx_q_traj[0], 'o-', mew=1, ms=8, mec='w', label="0")
ax.plot(t, approx_q_traj[1], 'o-', mew=1, ms=8, mec='w', label="1")
ax.plot(t, approx_q_traj[2], 'o-', mew=1, ms=8, mec='w', label="2")
ax.plot(t, approx_q_traj[3], 'o-', mew=1, ms=8, mec='w', label="3")
ax.set_xlabel('Time')
ax.set_ylabel('Approx state prob')
ax.legend()
fig.tight_layout()
fig.savefig('mtbd_state_probs_approx.png', dpi=200)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(t, exact_pD[0] / exact_pD_sum, 'o-', mew=1, ms=8, mec='w', label="0")
ax.plot(t, exact_pD[1] / exact_pD_sum, 'o-', mew=1, ms=8, mec='w', label="1")
ax.plot(t, exact_pD[2] / exact_pD_sum, 'o-', mew=1, ms=8, mec='w', label="2")
ax.plot(t, exact_pD[3] / exact_pD_sum, 'o-', mew=1, ms=8, mec='w', label="3")
ax.set_xlabel('Time')
ax.set_ylabel('Exact state prob')
ax.legend()
fig.tight_layout()
fig.savefig('mtbd_state_probs_exact.png', dpi=200)

