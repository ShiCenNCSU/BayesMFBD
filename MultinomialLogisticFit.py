#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:16:05 2023

@author: David
"""
import jax
import numpy as np
import jax.numpy as jnp
import time

import matplotlib.pyplot as plt
import seaborn as sns

"""
    MLR model of Bedford is fit by minimizing log loss of predicted variant vs. observed variant freqs
    BUT:
    We could imagine a model were we compute prob of observed sampling counts based on multinomial sampling probabilities
    
    How do we deal with variant introduction times?
        -Could fit variants individually so we don't need to estimate all fit values at once
        -This should be equivalent to estimating the fitness of a variant alongside the average fitness of other variants
        -But fit values would not be directly comparable
        -Alternatively we could estimate all at once but have to account for introduction times
"""

def plot_traj(times,traj):
    
    # Set up plot
    sns.set()
    sns.set_context("talk")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    num_vars = len(traj[:,0])
    for v in range(num_vars):
        ax.plot(times, traj[v,:], 'o-', mew=1, ms=8,
                mec='w', label=f'Variant {v}') #; beta={beta:.2f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Prevalence')
    ax.legend()
    fig.savefig('sim-traj.png', dpi=200)


def sim(f, init_counts, times):
    
    """
        Simulate variant dynamics based on their relative fitness
        Returns trajectories of both variant freqs and counts
        
        Question: should we just simulate deterministically so the only stochasticity is in the sampling of genotypes?
    """
    
    counts = init_counts
    N = np.sum(counts)
    freqs = counts / N
    
    num_vars = len(counts)
    num_times = len(times)
    freq_traj = np.zeros((num_vars,num_times))
    freq_traj[:,0] = freqs
    count_traj = np.zeros((num_vars,num_times))
    count_traj[:,0] = counts
    
    for i in range(1,len(times)):

        # We get expected freq at next time
        dt = times[i] - times[i-1]
        u = freqs * np.exp(f * dt) # was counts instead of freqs before
        exp_freqs = u / np.sum(u)
        freqs = exp_freqs
        
        # Sample new counts
        counts = np.random.multinomial(N, freqs)
        obsv_freqs =  counts / np.sum(counts)
        count_traj[:,i] = counts
        freq_traj[:,i] = obsv_freqs
    
    return count_traj, freq_traj   

def update(f, times, obsv, lr=0.01):
    
    return f - lr * jax.grad(loss_fn)(f, times, obsv)

@jax.jit
def update_jit(f, times, obsv, lr=0.01):
    
    return f - lr * jax.grad(loss_fn)(f, times, obsv)

def loss_fn_v1(f, init_counts, times, obsv):
    
    """
        Compute the likelihood (sum of squares) of observed data as the loss
    """
    
    counts = init_counts
    N = jnp.sum(counts)
    freqs = counts / N

    # Get expected freqs without any sampling noise
    loss = 0
    for i in range(1,len(times)):
        dt = times[i] - times[i-1]
        u = freqs * jnp.exp(f * dt)
        exp_freqs = u / jnp.sum(u)
        freqs = exp_freqs
        
        diffs = freqs - obsv[:,i]
        loss += jnp.sum(jnp.square(diffs))

    return loss

def loss_fn(f, times, obsv):
    
    """
        Compute the likelihood (sum of squares) of observed data as the loss
    """
    
    #counts = init_counts
    #N = jnp.sum(counts)
    #freqs = counts / N

    # Get expected freqs without any sampling noise
    loss = 0
    for i in range(1,len(times)):
        dt = times[i] - times[i-1]
        freqs = obsv[:,i-1] # curr freqs
        u = freqs * jnp.exp(f * dt)
        exp_freqs = u / jnp.sum(u)
        
        diffs = exp_freqs - obsv[:,i]
        loss += jnp.sum(jnp.square(diffs))

    return loss

if __name__ == '__main__':

    # Simulate with known fitness values
    f = np.array([0.1,0.05,0.01]) # fit vals
    #f = np.random.uniform(0.9, 1.1, 10)

    N = 1000
    init_freqs = np.ones(len(f)) * 1 / len(f)
    init_counts = np.random.multinomial(N, init_freqs)
    
    # Set up time steps for sims
    times = np.arange(0,11,1.0)
    
    # Sim data
    count_traj, freq_traj  = sim(f,init_counts,times)
    #plot_traj(times,freq_traj)

    # Get loss/likelihood
    obsv = freq_traj
    loss = loss_fn(f, times, obsv)

    # Time it
    runs = 100
    tic = time.perf_counter()
    for n in range(runs):
        loss = loss_fn(f, times, obsv)
        #print('jax like', jax_like)    
    toc = time.perf_counter()
    elapsed = (toc - tic) / runs
    print(f"Time per eval: {elapsed:0.5f} seconds")
    
    # Try training loop without jit-compiled update
    f = jnp.array([0.15,0.02,0.005])
    tic = time.perf_counter()
    #f = jnp.array(f)
    for episode in range(100):
        f = update(f, times, obsv, lr=0.01)
        print(str(episode), f)
    toc = time.perf_counter()
    elapsed = (toc - tic) / runs
    print(f"Time per eval no jit: {elapsed:0.4f} seconds")
    
    #Try training loop with jit-compiled update
    f = jnp.array([0.15,0.02,0.005])
    tic = time.perf_counter()
    for episode in range(100):
        f = update_jit(f, times, obsv, lr=0.01)
        print(str(episode), f)
    toc = time.perf_counter()
    elapsed = (toc - tic) / runs
    print(f"Time per eval with jit: {elapsed:0.4f} seconds")
    
    # Test gradients
    #f = jnp.array(f)
    #param_grad = jax.grad(loss_fn)(f, init_counts, times, obsv)
    #print('Likelihood gradients: ', param_grad)
    
    # Test jit version of loss_fn: seems to be slower in this case
    # log_like_jit = jax.jit(loss_fn) #,static_argnums=1)
    # runs = 100
    # tic = time.perf_counter()
    # for n in range(runs):
    #     jax_like = log_like_jit(f, init_counts, times, obsv)
    #     #print('jax like', jax_like)    
    # toc = time.perf_counter()
    # elapsed = (toc - tic) / runs
    # print(f"Time per eval: {elapsed:0.4f} seconds")
    
    
 