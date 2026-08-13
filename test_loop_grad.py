#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:29:58 2022

@author: david
"""
import jax
import numpy as np
import jax.numpy as jnp
import time

n_pops = 2

class Tree:
    """
        Simple class for storing reconciled trees along with their metadata used for post-processing
    """
    def __init__(self, n_pops=n_pops):
        
        """
            Maybe we can represent trees as a sequence of maps-products?
            At each event we:
                -We map, e.g.:
                    map = jnp.asarray([3, 2, 1, 0])
                    sprobs = sprobs[map]
                -Then multiply appropriate child probs (to reflect coal events)
                    
        
        """
        
        
        self.n_pops = n_pops
        
        sample_pop = 0
        init_probs = jnp.zeros(n_pops)
        self.sample_probs = init_probs.at[sample_pop].set(1.0)
        self.pi = jnp.array([0.75,0.25])

def first_finite_differences(f, x):
  eps = 1e-3
  indicators=jnp.zeros((n_pops,n_pops,n_pops))
  indicators = indicators.at[0,0,1].set(1.0)
  indicators = indicators.at[1,1,0].set(1.0)
  return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps)
                   for v in indicators])

def like(M,tr):
    
    """
        Compute likelihood of tree branch
    """
    
    sample_probs = tr.sample_probs
    pi = tr.pi
    
    Q = M - jnp.diag(jnp.sum(M,axis=1))
    Qdt = Q * 0.01
    
    # State prob update with jax.fori loop -- does not seem to speed things up substantially
    #body_fun = lambda i,x: x + jnp.matmul(Qdt,x)
    #sample_probs = jax.lax.fori_loop(0, 100, body_fun, sample_probs)
    
    # State prob update with python for loop
    for x in range(100):
        d_probs = jnp.matmul(Qdt,sample_probs)
        sample_probs = sample_probs + d_probs
    
    #print(sample_probs)
    
    return jnp.dot(sample_probs,pi)

d = np.zeros((n_pops, n_pops))
np.fill_diagonal(d, 0.1)
M = jnp.ones((n_pops,n_pops))*0.1 - d

tree = Tree(n_pops=n_pops)

L = like(M,tree)
print(L)

like_jit = jax.jit(like,static_argnums=1)
L = like_jit(M,tree)
print(L)

like_dx = jax.grad(like)
L_grads = like_dx(M,tree)
print(L_grads)

#like_dx_jit = jax.jit(like_dx)
#L_grads = like_dx_jit(M)
#print(L_grads)

#ffd_grads = first_finite_differences(like,M)
#print(ffd_grads)
   
# Time it
runs = 100
tic = time.perf_counter()
for n in range(runs):
    L = like(M,tree)
toc = time.perf_counter()
elapsed = (toc - tic) / runs
print(f"Time per eval: {elapsed:0.4f} seconds")

tic = time.perf_counter()
for n in range(runs):
    L = like_jit(M,tree)
toc = time.perf_counter()
elapsed = (toc - tic) / runs
print(f"Time per eval: {elapsed:0.8f} seconds")


# Test if we can compute gradients with respect to M using grad assuming known equilibrium freqs
# Then see if we can jit this function using LAX fori loop


