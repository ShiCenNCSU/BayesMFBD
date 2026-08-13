#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:16:05 2023

Core likelihood calculations for MTBD with random branch fitness effects
under a Brownian motion model

This was implmented on top of TreeLikeSiteEffects so we can still have multiple features impact fitnes

@author: David
"""
import numpy as np
import time
import sys
from ete3 import Tree
import TreeUtils
from VectorizedTree import VectorizedTree
from Bio import SeqIO
import pandas as pd
from pathlib import Path

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, init_to_value

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")
from jax import device_get, debug
import jax.numpy as jnp

class MTBD(object):
    
    def __init__(self,**params):
        
        # Initialize birth-death model parameters
        self.time_intervals = params['time_intervals']
        n = len(self.time_intervals)

        # Birth/transmission rate beta
        beta = params.get('beta', 1.0)
        try:
            self.beta = beta * np.ones(n)
        except ValueError:
            sys.exit("Initial params must be a scalar value or array with length = # of time intervals")
        
        # Feature-dependent fitness effects
        fit_effects = params.get('fit_effects',[1.0])
        self.n_features = len(fit_effects)
        try:
            #self.fit_effects = np.reshape(fit_effects,(self.n_features,1))
            self.fit_effects = np.reshape(fit_effects,(self.n_features))
        except ValueError:
            sys.exit("Initial fitnesss effects must be a scalar value or array")
        
        # Death/removal rate d
        d = params.get('d', 0.5)
        try:
            self.d = d * np.ones(n)
        except ValueError:
            sys.exit("Initial params must be a scalar value or array with length = # of time intervals")

        # Migration/transition rate gamma -- should be zero for models assuming known ancestral features
        gamma = params.get('gamma', 0.0)
        try:
            self.gamma = gamma * np.ones(n)
        except ValueError:
            sys.exit("Initial params must be a scalar value or array with length = # of time intervals")
        
        # Sampling fraction s
        s = params.get('s', 0.01)
        try:
            self.s = s * np.ones(n)
        except ValueError:
            sys.exit("Initial params must be a scalar value or array with length = # of time intervals")
  
        # Contemporaneous sampling at present rho (assumed to be a single value)"
        rho = params.get('rho', s)
        try:
            self.rho = rho * np.ones(n)
        except ValueError:
            sys.exit("Initial params must be a scalar value or array with length = # of time intervals")
            
        # Brownian motion variance scalar
        self.sigma = params.get('sigma', 0.01)
        self.branch_fit_effects = []
        
        # Global epistasis params
        self.ge_a = params.get('ge_a', 0.0)
        self.ge_b = params.get('ge_b', 0.0)

    def build(self,tree,features_dic):
    
        """
            Build model for a given input tree
            and distribute time-varying params among tree primitives
            
            Parameters: 
                params (dict): dictionary of estimated birth-death model params
                tree (ete3.Tree): input ete tree object
            
        """
        
        tree = TreeUtils.index_branches(tree) # only used for models with random branch effects
        tree, tree_times = TreeUtils.add_tree_times(tree)
        #child2parent = TreeUtils.get_branch_indexes(tree)
        parent_br_indexes = TreeUtils.get_parent_branch_indexes(tree)
        parent_dists = TreeUtils.get_parent_distances(tree)
        final_time = max(tree_times) # or present time
        self.time_intervals = final_time - self.time_intervals 
        
        # Convert ete tree into a vectorized data structure
        tree = VectorizedTree(tree,self.time_intervals,features_dic)
        
        """
            Pack everything in a dictionary for inference
        """
        tree_dict = {}
        
        # Gather edge-specific params from tree arrays
        tree_dict['edge_b'] = self.beta[tree.edge_arr['param_interval']]
        tree_dict['edge_gamma'] = self.gamma[tree.edge_arr['param_interval']]
        tree_dict['edge_d'] = self.d[tree.edge_arr['param_interval']]
        tree_dict['edge_s'] = self.s[tree.edge_arr['param_interval']]
        
        # Gather edge-specific times/intervals from tree array
        tree_dict['edge_time_step'] = tree.edge_arr['time_step']
        tree_dict['edge_back_time'] = tree.edge_arr['back_time']
        #tree_dict['edge_pE_interval'] = tree.edge_arr['pE_interval'] # only used for computing iterative pEs
        #back_times = tree.bkwd_interval_times
        #tree_dict['edge_pE_init_time'] = np.take(back_times, tree_dict['edge_pE_interval']) # only used for computing iterative pEs
        #tree_dict['pE_back_times'] = list(zip(back_times, np.arange(len(back_times)))) # only used for computing iterative pEs
        
        # Gather sample-specific parmas
        tree_dict['sample_s'] = self.s[tree.sample_arr['param_interval']]
        tree_dict['sample_d'] = self.d[tree.sample_arr['param_interval']]
        tree_dict['csa_rho'] = self.rho[tree.rho_arr['param_interval']]
        
        # Gather birth event-specific parmas
        tree_dict['birth_b'] = self.beta[tree.birth_arr['param_interval']]
        
        # Gather features for each edge/birth-event into 2D arrays
        tree_dict['edge_features'] = np.vstack(tree.edge_arr['features'])
        tree_dict['birth_features'] = np.vstack(tree.birth_arr['features'])
        
        # Gather time-intervals for each edge/birth-event
        tree_dict['edge_param_interval'] = tree.edge_arr['param_interval']
        tree_dict['birth_param_interval'] = tree.birth_arr['param_interval']
        
        # For random branch effects
        #tree_dict['child2parent'] = child2parent # dict mapping child idx to parent idx
        tree_dict['parent_br_indexes'] = np.array(parent_br_indexes) # parent idx for each child branch idx
        tree_dict['parent_dists'] = np.array(parent_dists) # parent idx for each child branch idx
        tree_dict['edge_branch_idx'] = tree.edge_arr['idx']
        tree_dict['birth_branch_idx'] = tree.birth_arr['idx']
        
        tree = tree_dict
        
        self.tree = tree_dict
        
        # Update beta for each lineage/birth-event according to feature-dependent fitness effects
        #tree = self._calc_fitness(tree,self.fit_effects)

        return tree
    
    def _calc_iterative_PEs(self, tree):
        
        t = tree
        pEs = []
        init_time = 0
        pE_init = jnp.ones(shape=len(t.edge_b)) * (1-self.rho[-1])
        
        """
            Iterate through time intervals backwards, starting with present
            and working back to time interval i = 0
        """
        for time, i in t.pE_back_times[::-1]:
            
            
            if time == 0:
                
                pE = pE_init
                
            else:
            
                """
                    TODO: generalize to allow all params to be lineage-dependent
                    Could be done by tiling constant params across a n_line x n_intervals array
                """
                b = self.beta[i]
                s = self.s[i]
                d = self.d[i]
                gamma = self.gamma[i]
                
                gbd_sum = gamma + b + d
                cnst_c = jnp.sqrt(jnp.square(gbd_sum) - 4 * d * (1 - s) * b)
                cnst_x = (-gbd_sum - cnst_c) / 2
                cnst_y = (-gbd_sum + cnst_c) / 2
                
                pE_num = (cnst_y + b * pE_init) * cnst_x * jnp.exp(-cnst_c * time) - cnst_y * (cnst_x + b * pE_init) * jnp.exp(-cnst_c * init_time)
                pE_denom = (cnst_y + b * pE_init) * jnp.exp(-cnst_c * time) - (cnst_x + b * pE_init) * jnp.exp(-cnst_c * init_time)
                pE = (-1 / b) * pE_num / pE_denom
                
            pEs.append(pE)
            pE_init = pE
            init_time = time
        
        all_pEs = jnp.transpose(jnp.stack(pEs[::-1])) # -1 reverse array
        
        return all_pEs

    def _calc_PEs(self,tree):
        
        """
            To simplify here we're not computing iterative pE's yet
        """
        #all_pEs = self._calc_iterative_PEs(tree)
        #new_pE_inits = jnp.take(all_pEs, t.edge_pE_interval) # this does not seem to work !!!
        new_pE_inits = jnp.ones(shape=len(tree['edge_b'])) * (1-self.rho[-1])
        
        t = tree
        time = t['edge_back_time'] # starting time is edge's event time (closest to present, in bkwds time)
        
        #init_time = t.edge_pE_init_time # time of nearest interval time at which we've computed pE values
        init_time = 0 # if not using iterative pEs
        
        gbd_sum = t['edge_gamma'] + t['edge_b'] + t['edge_d']
        cnst_c = jnp.sqrt(jnp.square(gbd_sum) - 4 * t['edge_d'] * (1 - t['edge_s']) * t['edge_b'])
        cnst_x = (-gbd_sum - cnst_c) / 2
        cnst_y = (-gbd_sum + cnst_c) / 2
        
        pE_num = (cnst_y + t['edge_b'] * new_pE_inits) * cnst_x * jnp.exp(-cnst_c * time) - cnst_y * (cnst_x + t['edge_b'] * new_pE_inits) * jnp.exp(-cnst_c * init_time)
        pE_denom = (cnst_y + t['edge_b'] * new_pE_inits) * jnp.exp(-cnst_c * time) - (cnst_x + t['edge_b'] * new_pE_inits) * jnp.exp(-cnst_c * init_time)
        
        """
            Lenora was using tf.safedivide which has no jax equivalent
            Could use:
                min_denom = 0.01
                pE_denom = jnp.where(pE_denom == 0.0, min_denom, pE_denom)
        """
        pEs = (-1 / t['edge_b']) * pE_num / pE_denom
        
        return cnst_x, cnst_y, cnst_c, pEs
    
    def _calc_fitness(self):
        
        """
            Compute fitness of each lineage based under global epistasis model
        """
        
        # Reset birth rates in tree
        self.tree['edge_b'] = self.tree['fixed_edge_b']
        self.tree['birth_b'] = self.tree['fixed_birth_b']
        
        fit_deviation =  self.branch_fit_effects[self.tree['edge_branch_idx']] - 1.0
        mut_fit_effects = self.ge_a + fit_deviation * self.ge_b
        mut_fit_effects = self.tree['edge_features'][:,0] * mut_fit_effects # assuming there is only one feature
        edge_fit_effects =  self.branch_fit_effects[self.tree['edge_branch_idx']] + mut_fit_effects
        
        fit_deviation =  self.branch_fit_effects[self.tree['birth_branch_idx']] - 1.0
        mut_fit_effects = self.ge_a + fit_deviation * self.ge_b
        mut_fit_effects = self.tree['birth_features'][:,0] * mut_fit_effects # assuming there is only one feature
        birth_fit_effects =  self.branch_fit_effects[self.tree['birth_branch_idx']] + mut_fit_effects
        
        self.tree['edge_b'] *= edge_fit_effects
        self.tree['birth_b'] *= birth_fit_effects
    
    def log_like(self,est_params):
        
        '''             
            Parameters: 
                params (dict): dictionary of estimated birth-death model params
                model ():  
                tree (VectorizedTree): vectorized tree to compute the log likelihood of
               
            Optional keyword arguments:
                blah (tuple): 
        '''

        # if estimating rand branch fit effects and ge_b
        #self.sigma = est_params['sigma']
        self.ge_a = est_params['ge_a']
        self.ge_b = est_params['ge_b']
        #self.branch_fit_effects = est_params['branch_fit_effects']
        
        # Update birth rates based on fitness effects
        self._calc_fitness()
        
        t = self.tree
        
        cnst_x, cnst_y, cnst_c, pEs = self._calc_PEs(t)
        
        # Calc prob density for all edges
        pD_denom = ((cnst_y + t['edge_b'] * pEs) * jnp.exp(-cnst_c * t['edge_time_step'])) - (cnst_x + t['edge_b'] * pEs)
        pD_intermed = (cnst_y - cnst_x) / pD_denom # see _calc_PEs for a safe division alternative
        pD = jnp.exp(-cnst_c * t['edge_time_step']) * jnp.square(pD_intermed)
        
        # Compute and sum log likelihood of all tree primitives
        line_like = jnp.sum(jnp.log(pD))
        sample_like = jnp.sum(jnp.log(t['sample_s'] * t['sample_d']))
        sample_like_csa = jnp.sum(jnp.log(t['csa_rho']))
        birth_like = jnp.sum(jnp.log(t['birth_b']))
        
        """
            Compute likelihood of random branch fitness effects 
            under a Brownian motion model of evolution
        """
        #Compute prob of fitness shifts between parent and child nodes under Brownian motion
        parent_fit_effects = self.branch_fit_effects[tree['parent_br_indexes']]
        fit_shifts = self.branch_fit_effects - parent_fit_effects
        epsilon = 0.005 # small value added to denom below to increase numerical stability when sigma * delta_t is << 1.0
        bm_probs = (-0.5 * fit_shifts**2) / (self.sigma * tree['parent_dists'] + epsilon) 
        bm_like = jnp.sum(bm_probs)

        loss = line_like + sample_like + sample_like_csa + birth_like + bm_like
        
        return loss
    
    def test_HMC(self, p_path, num_samples=5000):


        def HMC_model():
            
            ## Sample each parameter separately
            #sigma = numpyro.sample("sigma", dist.Uniform(0.0, 5.0)) # prior on Brownian motion variance scalar
            ge_a = numpyro.sample("ge_a", dist.Normal(0.0, 5.0)) # prior on slope of global epistasis
            ge_b = numpyro.sample("ge_b", dist.Normal(0.0, 5.0)) # prior on slope of global epistasis
            #n_br_effects = len(self.branch_fit_effects)
            #branch_fit_effects = numpyro.sample("branch_fit_effects", dist.Uniform(0.0, 2.0).expand((n_br_effects,)))

            # Pass the list of parameters to your likelihood function
            est_params = {'ge_a': ge_a,
                          'ge_b': ge_b}
            #est_params = {'branch_fit_effects': branch_fit_effects}
            likelihood_value = self.log_like(est_params)
    
            # Observe data with the calculated likelihood
            numpyro.factor("custom_likelihood_factor", likelihood_value)
    
    
        # Set up and run MCMC with NUTS
        initial_params = {"ge_a": jnp.array(0.0),
                          "ge_b": jnp.array(0.0)}
        #initial_params = {"branch_fit_effects": jnp.array(self.branch_fit_effects)}
    
        key = jax.random.PRNGKey(0)
        hmc_kernel = NUTS(HMC_model) # Important: needs to be function handle without calling using ()
        mcmc = MCMC(hmc_kernel, num_warmup=1000, num_samples=num_samples)
        mcmc.run(key, init_params=initial_params)
        mcmc.print_summary()
        # np.save("HMC.npy", samples.numpy())
    
        if not (p_path / "HMC_").exists():
            (p_path / "HMC_").mkdir(parents=True, exist_ok=True)
        
        #np.save((p_path / f"HMC_{prior}") / f"HMC_beta.npy", theta_traj)
    
        # Write posterior samples to csv
        samples = mcmc.get_samples()
        #sample_df = pd.DataFrame(samples['ge_b'])
        #sample_df.columns = ['ge_b'] # feature_names
        sample_df = pd.DataFrame(samples)
        sample_df.to_csv((p_path / "HMC_") / "HMC_samples.csv")
    
        # Write summary of posterior estimates to csv
        summ = summary(samples, group_by_chain=False)
        #df = pd.DataFrame(summ).T  # transpose so parameters are rows
        #est_df = pd.DataFrame(summ['ge_b'])
        #est_df.index = ['ge_b'] #feature_names
        est_df = pd.DataFrame(summ)
        est_df.to_csv((p_path / "HMC_") / "HMC_summary.csv")
    
if __name__ == '__main__':

    # Initialize birth-death model params
    beta = np.array([1.0,0.7,0.6])
    d = 0.5 # death rate
    gamma = 0.0 # no migration here
    s = 0.5 # sampling fraction upon removal
    rho = 0.5 # sampling fraction at present
    fit_effects = np.array(1*[1.0]) # np.random.uniform(0.9, 1.1, 10)
    sigma = 0.005 # scales variance in Brownian motion model (a small value epsilon gets added to this)
    
    # Global epistasis params
    ge_a = 0.
    ge_b = 0.
    
    # Set up time intervals in real time
    final_time = 20.0
    time_intervals = np.array([15.0, 10.0, 0.0]) # in distance from present

    params = {'beta': beta, 
              'fit_effects': fit_effects,
              'd': d, 
              'gamma': gamma, 
              's': s, 
              'rho': rho, 
              'sigma': sigma,
              'ge_a': ge_a,
              'ge_b': ge_b,
              'time_intervals': time_intervals}

    # Import tree
    #path = './test-sets/test-randomBranchEffects-sigma005-jan2026/'
    path = './test-sets/test-global-epi-ge_a=0.25_ge_b=-1.0_sigma005-jan2026/'
    sim_str = '009'
    tree_file = path + 'tree-' + sim_str + '.tre'
    input_tree = Tree(tree_file, format=1)
    
    fasta_file = path + 'tree-' + sim_str + '.fasta'
    features_dic = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        features_dic[record.id] = np.array(list(map(int,list(str(record.seq))))) # better way in numpy?

    # Initialize model and build model for a given tree
    model = MTBD(**params)
    tree = model.build(input_tree,features_dic)
    
    # Fix edge betas if not estimating 
    model.tree['fixed_edge_b'] = beta[tree['edge_param_interval']] # update base betas 
    model.tree['fixed_birth_b'] = beta[tree['birth_param_interval']] # update base betas 

    # Specify which parameters are to be estimated
    #est_params = beta
    n_branch_effects = len(tree['parent_br_indexes']) #max([*tree['child2parent']]) + 1
    branch_fit_effects = np.ones(n_branch_effects)
    branch_fit_effects = TreeUtils.set_rand_branch_effects(branch_fit_effects,input_tree, path + 'treeFitValues-' + sim_str + '.csv')
    model.branch_fit_effects = branch_fit_effects
    
    #est_params = {'branch_fit_effects': branch_fit_effects}
    #est_params = {'ge_b': ge_b}

    # Time it
    # runs = 1
    # tic = time.perf_counter()
    # for n in range(runs):
    #     like = model.log_like(est_params)
    #     print('jax like', like)    
    # toc = time.perf_counter()
    # elapsed = (toc - tic) / runs
    # print(f"Time per eval: {elapsed:0.4f} seconds")
    
    # Sample using HMC
    p_path = Path('./')
    num_samples = 2000
    model.test_HMC(p_path=p_path, num_samples=num_samples)
    
 