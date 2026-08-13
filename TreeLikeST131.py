#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 4 10:26:05 2026

Core likelihood calculations for MTBD model for E. coli ST131

Allows for feature-specific birth effects, feature-specific sampling effects, time-varying sampling or birth rates
and random branch fitness effects under a Brownian motion model 

This was implmented on top of TreeLikeSiteEffects and TreeLikeRandBranchEffects

@author: David
"""
import jax
import numpy as np
import pandas as pd
import jax.numpy as jnp
import time
import sys
from ete3 import Tree
import TreeUtils
from VectorizedTree import VectorizedTree
from Bio import SeqIO

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
        self.parent_fit_effects = []

    def build(self,tree,features_dic,sampling_dic=None):
    
        """
            Build model for a given input tree
            and distribute time-varying params among tree primitives
            
            Parameters: 
                params (dict): dictionary of estimated birth-death model params
                tree (ete3.Tree): input ete tree object
                
            NOTE: time intervals passed to VectorTree should be in past-to-present order
                in units of time (distance) from root.
            
        """
        
        tree = TreeUtils.index_branches(tree) # only used for models with random branch effects
        tree, tree_times = TreeUtils.add_tree_times(tree)
        #child2parent = TreeUtils.get_branch_indexes(tree)
        parent_br_indexes = TreeUtils.get_parent_branch_indexes(tree)
        parent_dists = TreeUtils.get_parent_distances(tree)
        
        # If time_intervals are given in distance from present (final time)
        #final_time = max(tree_times) # or present time
        #self.time_intervals = final_time - self.time_intervals 
        
        # Convert ete tree into a vectorized data structure
        tree = VectorizedTree(tree,self.time_intervals,features_dic,sampling_dict=sampling_dic)
        
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
        
        # Gather sampling features for each edge/sampling-event into 2D arrays
        if sampling_dic:
            tree_dict['edge_sampling_features'] = np.vstack(tree.edge_arr['sampling_features'])
            tree_dict['sample_sampling_features'] = np.vstack(tree.sample_arr['sampling_features'])
        
        # Gather time-intervals for each edge/birth-event
        tree_dict['sample_param_interval'] = tree.sample_arr['param_interval'] # added to allow time-varying sampling rates
        tree_dict['edge_param_interval'] = tree.edge_arr['param_interval']
        tree_dict['birth_param_interval'] = tree.birth_arr['param_interval']
        
        # For random branch effects
        #tree_dict['child2parent'] = child2parent # dict mapping child idx to parent idx
        tree_dict['parent_br_indexes'] = np.array(parent_br_indexes) # parent idx for each child branch idx
        tree_dict['parent_dists'] = np.array(parent_dists) # parent idx for each child branch idx
        tree_dict['edge_branch_idx'] = tree.edge_arr['idx']
        tree_dict['birth_branch_idx'] = tree.birth_arr['idx']
        
        tree = tree_dict
        
        # Update beta for each lineage/birth-event according to feature-dependent fitness effects
        #tree = self._calc_fitness(tree,self.fit_effects)

        return tree
    
    def _calc_fitness(self, tree, site_fit_effects):
        
        """
            Compute fitness of each lineage based on multiplicative fitness effects of their features
            Assumes beta is proportional to fitness
        """

        log_fit_effects = jnp.log(site_fit_effects) # log transform fitness effects so we "multiply" when we sum below
        edge_fit_effects = jnp.exp(jnp.matmul(tree['edge_features'],log_fit_effects)) # convert back to linear scale
        birth_fit_effects = jnp.exp(jnp.matmul(tree['birth_features'],log_fit_effects))
        
        """
            Multiply edge/birth site fit effects by appropriate branch fitness effects
            Now done in lieklihood function
        """
        #edge_fit_effects *= branch_fit_effects[tree['edge_branch_idx']]
        #birth_fit_effects *= branch_fit_effects[tree['birth_branch_idx']] 
        
        tree['edge_b'] *= edge_fit_effects
        tree['birth_b'] *= birth_fit_effects
        
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
    
    def log_like(self,est_params,tree):
        
        '''             
            Parameters: 
                params (dict): dictionary of estimated birth-death model params
                model ():  
                tree (VectorizedTree): vectorized tree to compute the log likelihood of
               
            Optional keyword arguments:
                blah (tuple): 
        '''

        if 'beta' in est_params:
            beta = est_params['beta']
            """
                Updating these edge/birth b values seems to give the tracer array conversion error
                But not if we're estimating beta???
            """
            # Set edge and birth event birth rates based on current values
            tree['edge_b'] = beta[tree['edge_param_interval']] # update base betas 
            tree['birth_b'] = beta[tree['birth_param_interval']] # update base betas
        else:
            beta = self.beta # necessary to reset this if not estimating?
            
        if 's' in est_params:
            s = est_params['s']
            tree['edge_s'] = s[tree['edge_param_interval']] # update edge sampling fractions 
            tree['sample_s'] =  s[tree['sample_param_interval']]
        else:
            s = self.s # necessary to reset this if not estimating?
        
        if 'fit_effects' in est_params:
            fit_effects = est_params['fit_effects']
        else:
            fit_effects = self.fit_effects
            
        if 'sampling_effects' in est_params:
            s_effects = est_params['sampling_effects']
            log_s_effects = jnp.log(s_effects) # log transform sampling effects so we "multiply" when we sum below
            edge_s_effects = jnp.exp(jnp.matmul(tree['edge_sampling_features'],log_s_effects)) # convert back to linear scale
            sample_s_effects = jnp.exp(jnp.matmul(tree['sample_sampling_features'],log_s_effects))
            tree['edge_s'] *= edge_s_effects
            tree['sample_s'] *= sample_s_effects    
            
        """
            This causes problems if we're not estimating branch_fit_effects
        """
        if 'branch_fit_effects' in est_params:
            branch_fit_effects = est_params['branch_fit_effects']
            # Rescale edge and birth event birth rates based on branch fit effects
            tree['edge_b'] *= branch_fit_effects[tree['edge_branch_idx']]
            tree['birth_b'] *= branch_fit_effects[tree['birth_branch_idx']]
            parent_fit_effects = branch_fit_effects[tree['parent_br_indexes']]
        else:
            branch_fit_effects = self.branch_fit_effects
            #tree['edge_b'] *= branch_fit_effects[tree['edge_branch_idx']]
            #tree['birth_b'] *= branch_fit_effects[tree['birth_branch_idx']]
            parent_fit_effects = self.parent_fit_effects
        
        # Update birth rates based on fitness effects
        tree = self._calc_fitness(tree, fit_effects)
        
        t = tree
        
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
        #parent_fit_effects = branch_fit_effects[tree['parent_br_indexes']]
        fit_shifts = branch_fit_effects - parent_fit_effects
        epsilon = 0.005 # small value added to denom below to increase numerical stability when sigma * delta_t is << 1.0
        bm_probs = (-0.5 * fit_shifts**2) / (self.sigma * tree['parent_dists'] + epsilon) 
        bm_like = jnp.sum(bm_probs)

        loss = -(line_like + sample_like + sample_like_csa + birth_like + bm_like)
        
        return loss

def update(est_params, tree, lr=0.01):
    
    #return est_params - lr * jax.grad(model.log_like)(est_params, tree)
    
    loss, grad = jax.value_and_grad(model.log_like)(est_params, tree)
    new_params = est_params - lr * grad
    return loss, new_params


@jax.jit
def update_jit(est_params, tree, lr=0.01):

    """
        Remember we can also use pytree tree_map function to apply
        gradients over different params in a dict
    """
    
    loss, grad = jax.value_and_grad(model.log_like)(est_params, tree)
    new_params = {}
    for param in est_params:
        new_params[param] = est_params[param] - lr * grad[param]    
    #new_params['beta'] = est_params['beta'] - lr * grad['beta']
    #new_params['branch_fit_effects'] = est_params['branch_fit_effects'] - lr * grad['branch_fit_effects']
    #new_params['s'] = est_params['s'] - lr * grad['s']
    #new_params = est_params - lr * grad
    
    return loss, new_params

#def run_on_sim_data():
# if __name__ == '__main__':
    
#     """
#         Run inference on simulated data
#     """

#     # Initialize birth-death model params
#     beta = np.array([1.0,0.7,0.6])
#     d = 0.5 # death rate
#     gamma = 0.0 # no migration here
#     s = np.array([0.5,0.5,0.5]) # sampling fraction upon removal
#     rho = 0.5 # sampling fraction at present
#     fit_effects = np.array(1*[1.0]) # np.random.uniform(0.9, 1.1, 10)
#     sigma = 0.005 # scales variance in Brownian motion model (a small value epsilon gets added to this)
    
#     # Set up time intervals in real time
#     #time_intervals = np.array([15.0, 10.0, 0.0]) # in distance from present at time 20.0
#     time_intervals = np.array([5.0,10.0,20.0]) # now in distance from root to be consistent

#     params = {'beta': beta, 
#               'fit_effects': fit_effects,
#               'd': d, 
#               'gamma': gamma, 
#               's': s, 
#               'rho': rho, 
#               'sigma': sigma,
#               'time_intervals': time_intervals}

#     # Import tree
#     path = './test-sets/test-randomBranchEffects-sigma005-jan2026/'
#     sim_str = '001'
#     tree_file = path + 'tree-' + sim_str + '.tre'
#     input_tree = Tree(tree_file, format=1)
    
#     fasta_file = path + 'tree-' + sim_str + '.fasta'
#     features_dic = {}
#     for record in SeqIO.parse(fasta_file, "fasta"):
#         features_dic[record.id] = np.array(list(map(int,list(str(record.seq))))) # better way in numpy?

#     # Initialize model and build model for a given tree
#     model = MTBD(**params)
#     tree = model.build(input_tree,features_dic)

#     # Set branch fit effects
#     n_branch_effects = len(tree['parent_br_indexes']) #max([*tree['child2parent']]) + 1
#     branch_fit_effects = np.ones(n_branch_effects) # set init values all to 1.0
#     # Set to random or possibly true values
#     branch_fit_effects = TreeUtils.set_rand_branch_effects(branch_fit_effects,input_tree, path + 'treeFitValues-' + sim_str + '.csv')
#     model.branch_fit_effects = branch_fit_effects
#     model.parent_fit_effects = branch_fit_effects[tree['parent_br_indexes']]
    
#     # Specify which parameters are to be estimated (can be a dictionary)
#     est_params = {'beta': beta}
    
#     #est_params = {'branch_fit_effects': branch_fit_effects}
    
#     est_params = {'branch_fit_effects': branch_fit_effects,
#                  'beta': beta}

#     # Time it
#     # runs = 10
#     # tic = time.perf_counter()
#     # for n in range(runs):
#     #     like = model.log_like(est_params,tree)
#     #     print('jax like', like)    
#     # toc = time.perf_counter()
#     # elapsed = (toc - tic) / runs
#     # print(f"Time per eval: {elapsed:0.4f} seconds")
    
#     # Test gradients
#     # param_grad = jax.grad(model.log_like)(est_params, tree)
#     # print('Likelihood gradients: ', param_grad)
    
#     # Try training loop with jit-compiled update
#     # est_params = jnp.ones(n_branch_effects)
#     tic = time.perf_counter()
#     n_train_steps = 10000
#     for episode in range(n_train_steps):
#         loss, est_params = update_jit(est_params, tree, lr=0.001)
#         beta_estimate = est_params['beta']
#         avg_estimate = np.mean(est_params['branch_fit_effects'])
#         if episode % 100 == 0:
#             #print('Episode =',str(episode),' Loss =',str(loss),' Avg. estimate:', str(avg_estimate))
#             #print('Episode =',str(episode),' Loss =',str(loss),' beta:', str(beta_estimate))
#             print('Episode =',str(episode),' Loss =',str(loss),' Avg. estimate:', str(avg_estimate), ' beta', str(beta_estimate))
#     toc = time.perf_counter()
#     elapsed = (toc - tic) / n_train_steps
#     print(f"Time per eval with jit: {elapsed:0.4f} seconds")
    
#     out_file = 'estimatedFitValues-' + sim_str + '.csv'
#     est_branch_effects = est_params['branch_fit_effects']
#     TreeUtils.write_rand_branch_effects(input_tree,est_branch_effects,out_file)
 
#def run_on_real_data():
if __name__ == '__main__':  
    
    """
        Run inference on simulated data
    """

    # Initialize birth-death model params
    beta = np.array([1.000001,1.000001,1.000001])
    d = 1.0 # death rate
    gamma = 0.0 # no migration here
    s = np.array([0.0001,0.0001,0.0001]) # sampling fraction upon removal
    rho = 0.0001 # sampling fraction at present
    fit_effects = np.array(39*[1.0]) # 79 total features, 39 if only using AMR features
    sampling_effects = np.array(1*[1.0]) # 16 bioproject features plus 1 biospecimin type feature
    sigma = 0.005 # scales variance in Brownian motion model (a small value epsilon gets added to this)
    
    # Set up time intervals in real time
    #time_intervals = np.array([1862.125219,
    #                           2003,
    #                           2013])
    
    # Set up time intervals in distance from root
    root_time = 1862.125219
    final_sample_time = 2023.0
    time_intervals = np.array([2003 - root_time,
                               2013 - root_time,
                               final_sample_time - root_time])

    params = {'beta': beta, 
              'fit_effects': fit_effects,
              'sampling_effects': sampling_effects,
              'd': d, 
              'gamma': gamma, 
              's': s, 
              'rho': rho, 
              'sigma': sigma,
              'time_intervals': time_intervals}

    
    # Set input tree / features files
    path = './st131-data/input/'
    tree_file = path + 'named.tree_lsd.date.noref.pruned_unannotated.nwk'
    meta_file = path + 'meta_features_marginal.csv'
    feature_file = path + 'genetic_features_marginal.csv'
    
    # Import tree
    input_tree = Tree(tree_file, format=1)

    # Import features with fitness effects
    features_dic = {}
    feature_df = pd.read_csv(feature_file, sep=',', index_col=0)
    # If only want to retain AMR features
    amr_features = [x for x in feature_df.columns if '_AMR' in x]
    feature_df = feature_df[amr_features]
    for index, row in feature_df.iterrows():
        features_dic[index] = row.to_numpy()
    
    # Import features with sampling effects
    sampling_dic = {}
    sampling_df = pd.read_csv(meta_file, sep=',', index_col=0)
    sample_type_features = ['blood_META']
    sampling_df = sampling_df[sample_type_features]
    for index, row in sampling_df.iterrows():
        sampling_dic[index] = row.to_numpy()
        
    # Initialize model and build model for a given tree
    model = MTBD(**params)
    tree = model.build(input_tree,features_dic,sampling_dic=sampling_dic)
    
    # Set branch fit effects
    n_branch_effects = len(tree['parent_br_indexes'])
    branch_fit_effects = np.ones(n_branch_effects) # set init values all to 1.0
    model.branch_fit_effects = branch_fit_effects
    model.parent_fit_effects = branch_fit_effects[tree['parent_br_indexes']]
    
    # Specify which parameters are to be estimated (can be a dictionary)
    est_params = {'fit_effects': fit_effects}
    
    # Test likelihood
    like = model.log_like(est_params,tree)
    print('jax like:', like)
    
    # Try training loop with jit-compiled update
    tic = time.perf_counter()
    n_train_steps = 20000
    for episode in range(n_train_steps):
        loss, est_params = update_jit(est_params, tree, lr=0.000001)
        fit_effects_estimate = est_params['fit_effects']
        if episode % 100 == 0:
            print('Episode =',str(episode),' Loss =',str(loss),' Fit effects:', str(fit_effects_estimate))
            #print('Sampling frac s =', str(est_params['s']))
    toc = time.perf_counter()
    elapsed = (toc - tic) / n_train_steps
    print(f"Time per eval with jit: {elapsed:0.4f} seconds")
    
    # Get ML estimates
    fit_ests = dict(zip(amr_features, fit_effects_estimate.tolist()))
    for param, est in fit_ests.items():
        print(f"{param}: {est:.4f}")


# if __name__ == '__main__':

#     run_on_sim_data()
    
#     #run_on_real_data()


    
 