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
import jax.nn as jnn
import sys
from ete3 import Tree
import TreeUtils
from VectorizedTree import VectorizedTree
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

jnp.set_printoptions(threshold=jnp.inf, linewidth=10_000)
jax.config.update("jax_enable_x64", True)


from itertools import combinations


DEFAULT_QRDR_FEATURES = (
    "gyrAS83L_AMR",
    "parCE84V_AMR",
    "gyrAD87NparCS80I_AMR",
    "parEI529L_AMR",
    "blaCTXM27_AMR",
    "tet_AMR",
    "mdtM_AMR",
    "glpTE448K_AMR",
    "blaCTXM15_AMR",
    "aac3_AMR",
    "aac6_AMR",
    "sul_AMR",
    "aac6IbcrblaOXAOH_AMR",
)

DEFAULT_SAMPLING_FEATURES = ("blood_META",)

# Interaction flags. Set these before running the script to control the
# fitness-effect design matrix.
INCLUDE_PAIRWISE_INTERACTIONS = True
INCLUDE_THREE_WAY_INTERACTIONS = True

# Nonlinear fitness flag. When False, use the linear multiplicative fitness
# model and skip B-spline coefficient estimation/output.
USE_NONLINEAR_FITNESS = False


@dataclass(frozen=True)
class ST131Config:
    input_dir: Path = Path("./st131-data/input")
    tree_filename: str = "named.tree_lsd.date.noref.pruned_unannotated.nwk"
    meta_filename: str = "meta_features_marginal.csv"
    feature_filename: str = "genetic_features_marginal.csv"
    output_dir: Path = Path("./ST131_HMC_fitEffects_all")
    root_time: float = 1862.125219
    earliest_sample_time: float = 1985.9
    birth_rate_changepoints: tuple[float, ...] = (2003.0, 2013.0)
    final_sample_time: float = 2023.0
    beta: tuple[float, ...] = (1.000001, 1.000001, 1.000001, 1.000001)
    d: float = 1.0
    gamma: float = 0.0
    s: tuple[float, ...] = (0.0, 0.0001, 0.0001, 0.0001)
    rho: float = 0.0001
    feature_names: tuple[str, ...] = DEFAULT_QRDR_FEATURES
    sampling_feature_names: tuple[str, ...] = DEFAULT_SAMPLING_FEATURES
    interaction_features: Optional[tuple[str, ...]] = None
    include_two_way_interactions: bool = INCLUDE_PAIRWISE_INTERACTIONS
    include_three_way_interactions: bool = INCLUDE_THREE_WAY_INTERACTIONS
    use_nonlinear_fitness: bool = USE_NONLINEAR_FITNESS
    n_splines: int = 5
    num_warmup: int = 1000
    num_samples: int = 2000
    random_seed: int = 0
    test_likelihood_only: bool = False

    @property
    def tree_file(self) -> Path:
        return self.input_dir / self.tree_filename

    @property
    def meta_file(self) -> Path:
        return self.input_dir / self.meta_filename

    @property
    def feature_file(self) -> Path:
        return self.input_dir / self.feature_filename

    @property
    def time_intervals(self) -> np.ndarray:
        times = (
            self.earliest_sample_time,
            *self.birth_rate_changepoints,
            self.final_sample_time,
        )
        return np.array([time - self.root_time for time in times])


def build_default_config() -> ST131Config:
    return ST131Config()


def make_full_knots(internal_knots, boundary_knots, degree=3):
    """
    internal_knots: (K,) JAX array
    boundary_knots: (2,) JAX array, [min_x, max_x]
    degree: spline degree, 3 = cubic
    """
    t0, t1 = boundary_knots
    return jnp.concatenate([
        jnp.repeat(t0, degree + 1),
        internal_knots,
        jnp.repeat(t1, degree + 1),
    ])


def degree0_basis_with_right_boundary(x, t):
    """
    x: (n,)
    t: (m,) full knot vector
    returns: B0, shape (n, m-1)
    """
    x = jnp.asarray(x)
    t = jnp.asarray(t)

    X = x[:, None]      # (n, 1)
    T = t[None, :]      # (1, m)
    m = t.shape[0]

    # Standard half-open intervals: [t[i], t[i+1])
    B0 = jnp.where((X >= T[:, :-1]) & (X < T[:, 1:]), 1.0, 0.0)  # (n, m-1)

    # For x == last_knot, force the last basis to be 1.
    last_knot = t[-1]
    on_right_boundary = (x == last_knot)  # (n,)

    # Replace last column: if on_right_boundary, use 1; else keep prior value.
    B0 = B0.at[:, -1].set(jnp.where(on_right_boundary, 1.0, B0[:, -1]))

    return B0


def bspline_basis_jax(x, full_knots, degree=3):
    """
    Evaluate B-spline basis of given degree at points x.

    x: (n,) JAX array
    full_knots: (m,) JAX array, open knot vector (boundary repeated degree+1 times)
    degree: int, e.g. 3 for cubic splines

    Returns: B, shape (n, n_basis)
        where n_basis = len(full_knots) - degree - 1
    """
    x = jnp.asarray(x)
    t = jnp.asarray(full_knots)
    m = t.shape[0]

    # degree 0 basis with right-boundary fix
    B = degree0_basis_with_right_boundary(x, t)  # (n, m-1)

    # Recursive Cox–de Boor for k = 1 .. degree
    X = x[:, None]
    T = t[None, :]
    for k in range(1, degree + 1):
        B_prev = B  # (n, m-k)

        # At degree k, basis count = m - k - 1
        # left term
        left_num = X - T[:, :m - k - 1]             # (n, m-k-1)
        left_den = T[:, k:m - 1] - T[:, :m - k - 1] # (n, m-k-1)
        safe_left_den = jnp.where(left_den > 0, left_den, 1.0)

        # left = jnp.where(left_den > 0, left_num / left_den, 0.0) * B_prev[:, :m - k - 1]
        left = jnp.where(left_den > 0, left_num / safe_left_den, 0.0) * B_prev[:, :m - k - 1]


        # right term
        right_num = T[:, k + 1:m] - X               # (n, m-k-1)
        right_den = T[:, k + 1:m] - T[:, 1:m - k]   # (n, m-k-1)
        safe_right_den = jnp.where(right_den > 0, right_den, 1.0)
        # right = jnp.where(right_den > 0, right_num / right_den, 0.0) * B_prev[:, 1:m - k]
        right = jnp.where(right_den > 0, right_num / safe_right_den, 0.0) * B_prev[:, 1:m - k]

        B = left + right  # (n, m-k-1)

    return B  # (n, n_basis), n_basis = m - degree - 1


def generate_interactions(X, feature_names, selected_features=None,
                          include_two_way=True,
                          include_three_way=False):
    """
    Generate interaction terms from a feature matrix.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_features)

    feature_names : list[str]
        Names of all columns in X

    selected_features : list[str] or None
        Features to use for interactions.
        If None, use all features.

    include_two_way : bool
        Whether to generate pairwise interactions

    include_three_way : bool
        Whether to generate 3-way interactions

    Returns
    -------
    X_new : np.ndarray
        Original matrix concatenated with interaction columns

    new_feature_names : list[str]
        Names of original + interaction columns
    """

    feature_to_idx = {f: i for i, f in enumerate(feature_names)}

    if selected_features is None:
        selected_features = feature_names

    selected_idx = [feature_to_idx[f] for f in selected_features]

    interaction_cols = []
    interaction_names = []

    # -------------------------
    # Two-way interactions
    # -------------------------
    if include_two_way:
        for i, j in combinations(selected_idx, 2):

            col = X[:, i] * X[:, j]

            interaction_cols.append(col)

            interaction_names.append(
                f"{feature_names[i]}:{feature_names[j]}"
            )

    # -------------------------
    # Three-way interactions
    # -------------------------
    if include_three_way:
        for i, j, k in combinations(selected_idx, 3):

            col = X[:, i] * X[:, j] * X[:, k]

            interaction_cols.append(col)

            interaction_names.append(
                f"{feature_names[i]}:{feature_names[j]}:{feature_names[k]}"
            )

    # Stack interaction columns
    if interaction_cols:
        interaction_matrix = np.column_stack(interaction_cols)

        X_new = np.hstack([X, interaction_matrix])

    else:
        X_new = X.copy()

    new_feature_names = feature_names + interaction_names

    return X_new, new_feature_names


def feature_names_with_interactions(feature_names: Sequence[str],
                                    selected_features: Optional[Sequence[str]] = None,
                                    include_two_way: bool = True,
                                    include_three_way: bool = False) -> list[str]:
    feature_names = list(feature_names)
    feature_to_idx = {f: i for i, f in enumerate(feature_names)}
    if selected_features is None:
        selected_features = feature_names

    selected_idx = [feature_to_idx[f] for f in selected_features]
    interaction_names = []

    if include_two_way:
        for i, j in combinations(selected_idx, 2):
            interaction_names.append(f"{feature_names[i]}:{feature_names[j]}")

    if include_three_way:
        for i, j, k in combinations(selected_idx, 3):
            interaction_names.append(
                f"{feature_names[i]}:{feature_names[j]}:{feature_names[k]}"
            )

    return feature_names + interaction_names





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
        self.feature_names = list(params.get('feature_names', []))
        self.interaction_features = params.get('interaction_features')
        self.include_two_way_interactions = params.get('include_two_way_interactions', True)
        self.include_three_way_interactions = params.get('include_three_way_interactions', False)
        self.use_nonlinear_fitness = params.get('use_nonlinear_fitness', True)
        self.n_splines = params.get('n_splines', 5)

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

        edge_features_inter, _ = generate_interactions(
            tree_dict['edge_features'],
            feature_names=self.feature_names,
            selected_features=self.interaction_features,
            include_two_way=self.include_two_way_interactions,
            include_three_way=self.include_three_way_interactions,
        )
        birth_features_inter, _ = generate_interactions(
            tree_dict['birth_features'],
            feature_names=self.feature_names,
            selected_features=self.interaction_features,
            include_two_way=self.include_two_way_interactions,
            include_three_way=self.include_three_way_interactions,
        )
        # print(edge_features_inter.shape)

        tree_dict['edge_features'] = edge_features_inter
        tree_dict['birth_features'] = birth_features_inter

        
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
        # print(parent_br_indexes)
        tree_dict['parent_dists'] = np.array(parent_dists) # parent idx for each child branch idx
        tree_dict['edge_branch_idx'] = tree.edge_arr['idx']
        tree_dict['birth_branch_idx'] = tree.birth_arr['idx']
        
        tree = tree_dict
        
        self.tree = tree
        
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

        # debug.print("edge_birth:{}", t['edge_b'])
        # debug.print("cnst_c:{}",  cnst_c)
        # debug.print("t['edge_s']:{}", t['edge_s'])
        # debug.print("pE_denom:{}", pE_denom)
        
        """
            Lenora was using tf.safedivide which has no jax equivalent
            Could use:
                min_denom = 0.01
                pE_denom = jnp.where(pE_denom == 0.0, min_denom, pE_denom)
        """
        pEs = (-1 / t['edge_b']) * pE_num / pE_denom
        
        return cnst_x, cnst_y, cnst_c, pEs
    
    def _calc_fitness(self, site_fit_effects, base_fit, bs_coef=None):

        if bs_coef is None:
            """
                Compute fitness of each lineage based on multiplicative fitness effects of their features
                Assumes beta is proportional to fitness
            """

            #log_fit_effects = jnp.log(site_fit_effects) # log transform fitness effects so we "multiply" when we sum below
            # log_fit_effects = site_fit_effects # site_fit_effects are aleady assumed to be log transformed for HMC
            log_fit_effects = site_fit_effects[0:self.n_features]

            edge_fit_effects = jnp.exp(jnp.matmul(self.tree['edge_features'],log_fit_effects) + base_fit) # convert back to linear scale
            birth_fit_effects = jnp.exp(jnp.matmul(self.tree['birth_features'],log_fit_effects)+ base_fit)

            self.tree['edge_b'] *= edge_fit_effects
            self.tree['birth_b'] *= birth_fit_effects
        else:
            log_fit_effects = site_fit_effects[0:self.n_features]
            edge_linear_effect = jnp.matmul(self.tree['edge_features'],log_fit_effects) + base_fit
            birth_linear_effect = jnp.matmul(self.tree['birth_features'],log_fit_effects) + base_fit


            n_splines = self.n_splines
            degree = 3
            boundary_knots = jnp.array([0.0, 1.0])
            n_internal_knots = n_splines + 1 - degree - 2
            internal_knots = jnp.linspace(0.0, 1.0, num=n_internal_knots + 2)
            internal_knots = internal_knots[1:(n_internal_knots + 1)]
            full_knots = make_full_knots(internal_knots, boundary_knots, degree=degree)

            edge_linear_effect_expit = jax.scipy.special.expit(edge_linear_effect)
            birth_linear_effect_expit = jax.scipy.special.expit(birth_linear_effect)

            edge_spline_basis = bspline_basis_jax(edge_linear_effect_expit, full_knots, degree=degree)
            birth_splines_basis = bspline_basis_jax(birth_linear_effect_expit, full_knots, degree=degree)

            self.tree['edge_b'] *= edge_spline_basis @ bs_coef
            self.tree['birth_b'] *= birth_splines_basis @ bs_coef
        
    
    def log_like(self,est_params):
        
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
            self.tree['edge_b'] = beta[self.tree['edge_param_interval']] # update base betas 
            self.tree['birth_b'] = beta[self.tree['birth_param_interval']] # update base betas
        else:
            beta = self.beta # necessary to reset this if not estimating?
            self.tree['edge_b'] = beta[self.tree['edge_param_interval']] # update base betas 
            self.tree['birth_b'] = beta[self.tree['birth_param_interval']] # update base betas
            
        if 's' in est_params:
            s = est_params['s']
            s.at[0].set(0.0) # hacky way to constrain sampling fraction to zero before first sample
            self.tree['edge_s'] = s[self.tree['edge_param_interval']] # update edge sampling fractions 
            self.tree['sample_s'] =  s[self.tree['sample_param_interval']]
        else:
            s = self.s # necessary to reset this if not estimating?
        
        if 'fit_effects' in est_params:
            fit_effects = est_params['fit_effects']
        else:
            fit_effects = self.fit_effects
            
        if 'sampling_effects' in est_params:
            s_effects = est_params['sampling_effects']
            log_s_effects = jnp.log(s_effects) # log transform sampling effects so we "multiply" when we sum below
            edge_s_effects = jnp.exp(jnp.matmul(self.tree['edge_sampling_features'],log_s_effects)) # convert back to linear scale
            sample_s_effects = jnp.exp(jnp.matmul(self.tree['sample_sampling_features'],log_s_effects))
            self.tree['edge_s'] *= edge_s_effects
            self.tree['sample_s'] *= sample_s_effects    
        
        # Sigma scales variance in Brownian motion model
        if 'sigma' in est_params:
            sigma = est_params['sigma']
        else:
            sigma = self.sigma

        if 'psi' in est_params:
            psi = est_params['psi']
        else:
            psi = 1
            
        """
            This causes problems if we're not estimating branch_fit_effects
        """
        if 'branch_fit_effects' in est_params:
            #branch_fit_effects = est_params['branch_fit_effects']
            branch_fit_effects = jnp.exp(est_params['branch_fit_effects']) # if estimated on a log scale
            # Rescale edge and birth event birth rates based on branch fit effects
            self.tree['edge_b'] *= branch_fit_effects[self.tree['edge_branch_idx']]
            self.tree['birth_b'] *= branch_fit_effects[self.tree['birth_branch_idx']]
            parent_fit_effects = branch_fit_effects[self.tree['parent_br_indexes'].astype(int)]
        else:
            branch_fit_effects = self.branch_fit_effects
            #tree['edge_b'] *= branch_fit_effects[tree['edge_branch_idx']]
            #tree['birth_b'] *= branch_fit_effects[tree['birth_branch_idx']]
            parent_fit_effects = self.parent_fit_effects

        if self.use_nonlinear_fitness and 'bs_coef' in est_params:
            bs_coef = est_params['bs_coef']
        else:
            bs_coef = None
        
        # Update birth rates based on fitness effects
        base_fit = est_params['base_fit']
        self._calc_fitness(fit_effects, base_fit, bs_coef)
        
        t = self.tree
        
        cnst_x, cnst_y, cnst_c, pEs = self._calc_PEs(t)
        
        # Calc prob density for all edges
        pD_denom = ((cnst_y + t['edge_b'] * pEs) * jnp.exp(-cnst_c * t['edge_time_step'])) - (cnst_x + t['edge_b'] * pEs)
        pD_intermed = (cnst_y - cnst_x) / pD_denom # see _calc_PEs for a safe division alternative
        pD = jnp.exp(-cnst_c * t['edge_time_step']) * jnp.square(pD_intermed)
        # debug.print("zero_birth:{}", jnp.sum(t['edge_b']==0))
        # debug.print("pEs:{}", pEs)
        
        """
            WARNING: pD values can return zero
            The jnp.exp(-cnst_c * t['edge_time_step']) term in pD calculation
            can go to zero (especially along very long branches) 
            which will cause -Inf values when we compute log(pD) below
        """
        #zero_pD_indexes = np.where(pD == 0)[0] # to catch zero values while debugging
        min_pD = 1e-06
        pD = jnp.where(pD == 0.0, min_pD, pD)
        
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
        # fit_shifts = branch_fit_effects - parent_fit_effects

        # print(t['parent_dists'])
        fit_shifts = jnp.log(branch_fit_effects) - psi * jnp.log(parent_fit_effects)
        # # debug.print("fit_shifts: {}",fit_shifts)
        epsilon = 1e-10 #0.005 # small value added to denom below to increase numerical stability when sigma * delta_t is << 1.0
        #
        const_term = -0.5/jnp.log(sigma)
        # bm_probs = const_term + (-0.5 * fit_shifts**2) / (sigma * t['parent_dists']+epsilon)
        bm_probs = const_term + (-0.5 * fit_shifts ** 2) / (sigma + epsilon)
        bm_like = jnp.sum(bm_probs)
        #
        bm_like = 0


        #loss = -(line_like + sample_like + sample_like_csa + birth_like + bm_like)
        
        """
            Print out all components of like using debug
        """
        # debug.print("line like:{}", line_like)
        # debug.print("sample like:{}", sample_like)
        # debug.print("sample csa like:{}", sample_like_csa)
        # debug.print("birth like:{}", birth_like)
        # debug.print("bm like:{}", bm_like)
        
        # Loss should not be negative for HMC
        loss = line_like + sample_like + sample_like_csa + birth_like + bm_like
        
        return loss

def load_st131_data(config):
    input_tree = Tree(str(config.tree_file), format=1)

    feature_df = pd.read_csv(config.feature_file, sep=',', index_col=0)
    feature_df = feature_df[list(config.feature_names)]
    features_dic = {
        index: row.to_numpy()
        for index, row in feature_df.iterrows()
    }

    sampling_df = pd.read_csv(config.meta_file, sep=',', index_col=0)
    sampling_df = sampling_df[list(config.sampling_feature_names)]
    sampling_dic = {
        index: row.to_numpy()
        for index, row in sampling_df.iterrows()
    }

    return input_tree, features_dic, sampling_dic


def model_params_from_config(config, n_fit_features):
    return {
        'beta': np.array(config.beta),
        'fit_effects': np.ones(n_fit_features),
        'sampling_effects': np.ones(len(config.sampling_feature_names)),
        'd': config.d,
        'gamma': config.gamma,
        's': np.array(config.s),
        'rho': config.rho,
        'time_intervals': config.time_intervals,
        'feature_names': config.feature_names,
        'interaction_features': config.interaction_features,
        'include_two_way_interactions': config.include_two_way_interactions,
        'include_three_way_interactions': config.include_three_way_interactions,
        'use_nonlinear_fitness': config.use_nonlinear_fitness,
        'n_splines': config.n_splines,
    }


def build_model(config, input_tree, features_dic, sampling_dic):
    fit_feature_names = feature_names_with_interactions(
        config.feature_names,
        selected_features=config.interaction_features,
        include_two_way=config.include_two_way_interactions,
        include_three_way=config.include_three_way_interactions,
    )
    params = model_params_from_config(config, len(fit_feature_names))

    model = MTBD(**params)
    tree = model.build(input_tree, features_dic, sampling_dic=sampling_dic)

    n_branch_effects = len(tree['parent_br_indexes'])
    branch_fit_effects = np.ones(n_branch_effects)
    model.branch_fit_effects = branch_fit_effects
    model.parent_fit_effects = branch_fit_effects[
        tree['parent_br_indexes'].astype(int)
    ]

    return model, tree, n_branch_effects, fit_feature_names


def make_hmc_model(model, n_fit_features, n_branch_effects, n_splines,
                   use_nonlinear_fitness):
    def hmc_model():
        var = numpyro.sample("var", dist.InverseGamma(0.1, 0.1))
        fit_effects = numpyro.sample(
            "fit_effects",
            dist.Laplace(
                jnp.zeros(n_fit_features),
                var.repeat(n_fit_features),
            ).to_event(1),
        )

        branch_fit_effects_var = numpyro.sample(
            "branch_fit_effects_var",
            dist.InverseGamma(1, 1),
        )
        branch_fit_effects_sd = jnp.full(
            (n_branch_effects,),
            jnp.sqrt(branch_fit_effects_var),
        )
        branch_fit_effects = numpyro.sample(
            "branch_fit_effects",
            dist.Normal(loc=0, scale=branch_fit_effects_sd).to_event(1),
        )

        base_fit_exp = numpyro.sample("base_fit_exp", dist.Uniform(0.5, 1.5))
        base_fit = jnp.log(base_fit_exp)
        numpyro.deterministic("base_fit", base_fit)

        est_params = {
            'var': var,
            'fit_effects': fit_effects,
            'branch_fit_effects': branch_fit_effects,
            'branch_fit_effects_var': branch_fit_effects_var,
            'base_fit': base_fit,
        }

        if use_nonlinear_fitness:
            bs_sigma = 1
            alpha = numpyro.sample(
                "alpha",
                dist.Normal(0., jnp.sqrt(bs_sigma)).expand([n_splines]).to_event(1),
            )
            bs_coef = jnp.cumsum(jnn.softplus(alpha), axis=-1)
            numpyro.deterministic("bs_coef", bs_coef)
            est_params['bs_coef'] = bs_coef

        likelihood_value = model.log_like(est_params)
        numpyro.factor("custom_likelihood_factor", likelihood_value)

    return hmc_model


def make_initial_params(n_fit_features, n_branch_effects, n_splines,
                        use_nonlinear_fitness):
    branch_fit_effects = np.ones(n_branch_effects)
    initial_params = {
        'var': jnp.array(0.1),
        'fit_effects': jnp.array([0.001] * n_fit_features),
        'branch_fit_effects': jnp.log(branch_fit_effects),
        'branch_fit_effects_var': jnp.array(0.1),
        'base_fit_exp': jnp.array(1.0),
    }

    if use_nonlinear_fitness:
        initial_params['alpha'] = jnp.ones(n_splines)

    return initial_params


def test_likelihood(model, n_fit_features, n_branch_effects):
    est_params = {
        'var': 0.1,
        'fit_effects': np.random.uniform(-0.5, 0.5, size=n_fit_features),
        'branch_fit_effects': np.random.uniform(-0.5, 0.5, size=n_branch_effects),
        'branch_fit_effects_var': 0.1,
        'base_fit': 1.2,
    }
    print(model.log_like(est_params))


def run_hmc(config, model, n_fit_features, n_branch_effects):
    hmc_model = make_hmc_model(
        model,
        n_fit_features=n_fit_features,
        n_branch_effects=n_branch_effects,
        n_splines=config.n_splines,
        use_nonlinear_fitness=config.use_nonlinear_fitness,
    )
    initial_params = make_initial_params(
        n_fit_features=n_fit_features,
        n_branch_effects=n_branch_effects,
        n_splines=config.n_splines,
        use_nonlinear_fitness=config.use_nonlinear_fitness,
    )

    key = jax.random.PRNGKey(config.random_seed)
    hmc_kernel = NUTS(hmc_model)
    mcmc = MCMC(
        hmc_kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
    )
    mcmc.run(key, init_params=initial_params)
    mcmc.print_summary()
    return mcmc


def output_metadata(config, n_branch_effects, fit_feature_names,
                    use_nonlinear_fitness):
    exp_dict = {
        'var': False,
        's': False,
        'fit_effects': True,
        'branch_fit_effects': True,
        'branch_fit_effects_var': False,
        'base_fit': True,
        'sigma': False,
        'psi': False,
        'bs_coef': False,
    }
    sample_names_dict = {
        'var': ['var'],
        's': ['s0', 's1', 's2', 's3'],
        'fit_effects': fit_feature_names,
        'branch_fit_effects': list(range(n_branch_effects)),
        'branch_fit_effects_var': ['branch_fit_effects_var'],
        'base_fit': ['base_fit'],
    }
    params_to_write = [
        'var',
        'fit_effects',
        'branch_fit_effects',
        'branch_fit_effects_var',
        'base_fit',
    ]

    if use_nonlinear_fitness:
        sample_names_dict['bs_coef'] = [
            f'bs_coef_{i}' for i in range(config.n_splines)
        ]
        params_to_write.append('bs_coef')

    return exp_dict, sample_names_dict, params_to_write


def write_hmc_outputs(config, mcmc, n_branch_effects, fit_feature_names):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    samples = mcmc.get_samples()
    exp_dict, sample_names_dict, params_to_write = output_metadata(
        config,
        n_branch_effects,
        fit_feature_names,
        config.use_nonlinear_fitness,
    )

    for param in params_to_write:
        sample_df = pd.DataFrame(samples[param])
        if exp_dict[param]:
            sample_df = np.exp(sample_df)
        sample_df.columns = sample_names_dict[param]
        sample_df.to_csv(config.output_dir / f'HMC_samples_{param}.csv')

    summ = summary(samples, group_by_chain=False)
    cols_to_transform = ['mean', 'std', 'median', '5.0%', '95.0%']
    for param in params_to_write:
        if param in ['var', 'sigma', 'base_fit', 'branch_fit_effects_var', 'psi']:
            est_df = pd.DataFrame(summ[param], index=[0])
        else:
            est_df = pd.DataFrame(summ[param])
        if exp_dict[param]:
            est_df[cols_to_transform] = est_df[cols_to_transform].apply(np.exp)
        est_df.index = sample_names_dict[param]
        est_df.to_csv(config.output_dir / f'HMC_summary_{param}.csv')


def main(config=None):
    config = config or build_default_config()
    input_tree, features_dic, sampling_dic = load_st131_data(config)
    model, tree, n_branch_effects, fit_feature_names = build_model(
        config,
        input_tree,
        features_dic,
        sampling_dic,
    )

    if config.test_likelihood_only:
        test_likelihood(model, len(fit_feature_names), n_branch_effects)
        return

    mcmc = run_hmc(config, model, len(fit_feature_names), n_branch_effects)
    write_hmc_outputs(config, mcmc, n_branch_effects, fit_feature_names)


if __name__ == '__main__':
    main()
 
