#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:22:15 2023

@author: David
"""

from ete3 import Tree
import TreeUtils
import numpy as np


class VectorizedTree(object):
    
    def __init__(self,tree,time_intervals,features_dict,sampling_dict=None):
        
        """
            events: 1=birth; 2=poisson_sample; 3=rho_sample; 4=edge
            
            TODO: rho-sampling (CSA) type never implemented
            
        """
        
        self.param_interval_times = time_intervals
        self.time_indexes = np.arange(0,len(time_intervals))
        self.present_time = max(self.param_interval_times)
        self.bkwd_interval_times = self.present_time - self.param_interval_times
        self.features_dict = features_dict
        self.sampling_dict = sampling_dict
        
        birth_times = []
        event_times = []
        events = []
        #parent_idx = []
        idx = []
        ft_names = []
        for node in tree.traverse("postorder"):
            
            "Add branch index for retriving branch-specific fitness values"
            node_index = node.branch_idx
            
            ft_name = node.name
            
            "Root is at time zero with time increasing towards tips"
            node_event_time = node.time # time closer to present
            birth_time = node_event_time - node.dist  # time further in past
            
            if len(node.children) > 1:
                record = True
                event = 1
                
            elif node.is_leaf():
                record = True
                if np.isclose(node_event_time, self.present_time, atol=0.001):
                    event = 3 # rho_sample
                else:
                    event = 2
                
            else:
                record = False
                
            if record:
                events.append(event)
                birth_times.append(birth_time)
                event_times.append(node_event_time)
                #parent_idx.append(parent_index)
                idx.append(node_index)
                ft_names.append(ft_name)
            
            # SAVE EDGE INFO
            events.append(4)
            birth_times.append(birth_time)
            event_times.append(node_event_time)
            #parent_idx.append(parent_index)
            idx.append(node_index)
            ft_names.append(ft_name)
            
        time_steps = [e - b for e, b in zip(event_times, birth_times)]
        back_times = [self.present_time - t for t in event_times]
        param_intervals = [self.getParamInterval(t) for t in event_times]
        pE_intervals = [p for p in param_intervals] # was p+1
        sequences = [self.features_dict[name] for name in ft_names] # tolist()?
        
        # New for sampling features
        if self.sampling_dict:
            sampling_feature_list = [self.sampling_dict[name] for name in ft_names]
        else:
            sampling_feature_list = []
            
        """
           Need to put in numpy array lists
           so we can grab tree events by type
        """
        arr_list = list(zip(
            idx,
            sequences,
            sampling_feature_list,
            birth_times,
            event_times,
            back_times,
            events,
            param_intervals,
            pE_intervals,
            time_steps
        ))
        
        arr = np.array(arr_list, dtype=[
            ('idx', int),
            ('features', object),
            ('sampling_features', object),
            ('birth_time', float),
            ('event_time', object),
            ('back_time', float),
            ('event', int),
            ('param_interval', int),
            ('pE_interval', int),
            ('time_step', float)
        ])
        self.arr = arr
        
        self.edge_arr = arr[arr['event'] == 4]
        self.birth_arr = arr[arr['event'] == 1]
        self.sample_arr = arr[arr['event'] == 2]
        self.rho_arr = arr[arr['event'] == 3]
        
        
    def getParamInterval(self,time):
        """
            Given an event time, returns the index of the
            parameter interval that time is contained in
        """
        try:
            #param_interval = np.where(time > self.param_interval_times)[0][-1] # this is what Lenora had
            param_interval = self.time_indexes[np.where(self.param_interval_times >= time)][0]
        except:
            param_interval = 0
        
        return param_interval
            
if __name__ == '__main__':
    
    path = './test-sets/testTF_randomSiteEffects_june2020/'
    tree_file = path + 'tree-000.tre'
    fasta_file = path + 'tree-000.fasta'
    
    "Initial birth-death model params"
    beta = np.array([1.0,0.7,0.6])
    d = 0.5 # death rate
    gamma = 0.0 # no migration here
    s = 0.5 # sampling fraction upon removal
    rho = 0.5 # sampling fraction at present
    dt = 1.0 # time step interval for update pE's along branch 
    params = {'beta': beta, 'd': d, 'gamma': gamma, 's': s, 'rho': rho, 'dt': dt, 'time_intervals': 0}
    
    "Provide dict with estimated params"
    est_params = {'site_effects': False, 'beta': True, 'd': False, 'gamma': False, 's': False, 'rho': False}
  
    tree = Tree(tree_file, format=1)
        
    "Set up tree for run"
    tree, tree_times = TreeUtils.add_tree_times(tree)
    final_time = max(tree_times)
    time_intervals = np.array([final_time - 15.0, final_time - 10.0, final_time])
    params.update(time_intervals = time_intervals)
    
    "Convert tree to TensorTree object"
    tree = TreeUtils.index_branches(tree) # only used for models with random branch effects
    tt = VectorizedTree(tree,time_intervals)
            
