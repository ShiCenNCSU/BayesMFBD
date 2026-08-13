#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 13:28:10 2026

@author: david
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats

est_df = pd.read_csv('estimatedFitValues-001.csv',index_col=0)
#path = './test-sets/test-randomBranchEffects-sigma005-jan2026/'
path = './test-sets/test-global-epi-ge_b-4_sigma005-jan2026/'
sim_file = path + 'treeFitValues-001.csv'
sim_df = pd.read_csv(sim_file,index_col=0)

sns.set_theme(style="white")
#sns.set_palette("colorblind")

est_df = est_df.rename(columns={'Feature': 'est_fitness'})
sim_df = sim_df.rename(columns={'Feature': 'sim_fitness'})
df = est_df.join(sim_df)

png_file = 'test_randomBrachEffects_estimated_vs_sim_fit.png'
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
sns.scatterplot(x = "sim_fitness", y = "est_fitness", data=df, color='grey', alpha=1.0)
ax.set_xlabel('True fitness',fontsize=12)
ax.set_ylabel('Estimated fitness',fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(png_file, dpi=200)

pearson_r = scipy.stats.pearsonr(df['sim_fitness'], df['est_fitness'])
print("Pearson corr = ", pearson_r)