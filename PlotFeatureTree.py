"""
Created on Mon Mar  8 09:41:09 2021

@author: david
"""

import balticmod as bt
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns

"Load in ancestral features as df before encoding"
base_dir = Path(__file__).parent / 'st131-data' / 'input'

features_file = str(base_dir / 'genetic_features_marginal.csv')
mtt_file = str(base_dir / "named.tree_lsd.date.noref.pruned_unannotated.nwk") # tree with internal labels

fig_file = 'ST131_2.png'

df = pd.read_csv(features_file,sep=",",index_col='node')
# df = pd.read_csv(features_file,sep=",")

"Load tree"
tree=bt.loadNewick(mtt_file,absoluteTime=False)
tree.traverse_tree() ## required to set heights
absolute_time = 1.0
tree.setAbsoluteTime(absolute_time)

# "Get mapping between host indexes and names"
# host_states = df['node'].unique()
# host_int2label = {index:state for index, state in enumerate(host_states)}

"Create state-to-integer mapping for each feature"
max_num_states = 0
#features=['HOST','AA4', 'AA6', 'AA9'] # keys for features
# features = df.columns.to_list()
# aa_features = [f for f in features if f[:2] == 'AA']
# features = ['HOST'] + aa_features
features = ["gyrAS83L_AMR", "parCE84V_AMR",
         "gyrAD87NparCS80I_AMR", "parEI529L_AMR"]
for feature in features:
    #states = df[feature].unique()
    f_counts = df[feature].value_counts(normalize=True)
    states = f_counts.index.to_list() # sort states so major variant always has the same color
    num_states = len(states)
    if num_states > max_num_states:
        max_num_states = num_states
    state2int = {state:index for index, state in enumerate(states)}
    df[feature] = df[feature].apply(lambda x: state2int[x])
cmap = sns.color_palette("muted", max_num_states) #mpl.cm.get_cmap('tab10', 10)
# cmap[1] = "yellow"
cmap[0] = "#F0E442"
cmap[1] = "#4878D0"

#fig,ax = plt.subplots(figsize=(20,20),facecolor='w')
# fig = plt.subplots(figsize=(7,20),facecolor='w')
fig = plt.figure(figsize=(7,20), facecolor='w')

"""
    Set up plot grid
    Set width_ratios
"""
gs = GridSpec(1,2,width_ratios=[2,1],wspace=0.05)
ax_tree = plt.subplot(gs[0])
ax_genome = plt.subplot(gs[1],sharey=ax_tree)


"Set attributes"
x_attr=lambda k: k.absoluteTime ## x coordinate of branches will be absoluteTime attribute
b_func=lambda k: 1.0 ## branch width
c_func=lambda k: 'dimgrey' #'darkorange' if k.traits['type']=='1' else 'steelblue' ## colour of branches
s_func=lambda k: 10 ## size of tips
z_func=lambda k: 100

"Plot tree"
tree.plotTree(ax_tree,branchWidth=b_func,x_attr=x_attr,colour_function=c_func) ## plot tree branches as regular black tree
tree.plotPoints(ax_tree,x_attr=x_attr,size_function=s_func,colour_function=c_func,zorder_function=z_func) ## plot circles at tips

# "Rename tips"
# tip_labels = df.index.to_list()
# short_labels = ['-'.join(label.split('-')[:2]) for label in tip_labels]
# label_map = {label:short_label for label, short_label in zip(tip_labels,short_labels)}
# df = df.rename(index=label_map)

# for k in tree.Objects: # iterate over branches
#     if k.branchType=='leaf':
#         #print(k.numName)
#         k.numName = label_map[k.numName]

"Add tip labels"
tip_font_size = 16
text_func = lambda k: k.numName.replace('_',' ')
target_func = lambda k: k.is_leaf()
position_func = lambda k: (1.001, k.y-0.15)
# tree.addText(ax_tree, text=text_func, position=position_func,fontsize=tip_font_size)

"""
    Plot features in character matrix
"""
clean_feature_names = [feature.replace("_AMR", "") for feature in features]
for k in tree.Objects: # iterate over branches
    if k.branchType=='leaf':
        for f in range(len(features)): ## iterate over trait keys
            ftype = df.loc[k.numName][features[f]]
            c=cmap[int(round(ftype, 0))]
            lineage=plt.Rectangle((f,k.y-0.5),1,1,facecolor=c,edgecolor='none') ## rectangle with height and width 1, at y position of tip and at the index of the key
            ax_genome.add_patch(lineage) ## add coloured rectangle to plot
ax_genome.set_xticks(np.arange(0.5,len(features)+0.5))
ax_genome.set_xticklabels(clean_feature_names,rotation=90)
[ax_genome.axvline(x,color='w') for x in range(len(features))]

"Add legend for map colors"
import matplotlib.patches as mpatches
#handles = [mpatches.Patch(color='lightgrey', label='WT'), mpatches.Patch(color='darkorchid', label='RB')]
#legend1 = ax_tree.legend(handles=handles,prop={'size': 22},loc='lower left',bbox_to_anchor=(0.03, 0.14)) #was 1.32

# handles = [mpatches.Patch(color=cmap[i], label=host_int2label[i]) for i in range(len(host_states))]
# ax_tree.legend(handles=handles,prop={'size': 22},loc='lower left',bbox_to_anchor=(0.03, 0.25)) #was 1.32

"Add original legend back in"
#ax_tree.add_artist(legend1)

"Set x and y lims"
ax_genome.set_xlim(0,len(features))
ax_tree.set_ylim(-0.5,tree.ySpan+1)

"Turn axis spines invisible"
[ax_tree.spines[loc].set_visible(False) for loc in ['top','right','left','bottom']] ## no axes
[ax_genome.spines[loc].set_visible(False) for loc in ['top','right','left','bottom']] ## no axes

ax_tree.set_xticks([])
ax_tree.set_yticks([])
ax_tree.tick_params(axis='both', length=0, labelbottom=False, labelleft=False)

ax_genome.tick_params(axis='both', length=0, labelsize=12, labelleft=False)
ax_genome.xaxis.set_ticks_position('top')

#ax_tree.grid(axis='x')

# plt.savefig(fig_file, dpi=300)
fig.savefig(fig_file, dpi=300)





