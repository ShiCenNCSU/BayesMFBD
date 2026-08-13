---
Completed: false
Necessary: true
Important: true
Immediate: true
---
## input
- **named.tree_lsd.date.noref.pruned_unannotated.nwk**
	- *data/named.tree_lsd.date.noref.pruned_unannotated.nwk*
	- Original tree output of LSD, pruned of reference node, unannotated
- **3_interval_tree.nwk**
	- *data_new/3_interval_tree/phylo.nwk*
	- Above tree, with branches split at time changepoints: either of the two birth rate changepoints or one of the sampling rate changepoints
- **meta_features_marginal.csv**
	- *data_new/meta_features_marginal.csv*
	- PastML output, marginal states, of bioproject and specimen type. Urine is used as the dummy/baseline variable for specimen type, and so we specify the probability that a phylogeny piece was collected from blood. The bioproject dummy/baseline variable is PRJNA248737 (the oldest one).
		- see **other/bioproject_times.csv**
- **genetic_features_marginal.csv**
	- *data_new/functional_groups_corr_pastml/marginal_states.csv*
	- PastML output, marginal states, of AMR features. Tip features were considered present when there was a >90% overlap with reference sequence -- this is different than original analysis, but doesn't change too many of them, especially with marginal probabilities. The AMR features were then divided into functional groups based on AMR type and phenotype implications.  All features with >95% co-correlation were then further grouped. Ancestral states were then inferred with PastML.
		- see **other/final_feature_info.csv, other/feature_info.html**
- **sampling_mask.csv**
	- *data_new/sampling_mask.csv*
	- Specifies, for each phylogeny piece, for each interval, whether or not it was possible that it would be sampled during that interval (whether its bioproject was active during those times). For internal branches, these are marginal probabilities based on the PastML bioproject reconstruction.

## other
- **pastml_ancestral_clade.csv**
	- I don't explicitly use these as features in the model, but do post-hoc analysis with them.
- **final_feature_info.csv** and **feature_info.html**
	- Defining what the shortened group names mean, what functional AMR group they are in, total count and percent, resistance conferred, etc.
- **bioproject_times.csv**
	- Specifies starting and ending estimated sampling window of bioprojects based on when first and last specimens were collected (true_min_time and true_max_time). These are adjusted slightly for use such that we allow for sampling a little before and a little after (min_time and max_time).
- **fit_model_params.json** and **config.yaml**
	- Input into the model fitting code. I don't think this will be super helpful, but wanted to include. fit_model_params is the current version, I think config.yaml was older. There are other json files, like one called *brownian_fit_setup.json* that do the rest of the setup.