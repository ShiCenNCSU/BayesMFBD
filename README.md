# MFBD-Bayes

MFBD-Bayes contains JAX implementations of likelihood calculations for
multi-type birth-death phylodynamic models on phylogenetic trees. The project
includes model variants for feature-dependent fitness effects, random branch
effects, global epistasis, and an E. coli ST131 analysis workflow.

This repository is a cleaned release copy of the research workspace. Large
generated HMC output directories, local IDE files, Python caches, and TeX build
intermediates are intentionally excluded.

## Repository Contents

- `TreeLike*.py`, `HMC*.py`, `SCARLikelihood.py`: model and inference code.
- `TreeUtils.py`, `VectorizedTree.py`: tree preprocessing utilities.
- `st131-data/`: input data used by the ST131 example workflow.
- `test-sets/`: compact simulated data sets used by development scripts.
- `test_basic_grad.py`, `test_loop_grad.py`, `test_decoupling.py`: small
  numerical checks and examples.
- `tex/`: derivation notes and rendered PDFs for MTBD approximations.

## Installation

The recommended setup uses Conda:

```bash
conda env create -f environment.yml
conda activate MFBD-Bayes
```

JAX installation can depend on your platform and accelerator. If Conda cannot
resolve `jax` or `jaxlib` for your machine, install the environment first and
then follow the official JAX installation instructions for your platform.

## Basic Checks

Run the lightweight Python checks from the repository root:

```bash
python test_basic_grad.py
python test_loop_grad.py
python test_decoupling.py
```

The full HMC scripts are compute-intensive and write posterior samples and
summaries into `ST131_HMC_*` output directories, which are ignored by Git.

## ST131 Workflow

The ST131 scripts expect input files under `st131-data/input/`. For example:

```bash
python HMCST131.py
```

## Release Notes

Before publishing publicly, choose and add a license file. Without a license,
GitHub users can view the source but do not receive explicit reuse rights.
