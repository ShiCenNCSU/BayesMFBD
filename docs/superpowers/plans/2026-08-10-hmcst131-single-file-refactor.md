# HMCST131 Single-File Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `HMCST131.py` into a clearer single-file script while preserving the runnable entry point.

**Architecture:** Keep all logic in `HMCST131.py`, but separate configuration, helpers, model code, data loading, inference, output writing, and `main()`. Replace implicit module globals with dataclass config values and closure-local values.

**Tech Stack:** Python 3.10, JAX, NumPy, pandas, ete3, NumPyro.

## Global Constraints

- Keep `HMCST131.py` as a single file.
- Do not create new runtime modules.
- Preserve `python HMCST131.py` as the default entry point.
- Keep default ST131 input path `st131-data/input`.
- Keep default output directory `ST131_HMC_fitEffects_all`.
- Do not run the expensive HMC workflow during verification.

---

### Task 1: Add Configuration And Feature-Name Helpers

**Files:**
- Modify: `HMCST131.py`

**Interfaces:**
- Produces: `DEFAULT_QRDR_FEATURES`, `ST131Config`, `feature_names_with_interactions()`, `build_default_config()`.

- [ ] **Step 1: Add dataclass imports and constants**

Add `from dataclasses import dataclass` and define `DEFAULT_QRDR_FEATURES` as a tuple of the current QRDR feature strings.

- [ ] **Step 2: Add `ST131Config`**

Create a dataclass with paths, time settings, model parameters, feature names, interaction flags, spline count, HMC warmup/sample counts, output path, and random seed.

- [ ] **Step 3: Add derived helpers**

Add `feature_names_with_interactions(feature_names, selected_features=None, include_two_way=True, include_three_way=False)` and `build_default_config()`.

### Task 2: Remove Implicit Feature Globals From `MTBD`

**Files:**
- Modify: `HMCST131.py`

**Interfaces:**
- Consumes: `ST131Config` fields passed through `params`.
- Produces: `MTBD.build()` that uses instance feature metadata instead of global `QRDR_features`.

- [ ] **Step 1: Store feature metadata in `MTBD.__init__`**

Read `feature_names`, `interaction_features`, `include_two_way_interactions`, `include_three_way_interactions`, and `n_splines` from `params`.

- [ ] **Step 2: Update `MTBD.build()`**

Replace direct references to `QRDR_features` with `self.feature_names` and interaction flags.

- [ ] **Step 3: Update spline code**

Replace the global `b` reference in `_calc_fitness()` with `self.n_splines`.

### Task 3: Extract Main Workflow Functions

**Files:**
- Modify: `HMCST131.py`

**Interfaces:**
- Consumes: `ST131Config`, `MTBD`.
- Produces: `load_st131_data()`, `build_model()`, `make_hmc_model()`, `make_initial_params()`, `run_hmc()`, `write_hmc_outputs()`, and `main()`.

- [ ] **Step 1: Move data loading into `load_st131_data(config)`**

Load the tree, feature dictionary, and sampling dictionary from configured paths.

- [ ] **Step 2: Move model initialization into `build_model(config, input_tree, features_dic, sampling_dic)`**

Create `MTBD`, call `build()`, set branch effects and parent effects, and return `(model, tree, n_branch_effects)`.

- [ ] **Step 3: Convert `HMC_model()` into a factory**

Implement `make_hmc_model(model, n_fit_features, n_branch_effects, n_splines)` returning an inner NumPyro model function.

- [ ] **Step 4: Move initial values and HMC execution into helpers**

Create `make_initial_params()` and `run_hmc()`.

- [ ] **Step 5: Move CSV writing into `write_hmc_outputs()`**

Use derived fit-effect names and branch-effect indexes to label outputs.

- [ ] **Step 6: Replace the script block with `main()`**

Keep only `if __name__ == "__main__": main()`.

### Task 4: Verify

**Files:**
- Modify: `HMCST131.py`

**Interfaces:**
- Consumes: the refactored single-file script.
- Produces: syntax-validated source file.

- [ ] **Step 1: Compile the refactored file**

Run:

```bash
mkdir -p .pycache_check
PYTHONPYCACHEPREFIX=.pycache_check python3 -m py_compile HMCST131.py
rm -rf .pycache_check
```

Expected: command exits with status 0.

- [ ] **Step 2: Scan for stale globals**

Run:

```bash
rg -n "QRDR_features|n_branch_effects|n_fit_features|^def HMC_model|^if __name__" HMCST131.py
```

Expected: no stale global `QRDR_features`, `n_branch_effects`, or `n_fit_features`; the entry point remains.
