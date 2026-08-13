# HMCST131 Single-File Refactor Design

## Goal

Refactor `HMCST131.py` into a clearer single-file script without changing the
public entry point or splitting code into new modules.

## Scope

The refactor keeps the MTBD likelihood implementation, spline helpers, ST131
data loading, NumPyro model construction, HMC execution, and CSV export in
`HMCST131.py`. The file remains runnable with:

```bash
python HMCST131.py
```

## Design

The file will be organized into explicit sections:

- Imports and global runtime configuration.
- Default ST131 constants.
- `ST131Config`, a dataclass for paths, feature choices, model constants, HMC
  settings, and output behavior.
- Feature and spline helper functions.
- The `MTBD` model class.
- Data-loading and model-building functions.
- NumPyro HMC model factory.
- Output-writing helpers.
- `main()`.

The current implicit globals (`model`, `QRDR_features`, `b`,
`n_branch_effects`, `n_fit_features`) will be replaced by config values or
closure-local values where practical. `MTBD` will receive the feature names and
interaction settings through its params so `build()` no longer depends on a
module-level `QRDR_features` variable.

## Behavior

Default behavior should stay aligned with the current script: load ST131 inputs
from `st131-data/input`, run NUTS, and write HMC CSV outputs to
`ST131_HMC_fitEffects_all`. The refactor will also derive the fit-effect count
and output column names from the configured features and interaction settings,
instead of maintaining those values by hand in multiple places.

## Verification

Because the active Python environment does not have project dependencies
installed, verification will use syntax compilation with
`PYTHONPYCACHEPREFIX` redirected inside the workspace. If dependencies are
available later, the next runtime check is `python HMCST131.py` in the Conda
environment.
