#!/bin/bash -l
# Submit from the PhyloJAX directory: sbatch HMCST131Candidates.sbatch
# Each task fits one candidate; IDs 0/1 are linear reduced/full, then
# 2/3 unconstrained, 4/5 monotone, 6/7 unimodal, and 8/9 GP.
# Override resources with sbatch flags and settings with environment variables.
# Defaults: four sequential chains, 2,000 warmup + 5,000 retained draws per
# chain, target acceptance 0.95, and maximum tree depth 12.
# The runner saves diagnostics and pointwise log_lik for WAIC in each archive.
# R2D2 example: sbatch --array=9 --export=ALL,PRIOR=R2D2 HMCST131Candidates.sbatch \
#     --r2d2-mean-r2 0.2 --r2d2-precision 5 --r2d2-scale 1
# Soft spike-and-slab: sbatch --array=9 --export=ALL,PRIOR=SpikeSlab HMCST131Candidates.sbatch \
#     --spike-slab-inclusion-prob 0.1 --spike-slab-spike-scale 0.1 --spike-slab-scale 1
# Learn shared p: sbatch --array=9 --export=ALL,PRIOR=SpikeSlab HMCST131Candidates.sbatch \
#     --spike-slab-learn-inclusion-prob --spike-slab-p-alpha 1 --spike-slab-p-beta 1
# Spike/slab SDs apply equally to raw mutation and interaction coefficients.
# Copy R2D2Prior.py and SpikeSlabPrior.py alongside the updated runner.

#SBATCH --job-name=st131_candidates
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9%5
#SBATCH --output=st131_candidates_%A_%a.out
#SBATCH --error=st131_candidates_%A_%a.err
##SBATCH --account=your_account
##SBATCH --partition=your_partition

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ ! "${TASK_ID}" =~ ^[0-9]$ ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID must be in 0-9; submit this script with sbatch." >&2
    exit 2
fi
FAMILIES=(linear bspline_unconstrained bspline_monotone bspline_unimodal gp_monotone)
FEATURE_SETS=(reduced full)
FAMILY="${FAMILIES[TASK_ID / 2]}"
FEATURE_SET="${FEATURE_SETS[TASK_ID % 2]}"

# These environment defaults match the existing NCSU Hazel job configuration.
# Set CONDA_MODULE='' if the Python environment needs no module.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-${PWD}}}"
CONDA_MODULE="${CONDA_MODULE-conda}"
CONDA_ENV="${CONDA_ENV:-/usr/local/usrapps/bjreich/scen2/phydy}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
cd "${PROJECT_DIR}"
PROJECT_DIR="$(pwd -P)"
if [[ ! -f "${PROJECT_DIR}/HMCST131Candidates.py" ]]; then
    echo "ERROR: Set PROJECT_DIR to the directory containing HMCST131Candidates.py." >&2
    exit 2
fi
if [[ -n "${CONDA_MODULE}" ]]; then
    if ! command -v module >/dev/null 2>&1; then
        echo "ERROR: module command unavailable; load your environment or set CONDA_MODULE=''." >&2
        exit 2
    fi
    module load "${CONDA_MODULE}"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python not executable: ${PYTHON_BIN}; set PYTHON_BIN or CONDA_ENV." >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

# A separate directory per array submission avoids replacing local results.
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/ST131_HMC_candidates_cluster/${SLURM_ARRAY_JOB_ID:-local}}"
NUM_WARMUP="${NUM_WARMUP:-2000}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
NUM_CHAINS="${NUM_CHAINS:-4}"
TARGET_ACCEPT="${TARGET_ACCEPT:-0.95}"
MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-12}"
COMMAND=(
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/HMCST131Candidates.py"
    --model "${FAMILY}" --feature-set "${FEATURE_SET}"
    --num-warmup "${NUM_WARMUP}" --num-samples "${NUM_SAMPLES}"
    --num-chains "${NUM_CHAINS}" --seed "${SEED:-0}"
    --target-accept "${TARGET_ACCEPT}" --max-tree-depth "${MAX_TREE_DEPTH}"
    --prior "${PRIOR:-Laplace}" --output-dir "${OUTPUT_DIR}"
    --no-progress
    "$@"
)
trap 'status=$?; if (( status != 0 )); then echo "ERROR: ${FAMILY}/${FEATURE_SET} failed with exit code ${status}" >&2; fi' EXIT

echo "[$(date)] Array ${SLURM_ARRAY_JOB_ID:-local}, task ${TASK_ID}: ${FAMILY}/${FEATURE_SET}"
echo "Project: ${PROJECT_DIR}; Python: ${PYTHON_BIN}; host: $(hostname)"
echo "Output: ${OUTPUT_DIR}/${FAMILY}_${FEATURE_SET}"
echo "Sampling: ${NUM_CHAINS} sequential chains; ${NUM_WARMUP} warmup and ${NUM_SAMPLES} retained draws per chain"
echo "NUTS: target acceptance ${TARGET_ACCEPT}; maximum tree depth ${MAX_TREE_DEPTH}"
printf 'Command: '
printf '%q ' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
echo "[$(date)] Finished ${FAMILY}/${FEATURE_SET}"
