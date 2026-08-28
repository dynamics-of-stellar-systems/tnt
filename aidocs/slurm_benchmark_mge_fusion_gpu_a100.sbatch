#!/bin/bash
# Submits the MGE quadrature-fusion benchmark as a non-interactive SLURM
# batch job on a VSC-5 A100 GPU node -- see
# aidocs/benchmark_mge_fusion.py's own module docstring for what's being
# measured, and slurm_benchmark_mge_fusion_gpu_a40.sbatch for the A40
# variant. No interactive-session profile existed for this one (unlike
# the A40/CPU profiles under ~/bin/) -- partition/QOS confirmed available
# to this project directly via `sinfo` and
# `sacctmgr show qos | grep p71867`: zen3_0512_a100x2, a full node (2x
# A100).
#
# Usage (from the login node, in a checkout of this repo):
#   sbatch aidocs/slurm_benchmark_mge_fusion_gpu_a100.sbatch
#
# Check progress with `squeue -u $USER` and `tail -f
# aidocs/slurm-mge-bench-gpu-a100-<jobid>.out`; the CSV itself is flushed
# after every row, so it's readable while the job is still running. Same
# batch-size range as the A40 script (up to 10,000,000) for direct
# comparability between the two GPUs; array-fused's OOM at large N/batch
# is caught and recorded per-row rather than crashing the whole sweep, so
# this doesn't need adjusting down just because A100/A40 may have
# different memory headroom.

#SBATCH --job-name=mge-bench-gpu-a100
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=p71867_a100dual
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --time=3:00:00
#SBATCH --output=aidocs/slurm-mge-bench-gpu-a100-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="aidocs/benchmark_mge_fusion_gpu_a100_${SLURM_JOB_NODELIST}_${TIMESTAMP}.csv"

uv run python aidocs/benchmark_mge_fusion.py \
    --n 5 25 125 \
    --batch 1 10 100 1000 10000 100000 1000000 10000000 \
    --repeats 5 \
    --output "$OUTPUT"

echo "Wrote $OUTPUT"
