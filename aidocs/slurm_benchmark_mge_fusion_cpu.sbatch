#!/bin/bash
# Submits the MGE quadrature-fusion benchmark as a non-interactive SLURM
# batch job on a VSC-5 CPU node -- see aidocs/benchmark_mge_fusion.py's own
# module docstring for what's being measured. Uses the same
# partition/QOS as the interactive `vsc5-attach-compute-cpu` profile
# (~/bin/vsc5-attach-compute-cpu on the login node): zen3_0512, 8 cores,
# 32GB, one node.
#
# Usage (from the login node, in a checkout of this repo):
#   sbatch aidocs/slurm_benchmark_mge_fusion_cpu.sbatch
#
# Check progress with `squeue -u $USER` and `tail -f
# aidocs/slurm-mge-bench-cpu-<jobid>.out`; the CSV itself is flushed after
# every row, so it's readable while the job is still running.

#SBATCH --job-name=mge-bench-cpu
#SBATCH --partition=zen3_0512
#SBATCH --qos=p71867_0512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=aidocs/slurm-mge-bench-cpu-%j.out

set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="aidocs/benchmark_mge_fusion_cpu_${SLURM_JOB_NODELIST}_${TIMESTAMP}.csv"

uv run python aidocs/benchmark_mge_fusion.py \
    --n 5 25 125 \
    --batch 1 10 100 1000 10000 100000 1000000 \
    --repeats 5 \
    --output "$OUTPUT"

echo "Wrote $OUTPUT"
