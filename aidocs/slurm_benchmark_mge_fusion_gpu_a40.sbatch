#!/bin/bash
# Submits the MGE quadrature-fusion benchmark as a non-interactive SLURM
# batch job on a VSC-5 A40 GPU node -- see aidocs/benchmark_mge_fusion.py's
# own module docstring for what's being measured. Uses the same
# partition/QOS as the interactive `vsc5-attach-compute-gpu` profile
# (~/bin/vsc5-attach-compute-gpu on the login node): zen2_0256_a40x2, a
# full node (2x A40). See slurm_benchmark_mge_fusion_gpu_a100.sbatch for
# the A100 variant -- only one GPU generation had been benchmarked before
# that was added; the two aren't assumed to behave the same.
#
# Usage (from the login node, in a checkout of this repo):
#   sbatch aidocs/slurm_benchmark_mge_fusion_gpu_a40.sbatch
#
# Check progress with `squeue -u $USER` and `tail -f
# aidocs/slurm-mge-bench-gpu-a40-<jobid>.out`; the CSV itself is flushed
# after every row, so it's readable while the job is still running. Batch
# sizes go up to 10,000,000 here (vs. 1,000,000 on the CPU job) --
# array-fused's (nodes, N, batch) tensor is the most likely to hit a real
# memory ceiling at that size; the script catches JAX/XLA's own OOM error
# and records it rather than crashing the whole sweep, so
# composite/scan-fused's rows at the same batch size are unaffected either
# way.

#SBATCH --job-name=mge-bench-gpu-a40
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=p71867_a40dual
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --time=3:00:00
#SBATCH --output=aidocs/slurm-mge-bench-gpu-a40-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="aidocs/benchmark_mge_fusion_gpu_a40_${SLURM_JOB_NODELIST}_${TIMESTAMP}.csv"

uv run python aidocs/benchmark_mge_fusion.py \
    --n 5 25 125 \
    --batch 1 10 100 1000 10000 100000 1000000 10000000 \
    --repeats 5 \
    --output "$OUTPUT"

echo "Wrote $OUTPUT"
