#!/bin/bash
# Submits the MGE quadrature-fusion benchmark as a non-interactive SLURM
# batch job on a VSC-5 CPU node -- see aidocs/benchmark_mge_fusion.py's own
# module docstring for what's being measured. Same partition/QOS as the
# interactive `vsc5-attach-compute-cpu` profile
# (~/bin/vsc5-attach-compute-cpu on the login node), zen3_0512 -- but
# --exclusive for the whole node's 256 cores/500GB (`sinfo -p zen3_0512`),
# not that profile's 8 cores/32GB. That profile is sized for a lightweight
# interactive dev session; this benchmark's whole point is measuring how
# each implementation's runtime scales with available parallelism (the
# original investigation found composite's per-component dispatch overhead
# specifically gets worse as core count rises), so it deliberately wants
# everything the node has, not a capped slice of it. JAX's CPU backend
# picks up however many cores the OS reports as available on its own --
# nothing here needs to tell it to use more.
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
#SBATCH --exclusive
#SBATCH --time=3:00:00
#SBATCH --output=aidocs/slurm-mge-bench-cpu-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="aidocs/benchmark_mge_fusion_cpu_${SLURM_JOB_NODELIST}_${TIMESTAMP}.csv"

uv run python aidocs/benchmark_mge_fusion.py \
    --n 5 25 125 \
    --batch 1 10 100 1000 10000 100000 1000000 \
    --repeats 5 \
    --output "$OUTPUT"

echo "Wrote $OUTPUT"
