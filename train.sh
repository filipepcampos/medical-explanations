#!/bin/bash
#
#SBATCH --partition=gpu_min8gb    # Debug partition
#SBATCH --qos=gpu_min8gb          # Debug QoS level
#SBATCH --job-name=train_explanations     # Job name
#SBATCH -o /nas-ctm01/homes/fpcampos/slurm_logs/DPDM/%x_%j_%N.out  # File containing STDOUT output
#SBATCH -e /nas-ctm01/homes/fpcampos/slurm_logs/DPDM/%x_%j_%N.err  # File containing STDERR output

echo "Job started"

python3 src/train_synthetic.py