#!/bin/bash
#
#SBATCH --partition=gpu_min8gb   # Debug partition
#SBATCH --qos=gpu_min8gb         # Debug QoS level
#SBATCH --job-name=fl_train     # Job name
#SBATCH -o %x_%j_%N.out  # File containing STDOUT output
#SBATCH -e %x_%j_%N.err  # File containing STDERR output

echo "Job started"

python3 src/sim.py
