#!/bin/bash
#
#SBATCH --partition=cpu_7cores    # Debug partition
#SBATCH --qos=cpu_7cores          # Debug QoS level
#SBATCH --job-name=fl_train     # Job name
#SBATCH -o %x_%j_%N.out  # File containing STDOUT output
#SBATCH -e %x_%j_%N.err  # File containing STDERR output

echo "Job started"

python3 src/sim.py
