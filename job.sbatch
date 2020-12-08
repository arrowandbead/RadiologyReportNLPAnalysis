#!/bin/bash

#SBATCH --time=30:00

# Use 4 nodes with 48 tasks each, for 80 MPI tasks:
#SBATCH --nodes=1
#SBATCH --tasks-per-node=48
#SBATCH --cpus-per-task=1

# Specify a job name:
#SBATCH -J RadiologyReportNLPAnalysis

#SBATCH --output=nlp_%j.log

# Load Modules
module load python/3.7.4
source /users/psaluja/scratch/RadiologyReportNLPAnalysis/environment/bin/activate

# Fetch System Details
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""

#SBATCH --mail-type BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user 5803022063@tmomail.net

# Fire up tests
cd /users/psaluja/scratch/RadiologyReportNLPAnalysis/code

python biobert_model.py

