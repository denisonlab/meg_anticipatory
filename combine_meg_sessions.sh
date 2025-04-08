#!/bin/bash -l

# Set SCC project
#$ -P rdenlab

# Request resources
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -l h_rt=4:00:00

# Name the job
#$ -N combine_meg

# Combine output and error files
#$ -j y

# Specify output file
#$ -o ./combine_meg.log


# Print job info
echo "=========================================================="
echo "Starting MEG data combination job at $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $HOSTNAME"
echo "Current directory: $PWD"
echo "=========================================================="

# Set up environment
export PYTHONPATH="/projectnb/rdenlab/Users/Lizzy:$PYTHONPATH"
PYTHON_PATH="/projectnb/viscog01/lizzyjoo/.conda/envs/mne/bin/python"

echo "Using Python interpreter: $PYTHON_PATH"

# Run the Python script to combine results
$PYTHON_PATH /projectnb/rdenlab/Users/Lizzy/combine_meg_sessions.py

echo "=========================================================="
echo "MEG data combination completed at $(date)"
echo "=========================================================="

# Empty line at end of file (as recommended in the BU guide)