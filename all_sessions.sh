#!/bin/bash -l

# Request resources
#$ -pe omp 8
#$ -l h_rt=24:00:00
#$ -l mem_per_core=4G
#$ -N meg_analysis
#$ -j y
#$ -o meg_output.log
#$ -cwd

# Print job info
echo "=========================================="
echo "Starting MEG analysis job at $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $HOSTNAME"
echo "Current directory: $PWD"
echo "=========================================="

# Set up environment - specify the correct path to your conda environment
CONDA_PATH="/projectnb/viscog01/lizzyjoo/.conda"
CONDA_ENV="mne"

# Add custom modules to Python path
export PYTHONPATH="/projectnb/rdenlab/Users/Lizzy:$PYTHONPATH"

# Define Python interpreter path
PYTHON_PATH="${CONDA_PATH}/envs/${CONDA_ENV}/bin/python"

echo "Using Python: $PYTHON_PATH"
echo "PYTHONPATH: $PYTHONPATH"

# Run the Python script
echo "Running all_sessions.py..."
$PYTHON_PATH /projectnb/rdenlab/Users/Lizzy/all_sessions.py

# Print completion message
echo "=========================================="
echo "MEG analysis job completed at $(date)"
echo "=========================================="