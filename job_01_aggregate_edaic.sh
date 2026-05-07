#!/bin/bash -l
#SBATCH --job-name=edaic_aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=%x.o%j

# Step 1: Aggregate E-DAIC raw OpenFace frame-level CSVs → per-participant features.
# Run ONCE before any regression or cross-corpus scripts.

source /home/woody/empk/empk004h/software/private/myenv/bin/activate
SCRIPT_DIR="/home/hpc/empk/empk004h/depression-detection/KDD_paper"
srun python "${SCRIPT_DIR}/01_aggregate_edaic_video_features.py" \
    --min_confidence 0.5 \
    --min_frames 50
