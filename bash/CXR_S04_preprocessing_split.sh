#!/bin/bash
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu
#SBATCH --error=../eo/S04.err
#SBATCH --output=../eo/S04.out
#SBATCH --job-name=S04.job
#SBATCH --mem-per-cpu=1G
#SBATCH -t 120
#SBATCH -c 1
python ../scripts/CXR_S04_preprocessing_split.py
