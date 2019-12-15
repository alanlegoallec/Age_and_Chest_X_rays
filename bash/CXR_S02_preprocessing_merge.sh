#!/bin/bash
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu
#SBATCH --error=../eo/S02.err
#SBATCH --output=../eo/S02.out
#SBATCH --job-name=S02.job
#SBATCH --mem-per-cpu=1G
#SBATCH -t 60
#SBATCH -c 1
python ../scripts/CRX_S02_preprocessing_merge.py
