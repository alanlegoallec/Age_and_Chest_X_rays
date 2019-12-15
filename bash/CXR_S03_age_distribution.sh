#!/bin/bash
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu
#SBATCH --error=../eo/S03.err
#SBATCH --output=../eo/S03.out
#SBATCH --job-name=S03.job
#SBATCH --mem-per-cpu=1G
#SBATCH -t 20
#SBATCH -c 1
python ../scripts/CRX_S03_age_distribution.py
