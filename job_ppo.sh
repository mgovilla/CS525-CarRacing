#!/bin/bash

#SBATCH --mem 16G
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -C A100
#SBATCH --job-name="trainppo"

python train_ppo.py