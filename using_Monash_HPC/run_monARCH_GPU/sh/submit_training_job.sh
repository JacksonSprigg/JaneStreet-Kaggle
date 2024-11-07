#!/bin/bash
#SBATCH --job-name=ML_Training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00

#SBATCH --output=/monfs01/projects/ys68/JaneStreet-Kaggle/using_Monash_HPC/run_monARCH_GPU/slurm_outputs/SLURM%j.out

# Load necessary modules
module load cuda

nvidia-smi

# Set environment variables for LightGBM GPU
export CUDA_VISIBLE_DEVICES=0,1

srun python /monfs01/projects/ys68/JaneStreet-Kaggle/training/run.py