#!/bin/bash
#SBATCH --job-name=ML_Inference
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00

#SBATCH --output=/monfs01/projects/ys68/XRD_ML/monash_HPC_setup/run_monARCH_GPU/slurm_outputs/SLURM%j.out

# Load necessary modules
module load cuda

nvidia-smi
deviceQuery

srun python /monfs01/projects/ys68/XRD_ML/src/inference/inference.py