#!/usr/bin/env bash
#abc#SBATCH --job-name=CONVERT        # nom du job
#SBATCH --partition=gpu_p2       # demande d'allocation sur la partition GPU
#SBATCH --ntasks=3                  # nombre de tâche (un unique processus ici)
#SBATCH --nodes=3
#SBATCH --gres=gpu:8                # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --cpus-per-task=24          # nombre de coeurs à réserver (un quart du noeud)
#SBATCH --time=20:00:00             # temps exécution maximum demande (HH:MM:SS)
#SBATCH -A qsb@gpu
source ~/.bashrc

srun   -n 1 -N 1 ./run_step.sh /gpfsscratch/rech/qdh/usk17na/Datasets/JTA-Dataset/anns_4 4 &
srun   -n 1 -N 1 ./run_step.sh /gpfsscratch/rech/qdh/usk17na/Datasets/JTA-Dataset/anns_8 8 &
srun   -n 1 -N 1 ./run_step.sh /gpfsscratch/rech/qdh/usk17na/Datasets/JTA-Dataset/anns_16 16 &

wait
