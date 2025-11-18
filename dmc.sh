#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=italo.carneiro@fisica.ufc.br
#SBATCH --job-name=dmc
#SBATCH --output=dmc.out
#SBATCH --error=dmc.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=35

g++ -std=c++17 -fopenmp -g -o dmc main.cpp dmc.cpp

./dmc

python3 plot.py