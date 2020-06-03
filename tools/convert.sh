#!/usr/bin/env bash
#OAR -p gpu='NO' and host='nef053.inria.fr'
#OAR -l /nodes=1,walltime=24
#OAR --notify mail:juan-diego.gonzales-zuniga@inria.fr

conda activate torcheccv
python convert_ann.py --out_dir_path /gpfsscratch/rech/qdh/usk17na/Datasets/JTA-Dataset/anns_4 --clip_length 4
python convert_ann.py --out_dir_path /gpfsscratch/rech/qdh/usk17na/Datasets/JTA-Dataset/anns_8 --clip_length 8
python convert_ann.py --out_dir_path /gpfsscratch/rech/qdh/usk17na/Datasets/JTA-Dataset/anns_16 --clip_length 16

