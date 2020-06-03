#!/usr/bin/env bash
source ~/.bashrc
conda activate torch1.4

python convert_ann.py --out_dir_path $1 --clip_length $2
