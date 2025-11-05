#!/usr/bin/env bash
set -e

python src/train_lm.py   --data_path data/tiny_shakespeare_sample.txt   --out_dir results/lm_base   --seed 3407   --batch_size 64   --block_size 256   --d_model 256   --n_head 4   --n_layer 4   --ffn_hidden 1024   --lr 3e-4   --warmup_steps 200   --max_steps 2500
