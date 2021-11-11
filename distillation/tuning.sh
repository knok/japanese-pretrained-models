#!/bin/sh

python gpt2-train.py \
    --student_type gpt2 \
    --student_config distilgpt2-ja.json \
    --teacher_type gpt2 \
    --teacher_name rinna/japanese-gpt2-medium \
    --alpha_ce 5.0 --alpha_cos 1.0  \
    --freeze_pos_embs \
    --dump_path data/distilgpt2-e3 \
    --n_epoch 3 \
    --force
  
