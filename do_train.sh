#!/usr/bin/env bash

python main.py \
  --task FB13 \
  --model TransE \
  --do_train \
  --do_test \
  --epoch 100 \
  --batch 128 \
  --margin 1.0 \
  --entity_em_size 200 \
  --relation_em_size 200 \
  --optimizer Adam \
  --lr 1e-3 \
  --pre_train_epochs 20 \
  --early_stop 5 \
  --early_stop_delta 0.001
