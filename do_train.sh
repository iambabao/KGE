#!/usr/bin/env bash

python main.py \
  --task FB13 \
  --model TransE \
  --do_train \
  --do_eval \
  --do_test \
  --epoch 500 \
  --batch 128 \
  --margin 4.0 \
  --entity_em_size 200 \
  --relation_em_size 200 \
  --optimizer SGD \
  --lr 0.5 \
  --pre_train_epochs 100 \
  --early_stop 0 \
  --early_stop_delta 0.01
