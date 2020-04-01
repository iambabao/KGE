#!/usr/bin/env bash

python main.py \
  --task FB13 \
  --model TransE \
  --do_test \
  --batch 128 \
  --entity_em_size 200 \
  --relation_em_size 200 \
  --optimizer SGD
