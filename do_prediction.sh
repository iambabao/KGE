#!/usr/bin/env bash

python prediction.py \
  --task FB13 \
  --model TransE \
  --top_k 10 \
  --entity_em_size 200 \
  --relation_em_size 200
