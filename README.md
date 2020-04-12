# KGE
Practice on knowledge graph embedding

This repo is implemented with TensorFlow 1.14

## Data Preparation
You can download data from [OpenKE](https://github.com/thunlp/OpenKE), put `benchmarks` in `data` and run:
```shell script
python preprocess.py
```

## How to use
You can run `do_*.sh` to train, evaluate and test models.

`do_classification.sh` run the triple classification task.

`do_prediction.sh` run the link prediction task.
