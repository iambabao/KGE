# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/3/29 18:07
@Desc       : 
"""

import os
import argparse
import tensorflow as tf
from operator import itemgetter

from src.config import Config
from src.data_reader import DataReader
from src.model import get_model
from src.utils import read_json_dict, save_json

parser = argparse.ArgumentParser()
parser.add_argument('--task', '-t', type=str, required=True)
parser.add_argument('--model', '-m', type=str, required=True)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--entity_em_size', type=int, default=200)
parser.add_argument('--relation_em_size', type=int, default=200)
parser.add_argument('--model_file', type=str)
args = parser.parse_args()

config = Config('.', args.task, args.model,
                top_k=args.top_k,
                entity_em_size=args.entity_em_size, relation_em_size=args.relation_em_size)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True


def link_prediction(sess, model, test_data, side, verbose=True):
    step = 0
    hits_k = 0
    for sid, pid, oid in zip(*test_data):
        if side == 'left':
            sid, pid, oid = oid, pid ,sid
        sid_batch = [sid] * len(config.id_2_entity)
        pid_batch = [pid] * len(config.id_2_entity)
        oid_batch = list(config.id_2_entity)
        distance = sess.run(
            model.pos_dis,
            feed_dict={
                model.batch_size: len(sid_batch),
                model.sid: sid_batch,
                model.pid: pid_batch,
                model.oid: oid_batch,
                model.training: False
            }
        )
        top_k = dict(sorted([(i, j) for i, j in zip(oid_batch, distance.tolist())], key=itemgetter(1))[:config.top_k])
        if oid in top_k.keys():
            hits_k += 1

        step += 1
        if verbose:
            print('\rprocessing {} side: {:>6d}'.format(side, step + 1), end='')
    print()

    return hits_k / step


def main():
    print('preparing data...')
    config.entity_2_id, config.id_2_entity = read_json_dict(config.entity_dict)
    config.relation_2_id, config.id_2_relation = read_json_dict(config.relation_dict)
    config.num_entity = len(config.entity_2_id)
    config.num_relation = len(config.relation_2_id)

    data_reader = DataReader(config)

    print('building model...')
    model = get_model(config)
    saver = tf.train.Saver(max_to_keep=10)

    print('loading data...')
    test_data = data_reader.read_test_data_wo_negative_sampling()

    with tf.Session(config=sess_config) as sess:
        model_file = args.model_file
        if model_file is None:
            model_file = tf.train.latest_checkpoint(os.path.join(config.result_dir, config.task_name, config.current_model))
        if model_file is not None:
            print('loading model from {}...'.format(model_file))
            saver.restore(sess, model_file)

            right_score = link_prediction(sess, model, test_data, side='right', verbose=True)
            left_score = link_prediction(sess, model, test_data, side='left', verbose=True)
            hits_score = {
                'hits@{}_right'.format(config.top_k): right_score,
                'hits@{}_left'.format(config.top_k): left_score
            }
            print(hits_score)

            save_json(hits_score, os.path.join(config.result_dir, config.task_name, config.current_model, 'prediction.json'))
        else:
            print('model not found!')


if __name__ == '__main__':
    main()
    print('done')
