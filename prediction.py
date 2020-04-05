# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/3/29 23:41
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
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--model_file', type=str)
args = parser.parse_args()

config = Config('.', args.task, args.model,
                top_k=args.top_k,
                entity_em_size=args.entity_em_size, relation_em_size=args.relation_em_size,
                optimizer=args.optimizer)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True


def link_prediction(sess, model, test_data, all_triples, side, verbose=True):
    step = 0
    rank_raw = 0
    rank_filter = 0
    hits_k_raw = 0
    hits_k_filter = 0
    for pos_s, pos_p, pos_o in zip(*test_data):
        if side == 'right':
            pos_s_batch = [pos_s] * len(config.id_2_entity)
            pos_p_batch = [pos_p] * len(config.id_2_entity)
            pos_o_batch = list(config.id_2_entity.keys())
        else:
            pos_s_batch = list(config.id_2_entity.keys())
            pos_p_batch = [pos_p] * len(config.id_2_entity)
            pos_o_batch = [pos_o] * len(config.id_2_entity)

        distance = sess.run(
            model.pos_dis,
            feed_dict={
                model.batch_size: len(pos_s_batch),
                model.pos_s: pos_s_batch,
                model.pos_p: pos_p_batch,
                model.pos_o: pos_o_batch,
                model.training: False
            }
        )
        if side == 'right':
            predicted = sorted([(i, j) for i, j in zip(pos_o_batch, distance.tolist())], key=itemgetter(1))
        else:
            predicted = sorted([(i, j) for i, j in zip(pos_s_batch, distance.tolist())], key=itemgetter(1))

        skip = 0
        for i in range(len(predicted)):
            current_entity, current_distance = predicted[i]
            if (side == 'right' and current_entity == pos_o) or (side == 'left' and current_entity == pos_s):
                rank_raw += i + 1
                rank_filter += i + 1 - skip
                if i < config.top_k:
                    hits_k_raw += 1
                if i - skip < config.top_k:
                    hits_k_filter += 1
                break
            if side == 'right' and (pos_s, pos_p, current_entity) in all_triples:
                skip += 1
            elif side == 'left' and (current_entity, pos_p, pos_s) in all_triples:
                skip += 1

        step += 1
        if verbose:
            print('\rprocessing {} side: {:>6d}'.format(side, step + 1), end='')
    print()

    return rank_raw / step, rank_filter / step, hits_k_raw / step, hits_k_filter / step


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
    train_data = data_reader.read_train_data()
    valid_data = data_reader.read_valid_data()
    test_data = data_reader.read_test_data()
    all_triples = set()
    for sid, pid, oid in zip(*train_data):
        all_triples.add((sid, pid, oid))
    for sid, pid, oid in zip(*valid_data):
        all_triples.add((sid, pid, oid))
    for sid, pid, oid in zip(*test_data):
        all_triples.add((sid, pid, oid))

    with tf.Session(config=sess_config) as sess:
        model_file = args.model_file
        if model_file is None:
            model_file = tf.train.latest_checkpoint(os.path.join(config.result_dir, config.task_name, config.current_model))
        if model_file is not None:
            print('loading model from {}...'.format(model_file))
            saver.restore(sess, model_file)

            right_score = link_prediction(sess, model, test_data, all_triples, side='right', verbose=True)
            left_score = link_prediction(sess, model, test_data, all_triples, side='left', verbose=True)
            hits_score = {
                'mean_rank_raw_right': right_score[0],
                'mean_rank_filter_right': right_score[1],
                'hits@{}_raw_right'.format(config.top_k): right_score[2],
                'hits@{}_filter_right'.format(config.top_k): right_score[3],
                'mean_rank_raw_left': left_score[0],
                'mean_rank_filter_left': left_score[1],
                'hits@{}_raw_left'.format(config.top_k): left_score[2],
                'hits@{}_filter_left'.format(config.top_k): left_score[3]
            }
            for k, v in hits_score.items():
                print('{}: {}'.format(k, v))

            save_json(hits_score, os.path.join(config.result_dir, config.task_name, config.current_model, 'prediction.json'))
        else:
            print('model not found!')


if __name__ == '__main__':
    main()
    print('done')
