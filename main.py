# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/3/29 0:06
@Desc       : 
"""

import os
import time
import argparse
import tensorflow as tf

from src.config import Config
from src.data_reader import DataReader
from src.model import get_model
from src.utils import makedirs, read_json_dict, save_json, save_json_lines, print_title, make_batch_iter

parser = argparse.ArgumentParser()
parser.add_argument('--task', '-t', type=str, required=True)
parser.add_argument('--model', '-m', type=str, required=True)
parser.add_argument('--do_train', action='store_true', default=False)
parser.add_argument('--do_eval', action='store_true', default=False)
parser.add_argument('--do_test', action='store_true', default=False)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--margin', type=float, default=1.0)
parser.add_argument('--entity_em_size', type=int, default=200)
parser.add_argument('--relation_em_size', type=int, default=200)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--model_file', type=str)
parser.add_argument('--log_steps', type=int, default=100)
parser.add_argument('--save_steps', type=int, default=1000)
parser.add_argument('--pre_train_epochs', type=int, default=0)
parser.add_argument('--early_stop', type=int, default=0)
parser.add_argument('--early_stop_delta', type=float, default=0.00)
args = parser.parse_args()

config = Config('.', args.task, args.model,
                num_epoch=args.epoch, batch_size=args.batch, threshold=args.margin,
                entity_em_size=args.entity_em_size, relation_em_size=args.relation_em_size,
                optimizer=args.optimizer, lr=args.lr, dropout=args.dropout)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True


def save_result(result, result_file):
    pos_dis, neg_dis = result
    result = [{
        'pos_dis': pos,
        'neg_dis': neg
    } for pos, neg in zip(pos_dis, neg_dis)]

    save_json_lines(result, result_file)


def run_test(sess, model, test_data, verbose=True):
    pos_dis = []
    neg_dis = []
    batch_iter = make_batch_iter(list(zip(*test_data)), config.batch_size, shuffle=False, verbose=verbose)
    for step, batch in enumerate(batch_iter):
        sid, pid, oid, neg_sid, neg_pid, neg_oid = list(zip(*batch))

        _pos_dis, _neg_dis, loss, accuracy = sess.run(
            [model.pos_dis, model.neg_dis, model.loss, model.accuracy],
            feed_dict={
                model.batch_size: len(sid),
                model.sid: sid,
                model.pid: pid,
                model.oid: oid,
                model.neg_sid: neg_sid,
                model.neg_pid: neg_pid,
                model.neg_oid: neg_oid,
                model.training: False
            }
        )
        pos_dis.extend(_pos_dis.tolist())
        neg_dis.extend(_neg_dis.tolist())

        if verbose:
            print('\rprocessing batch: {:>6d}'.format(step + 1), end='')
    print()

    return pos_dis, neg_dis


def run_evaluate(sess, model, valid_data, valid_summary_writer=None, verbose=True):
    steps = 0
    pos_dis = []
    neg_dis = []
    total_loss = 0.0
    total_accuracy = 0.0
    batch_iter = make_batch_iter(list(zip(*valid_data)), config.batch_size, shuffle=False, verbose=verbose)
    for batch in batch_iter:
        sid, pid, oid, neg_sid, neg_pid, neg_oid = list(zip(*batch))

        _pos_dis, _neg_dis, loss, accuracy, global_step, summary = sess.run(
            [model.pos_dis, model.neg_dis, model.loss, model.accuracy, model.global_step, model.summary],
            feed_dict={
                model.batch_size: len(sid),
                model.sid: sid,
                model.pid: pid,
                model.oid: oid,
                model.neg_sid: neg_sid,
                model.neg_pid: neg_pid,
                model.neg_oid: neg_oid,
                model.training: False
            }
        )
        pos_dis.extend(_pos_dis.tolist())
        neg_dis.extend(_neg_dis.tolist())

        steps += 1
        total_loss += loss
        total_accuracy += accuracy
        if verbose:
            print('\rprocessing batch: {:>6d}'.format(steps + 1), end='')
        if steps % args.log_steps == 0 and valid_summary_writer is not None:
            valid_summary_writer.add_summary(summary, global_step)
    print()

    return (pos_dis, neg_dis), total_loss / steps, total_accuracy / steps


def run_train(sess, model, train_data, valid_data, saver,
              train_summary_writer=None, valid_summary_writer=None, verbose=True):
    flag = 0
    valid_log = 0.0
    best_valid_log = 0.0
    valid_log_history = {'loss': [], 'accuracy': [], 'global_step': []}
    global_step = 0
    for i in range(config.num_epoch):
        print_title('Train Epoch: {}'.format(i + 1))
        steps = 0
        total_loss = 0.0
        total_accuracy = 0.0
        batch_iter = make_batch_iter(list(zip(*train_data)), config.batch_size, shuffle=True, verbose=verbose)
        for batch in batch_iter:
            start_time = time.time()
            sid, pid, oid, neg_sid, neg_pid, neg_oid = list(zip(*batch))

            _, loss, accuracy, global_step, summary = sess.run(
                [model.train_op, model.loss, model.accuracy, model.global_step, model.summary],
                feed_dict={
                    model.batch_size: len(sid),
                    model.sid: sid,
                    model.pid: pid,
                    model.oid: oid,
                    model.neg_sid: neg_sid,
                    model.neg_pid: neg_pid,
                    model.neg_oid: neg_oid,
                    model.training: True
                }
            )

            steps += 1
            total_loss += loss
            total_accuracy += accuracy
            if verbose:
                print('\rafter {:>6d} batch(s), train loss is {:>.4f}, train accuracy is {:>.4f}, {:>.4f}s/batch'
                      .format(steps, loss, accuracy, time.time() - start_time), end='')
            if steps % args.log_steps == 0 and train_summary_writer is not None:
                train_summary_writer.add_summary(summary, global_step)
            if global_step % args.save_steps == 0:
                # evaluate saved models after pre-train epochs
                if i < args.pre_train_epochs:
                    saver.save(sess, config.model_file, global_step=global_step)
                else:
                    result, valid_loss, valid_accuracy = run_evaluate(
                        sess, model, valid_data, valid_summary_writer, verbose=False
                    )
                    print_title('Valid Result', sep='*')
                    print('average valid loss is {:>.4f}, average valid accuracy is {:>.4f}'.format(valid_loss, valid_accuracy))

                    print_title('Saving Results')
                    save_result(result, config.valid_result)

                    if valid_accuracy >= best_valid_log:
                        best_valid_log = valid_accuracy
                        saver.save(sess, config.model_file, global_step=global_step)

                    # early stop
                    if valid_accuracy - args.early_stop_delta >= valid_log:
                        flag = 0
                    elif flag < args.early_stop:
                        flag += 1
                    elif args.early_stop:
                        return valid_log_history

                    valid_log = valid_accuracy
                    valid_log_history['loss'].append(valid_loss)
                    valid_log_history['accuracy'].append(valid_accuracy)
                    valid_log_history['global_step'].append(int(global_step))
        print()
        print_title('Train Result')
        print('average train loss is {:>.4f}, average train accuracy is {:>.4f}'.format(total_loss / steps, total_accuracy / steps))
    saver.save(sess, config.model_file, global_step=global_step)

    return valid_log_history


def main():
    makedirs(config.temp_dir)
    makedirs(os.path.join(config.result_dir, config.task_name))
    makedirs(config.train_log_dir)
    makedirs(config.valid_log_dir)

    print('preparing data...')
    config.entity_2_id, config.id_2_entity = read_json_dict(config.entity_dict)
    config.relation_2_id, config.id_2_relation = read_json_dict(config.relation_dict)
    config.num_entity = len(config.entity_2_id)
    config.num_relation = len(config.relation_2_id)

    data_reader = DataReader(config)

    print('building model...')
    model = get_model(config)
    saver = tf.train.Saver(max_to_keep=10)

    if args.do_train:
        print('saving config...')
        save_json(config.to_dict(), os.path.join(config.result_dir, config.task_name, config.current_model, 'config.json'))

        print('loading data...')
        train_data = data_reader.read_train_data()
        valid_data = data_reader.read_valid_data()

        print_title('Trainable Variables')
        for v in tf.trainable_variables():
            print(v)

        print_title('Gradients')
        for g in model.gradients:
            print(g)

        with tf.Session(config=sess_config) as sess:
            model_file = args.model_file
            if model_file is None:
                model_file = tf.train.latest_checkpoint(config.result_dir)
            if model_file is not None:
                print('loading model from {}...'.format(model_file))
                saver.restore(sess, model_file)
            else:
                print('initializing from scratch...')
                tf.global_variables_initializer().run()

            train_writer = tf.summary.FileWriter(config.train_log_dir, sess.graph)
            valid_writer = tf.summary.FileWriter(config.valid_log_dir, sess.graph)

            valid_log_history = run_train(sess, model, train_data, valid_data, saver,
                                          train_writer, valid_writer, verbose=True)
            save_json(valid_log_history, os.path.join(config.result_dir, config.task_name, config.current_model, 'valid_log_history.json'))

    if args.do_eval:
        print('loading data...')
        valid_data = data_reader.read_valid_data()

        with tf.Session(config=sess_config) as sess:
            model_file = args.model_file
            if model_file is None:
                model_file = tf.train.latest_checkpoint(os.path.join(config.result_dir, config.task_name, config.current_model))
            if model_file is not None:
                print('loading model from {}...'.format(model_file))
                saver.restore(sess, model_file)

                result, valid_loss, valid_accuracy = run_evaluate(sess, model, valid_data, verbose=True)
                print('average valid loss is {:>.4f}, average valid accuracy is {:>.4f}'.format(valid_loss, valid_accuracy))

                print_title('Saving Results')
                save_result(result, config.valid_result)
            else:
                print('model not found!')

    if args.do_test:
        print('loading data...')
        test_data = data_reader.read_test_data()

        with tf.Session(config=sess_config) as sess:
            model_file = args.model_file
            if model_file is None:
                model_file = tf.train.latest_checkpoint(os.path.join(config.result_dir, config.task_name, config.current_model))
            if model_file is not None:
                print('loading model from {}...'.format(model_file))
                saver.restore(sess, model_file)

                result = run_test(sess, model, test_data, verbose=True)

                print_title('Saving Results')
                save_result(result, config.test_result)
            else:
                print('model not found!')


if __name__ == '__main__':
    main()
    print('done')
