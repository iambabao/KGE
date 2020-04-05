# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/3/29 18:07
@Desc       : 
"""

import os
import argparse
from operator import itemgetter

from src.config import Config
from src.utils import print_title, read_json_lines, save_json

parser = argparse.ArgumentParser()
parser.add_argument('--task', '-t', type=str, required=True)
parser.add_argument('--model', '-m', type=str, required=True)
args = parser.parse_args()

config = Config('.', args.task, args.model)


def get_best_threshold(result_file):
    results = []
    for line in read_json_lines(result_file):
        results.append((line['pos_dis'], 1))
        results.append((line['neg_dis'], 0))
    results = sorted(results, key=itemgetter(0))

    true_positive = 0
    true_negative = len(results) // 2
    best_accuracy = (true_positive + true_negative) / len(results)
    best_threshold = 0.0
    for dis, label in results:
        if label == 0:
            true_negative -= 1
        else:
            true_positive += 1
        accuracy = (true_positive + true_negative) / len(results)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_threshold = dis

    return best_threshold


def get_metrics(result_file, threshold):
    tp, tn, fp, fn = 0, 0, 0, 0
    for line in read_json_lines(result_file):
        if line['pos_dis'] <= threshold:
            tp += 1
        else:
            fn += 1
        if line['neg_dis'] > threshold:
            tn += 1
        else:
            fp += 1

    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp),
        'recall': tp / (tp + fn)
    }
    metrics['f1'] = (2 * metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])

    return metrics


def main():
    print_title('Get Threshold')
    threshold = get_best_threshold(config.valid_result)
    print(threshold)

    print_title('Result')
    metrics = get_metrics(config.test_result, threshold)
    metrics['threshold'] = threshold
    for k, v in metrics.items():
        print('{}: {}'.format(k, v))

    save_json(metrics, os.path.join(config.result_dir, config.task_name, config.current_model, 'classification.json'))


if __name__ == '__main__':
    main()
    print('done')
