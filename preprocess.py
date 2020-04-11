# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/3/28 23:18
@Desc       : 
"""

import os
from collections import defaultdict

from src.config import Config
from src.utils import makedirs, save_json, save_json_dict, save_json_lines


def convert_data(config):
    def convert_dict(filename):
        key_2_id = {}
        id_2_key = {}
        with open(filename, 'r', encoding='utf-8') as fin:
            fin.readline()
            for line in fin:
                key, id_ = line.strip().split()
                key_2_id[key] = int(id_)
                id_2_key[int(id_)] = key
        return key_2_id, id_2_key

    entity_2_id, id_2_entity = convert_dict(os.path.join(config.data_dir, 'benchmarks', config.task_name, 'entity2id.txt'))
    save_json_dict(entity_2_id, config.entity_dict)

    relation_2_id, id_2_relation = convert_dict(os.path.join(config.data_dir, 'benchmarks', config.task_name, 'relation2id.txt'))
    save_json_dict(relation_2_id, config.relation_dict)

    def convert_spo(filename):
        data = []
        relation_mapping = defaultdict(set)
        with open(filename, 'r', encoding='utf-8') as fin:
            fin.readline()
            for line in fin:
                sid, oid, pid = line.strip().split()
                line = {
                    's': id_2_entity[int(sid)],
                    'p': id_2_relation[int(pid)],
                    'o': id_2_entity[int(oid)]
                }
                data.append(line)
                relation_mapping[line['p']].add((line['s'], line['o']))

        replace_prob = {}
        for p, so in relation_mapping.items():
            s_dict = defaultdict(set)
            o_dict = defaultdict(set)
            for s, o in so:
                s_dict[s].add(o)
                o_dict[o].add(s)
            tph = sum([len(s_dict[s]) for s in s_dict]) / len(s_dict)
            hpt = sum([len(o_dict[o]) for o in o_dict]) / len(o_dict)
            replace_prob[p] = {'s': tph / (tph + hpt), 'o': hpt / (tph + hpt)}

        return data, replace_prob

    train_data, replace_prob = convert_spo(os.path.join(config.data_dir, 'benchmarks', config.task_name, 'train2id.txt'))
    save_json_lines(train_data, config.train_data)
    save_json(replace_prob, config.replace_dict)
    valid_data, _ = convert_spo(os.path.join(config.data_dir, 'benchmarks', config.task_name, 'valid2id.txt'))
    save_json_lines(valid_data, config.valid_data)
    test_data, _ = convert_spo(os.path.join(config.data_dir, 'benchmarks', config.task_name, 'test2id.txt'))
    save_json_lines(test_data, config.test_data)


def main():
    tasks = ['FB13', 'FB15K', 'FB15K237', 'WN11', 'WN18', 'WN18RR', 'NELL-995', 'YAGO3-10']
    for task in tasks:
        config = Config('', task, '')
        makedirs(os.path.join(config.data_dir, task))
        print('converting {} data'.format(task))
        convert_data(config)


if __name__ == '__main__':
    main()
