# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/2/20 14:04
@Desc       :
"""

import random

from src.utils import read_json, read_json_lines


def negative_sampling_v0(s, p, o, entities, num_neg):
    neg_set = set()
    while len(neg_set) < num_neg:
        # randomly replace subject or object with random entity in all entity set
        if random.randint(0, 2):
            neg_set.add((random.choice(entities), p, o))
        else:
            neg_set.add((s, p, random.choice(entities)))
    return neg_set


def negative_sampling_v1(s, p, o, relation_mapping, num_neg):
    neg_set = set()
    while len(neg_set) < num_neg:
        # replace subject or object according to probability
        s_set = relation_mapping[p]['s']
        o_set = relation_mapping[p]['o']
        if random.randint(0, len(s_set) + len(o_set)) < len(s_set):
            neg_set.add((random.choice(s_set), p, o))
        else:
            neg_set.add((s, p, random.choice(o_set)))
    return neg_set


class DataReader:
    def __init__(self, config):
        self.config = config
        self.relation_mapping = read_json(self.config.relation_mapping)

    def _read_data_with_negative_sampling(self, filename, num_neg):
        sid = []
        pid = []
        oid = []
        neg_sid = []
        neg_pid = []
        neg_oid = []

        counter = 0
        for line in read_json_lines(filename):
            s = line['s']
            p = line['p']
            o = line['o']

            # for neg_s, neg_p, neg_o in negative_sampling_v0(s, p, o, list(self.config.entity_2_id.keys()), num_neg):
            for neg_s, neg_p, neg_o in negative_sampling_v1(s, p, o, self.relation_mapping, num_neg):
                sid.append(self.config.entity_2_id[s])
                pid.append(self.config.relation_2_id[p])
                oid.append(self.config.entity_2_id[o])
                neg_sid.append(self.config.entity_2_id[neg_s])
                neg_pid.append(self.config.relation_2_id[neg_p])
                neg_oid.append(self.config.entity_2_id[neg_o])

            counter += 1
            if counter % 10000 == 0:
                print('\rprocessing file {}: {:>6d}'.format(filename, counter), end='')
        print()

        return sid, pid, oid, neg_sid, neg_pid, neg_oid

    def _read_data_wo_negative_sampling(self, filename):
        sid = []
        pid = []
        oid = []

        counter = 0
        for line in read_json_lines(filename):
            s = line['s']
            p = line['p']
            o = line['o']

            sid.append(self.config.entity_2_id[s])
            pid.append(self.config.relation_2_id[p])
            oid.append(self.config.entity_2_id[o])

            counter += 1
            if counter % 10000 == 0:
                print('\rprocessing file {}: {:>6d}'.format(filename, counter), end='')
        print()

        return sid, pid, oid

    def read_train_data(self, num_neg):
        return self._read_data_with_negative_sampling(self.config.train_data, num_neg)

    def read_valid_data(self, num_neg):
        return self._read_data_with_negative_sampling(self.config.valid_data, num_neg)

    def read_test_data(self, num_neg):
        return self._read_data_with_negative_sampling(self.config.test_data, num_neg)

    def read_train_data_wo_negative_sampling(self):
        return self._read_data_wo_negative_sampling(self.config.train_data)

    def read_valid_data_wo_negative_sampling(self):
        return self._read_data_wo_negative_sampling(self.config.valid_data)

    def read_test_data_wo_negative_sampling(self):
        return self._read_data_wo_negative_sampling(self.config.test_data)
