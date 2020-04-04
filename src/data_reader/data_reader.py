# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/2/20 14:04
@Desc       :
"""

from src.utils import read_json_lines


class DataReader:
    def __init__(self, config):
        self.config = config

    def _read_data_with_negative_sampling(self, filename):
        pos_s = []
        pos_p = []
        pos_o = []

        counter = 0
        for line in read_json_lines(filename):
            s = line['s']
            p = line['p']
            o = line['o']

            pos_s.append(self.config.entity_2_id[s])
            pos_p.append(self.config.relation_2_id[p])
            pos_o.append(self.config.entity_2_id[o])

            counter += 1
            if counter % 10000 == 0:
                print('\rprocessing file {}: {:>6d}'.format(filename, counter), end='')
        print()

        return pos_s, pos_p, pos_o

    def read_train_data(self):
        return self._read_data_with_negative_sampling(self.config.train_data)

    def read_valid_data(self):
        return self._read_data_with_negative_sampling(self.config.valid_data)

    def read_test_data(self):
        return self._read_data_with_negative_sampling(self.config.test_data)
