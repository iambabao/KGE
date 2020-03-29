# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/2/20 20:42
@Desc       :
"""

from .trans_e import TransE
from .trans_h import TransH
from .trans_r import TransR

model_list = {
    'TransE': TransE,
    'TransH': TransH,
    'TransR': TransR
}


def get_model(config):
    assert config.current_model in model_list

    return model_list[config.current_model](config)
