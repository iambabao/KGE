# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/3/29 0:07
@Desc       : 
"""

import tensorflow as tf


class TransH:
    def __init__(self, config):
        self.num_entity = config.num_entity
        self.num_relation = config.num_relation

        self.entity_em_size = config.entity_em_size
        self.relation_em_size = config.relation_em_size
        self.margin = config.threshold
        self.lr = config.lr
        self.dropout = config.dropout
        self.l2_rate = config.l2_rate

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.sid = tf.placeholder(tf.int32, [None], name='sid')
        self.pid = tf.placeholder(tf.int32, [None], name='pid')
        self.oid = tf.placeholder(tf.int32, [None], name='oid')
        self.neg_sid = tf.placeholder(tf.int32, [None], name='neg_sid')
        self.neg_pid = tf.placeholder(tf.int32, [None], name='neg_pid')
        self.neg_oid = tf.placeholder(tf.int32, [None], name='neg_oid')
        self.training = tf.placeholder(tf.bool, [], name='training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.entity_embedding = tf.keras.layers.Embedding(self.num_entity, self.entity_em_size, name='entity_embedding')
        self.relation_embedding = tf.keras.layers.Embedding(self.num_relation, self.relation_em_size, name='relation_embedding')
        self.normal_vector = tf.keras.layers.Embedding(self.num_relation, self.entity_em_size, name='normal_vector')
        self.s_dropout = tf.keras.layers.Dropout(self.dropout)
        self.p_dropout = tf.keras.layers.Dropout(self.dropout)
        self.o_dropout = tf.keras.layers.Dropout(self.dropout)
        self.norm_dropout = tf.keras.layers.Dropout(self.dropout)
        self.neg_s_dropout = tf.keras.layers.Dropout(self.dropout)
        self.neg_p_dropout = tf.keras.layers.Dropout(self.dropout)
        self.neg_o_dropout = tf.keras.layers.Dropout(self.dropout)
        self.neg_norm_dropout = tf.keras.layers.Dropout(self.dropout)

        if config.optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif config.optimizer == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif config.optimizer == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.lr)
        elif config.optimizer == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        else:
            assert False

        self.pos_dis, self.neg_dis = self.forward()
        margin_loss = tf.reduce_mean(tf.maximum(0.0, self.pos_dis - self.neg_dis + self.margin))
        constrain_loss = self.get_constrain_loss()
        self.loss = margin_loss + self.l2_rate * constrain_loss
        self.accuracy = tf.reduce_mean(tf.cast(tf.less(self.pos_dis, self.neg_dis), tf.float32))
        self.gradients, self.train_op = self.get_train_op()

        tf.summary.scalar('learning_rate', self.lr() if callable(self.lr) else self.lr)
        tf.summary.scalar('margin_loss', margin_loss)
        tf.summary.scalar('constrain_loss', constrain_loss)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.summary = tf.summary.merge_all()

    def forward(self):
        # embedding
        sem, pem, oem, norm_em, neg_sem, neg_pem, neg_oem, neg_norm_em = self.embedding_layer()

        # projection
        sem = sem - tf.reduce_sum(norm_em * sem, axis=-1, keepdims=True) * norm_em
        oem = oem - tf.reduce_sum(norm_em * oem, axis=-1, keepdims=True) * norm_em
        neg_sem = neg_sem - tf.reduce_sum(norm_em * neg_sem, axis=-1, keepdims=True) * norm_em
        neg_oem = neg_oem - tf.reduce_sum(norm_em * neg_oem, axis=-1, keepdims=True) * norm_em

        # normalize
        sem = tf.math.l2_normalize(sem, axis=-1)
        pem = tf.math.l2_normalize(pem, axis=-1)
        oem = tf.math.l2_normalize(oem, axis=-1)
        neg_sem = tf.math.l2_normalize(neg_sem, axis=-1)
        neg_pem = tf.math.l2_normalize(neg_pem, axis=-1)
        neg_oem = tf.math.l2_normalize(neg_oem, axis=-1)

        # calculate distance
        pos_dis = tf.norm(sem + pem - oem, ord=1, axis=-1)
        neg_dis = tf.norm(neg_sem + neg_pem - neg_oem, ord=1, axis=-1)

        return pos_dis, neg_dis

    def get_train_op(self):
        gradients = tf.gradients(self.loss, tf.trainable_variables())
        train_op = self.optimizer.apply_gradients(zip(gradients, tf.trainable_variables()), self.global_step)

        return gradients, train_op

    def embedding_layer(self):
        with tf.device('/cpu:0'):
            sem = self.s_dropout(self.entity_embedding(self.sid), training=self.training)
            pem = self.p_dropout(self.relation_embedding(self.pid), training=self.training)
            oem = self.o_dropout(self.entity_embedding(self.oid), training=self.training)
            norm_em = self.norm_dropout(self.normal_vector(self.pid), training=self.training)
            neg_sem = self.neg_s_dropout(self.entity_embedding(self.neg_sid), training=self.training)
            neg_pem = self.neg_p_dropout(self.relation_embedding(self.neg_pid), training=self.training)
            neg_oem = self.neg_o_dropout(self.entity_embedding(self.neg_oid), training=self.training)
            neg_norm_em = self.neg_norm_dropout(self.normal_vector(self.neg_pid), training=self.training)

        return sem, pem, oem, norm_em, neg_sem, neg_pem, neg_oem, neg_norm_em

    def get_constrain_loss(self):
        sem = self.entity_embedding(self.sid)
        oem = self.entity_embedding(self.oid)
        norm_em = self.norm_dropout(self.normal_vector(self.pid), training=self.training)

        s_loss = tf.maximum(0.0, tf.norm(sem, ord=2, axis=-1) ** 2 - 1.0)
        o_loss = tf.maximum(0.0, tf.norm(oem, ord=2, axis=-1) ** 2 - 1.0)
        norm_loss = tf.maximum(0.0, tf.norm(norm_em, ord=2, axis=-1) ** 2 - 1.0)

        return tf.reduce_mean(s_loss + o_loss + norm_loss)
