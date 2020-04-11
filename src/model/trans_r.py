# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/3/29 0:07
@Desc       : 
"""

import tensorflow as tf


class TransR:
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
        self.pos_s = tf.placeholder(tf.int32, [None], name='pos_s')
        self.pos_p = tf.placeholder(tf.int32, [None], name='pos_p')
        self.pos_o = tf.placeholder(tf.int32, [None], name='pos_o')
        self.neg_s = tf.placeholder(tf.int32, [None, None], name='neg_s')
        self.neg_p = tf.placeholder(tf.int32, [None, None], name='neg_p')
        self.neg_o = tf.placeholder(tf.int32, [None, None], name='neg_o')
        self.training = tf.placeholder(tf.bool, [], name='training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.entity_embedding = tf.keras.layers.Embedding(self.num_entity, self.entity_em_size, name='entity_embedding')
        self.relation_embedding = tf.keras.layers.Embedding(self.num_relation, self.relation_em_size, name='relation_embedding')
        self.projection_vector = tf.keras.layers.Embedding(self.num_relation, self.entity_em_size * self.relation_em_size, name='projection_vector')
        self.pos_s_dropout = tf.keras.layers.Dropout(self.dropout)
        self.pos_p_dropout = tf.keras.layers.Dropout(self.dropout)
        self.pos_o_dropout = tf.keras.layers.Dropout(self.dropout)
        self.pos_vector_dropout = tf.keras.layers.Dropout(self.dropout)
        self.neg_s_dropout = tf.keras.layers.Dropout(self.dropout)
        self.neg_p_dropout = tf.keras.layers.Dropout(self.dropout)
        self.neg_o_dropout = tf.keras.layers.Dropout(self.dropout)
        self.neg_vector_dropout = tf.keras.layers.Dropout(self.dropout)

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

        self.lr = tf.train.exponential_decay(self.lr, self.global_step, decay_steps=2000, decay_rate=0.99)
        self.pos_dis, self.neg_dis = self.forward()
        margin_loss = tf.reduce_mean(tf.maximum(0.0, tf.reshape(self.pos_dis, [-1, 1]) - self.neg_dis + self.margin))
        constrain_loss = self.get_constrain_loss()
        self.loss = margin_loss + self.l2_rate * constrain_loss
        self.accuracy = tf.reduce_mean(tf.cast(tf.less(tf.reshape(self.pos_dis, [-1, 1]), self.neg_dis), tf.float32))
        self.gradients, self.train_op = self.get_train_op()

        tf.summary.scalar('learning_rate', self.lr() if callable(self.lr) else self.lr)
        tf.summary.scalar('margin_loss', margin_loss)
        tf.summary.scalar('constrain_loss', constrain_loss)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.summary = tf.summary.merge_all()

    def forward(self):
        # embedding
        pos_s_em, pos_p_em, pos_o_em, pos_vector, neg_s_em, neg_p_em, neg_o_em, neg_vector = self.embedding_layer()

        # projection
        pos_matrix = tf.reshape(pos_vector, [self.batch_size, self.entity_em_size, self.relation_em_size])
        pos_s_em = tf.einsum('ij,ijk->ik', pos_s_em, pos_matrix)
        pos_o_em = tf.einsum('ij,ijk->ik', pos_o_em, pos_matrix)
        neg_matrix = tf.reshape(neg_vector, [self.batch_size, -1, self.entity_em_size, self.relation_em_size])
        neg_s_em = tf.einsum('ijk,ijkl->ijl', neg_s_em, neg_matrix)
        neg_o_em = tf.einsum('ijk,ijkl->ijl', neg_o_em, neg_matrix)

        # normalize
        pos_s_em = tf.math.l2_normalize(pos_s_em, axis=-1)
        pos_p_em = tf.math.l2_normalize(pos_p_em, axis=-1)
        pos_o_em = tf.math.l2_normalize(pos_o_em, axis=-1)
        neg_s_em = tf.math.l2_normalize(neg_s_em, axis=-1)
        neg_p_em = tf.math.l2_normalize(neg_p_em, axis=-1)
        neg_o_em = tf.math.l2_normalize(neg_o_em, axis=-1)

        # calculate distance
        pos_dis = tf.norm(pos_s_em + pos_p_em - pos_o_em, ord=1, axis=-1)
        neg_dis = tf.norm(neg_s_em + neg_p_em - neg_o_em, ord=1, axis=-1)

        return pos_dis, neg_dis

    def get_train_op(self):
        gradients = tf.gradients(self.loss, tf.trainable_variables())
        train_op = self.optimizer.apply_gradients(zip(gradients, tf.trainable_variables()), self.global_step)

        return gradients, train_op

    def embedding_layer(self):
        pos_s_em = self.pos_s_dropout(self.entity_embedding(self.pos_s), training=self.training)
        pos_p_em = self.pos_p_dropout(self.relation_embedding(self.pos_p), training=self.training)
        pos_o_em = self.pos_o_dropout(self.entity_embedding(self.pos_o), training=self.training)
        pos_vector = self.pos_vector_dropout(self.projection_vector(self.pos_p), training=self.training)
        neg_s_em = self.neg_s_dropout(self.entity_embedding(self.neg_s), training=self.training)
        neg_p_em = self.neg_p_dropout(self.relation_embedding(self.neg_p), training=self.training)
        neg_o_em = self.neg_o_dropout(self.entity_embedding(self.neg_o), training=self.training)
        neg_vector = self.neg_vector_dropout(self.projection_vector(self.neg_p), training=self.training)

        return pos_s_em, pos_p_em, pos_o_em, pos_vector, neg_s_em, neg_p_em, neg_o_em, neg_vector

    def get_constrain_loss(self):
        s_em = self.entity_embedding(self.pos_s)
        p_em = self.relation_embedding(self.pos_p)
        o_em = self.entity_embedding(self.pos_o)
        projection_vector = self.projection_vector(self.pos_p)
        projection_matrix = tf.reshape(projection_vector, [self.batch_size, self.entity_em_size, self.relation_em_size])
        projected_s_em = tf.einsum('ij,ijk->ik', s_em, projection_matrix)
        projected_o_em = tf.einsum('ij,ijk->ik', o_em, projection_matrix)

        s_loss = tf.maximum(0.0, tf.reduce_sum(s_em ** 2, axis=-1) - 1.0)
        p_loss = tf.maximum(0.0, tf.reduce_sum(p_em ** 2, axis=-1) - 1.0)
        o_loss = tf.maximum(0.0, tf.reduce_sum(o_em ** 2, axis=-1) - 1.0)
        projected_s_loss = tf.maximum(0.0, tf.reduce_sum(projected_s_em ** 2, axis=-1) - 1.0)
        projected_o_loss = tf.maximum(0.0, tf.reduce_sum(projected_o_em ** 2, axis=-1) - 1.0)

        return tf.reduce_mean(s_loss + p_loss + o_loss + projected_s_loss + projected_o_loss)
