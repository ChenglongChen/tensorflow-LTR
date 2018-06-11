
import time
import numpy as np
import tensorflow as tf

import utils
from tf_common.nn_module import resnet_block, dense_block
from tf_common.nadam import NadamOptimizer


class BaseRankModel(object):

    def __init__(self, model_name, params, logger, training=True):
        self.model_name = model_name
        self.params = params
        self.logger = logger
        utils._makedirs(self.params["offline_model_dir"], force=training)

        self._init_tf_vars()
        self.loss, self.num_pairs, self.score, self.train_op = self._build_model()

        self.sess, self.saver = self._init_session()


    def _init_tf_vars(self):
        #### input
        self.training = tf.placeholder(tf.bool, shape=[], name="training")
        self.feature = tf.placeholder(tf.float32, shape=[None, self.params["feature_dim"]], name="feature")
        self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")
        self.sorted_label = tf.placeholder(tf.float32, shape=[None, 1], name="sorted_label")
        self.index_range = tf.placeholder(tf.float32, shape=[None, 1], name="index_range")
        self.qid = tf.placeholder(tf.float32, shape=[None, 1], name="qid")
        #### training
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.params["init_lr"], self.global_step,
                                                        self.params["decay_steps"], self.params["decay_rate"])
        self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")

    def _build_model(self):
        return None, None, None, None


    def _score_fn(self, x, reuse=False):
        # deep
        hidden_units = [self.params["fc_dim"] * 4, self.params["fc_dim"] * 2, self.params["fc_dim"]]
        dropouts = [self.params["fc_dropout"]] * len(hidden_units)
        out = dense_block(x, hidden_units=hidden_units, dropouts=dropouts, densenet=False, reuse=reuse,
                                   training=self.training, seed=self.params["random_seed"])
        # score
        score = tf.layers.dense(out, 1, activation=None,
                               kernel_initializer=tf.glorot_uniform_initializer(seed=self.params["random_seed"]))
        return score


    def _get_train_op(self, loss):
        """
        for model that gradient can be computed with respect to loss, e.g., LogisticRegression and RankNet
        """
        with tf.name_scope("optimization"):
            if self.params["optimizer_type"] == "nadam":
                optimizer = NadamOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                           beta2=self.params["beta2"], epsilon=1e-8,
                                           schedule_decay=self.params["schedule_decay"])
            elif self.params["optimizer_type"] == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                                   beta2=self.params["beta2"], epsilon=1e-8)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=self.global_step)

        return train_op


    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 1})
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 4
        config.inter_op_parallelism_threads = 4
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        # max_to_keep=None, keep all the models
        saver = tf.train.Saver(max_to_keep=None)
        return sess, saver


    def save_session(self):
        self.saver.save(self.sess, self.params["offline_model_dir"] + "/model.checkpoint")


    def restore_session(self):
        self.saver.restore(self.sess, self.params["offline_model_dir"] + "/model.checkpoint")


    def _get_batch_index(self, seq, step):
        n = len(seq)
        res = []
        for i in range(0, n, step):
            res.append(seq[i:i + step])
        # last batch
        if len(res) * step < n:
            res.append(seq[len(res) * step:])
        return res


    def _get_feed_dict(self, X, idx, training=False):
        feed_dict = {
            self.feature: X["feature"][idx],
            self.label: X["label"][idx].reshape((-1, 1)),
            self.qid: X["qid"][idx].reshape((-1, 1)),
            self.training: training,
            self.batch_size: len(idx),
        }
        return feed_dict


    def fit(self, X, validation_data, shuffle=False):
        qid_unique = np.unique(X["qid"])
        num_qid_unique = len(qid_unique)
        start_time = time.time()
        l = X["feature"].shape[0]
        self.logger.info("fit on %d sample" % l)
        train_idx_shuffle = np.arange(num_qid_unique)
        total_loss = 0.
        loss_decay = 0.9
        total_batch = 0
        for epoch in range(self.params["epoch"]):
            self.logger.info("epoch: %d" % (epoch + 1))
            np.random.seed(epoch)
            if shuffle:
                np.random.shuffle(train_idx_shuffle)
            batches = self._get_batch_index(train_idx_shuffle, self.params["batch_size"])
            for i, idx in enumerate(batches):
                ind = utils._get_intersect_index(X["qid"], qid_unique[idx])
                feed_dict = self._get_feed_dict(X, ind, training=True)
                loss, lr, opt = self.sess.run((self.loss, self.learning_rate, self.train_op), feed_dict=feed_dict)
                total_loss = loss_decay * total_loss + (1. - loss_decay) * loss
                total_batch += 1
                if validation_data is not None and total_batch % self.params["eval_every_num_update"] == 0:
                    valid_loss = self.evaluate(validation_data)
                    self.logger.info(
                        "[epoch-%d, batch-%d] train-loss=%.5f, valid-loss=%.5f, lr=%.5f [%.1f s]" % (
                            epoch + 1, total_batch, total_loss, valid_loss,
                            lr, time.time() - start_time))
                else:
                    self.logger.info("[epoch-%d, batch-%d] train-loss=%.5f, lr=%.5f [%.1f s]" % (
                        epoch + 1, total_batch, total_loss,
                        lr, time.time() - start_time))


    def predict(self, X):
        l = X["feature"].shape[0]
        train_idx = np.arange(l)
        batches = self._get_batch_index(train_idx, self.params["batch_size"])
        y_pred = []
        y_pred_append = y_pred.append
        for idx in batches:
            feed_dict = self._get_feed_dict(X, idx, training=False)
            pred = self.sess.run((self.score), feed_dict=feed_dict)
            y_pred_append(pred)
        y_pred = np.vstack(y_pred).reshape((-1, 1))
        return y_pred


    def evaluate(self, X):
        qid_unique = np.unique(X["qid"])
        num_qid_unique = len(qid_unique)
        train_idx = np.arange(num_qid_unique)
        batches = self._get_batch_index(train_idx, self.params["batch_size"])
        valid_loss = 0.
        valid_num_pairs = 0.
        for idx in batches:
            ind = utils._get_intersect_index(X["qid"], qid_unique[idx])
            feed_dict = self._get_feed_dict(X, ind, training=False)
            loss, num_pairs = self.sess.run((self.loss, self.num_pairs), feed_dict=feed_dict)
            valid_loss += (loss * num_pairs)
            valid_num_pairs += num_pairs
        return valid_loss/(valid_num_pairs)


class LogisticRegression(BaseRankModel):

    def __init__(self, model_name, params, logger, training=True):
        super(LogisticRegression, self).__init__(model_name, params, logger, training)

    def _build_model(self):
        # score
        score = logits = self._score_fn(self.feature)

        logloss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.label)
        loss = tf.reduce_mean(logloss)
        num_pairs = tf.shape(self.feature)[0]

        return loss, num_pairs, score, self._get_train_op(loss)


class RankNet(BaseRankModel):

    def __init__(self, model_name, params, logger, training=True):
        super(RankNet, self).__init__(model_name, params, logger, training)


    def _build_model(self):
        if self.params["factorization"]:
            return self._build_factorized_model()
        else:
            return self._build_unactorized_model()


    def _build_unactorized_model(self):
        # score
        score = self._score_fn(self.feature)

        #
        S_ij = self.label - tf.transpose(self.label)
        S_ij = tf.maximum(tf.minimum(1., S_ij), -1.)
        P_ij = (1 / 2) * (1 + S_ij)
        s_i_minus_s_j = logits = score - tf.transpose(score)

        logloss = tf.nn.sigmoid_cross_entropy_with_logits(logits=s_i_minus_s_j, labels=P_ij)

        # only extracted the loss of pairs of the same group
        mask1 = tf.equal(self.qid - tf.transpose(self.qid), 0)
        mask1 = tf.cast(mask1, tf.float32)
        # exclude the pair of sample and itself
        n = tf.shape(self.feature)[0]
        mask2 = tf.ones([n, n]) - tf.diag(tf.ones([n]))
        mask = mask1 * mask2
        num_pairs = tf.reduce_sum(mask)

        loss = tf.reduce_sum(logloss * mask) / num_pairs

        return loss, num_pairs, score, self._get_train_op(loss)


    def _build_factorized_model(self):
        # score
        score = self._score_fn(self.feature)

        #
        S_ij = self.label - tf.transpose(self.label)
        S_ij = tf.maximum(tf.minimum(1., S_ij), -1.)
        P_ij = (1 / 2) * (1 + S_ij)
        s_i_minus_s_j = logits = score - tf.transpose(score)
        sigma = self.params["sigma"]
        lambda_ij = sigma * ((1 / 2) * (1 - S_ij) - tf.nn.sigmoid(-sigma*s_i_minus_s_j))

        logloss = tf.nn.sigmoid_cross_entropy_with_logits(logits=s_i_minus_s_j, labels=P_ij)

        # only extracted the loss of pairs of the same group
        mask1 = tf.equal(self.qid - tf.transpose(self.qid), 0)
        mask1 = tf.cast(mask1, tf.float32)
        # exclude the pair of sample and itself
        n = tf.shape(self.feature)[0]
        mask2 = tf.ones([n, n]) - tf.diag(tf.ones([n]))
        mask = mask1 * mask2
        num_pairs = tf.reduce_sum(mask)

        loss = tf.reduce_sum(logloss * mask) / num_pairs

        lambda_ij = lambda_ij * mask

        def jacobian(y_flat, x):
            """
            https://github.com/tensorflow/tensorflow/issues/675
            """
            loop_vars = [
                tf.constant(0, tf.int32),
                tf.TensorArray(tf.float32, size=self.batch_size),
            ]

            _, jacobian = tf.while_loop(
                lambda j, _: j < n,
                lambda j, result: (j + 1, result.write(j, tf.gradients(y_flat[j], x))),
                loop_vars)

            return jacobian.stack()

        def _get_derivative(score, Wk):
            # dsi_dWk = tf.map_fn(lambda s: tf.gradients(s, [Wk])[0], score) # do not work
            dsi_dWk = jacobian(score, Wk)
            dsi_dWk_minus_dsj_dWk = tf.expand_dims(dsi_dWk, 1) - tf.expand_dims(dsi_dWk, 0)
            shape = tf.concat(
                [tf.shape(lambda_ij), tf.ones([tf.rank(dsi_dWk_minus_dsj_dWk) - tf.rank(lambda_ij)], dtype=tf.int32)],
                axis=0)
            grad = tf.reduce_mean(tf.reshape(lambda_ij, shape) * dsi_dWk_minus_dsj_dWk, axis=[0, 1])
            return tf.reshape(grad, tf.shape(Wk))

        vars = tf.trainable_variables()
        grads = [_get_derivative(score, Wk) for Wk in vars]

        with tf.name_scope("optimization"):
            if self.params["optimizer_type"] == "nadam":
                optimizer = NadamOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                           beta2=self.params["beta2"], epsilon=1e-8,
                                           schedule_decay=self.params["schedule_decay"])
            elif self.params["optimizer_type"] == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                                   beta2=self.params["beta2"], epsilon=1e-8)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(zip(grads, vars))

        return loss, num_pairs, score, train_op
