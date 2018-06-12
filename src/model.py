
import time
import numpy as np
import pandas as pd
import tensorflow as tf

import utils
from metrics import ndcg, calc_err
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
        with tf.name_scope(self.model_name):
            #### input for training and inference
            self.feature = tf.placeholder(tf.float32, shape=[None, self.params["feature_dim"]], name="feature")
            self.training = tf.placeholder(tf.bool, shape=[], name="training")
            #### input for training
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")
            self.sorted_label = tf.placeholder(tf.float32, shape=[None, 1], name="sorted_label")
            self.qid = tf.placeholder(tf.float32, shape=[None, 1], name="qid")
            #### vars for training
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.params["init_lr"], self.global_step,
                                                            self.params["decay_steps"], self.params["decay_rate"])
            self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")


    def _build_model(self):
        return None, None, None, None


    def _score_fn_inner(self, x, reuse=False):
        # deep
        hidden_units = [self.params["fc_dim"] * 4, self.params["fc_dim"] * 2, self.params["fc_dim"]]
        dropouts = [self.params["fc_dropout"]] * len(hidden_units)
        out = dense_block(x, hidden_units=hidden_units, dropouts=dropouts, densenet=False, reuse=reuse,
                          training=self.training, seed=self.params["random_seed"])
        # score
        score = tf.layers.dense(out, 1, activation=None,
                                kernel_initializer=tf.glorot_uniform_initializer(seed=self.params["random_seed"]))

        return score


    def _score_fn(self, x, reuse=False):
        # https://stackoverflow.com/questions/45670224/why-the-tf-name-scope-with-same-name-is-different
        with tf.name_scope(self.model_name+"/"):
            score = self._score_fn_inner(x, reuse)
            # https://stackoverflow.com/questions/46980287/output-node-for-tensorflow-graph-created-with-tf-layers
            # add an identity node to output graph
            score = tf.identity(score, "score")

        return score


    def _jacobian(self, y_flat, x):
        """
        https://github.com/tensorflow/tensorflow/issues/675
        for ranknet and lambdarank
        """
        loop_vars = [
            tf.constant(0, tf.int32),
            tf.TensorArray(tf.float32, size=self.batch_size),
        ]

        _, jacobian = tf.while_loop(
            lambda j, _: j < self.batch_size,
            lambda j, result: (j + 1, result.write(j, tf.gradients(y_flat[j], x))),
            loop_vars)

        return jacobian.stack()


    def _get_derivative(self, score, Wk, lambda_ij):
        """
        for ranknet and lambdarank
        :param score:
        :param Wk:
        :param lambda_ij:
        :return:
        """
        # dsi_dWk = tf.map_fn(lambda s: tf.gradients(s, [Wk])[0], score) # do not work
        dsi_dWk = self._jacobian(score, Wk)
        dsi_dWk_minus_dsj_dWk = tf.expand_dims(dsi_dWk, 1) - tf.expand_dims(dsi_dWk, 0)
        shape = tf.concat(
            [tf.shape(lambda_ij), tf.ones([tf.rank(dsi_dWk_minus_dsj_dWk) - tf.rank(lambda_ij)], dtype=tf.int32)],
            axis=0)
        grad = tf.reduce_mean(tf.reshape(lambda_ij, shape) * dsi_dWk_minus_dsj_dWk, axis=[0, 1])
        return tf.reshape(grad, tf.shape(Wk))


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
        # write graph for freeze_graph.py
        tf.train.write_graph(self.sess.graph.as_graph_def(), self.params["offline_model_dir"], "graph.pb", as_text=True)
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
            self.sorted_label: np.sort(X["label"][idx].reshape((-1, 1)))[::-1],
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
                if total_batch % self.params["eval_every_num_update"] == 0:
                    loss_mean_train, err_mean_train, ndcg_mean_train, ndcg_all_mean_train = self.evaluate(X)
                    if validation_data is not None:
                        loss_mean_valid, err_mean_valid, ndcg_mean_valid, ndcg_all_mean_valid = self.evaluate(validation_data)
                        self.logger.info(
                            "[epoch-{}, batch-{}] -- Train Loss: {:5f} NDCG: {:5f} ({:5f}) ERR: {:5f}  -- Valid Loss: {:5f} NDCG: {:5f} ({:5f}) ERR: {:5f} -- {:5f} s".format(
                                epoch + 1, total_batch, loss_mean_train, ndcg_mean_train, ndcg_all_mean_train, err_mean_train,
                                loss_mean_valid, ndcg_mean_valid, ndcg_all_mean_valid, err_mean_valid, time.time() - start_time))
                    else:
                        self.logger.info(
                            "[epoch-{}, batch-{}] -- Train Loss: {:5f} NDCG: {:5f} ({:5f}) ERR: {:5f} -- {:5f} s".format(
                                epoch + 1, total_batch, loss_mean_train, ndcg_mean_train, ndcg_all_mean_train, err_mean_train,
                                time.time() - start_time))


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
        n = len(qid_unique)
        losses = np.zeros(n)
        ndcgs = np.zeros(n)
        ndcgs_all = np.zeros(n)
        errs = np.zeros(n)
        for e,qid in enumerate(qid_unique):
            ind = np.where(X["qid"] == qid)[0]
            feed_dict = self._get_feed_dict(X, ind, training=False)
            loss, score = self.sess.run((self.loss, self.score), feed_dict=feed_dict)
            df = pd.DataFrame({"label": X["label"][ind].flatten(), "score": score.flatten()})
            df.sort_values("score", ascending=False)

            losses[e] = loss
            ndcgs[e] = ndcg(df["label"])
            ndcgs_all[e] = ndcg(df["label"], top_ten=False)
            errs[e] = calc_err(df["label"])
        losses_mean = np.mean(losses)
        ndcgs_mean = np.mean(ndcgs)
        ndcgs_all_mean = np.mean(ndcgs_all)
        errs_mean = np.mean(errs)
        return losses_mean, errs_mean, ndcgs_mean, ndcgs_all_mean


class DNN(BaseRankModel):

    def __init__(self, model_name, params, logger, training=True):
        super(DNN, self).__init__(model_name, params, logger, training)

    def _build_model(self):
        # score
        score = logits = self._score_fn(self.feature)

        logloss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.label)
        loss = tf.reduce_mean(logloss)
        num_pairs = tf.shape(self.feature)[0]

        return loss, num_pairs, score, self._get_train_op(loss)


class LogisticRegression(DNN):

    def __init__(self, model_name, params, logger, training=True):
        super(LogisticRegression, self).__init__(model_name, params, logger, training)


    def _score_fn_inner(self, x, reuse=False):
        score = tf.layers.dense(x, 1, activation=None,
                                kernel_initializer=tf.glorot_uniform_initializer(seed=self.params["random_seed"]))
        return score


class RankNet(BaseRankModel):

    def __init__(self, model_name, params, logger, training=True):
        super(RankNet, self).__init__(model_name, params, logger, training)


    def _build_model(self):
        if self.params["factorization"]:
            return self._build_factorized_model()
        else:
            return self._build_unfactorized_model()


    def _build_unfactorized_model(self):
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

        vars = tf.trainable_variables()
        grads = [self._get_derivative(score, Wk, lambda_ij) for Wk in vars]

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


class LambdaRank(BaseRankModel):

    def __init__(self, model_name, params, logger, training=True):
        super(LambdaRank, self).__init__(model_name, params, logger, training)


    def _build_model(self):
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

        # multiply by delta ndcg
        # current dcg
        index = tf.reshape(tf.range(1., tf.cast(self.batch_size, dtype=tf.float32) + 1), tf.shape(self.label))
        cg_discount = tf.log(1. + index)
        rel = 2 ** self.label - 1
        sorted_rel = 2 ** self.sorted_label - 1
        dcg_m = rel / cg_discount
        dcg = tf.reduce_sum(dcg_m)
        # every possible swapped dcg
        stale_ij = tf.tile(dcg_m, [1, self.batch_size])
        new_ij = rel / tf.transpose(cg_discount)
        stale_ji = tf.transpose(stale_ij)
        new_ji = tf.transpose(new_ij)
        # new dcg
        dcg_new = dcg - stale_ij + new_ij - stale_ji + new_ji
        # delta ndcg
        # sorted_label = tf.contrib.framework.sort(self.label, direction="DESCENDING")
        dcg_max = tf.reduce_sum(sorted_rel / cg_discount)
        ndcg_delta = tf.abs(dcg_new - dcg) / dcg_max
        lambda_ij = lambda_ij * ndcg_delta

        vars = tf.trainable_variables()
        grads = [self._get_derivative(score, Wk, lambda_ij) for Wk in vars]

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


class ListNet(BaseRankModel):

    def __init__(self, model_name, params, logger, training=True):
        super(ListNet, self).__init__(model_name, params, logger, training)
