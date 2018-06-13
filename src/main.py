
import sys
import numpy as np

import utils
from model import LogisticRegression, DNN, RankNet, LambdaRank
from prepare_data import label_file_pat, group_file_pat, feature_file_pat

def load_data(type):

    labels = np.load(label_file_pat%type)
    qids = np.load(group_file_pat % type)
    features = np.load(feature_file_pat%type)

    X = {
        "feature": features,
        "label": labels,
        "qid": qids
    }
    return X


utils._makedirs("../logs")
logger = utils._get_logger("../logs", "tf-%s.log" % utils._timestamp())

params_common = {
    "batch_size": 256,
    "epoch": 20,
    "feature_dim": 46,

    "batch_sampling_method": "sample",
    "shuffle": True,

    "optimizer_type": "adam",
    "init_lr": 0.001,
    "beta1": 0.975,
    "beta2": 0.999,
    "decay_steps": 1000,
    "decay_rate": 0.9,
    "schedule_decay": 0.004,
    "random_seed": 2018,
    "eval_every_num_update": 100,
}


def train_lr():
    params = {
        "offline_model_dir": "../weights/lr",
    }
    params.update(params_common)

    X_train, X_valid = load_data("train"), load_data("vali")

    model = LogisticRegression("ranking", params, logger)
    model.fit(X_train, validation_data=X_valid)
    model.save_session()


def train_dnn():
    params = {
        "offline_model_dir": "../weights/dnn",

        # deep part score fn
        "fc_type": "fc",
        "fc_dim": 32,
        "fc_dropout": 0.,
    }
    params.update(params_common)

    X_train, X_valid = load_data("train"), load_data("vali")

    model = DNN("ranking", params, logger)
    model.fit(X_train, validation_data=X_valid)
    model.save_session()


def train_ranknet():
    params = {
        "offline_model_dir": "../weights/ranknet",

        # deep part score fn
        "fc_type": "fc",
        "fc_dim": 32,
        "fc_dropout": 0.,

        # ranknet param
        "factorization": True,
        "sigma": 1.,
    }
    params.update(params_common)

    X_train, X_valid = load_data("train"), load_data("vali")

    model = RankNet("ranking", params, logger)
    model.fit(X_train, validation_data=X_valid)
    model.save_session()


def train_lambdarank():
    params = {
        "offline_model_dir": "../weights/lambdarank",

        # deep part score fn
        "fc_type": "fc",
        "fc_dim": 32,
        "fc_dropout": 0.,

        # lambdarank param
        "sigma": 1.,
    }
    params.update(params_common)

    X_train, X_valid = load_data("train"), load_data("vali")

    model = LambdaRank("ranking", params, logger)
    model.fit(X_train, validation_data=X_valid)
    model.save_session()


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "lr":
            train_lr()
        elif sys.argv[1] == "dnn":
            train_dnn()
        elif sys.argv[1] == "ranknet":
            train_ranknet()
        elif sys.argv[1] == "lambdarank":
            train_lambdarank()
    else:
        train_lr()


if __name__ == "__main__":
    main()
