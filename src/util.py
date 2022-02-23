import numpy as np
from libsvm.svmutil import *


def softmax(tp):
    return np.exp(tp) / np.sum(np.exp(tp))


def gen_batches(data_num, batch_size):
    batch_num = data_num // batch_size
    size_list = [batch_size] * batch_num
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    res = list()
    b = 0
    for size in size_list:
        res.append(indexes[b: b + size])
        b += size
    return res


def load_data(file_name):
    if file_name == 'adult':
        Y_train, X_train = svm_read_problem("./data/adult/a8a")
        for idx, y in enumerate(Y_train):
            if y < 0:
                Y_train[idx] = 0
            else:
                Y_train[idx] = 1
        X_train_array = np.array(to_array(X_train))
        Y_test, X_test = svm_read_problem("./data/adult/a8a.t")
        for idx, y in enumerate(Y_test):
            if y < 0:
                Y_test[idx] = 0
            else:
                Y_test[idx] = 1
        X_test_array = np.array(to_array(X_test))
        return X_train_array, Y_train, X_test_array, Y_test
    else:
        raise Exception('this dataset is not supported yet')


def split_data(X_train, X_test, client_num):
    feature_num = len(X_test[0])
    each_client_f_num = feature_num // client_num
    X_train_s = [X_train[:, each_client_f_num * i: each_client_f_num * (i + 1)]
                 for i in range(client_num)]
    X_test_s = [X_test[:, each_client_f_num * i: each_client_f_num * (i + 1)]
                for i in range(client_num)]

    return X_train_s, X_test_s


def to_array(X_libsvm, file_name='adult'):
    feature_num = 0
    if file_name == 'adult':
        feature_num = 123
    res = []
    for X in X_libsvm:
        X_real = []
        X = dict(X)
        # 123 features
        for i in range(1, feature_num + 1):
            if X.get(i) != 1:
                X_real.append(0)
            else:
                X_real.append(1)
        res.append(X_real)
    return res
