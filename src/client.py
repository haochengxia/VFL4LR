import re
from turtle import shape
import numpy as np

class Client(object):
    def __init__(self, X_train, X_test, config) -> None:
        self.X_train = X_train
        self.X_test = X_test

        # Extract config info
        self.batch_size = config['batch_size']
        self.lr = config['learning_rate']
        self.class_num = config['class_num']

        # Init client's params
        self.batch_indexs = [0] * self.batch_size
        feature_num = len(X_test[0])
        # self.weight = np.zeros(shape=(self.class_num, feature_num))
        # self.embeding_grads = np.zeros(shape=(self.class_num, self.batch_size))
        self.weight = np.random.random(size=(self.class_num, feature_num))
        self.embeding_grads = np.random.random(size=(self.class_num, self.batch_size))
        self.id = None

    def set_id(self, id):
        self.id = id
    
    def set_batch_indexs(self, batch_indexs):
        self.batch_indexs = batch_indexs

    def set_embeding_grads(self, embeding_grads):
        self.embeding_grads = embeding_grads

    # Update weight param of the model
    def update_weight(self):
        X_batch = self.X_train[self.batch_indexs]
        # (class_num, feature_num) =
        # (class_num, batch_size) \dot (batch_size, feature_num)
        grad = np.sum((np.dot(self.embeding_grads, X_batch))) / self.batch_size
        # SGD
        self.weight -= self.lr * grad

    def get_embedding_data(self, period_type="batch"):
        """
        Return the embedding data, calculated on X.
            batch trainning - X_batch
            testing - X_test
        """
        if period_type == 'batch':
            X_batch = self.X_train[self.batch_indexs]
            res = np.dot(self.weight, X_batch.T) # (class_num, batch_size)
        else:
            # 'test'
            res = np.dot(self.weight, self.X_test.T) # (class_num, test_size)
        return res