from typing import List
from .client import Client
from .util import *


class Server(object):
    def __init__(self, Y_train, Y_test, config) -> None:
        # Save label data
        self.Y_train = Y_train
        self.Y_test = Y_test

        # Extract config info
        self.class_num = config['class_num']
        self.client_num = config['client_num']
        self.epoch_num = config['epoch_num']
        self.batch_size = config['batch_size']
        self.lr = config['learning_rate']

        self.data_num = len(Y_train)

        # Empty list used to collect clients
        self.clients = list()

        # Model param of server
        # self.bias = np.zeros(self.class_num)
        self.bias = np.random.random(size=self.class_num)

        # Collected embedding data (shape is (class_num, class_num))
        # = weight (shape is (class_num, feature_num)) \dot 
        # X_batch.T (shape is (feature_num, batch_size)) 
        self.embedding_data = np.zeros(shape=(self.client_num,
                                              self.class_num, self.batch_size))
        # For test eval
        self.test_embedding_data = np.zeros(shape=(self.client_num,
                                                   self.class_num, len(self.Y_test)))

        self.batch_indexes = [0] * self.batch_size

        # w.r.t each client's embedding data
        self.embedding_grads = np.zeros(shape=(self.class_num, self.batch_size))

    def attach_clients(self, clients: List[Client]):
        """ Attach clients to the server. 
        The server can access the client by id.
        """
        self.clients = clients

    def update_embedding_data(self, client: Client, period_type='batch'):
        """ Call client to calculate embedding data and send it to server.
        Server will receive it and save it.
        """
        if period_type == 'test':
            self.test_embedding_data[client.id] = client.get_embedding_data(period_type)
        if period_type == 'batch':
            self.embedding_data[client.id] = client.get_embedding_data(period_type)

    def send_embedding_grads(self, client: Client, grads):
        self.clients[client.id].set_embedding_grads(grads)

    def cal_batch_embedding_grads(self):
        """ Calculate grads w.r.t. embedding data
        """
        loss = 0
        grads = np.zeros(shape=(self.class_num, self.batch_size))
        aggr_embedding_data = np.sum(self.embedding_data, axis=0)  # shape (self.class_num, self.batch_size)
        for i in range(0, self.batch_size):
            # Ground truth
            y = self.Y_train[self.batch_indexes[i]]
            # y = X^T \dot weight + bias
            pred_prob = softmax(aggr_embedding_data[:, i] + self.bias)
            loss -= np.log(pred_prob[y])  # more right prob, less loss

            # Wrong direction, the higher the more deviant
            grads[:, i] = pred_prob
            # Right direction
            grads[y, i] -= 1

        self.embedding_grads = grads

        # Send it to clients
        for c in self.clients:
            self.send_embedding_grads(c, grads)

        return loss / self.batch_size

    def update_bias(self):
        self.bias -= self.lr * (np.sum(self.embedding_grads, axis=1) / self.batch_size)
