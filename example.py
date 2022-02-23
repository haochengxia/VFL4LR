from typing import List
from src import *

# load data
file_name = 'adult'
X_train, Y_train, X_test, Y_test = load_data(file_name)


# config
config = dict()
config['class_num'] = 2
config['client_num'] = 3
config['epoch_num'] = 5
config['batch_size'] = 2000
config["learning_rate"] = 0.2

# Split Data
X_train_s, X_test_s = split_data(X_train, X_test, config['client_num'])

# Init server
server = Server(Y_train, Y_test, config)

# Init clients
clients = list()
for i in range(config['client_num']):
    c = Client(X_train_s[i], X_test_s[i], config)
    c.set_id(i)
    clients.append(c)

server.attach_clients(clients)

# Train and Evaluation
vfl_lr_train(server, clients)
eval(server, clients)