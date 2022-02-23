from typing import List
from .server import Server, Client
from .util import *


def vfl_lr_train(server: Server, clients: List[Client]):
    test_loss, test_acc = evaluation(server, clients)
    print('[*info] Current Test Loss %f Current Test Acc: %f' % (test_loss, test_acc))
    for epoch in range(server.epoch_num):
        batch_num = server.data_num // server.batch_size
        # Divide batches
        batches = gen_batches(server.data_num, server.batch_size)
        for i in range(batch_num):
            batch_indexes = batches[i]
            # Init clients
            for c in clients:
                c.set_batch_indexes(batch_indexes)
                # Step 1: server calls clients to send embedding data and receive
                server.update_embedding_data(c)
            # Step 2: server calculates the gradient and broadcast it to clients
            loss = server.cal_batch_embedding_grads()

            test_loss, test_acc = evaluation(server, clients)
            if i % 3 == 0:
                print('[*info] Epoch %d Batch %d Current Train Loss: %f' % (epoch, i, loss))
                print('[*info] Current Test Loss %f Current Test Acc: %f' % (test_loss, test_acc))

            # Step 3: server updates the server side model
            server.update_bias()
            # Step 4: Clients update models
            for c in clients:
                c.update_weight()


def evaluation(server: Server, clients: List[Client]):
    # Show the performance on Test Dataset
    test_loss = 0
    test_acc = 0
    for c in clients:
        server.update_embedding_data(c, period_type="test")

    aggr_embedding_data = np.sum(server.test_embedding_data, axis=0)

    for idx, y in enumerate(server.Y_test):
        pred_prob = softmax(aggr_embedding_data[:, idx] + server.bias)
        test_loss -= np.log(pred_prob[y])
        if np.argmax(pred_prob) == y:
            test_acc += 1

    test_acc /= len(server.Y_test)

    return test_loss, test_acc
