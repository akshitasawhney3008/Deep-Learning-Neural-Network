import MyLogistics
import MyNeuralNetwork
import tensorflow as tf
import random
from tensorflow.python.framework import ops
import pickle as pkl
import pandas as pd
import numpy as np
import os

random.seed(0)

class ModelMyNeuralNetwork():
    def __init__(self, train_data, validation_data, test_data,num_hidden_nodes_for_shared_network,num_shared_layers,
                                   initial_weights,keep_prob, learning_rate, learning_rate_decay,num_iterations_per_decay, my_lambda,  batch_size, print_cost, num_epochs, mod_path, input_dimensions):

        X_train, Y_train, X_valid, Y_valid, X_test, Y_test = MyLogistics.Logistics.get_X_Y(train_data, validation_data, test_data)
        Y_train,Y_valid, Y_test = MyLogistics.Logistics.convert_to_one_hot(Y_train.astype('int'),6).T, \
                                  MyLogistics.Logistics.convert_to_one_hot(Y_valid.astype('int'),6).T, \
                                  MyLogistics.Logistics.convert_to_one_hot(Y_test.astype('int'),6).T

        ops.reset_default_graph()
        # Create tf session object
        my_session = tf.Session()
        costs = []
        x = tf.placeholder("float", shape=[None, input_dimensions], name='x')
        y = tf.placeholder('float', shape=[None, 6], name ='y')

        # create Parameters object
        params = MyLogistics.Parameters(num_hidden_nodes_for_shared_network,num_shared_layers,initial_weights,keep_prob, input_dimensions, my_lambda)

        my_nn = MyNeuralNetwork.MyNN(x, y, params)

        # Setup optimizer
        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        my_step = tf.Variable(0, trainable=False)
        increment_global_step = tf.assign(my_step, my_step + 1)
        my_lr = tf.train.inverse_time_decay(learning_rate, increment_global_step, num_iterations_per_decay, learning_rate_decay)
        trainer = tf.train.AdamOptimizer(learning_rate=my_lr)
        optimizer = trainer.minimize(my_nn.cost, global_step=my_step)

        # Initialize all the variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            # Run the initialization
            sess.run(init)
            num_of_batches = int(X_train.shape[0] / batch_size)

            for epoch in range(num_epochs):
                epoch_cost = 0       # Defines a cost related to an epoch

                list_of_indices = list(range(0, X_train.shape[0]))
                list_of_batch_indices = []
                # Creating random batches
                for batch_iter in range(num_of_batches):
                    sampled_indices = random.sample(list_of_indices, batch_size)
                    list_of_batch_indices.append(sampled_indices)
                    list_of_indices = list(set(list_of_indices) - set(sampled_indices))
                if len(list_of_indices) != 0:
                    list_of_batch_indices.append(list_of_indices)

                for minibatch in list_of_batch_indices:
                    # print(list_of_batch_indices.index(minibatch))
                    minibatch_x = X_train[minibatch,:]
                    minibatch_y = Y_train[minibatch,:]
                    _,minibatch_cost = sess.run([optimizer,my_nn.cost], feed_dict={x: minibatch_x, y: minibatch_y})

                    epoch_cost += minibatch_cost / num_of_batches

                cost, weights_shared, y_pred, y_factual, lr =  sess.run([my_nn.cost,my_nn.weights_shared,my_nn.output, my_nn.y, trainer. _lr],\
                                                               feed_dict={x:X_train,y:Y_train})

                if print_cost == True and epoch % 100 == 0:
                    correct_prediction = np.equal(np.argmax(y_pred, axis=1), np.argmax(y_factual, axis=1))
                    accuracy = np.sum(correct_prediction) / X_train.shape[0] * 100
                    print('train epoch, accuracy, cost',epoch, accuracy, cost, lr)

                cost, weights_shared, y_pred, y_factual = sess.run([my_nn.cost, my_nn.weights_shared, my_nn.output, my_nn.y], \
                    feed_dict={x: X_test, y: Y_test})
                # print(cost)


            # Calculate the correct predictions
                if print_cost == True and epoch % 100 == 0:
                    correct_prediction = np.equal(np.argmax(y_pred, axis=1), np.argmax(y_factual, axis=1))
                    accuracy = np.sum(correct_prediction) / X_test.shape[0] * 100
                    print('test epoch, accuracy, cost',epoch, accuracy, cost)


            # saver = tf.train.Saver()
            # save_path = saver.save(sess, os.path.join(mod_path, "{0}.ckpt".format("final")))
            # print("Model saved to: {0}".format(save_path))










