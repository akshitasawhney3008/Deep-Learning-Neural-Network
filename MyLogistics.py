import h5py
import numpy as np
import tensorflow as tf
import math

train_size = 0.90
validation_size = 0.00
test_size = 0.10


class Logistics:

    @staticmethod
    def load_dataset():
        train_dataset = h5py.File('datasets/train_signs.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

        test_dataset = h5py.File('datasets/test_signs.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

        classes = np.array(test_dataset["list_classes"][:])  # the list of classes

        train_set_y_orig = train_set_y_orig.reshape((train_set_y_orig.shape[0],1))
        test_set_y_orig = test_set_y_orig.reshape((test_set_y_orig.shape[0],1))

        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    @staticmethod
    def merge_data(train_data_x, train_data_y, test_data_x, test_data_y):
        np.random.seed(0)
        train_data = np.hstack((train_data_x,train_data_y))
        test_data = np.hstack((test_data_x,test_data_y))
        whole_data = np.vstack((train_data,test_data))

        np.random.shuffle(whole_data)

        return whole_data

    @staticmethod
    def flatten_normalize_array(xtrain, xtest):
        # Flatten each array
        xtrain_flatten = xtrain.reshape(xtrain.shape[0], -1)
        xtest_flatten = xtest.reshape(xtest.shape[0], -1)

        # Normalize image vectors
        xtrain = xtrain_flatten / 255
        xtest = xtest_flatten / 255
        return xtrain,xtest


    @staticmethod
    def divide_into_train_dev_test(whole_data, classes):
        num_rows = whole_data.shape[0]    #get the num of rows in whole data
        num_columns = whole_data.shape[1]   # get the num of columns in whole data
        train_row_count = int(num_rows * train_size)     #num of rows for train data
        train_count_eachsign = int(train_row_count/6)    # num of pictures for each sign in train data
        validation_row_count = int(num_rows * validation_size)     # num of rows for validation data
        validation_count_eachsign = int(validation_row_count/6)    # num of pictures for each sign in validation data
        # test_row_count = int(num_rows * test_size)                 # num of rows for test data
        # test_count_eachsign = int(test_row_count/6)                # num of pictures for each sign in test data
        unique_labels = classes               # labels in the target column: signs for numbers 0-5

        #select train images
        flag = 0
        for label in unique_labels:
            if flag == 0:
                train_indices = np.where(whole_data[:,-1] == label)[0][:train_count_eachsign]
                train_rows = whole_data[train_indices]
                X_train = train_rows

                # Y_train = train_labels_rows
                whole_data = np.delete(whole_data, train_indices,axis=0)
                flag = 1
            else:
                train_indices = np.where(whole_data[:,-1] == label)[0][:train_count_eachsign]
                train_rows = whole_data[train_indices]

                X_train=np.vstack((X_train,train_rows))
                whole_data = np.delete(whole_data, train_indices,axis=0)

        #select validation images
        flag = 0
        for label in unique_labels:
            if flag==0:
                validation_indices=  np.where(whole_data[:,-1]==label)[0][:validation_count_eachsign]
                validation_rows = whole_data[validation_indices]
                X_valid = validation_rows
                whole_data = np.delete(whole_data,validation_indices,axis=0)
                flag=1

            else:
                validation_indices = np.where(whole_data[:, -1] == label)[0][:validation_count_eachsign]
                validation_rows = whole_data[validation_indices]
                X_valid = np.vstack((X_valid,validation_rows))
                whole_data = np.delete(whole_data, validation_indices, axis=0)

        #rest images will be left for test
        X_test = whole_data.copy()

        return X_train,X_valid, X_test

    @staticmethod
    def get_X_Y(train_data, validation_data, test_data):
        return train_data[:,:-1], train_data[:,-1].reshape((1,-1)), validation_data[:,:-1], validation_data[:,-1].reshape((1,-1)), \
               test_data[:,:-1], test_data[:,-1].reshape((1,-1))

    @staticmethod
    def convert_to_one_hot(Y,C):
        Y=Y.reshape(-1)
        Y = np.eye(C)[Y].T
        return Y



class Parameters:
    def __init__(self, num_hidden_nodes_for_shared_network,num_shared_layers,initial_weights,keep_prob, input_dimensions, my_lambda):

        self.num_layers = num_shared_layers
        self.num_hidden_nodes = num_hidden_nodes_for_shared_network
        self.init_weights = initial_weights
        self.keep_prob = keep_prob
        self.input_dimensions = input_dimensions
        self.my_lambda = my_lambda
























