import MyLogistics
import MyTrainer
import pickle as pkl

scenario_switch= 2
data_dir = 'Data/'
objects_dir = 'objects/'
num_hidden_nodes_for_shared_network = [25,12,6]
num_shared_layers = 3
initial_weights = 0.01
keep_prob =1.0
learning_rate = 0.0001
batch_size = 32
print_cost = True
num_epochs = 2000
learning_rate_decay = 0.01
num_iterations_per_decay = 500
# my_lambda = 0.000005
my_lambda = 0
mod_path = 'Models/'


if scenario_switch == 1:

    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = MyLogistics.Logistics.load_dataset()
    train_set_x_orig_flatten, test_set_x_orig_flatten = MyLogistics.Logistics.flatten_normalize_array(train_set_x_orig, test_set_x_orig)
    whole_data= MyLogistics.Logistics.merge_data(train_set_x_orig_flatten,train_set_y_orig,test_set_x_orig_flatten,test_set_y_orig)
    train_data,validation_data,test_data = MyLogistics.Logistics.divide_into_train_dev_test(whole_data, classes)



    with open(objects_dir + 'train_data.pkl', 'wb') as f:
        pkl.dump(train_data, f)
    with open(objects_dir + 'valid_data.pkl', 'wb') as f:
        pkl.dump(validation_data, f)
    with open(objects_dir + 'test_data.pkl', 'wb') as f:
        pkl.dump(test_data, f)

else:
    with open(objects_dir + 'train_data.pkl', 'rb') as f:
        train_data = pkl.load(f)
    with open(objects_dir + 'valid_data.pkl', 'rb') as f:
        validation_data = pkl.load(f)
    with open(objects_dir + 'test_data.pkl', 'rb') as f:
        test_data= pkl.load(f)
    input_dimensions = train_data[:,:-1].shape[1]


    print("number of training examples = " + str(train_data.shape[0]))
    print("number of validation examples = " + str(validation_data.shape[0]))
    print("number of test examples = " + str(test_data.shape[0]))
    print("X_train shape: " + str(train_data.shape))
    print('X_valid shape' + str(validation_data.shape))
    # print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(test_data.shape))
    # print("Y_test shape: " + str(Y_test.shape))

    MyTrainer.ModelMyNeuralNetwork(train_data,validation_data,test_data, num_hidden_nodes_for_shared_network,num_shared_layers,
                                   initial_weights,keep_prob, learning_rate, learning_rate_decay,num_iterations_per_decay, my_lambda, batch_size,
                                   print_cost, num_epochs, mod_path, input_dimensions)







