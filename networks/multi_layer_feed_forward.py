import cPickle
import collections
from datetime import datetime
import numpy
import sys
import theano
import theano.tensor as T

import matplotlib.pyplot as plt

from abc import abstractmethod, ABCMeta

from elements.activation_functions import tanh
from layers.image_processing import ConvLayer
from layers.one_dimensional import HiddenLayer, SoftmaxLayer
from tools.dataset import CSVDataset, NumpyDataset


class BaseNetwork(object):
    """
    Base Class for all other Neural Networks.
    Best practice:
        - define network's input/output symbolic variables and also values for network's parameter (numpy) in '__init__'
        - save values for parameters into file in 'save'
        - load values for parameters from file in 'load'.
        - define network's architecture and create it's parameters in 'create_network' (use values)
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def create_network(self):
        """
        Creates a network based on parameters that are stored in network param_values.
        :return: None
        """

    @abstractmethod
    def save_network(self, file_address):
        """
        Saves network parameters for future uses.
        :return: None
        """

    @abstractmethod
    def load_network(self, file_address):
        """
        Loads network from given file.
        :return: None
        """

    @abstractmethod
    def get_error(self):
        """
        Returns network's error from end-user's point of view, Zero-One error is recommended.
        :return: A function that gets correct output and returns actual network's error.
        """

    @abstractmethod
    def get_cost(self):
        """
        Return network's cost.
        :return: A function that gets correct output and returns actual network's cost.
        """

    @abstractmethod
    def get_input(self):
        """
        :return: Theano symbolic variable of network's input.
        """

    @abstractmethod
    def get_output(self):
        """
        :return: Theano symbolic variable of network's output.
        """

    @abstractmethod
    def get_params(self):
        """
        :return: Theano symbolic variable of network's input.
        """

    @abstractmethod
    def train(self, dataset_address):
        """
        Train The network on a given dataset.
        """


class ShallowNetwork(BaseNetwork):
    """
    This is mostly for debugging and tutorial uses to show you how to define your networks.
    """
    def __init__(self, file_address=None, input_size=None, nof_middle=None, nof_output=None, l1_reg=None, l2_reg=None):
        self.x = T.matrix('x')    # the data is presented as sequence of floats
        self.y = T.imatrix('y')   # the labels are presented as 1D vector of [int] labels (Rejected or Not Rejected)

        if file_address is None:
            if input_size is None or nof_middle is None or l1_reg is None or l2_reg is None or nof_output is None:
                raise Exception("You should set a file name or all of input_size, nof_middle, l1 and l2 reg")

            self.input_size = input_size
            self.nof_middle = nof_middle
            self.nof_output = nof_output
            self.l1_reg = l1_reg
            self.l2_reg = l2_reg
            self.layer1_w_values = None
            self.layer1_b_values = None
            self.layer2_w_values = None
            self.layer2_b_values = None
        else:
            self.load_network(file_address)

        self.create_network()

    def create_network(self):
        # allocate symbolic variables for the data
        # construct the MLP class
        self.layer1 = HiddenLayer(n_in=self.input_size,
                                  n_out=self.nof_middle,
                                  layer_input=self.x,
                                  activation=tanh,
                                  w_values=self.layer1_w_values,
                                  b_values=self.layer1_b_values)

        self.layer2 = SoftmaxLayer(n_in=self.nof_middle,
                                   n_out=self.nof_output,
                                   layer_input=self.layer1.output,
                                   w_values=self.layer2_w_values,
                                   b_values=self.layer2_b_values)

        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        self.cost = (
            self.layer2.get_cost(self.y) +
            self.l1_reg * (self.layer1.get_l1() + self.layer2.get_l1()) +
            self.l2_reg * (self.layer1.get_l2_sqr() + self.layer2.get_l2_sqr())
        )

        self.error = self.layer2.get_error(self.y)

        self.params = self.layer1.get_params() + self.layer2.get_params()

    def save_network(self, file_address):
        network_dict = {"layer1": self.layer1.get_dict(),
                        "layer2": self.layer2.get_dict(),
                        "l1_reg": self.l1_reg,
                        "l2_reg": self.l2_reg}

        with open(file_address, "wb") as my_file:
            cPickle.dump(network_dict, my_file, protocol=2)

    def load_network(self, file_address):
        with open(file_address, "wb") as my_file:
            network_dict = cPickle.load(my_file)
            self.input_size = network_dict["layer1"]["n_in"]
            self.nof_middle = network_dict["layer1"]["n_out"]
            self.nof_output = network_dict["layer2"]["n_out"]
            self.l1_reg = network_dict["l1_reg"]
            self.l2_reg = network_dict["l2_reg"]
            self.layer1_w_values = network_dict["layer1"]["w_values"]
            self.layer1_b_values = network_dict["layer1"]["b_values"]
            self.layer2_w_values = network_dict["layer2"]["w_values"]
            self.layer2_b_values = network_dict["layer2"]["b_values"]

    def train(self, dataset_address):
        # Project settings
        batch_size = 20
        learning_rate = 0.01
        momentum_rate = 0.95
        n_epochs = 1000

        dataset = NumpyDataset(dataset_address)

        train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = dataset.get_dataset()

        # compute number of minibatches for training, validation and testing
        n_train_batches, n_valid_batches, n_test_batches = dataset.get_number_of_batches(batch_size)

        index = T.lscalar()

        test_model = theano.function(inputs=[index], outputs=self.get_error(),
                                     givens={
                                            self.get_input(): test_set_x[index * batch_size:(index + 1) * batch_size],
                                            self.get_output(): test_set_y[index * batch_size:(index + 1) * batch_size]
                                            })

        validate_model = theano.function(inputs=[index], outputs=self.get_cost(),
                                         givens=
                                         {
                                          self.get_input(): valid_set_x[index * batch_size:(index + 1) * batch_size],
                                          self.get_output(): valid_set_y[index * batch_size:(index + 1) * batch_size]
                                         })

        delta_params = [theano.shared(value=numpy.zeros(param.get_value(borrow=True).shape), borrow=True)
                        for param in self.get_params()]

        params_g = [T.grad(self.get_cost(), param) for param in self.get_params()]

        # Normal SGD
        # updates = [(param, param - learning_rate * param_g) for param, param_g in zip(self.get_params(), params_g)]

        # Momentum SGD
        updates = [(delta_param, -learning_rate*param_g + momentum_rate*delta_param)
                   for param_g, delta_param in zip(params_g, delta_params)]

        updates += [(param, param + delta_param) for param, delta_param in zip(self.get_params(), delta_params)]

        train_model = theano.function(inputs=[index], outputs=self.get_cost(), updates=updates,
                                      givens=
                                      {
                                            self.get_input(): train_set_x[index * batch_size: (index + 1) * batch_size],
                                            self.get_output(): train_set_y[index * batch_size: (index + 1) * batch_size]
                                      })

        ###############
        # TRAIN MODEL #
        ###############
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is considered significant
        validation_frequency = min(n_train_batches, patience / 2)

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        epoch = 0
        done_looping = False

        train_error_history = collections.OrderedDict()
        validation_error_history = collections.OrderedDict()

        while (epoch < n_epochs) and (not done_looping):
            epoch += 1

            for minibatch_index in xrange(n_train_batches):
                # iteration number
                iteration_number = (epoch - 1) * n_train_batches + minibatch_index

                # Save iteration history
                minibatch_avg_cost = train_model(minibatch_index)
                train_error_history[iteration_number] = minibatch_avg_cost

                # validate every validation_frequency time :)
                if (iteration_number + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print('\r epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.)),
                    sys.stdout.flush()

                    # Save validation history
                    validation_error_history[iteration_number] = this_validation_loss

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iteration_number * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iteration_number

                        # test it on the test set
                        test_losses = [test_model(i) for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                if patience <= iteration_number:
                    done_looping = True
                    break
        plt.plot(train_error_history.keys(), train_error_history.values(), 'r',
                 validation_error_history.keys(), validation_error_history.values(), 'b')

        print "\nTraining is Done"
        sys.stdout.flush()
        plt.savefig("assets/outputs/image-%s-%2.4f.jpg" % (str(datetime.now().replace(microsecond=0)), test_score * 100.))
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    def get_error(self):
        return self.error

    def get_cost(self):
        return self.cost

    def get_input(self):
        return self.x

    def get_output(self):
        return self.y

    def get_params(self):
        return self.params


class ConvNetwork(BaseNetwork):
    """
    This is mostly for debugging and tutorial uses to show you how to define your Convolutional networks.
    """
    def __init__(self, file_address=None, input_shape=None, nof_output=None, l1_reg=None, l2_reg=None):
        self.x = T.matrix('x')    # the data is presented as sequence of floats
        self.y = T.imatrix('y')   # the labels are presented as 1D vector of [int] labels (Rejected or Not Rejected)

        self.batch_size = 20
        # For now, we'll have two Conv layer, one fully-connceted layer, and one output (softmax) layer.

        self.x_reshaped = self.x.reshape((self.batch_size, 1, input_shape[0], input_shape[1]))

        if file_address is None:
            if input_shape is None or l1_reg is None or l2_reg is None or nof_output is None:
                raise Exception("You should set a file name or all of input_shape, nof_output, l1 and l2 reg")

            self.input_size = input_shape
            self.nof_output = nof_output
            self.l1_reg = l1_reg
            self.l2_reg = l2_reg
            self.layer1_w_values = None
            self.layer1_b_values = None
            self.layer2_w_values = None
            self.layer2_b_values = None
            self.layer3_w_values = None
            self.layer3_b_values = None
            self.layer4_w_values = None
            self.layer4_b_values = None

        else:
            self.load_network(file_address)

        self.create_network()

    def create_network(self):
        # allocate symbolic variables for the data
        # construct the MLP class
        nkerns = (20,50)

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        self.layer0 = ConvLayer(filter_shape=(nkerns[0], 1, 5, 5),
                                image_shape=(self.batch_size, 1, 28, 28),
                                layer_input=self.x_reshaped,
                                pool_size=(2, 2),
                                activation=tanh)

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        self.layer1 = ConvLayer(filter_shape=(nkerns[1], nkerns[0], 5, 5),
                                image_shape=(self.batch_size, nkerns[0], 12, 12),
                                layer_input=self.layer0.output,
                                pool_size=(2, 2),
                                activation=tanh)

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        layer2_input = self.layer1.output.flatten(2)

        self.layer2 = HiddenLayer(n_in=nkerns[1]*4*4,
                                  n_out=500,
                                  layer_input=layer2_input,
                                  activation=tanh,
                                  w_values=self.layer2_w_values,
                                  b_values=self.layer2_b_values)

        self.layer3 = SoftmaxLayer(n_in=500,
                                   n_out=10,
                                   layer_input=self.layer2.output,
                                   w_values=self.layer3_w_values,
                                   b_values=self.layer3_b_values)

        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        self.cost = (
            self.layer3.get_cost(self.y) +
            self.l1_reg * (self.layer0.get_l1() + self.layer1.get_l1() + self.layer2.get_l1() + self.layer3.get_l1()) +
            self.l2_reg * (self.layer0.get_l2_sqr() + self.layer1.get_l2_sqr() + self.layer2.get_l2_sqr() + self.layer3.get_l2_sqr())
        )

        self.error = self.layer3.get_error(self.y)

        self.params = self.layer0.get_params() + self.layer1.get_params() + self.layer2.get_params() + self.layer3.get_params()

    def save_network(self, file_address):
        network_dict = {"layer1": self.layer1.get_dict(),
                        "layer2": self.layer2.get_dict(),
                        "l1_reg": self.l1_reg,
                        "l2_reg": self.l2_reg}

        with open(file_address, "wb") as my_file:
            cPickle.dump(network_dict, my_file, protocol=2)

    def load_network(self, file_address):
        with open(file_address, "wb") as my_file:
            network_dict = cPickle.load(my_file)
            self.input_size = network_dict["layer1"]["n_in"]
            self.nof_middle = network_dict["layer1"]["n_out"]
            self.nof_output = network_dict["layer2"]["n_out"]
            self.l1_reg = network_dict["l1_reg"]
            self.l2_reg = network_dict["l2_reg"]
            self.layer1_w_values = network_dict["layer1"]["w_values"]
            self.layer1_b_values = network_dict["layer1"]["b_values"]
            self.layer2_w_values = network_dict["layer2"]["w_values"]
            self.layer2_b_values = network_dict["layer2"]["b_values"]

    def train(self, dataset_address):
        # Project settings
        batch_size = 20
        learning_rate = 0.01
        momentum_rate = 0.95
        n_epochs = 1000

        dataset = NumpyDataset(dataset_address)

        train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = dataset.get_dataset()

        # compute number of minibatches for training, validation and testing
        n_train_batches, n_valid_batches, n_test_batches = dataset.get_number_of_batches(batch_size)

        index = T.lscalar()

        test_model = theano.function(inputs=[index], outputs=self.get_error(),
                                     givens={
                                            self.get_input(): test_set_x[index * batch_size:(index + 1) * batch_size],
                                            self.get_output(): test_set_y[index * batch_size:(index + 1) * batch_size]
                                            })

        validate_model = theano.function(inputs=[index], outputs=self.get_cost(),
                                         givens=
                                         {
                                          self.get_input(): valid_set_x[index * batch_size:(index + 1) * batch_size],
                                          self.get_output(): valid_set_y[index * batch_size:(index + 1) * batch_size]
                                         })

        delta_params = [theano.shared(value=numpy.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX)
                                      , borrow=True)
                        for param in self.get_params()]

        params_g = [T.grad(self.get_cost(), param) for param in self.get_params()]

        # Normal SGD
        # updates = [(param, param - learning_rate * param_g) for param, param_g in zip(self.get_params(), params_g)]

        # Momentum SGD
        updates = [(delta_param, -learning_rate*param_g + momentum_rate*delta_param)
                   for param_g, delta_param in zip(params_g, delta_params)]

        updates += [(param, param + delta_param) for param, delta_param in zip(self.get_params(), delta_params)]

        train_model = theano.function(inputs=[index], outputs=self.get_cost(), updates=updates,
                                      givens=
                                      {
                                            self.get_input(): train_set_x[index * batch_size: (index + 1) * batch_size],
                                            self.get_output(): train_set_y[index * batch_size: (index + 1) * batch_size]
                                      })

        ###############
        # TRAIN MODEL #
        ###############
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is considered significant
        validation_frequency = min(n_train_batches, patience / 2)

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        epoch = 0
        done_looping = False

        train_error_history = collections.OrderedDict()
        validation_error_history = collections.OrderedDict()

        while (epoch < n_epochs) and (not done_looping):
            epoch += 1

            for minibatch_index in xrange(n_train_batches):
                # iteration number
                iteration_number = (epoch - 1) * n_train_batches + minibatch_index

                # Save iteration history
                minibatch_avg_cost = train_model(minibatch_index)
                train_error_history[iteration_number] = minibatch_avg_cost

                # validate every validation_frequency time :)
                if (iteration_number + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print('\r epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.)),
                    sys.stdout.flush()

                    # Save validation history
                    validation_error_history[iteration_number] = this_validation_loss

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iteration_number * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iteration_number

                        # test it on the test set
                        test_losses = [test_model(i) for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                if patience <= iteration_number:
                    done_looping = True
                    break
        plt.plot(train_error_history.keys(), train_error_history.values(), 'r',
                 validation_error_history.keys(), validation_error_history.values(), 'b')

        print "\nTraining is Done"
        sys.stdout.flush()
        plt.savefig("assets/outputs/image-%s-%2.4f.jpg" % (str(datetime.now().replace(microsecond=0)), test_score * 100.))
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    def get_error(self):
        return self.error

    def get_cost(self):
        return self.cost

    def get_input(self):
        return self.x

    def get_output(self):
        return self.y

    def get_params(self):
        return self.params
