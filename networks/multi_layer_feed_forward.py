import cPickle
import collections
import numpy
import sys
import theano
import theano.tensor as T

import matplotlib.pyplot as plt

from abc import abstractmethod, ABCMeta

from elements.activation_functions import tanh
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

        params_g = [T.grad(self.get_cost(), param) for param in self.get_params()]

        # Normal SGD
        updates = [(param, param - learning_rate * param_g) for param, param_g in zip(self.get_params(), params_g)]

        # Momentum SGD

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
                itteration_number = (epoch - 1) * n_train_batches + minibatch_index

                # Save itteration history
                minibatch_avg_cost = train_model(minibatch_index)
                train_error_history[itteration_number] = minibatch_avg_cost

                # validate every validation_frequency time :)
                if (itteration_number + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print('\r epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.)),
                    sys.stdout.flush()

                    # Save validation history
                    validation_error_history[itteration_number] = this_validation_loss

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, itteration_number * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = itteration_number

                        # test it on the test set
                        test_losses = [test_model(i) for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                if patience <= itteration_number:
                    done_looping = True
                    break
        plt.plot(train_error_history.keys(), train_error_history.values(), 'r',
                 validation_error_history.keys(), validation_error_history.values(), 'b')

        print validation_error_history.keys()
        plt.savefig("assets/outputs/image.jpg")
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
