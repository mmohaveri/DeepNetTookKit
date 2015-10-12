import cPickle
import theano.tensor as T

from abc import abstractmethod, ABCMeta

from elements.activation_functions import tanh
from layers.one_dimensional import HiddenLayer, SoftmaxLayer


class BaseNetwork:
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


class ShallowNetwork(BaseNetwork):
    """
    This is mostly for debugging and tutorial uses to show you how to define your networks.
    """
    def __init__(self, file_address=None, input_size=None, nof_middle=None, l1_reg=None, l2_reg=None):
        self.x = T.matrix('x')    # the data is presented as sequence of floats
        self.y = T.ivector('y')   # the labels are presented as 1D vector of [int] labels (Rejected or Not Rejected)

        if file_address is None:
            if input_size is None or nof_middle is None or l1_reg is None or l2_reg is None:
                raise Exception("You should set a file name or all of input_size, nof_middle, l1 and l2 reg")

            self.input_size = input_size
            self.nof_middle = nof_middle
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
                                   n_out=2,
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
            self.l1_reg = network_dict["l1_reg"]
            self.l2_reg = network_dict["l2_reg"]
            self.layer1_w_values = network_dict["layer1"]["w_values"]
            self.layer1_b_values = network_dict["layer1"]["b_values"]
            self.layer2_w_values = network_dict["layer2"]["w_values"]
            self.layer2_b_values = network_dict["layer2"]["b_values"]

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
