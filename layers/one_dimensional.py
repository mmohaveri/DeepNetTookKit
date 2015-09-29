import numpy
import theano
import theano.tensor as T
import time

from elements.cost_functions import l1_norm, l2_norm, l2_norm_sqr
from elements.erro_functions import negative_log_likelihood_error, zero_one_error

class BaseLayer:
    """
    Base class for all neural network layers.
    """

    def __init__(self, n_in, n_out, input, activation=None, rng=None, W=None, b=None, W_values=None, b_values=None):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared with another architecture or if the weights should be initialize in any other way than random.
                  If the layer should be standalone and the weitght should initialized to random, leave this to None.

        :type b: theano.tensor.TensorType
        :param b: Theano variable pointing to a set of biases that should be
                  shared with another architecture or if the biases should be initialize in any other way than random.
                  If the layer should be standalone and the biases should initialized to random, leave this to None.

        :type W_values: numpy.matrix
        :param W_values: initial value for W

        :type b_values: numpy.matrix
        :param b_values: initial value for b

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        :type activation: theano.Op or function that works on theano.tensor.TensorType
        :param activation: Non linearity to be applied in the hidden layer

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the architecture (one minibatch)
        """
        self.input = input

        self.n_in = n_in
        self.n_out = n_out

        self.activation = activation

        if rng in None:
            if DEBUG:
                self.rng = numpy.random.RandomState(1234)
            else:
                self.rng = numpy.random.RandomState(int(time.time()))
        else:
            self.rng = rng

        if W is None:
            if W_values in None:
                AbsBandOfWeights = numpy.sqrt(6. / (n_in + n_out))
                W_values = numpy.asarray(self.rng.uniform(low=-AbsBandOfWeights,\
                                                       high=AbsBandOfWeights,\
                                                       size=(n_in, n_out)),\
                                                       dtype = theano.config.floatX)
            self.W = theano.shared(value= W_values, name="W", borrow=True)
        else:
            self.W = W

        if b is None:
            if b_values is None:
                b_values = numpy.zeros((n_out,), dtype= theano.config.floatX)
            self.b = theano.shared(value=b_values, name="b", borrow=True)
        else:
            self.b = b

        if activation is None:
            self.activation = lamda x: x
        else:
            self.activation = activation

        self.params = [self.W , self.b]

    def get_l1(self):
        return l1_norm(self.W)

    def get_l2(self):
        return l2_norm(self.W)

    def get_l2_sqr(self):
        return l2_norm_sqr(self.W)


class SoftmaxLayer(BaseLayer):
    """
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, n_in, n_out, input, rng=None, W=None, b=None, W_values=None, b_values=None):
        super(SoftmaxLayer, self).__init__(n_in, n_out, input, rng=rng, W=W, b=b, W_values=W_values, b_values=b_values)
        self.P_y_given_x = T.nnet.softmax(T.dot(self.input, self.W)+self.b)
        self.y_prediction = T.argmax(self.P_y_given_x, axis = 1)
        self.output = self.y_prediction

    def get_cost(self, target):
        negative_log_likelihood_error(self.P_y_given_x, target)

    def get_error(self, target):
        zero_one_error(self.y_prediction, target)

class HiddenLayer(BaseLayer):
    def __init__(self, n_in, n_out, input, activation=None, rng=None, W=None, b=None, W_values=None, b_values=None):
        super(HiddenLayer, self).__init__(n_in, n_out, input, activation, rng, W, b, W_values, b_values)
        weightedSum = T.dot(self.input, self.W) + self.b
        self.output = self.activation(weightedSum)
