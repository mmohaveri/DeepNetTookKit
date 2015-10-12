import numpy
import theano
import theano.tensor as T
import time

from elements.cost_functions import l1_norm, l2_norm, l2_norm_sqr
from elements.error_functions import negative_log_likelihood_error, zero_one_error


class BaseLayer:
    """
    Base class for all neural network layers.
    """

    def __init__(self, n_in, n_out, layer_input, activation=None, rng=None, w=None, b=None, w_values=None, b_values=None):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type w: theano.tensor.TensorType
        :param w: Theano variable pointing to a set of weights that should be
                  shared with another architecture or if the weights should be initialize in any other way than random.
                  If the layer should be standalone and the weitght should initialized to random, leave this to None.

        :type b: theano.tensor.TensorType
        :param b: Theano variable pointing to a set of biases that should be
                  shared with another architecture or if the biases should be initialize in any other way than random.
                  If the layer should be standalone and the biases should initialized to random, leave this to None.

        :type w_values: numpy.matrix
        :param w_values: initial value for W

        :type b_values: numpy.matrix
        :param b_values: initial value for b

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        :type activation: theano.Op or function that works on theano.tensor.TensorType
        :param activation: Non linearity to be applied in the hidden layer

        :type layer_input: theano.tensor.TensorType
        :param layer_input: symbolic variable that describes the input of the architecture (one minibatch)
        """
        self.input = layer_input

        self.n_in = n_in
        self.n_out = n_out

        self.activation = activation

        if rng in None:
            self.rng = numpy.random.RandomState(int(time.time()))
        else:
            self.rng = rng

        if w is None:
            if w_values in None:
                abs_band_of_weights = numpy.sqrt(6. / (n_in + n_out))
                w_values = numpy.asarray(self.rng.uniform(low=-abs_band_of_weights,
                                                          high=abs_band_of_weights,
                                                          size=(n_in, n_out)),
                                         dtype=theano.config.floatX)

            self.w = theano.shared(value=w_values, name="W", borrow=True)
        else:
            self.w = w

        if b is None:
            if b_values is None:
                b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name="b", borrow=True)
        else:
            self.b = b

        if activation is None:
            self.activation = lambda x: x
        else:
            self.activation = activation

        self.params = [self.w, self.b]

    def get_l1(self):
        return l1_norm(self.w)

    def get_l2(self):
        return l2_norm(self.w)

    def get_l2_sqr(self):
        return l2_norm_sqr(self.w)

    def get_params(self):
        return self.params

    def get_dict(self):
        return {
                "n_in": self.n_in,
                "n_out": self.n_out,
                "w_values": self.w.get_value(),
                "b_values": self.b.get_value(),
               }


class SoftmaxLayer(BaseLayer):
    """
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, n_in, n_out, layer_input, rng=None, w=None, b=None, w_values=None, b_values=None):
        super(SoftmaxLayer, self).__init__(n_in, n_out, layer_input, rng=rng, w=w, b=b, w_values=w_values,
                                           b_values=b_values)
        self.P_y_given_x = T.nnet.softmax(T.dot(self.input, self.w)+self.b)
        self.y_prediction = T.argmax(self.P_y_given_x, axis=1)
        self.output = self.y_prediction

    def get_cost(self, target):
        return negative_log_likelihood_error(self.P_y_given_x, target)

    def get_error(self, target):
        return zero_one_error(self.y_prediction, target)


class HiddenLayer(BaseLayer):
    def __init__(self, n_in, n_out, layer_input, activation=None, rng=None, w=None, b=None, w_values=None, b_values=None):
        super(HiddenLayer, self).__init__(n_in, n_out, layer_input, activation, rng, w, b, w_values, b_values)
        weighted_sum = T.dot(self.input, self.w) + self.b
        self.output = self.activation(weighted_sum)
