import time

import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from elements.cost_functions import l1_norm, l2_norm, l2_norm_sqr


class ConvLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, filter_shape, image_shape, layer_input, pool_size=(2, 2),
                 activation=None, rng=None, w=None, b=None, do_down_sample=True,
                 w_values=None, b_values=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type layer_input: theano.tensor.dtensor4
        :param layer_input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps, filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps, image height, image width)

        :type pool_size: tuple or list of length 2
        :param pool_size: the down-sampling (pooling) factor (#rows, #cols)

        :type w: theano.tensor.TensorType
        :param w: Theano variable pointing to a set of weights that should be
                  shared with another architecture or if the weights should be initialize in any other way than random.
                  If the layer should be standalone and the weitght should initialized to random, leave this to None.

        :type b: theano.tensor.TensorType
        :param b: Theano variable pointing to a set of biases that should be
                  shared with another architecture or if the biases should be initialize in any other way than random.
                  If the layer should be standalone and the biases should initialized to random, leave this to None.

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer,
               if you want a simple linear neuron just set it to None

        :type do_down_sample: Boolean
        :param do_down_sample: indicate that the layer has down-sampling or not, by default it's true.

        :type w_values: numpy.matrix
        :param w_values: initial value for W

        :type b_values: numpy.matrix
        :param b_values: initial value for b
        """

        assert image_shape[1] == filter_shape[1]

        self.input = layer_input

        self.filter_shape = filter_shape
        self.image_shape = image_shape

        self.pool_size = pool_size
        self.do_down_sample = do_down_sample

        if rng is None:
            self.rng = numpy.random.RandomState(int(time.time()))
        else:
            self.rng = rng

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" / pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(pool_size))

        if w is None:
            if w_values is None:
                # initialize weights with random weights
                w_bound = numpy.sqrt(6. / (fan_in + fan_out))

                w_values = numpy.asarray(self.rng.uniform(low=-w_bound, high=w_bound, size=filter_shape),
                                         dtype=theano.config.floatX)

            self.w = theano.shared(value=w_values, borrow=True)
        else:
            self.w = w

        if b is None:
            if b_values is None:
                # the bias is a 1D tensor -- one bias per output feature map
                b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=self.input, filters=self.w, filter_shape=filter_shape, image_shape=image_shape)

        # Because we're doing down-sampling by Max Pooling (not average pooling)
        # order of down-sampling and applying activation function is not important.
        # So in order to reduce the number of activation function we have to compute,
        # we'll apply activation function on the output of dawn-sampling layer.

        # down-sample each feature map individually, using max-pooling
        if do_down_sample:
            pooled_out = downsample.max_pool_2d(input=conv_out, ds=pool_size, ignore_border=True)
        else:
            pooled_out = conv_out

        # Add the bias term. Since the bias is a vector (1D array),
        # we first reshape it to a tensor of shape (1, n_filters, 1, 1).
        # Each bias will thus be broadcaster across mini-batches and feature map width & height

        weighted_sum = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')

        if activation is None:
            self.activation = lambda x: x
        else:
            self.activation = activation

        # store parameters of this layer
        self.params = [self.w, self.b]

        self.output = activation(weighted_sum)

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
                "filter_shape": self.filter_shape,
                "image_shape": self.image_shape,
                "pool_size": self.pool_size,
                "do_down_sample": self.do_down_sample,
                "w_values": self.w.get_value(),
                "b_values": self.b.get_value(),
               }
