import theano
import theano.tensor as T

"""
A set of activation functions for Neural Network layers.
They're in the form of class so we can take advantage of constructor
to set initial value for some parameters.
"""

def tanh(x):
    """
    tanh function (-1 to 1)

    @input: x, theano shared variable.
    @output: element-wise tanh of x
    """
    return T.tanh(x)

def sigmoid(x):
    """
    sigmoid function (0 to 1, (tanh(x)+1)/2).

    @input: x, theano shared variable.
    @output: element-wise sigmoid of x
    """
    return (T.tanh(x)+1)/2

def linier(x):
    """
    linier function.

    @input: x, theano shared variable.
    @output: x
    """
    return x

def relu_generator(alpha=0):
    """
    this function returns a relu function with proper alpha value.

    @input: alpha, slope of negative side of ReLU.
    @output: ReLU function
    """
    def relu(x):
        """
        rectified linier function (-alpha*x if x<0, x if x>0).

        @input: x, theano shared variable.
        @output: x<0?-alpha*x:x
        """
        return T.nnet.relu(x, alpha)

    return relu

# TODO:
# add RBF activation function
#
# def RBF(x):
#   """
#   radial basis function.

#   @input: x, theano shared variable.
#   @output: Not Implimented
#   """
