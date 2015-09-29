"""
a set of cost functions for Neural Network layers.
"""

import theano.tensor as T

def zero_one_error(prediction, target):
    """
    Return a float representing number of errors (miss-classifications) in the mini-batch
    divided by number of elements in mini-batch.

    :type prediction: theano.tensor.TensorType
    :param prediction: predicted value

    :type target: theano.tensor.TensorType
    :param target: target value

    Note that both prediction and target are in form of a vector of Intigers
    that each element represent output for a specific input)
    """

    # safty checks first
    if prediction.ndim != target.ndim:
        raise TypeError( 'prediction and target should have the same size \n prediction: %d, target: %d'%(prediction.ndim, target.ndim))

    if not prediction.dtype.startswith('int'):
        raise TypeError("prediction should be int, it's %s"%prediction.dtype)

    if not target.dtype.startswith('int'):
        raise TypeError("target should be int, it's %s"%target.dtype)

    return T.mean(T.neq(prediction, target))

def negative_log_likelihood_error(cond_prob, target):
        """
        Return the mean of the negative log-likelihood of the prediction
        of predictions(conditional probability of a label given an input) under a given target distribution.

        :type cond_prob: theano.tensor.TensorType
        :param cond_prob: corresponds to a matrix with conditional probabilities of each label given inputs.

    :type target: theano.tensor.TensorType
    :param target: target value

    Note that cond_prob is is a matrix of floats (or doubles)
    each row of it shows conditional probability for outputs given an specific input.
    While target is a vector of Integers that each element represent output for a specific input
    """

    # y.shape[0] is (symbolically) the number of rows in y,
    # i.e., number of examples (call it n) in the minibatch
    # T.arange(y.shape[0]) is a symbolic vector which will contain
    # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
    # Log-Probabilities (call it LP) with one row per example and
    # one column per class LP[T.arange(y.shape[0]),y] is a vector
    # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
    # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
    # the mean (across minibatch examples) of the elements in v,
    # i.e., the mean log-likelihood across the minibatch.
    # for more clear explanation, check log-likelihood definition
    return -T.mean(T.log(cond_prob)[T.arange(target.shape[0]), target])

def squered_error(prediction, target):
    """
    Return the mean of the squared errorof the prediction
    of this model under a given target distribution.

    :type prediction: theano.tensor.TensorType
    :param prediction: corresponds to a vector that gives for each example the predicted label

    :type label: theano.tensor.TensorType
    :param label: corresponds to a vector that gives for each example the correct label
    """

    return T.mean((prediction-target)**2)

# TODO:
#
# def cross_entropy_error(prediction, target):
#     """
#     Return the mean of the cross entropy error of the prediction in the mini-batch.

#     :type prediction: theano.tensor.TensorType
#     :param prediction: predicted value

#     :type target: theano.tensor.TensorType
#     :param target: target value

#     Note that both prediction and target are in form of a vector of Intigers
#     that each element represent output for a specific input)

#     """

#     # note : we sum over the size of a datapoint; if we are using
#     #               minibatches, L will be a vector, with one entry per
#     #               example in minibatch
#     return T.mean(- T.sum(target * T.log(prediction) + (1 - target) * T.log(1 - prediction), axis=1))
