import numpy as np
import theano
import theano.tensor as T
import mlp

class SVMLayer(object):
    """
    SVM-like layer
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)

        # parameters of the model
        self.params = [self.W, self.b]

        self.output = T.dot(input, self.W) + self.b

        self.y_pred = T.argmax(self.output, axis=1)

    def hinge(self, u):
            return T.maximum(0, 1 - u)

    def cost(self, y_h):
        """ return the one-vs-all svm cost
        given ground-truth y in one-hot {-1, 1} form """
        y_h_printed = theano.printing.Print('this is important')(T.max(y_h))
        margin = y_h * self.output
        cost = self.hinge(margin).mean(axis=0).sum()
        return cost

    def errors(self, y):
        """ compute zero-one loss
        note, y is in integer form, not one-hot
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def classify_images(input, filters, W1, b1, W2, b2, n_in, n_out):

    filters_reshaped = filters.reshape((1, filters.size))
    h_input = input.flatten(2)
    hiddenLayer = mlp.HiddenLayer(input=h_input, W=W1, b=b1, n_in=n_in, n_out=n_out, activation=T.tanh, rng=0)
    p_y_given_x = T.nnet.softmax(T.dot(hiddenLayer.output, W2) + b2)
    y_pred = T.argmax(p_y_given_x, axis=1)

    # two functions for calculating the result and confidence/probability per class
    f_pred = theano.function([h_input], y_pred)
    f_prob = theano.function([h_input], p_y_given_x)

    result_pred = f_pred(filters_reshaped)[0]
    result_prob = f_prob(filters_reshaped)[0]

    return result_pred, result_prob

    output = T.dot(input, W) + b
    y_pred = T.argmax(output, axis=1)

    # two functions for calculating the result and confidence/probability per class
    f_pred = theano.function([h_input], y_pred)
    f_prob = theano.function([h_input], p_y_given_x)

    result_pred = f_pred(filters_reshaped)[0]
    result_prob = f_prob(filters_reshaped)[0]

    return result_pred, result_prob


def y_one_hot(data_y, n_classes, borrow=True):
    """
    Create one-hot array target for the n_classes target data_y
    Return it as a shared tensor
    :param data_y:
    :param n_classes:
    :param borrow:
    :return:
    """

    # one-hot encoded labels as {-1, 1}
    y_h = -1 * np.ones((data_y.shape[0], n_classes))
    y_h[np.arange(data_y.shape[0]), data_y] = 1
    shared_y_h = theano.shared(np.asarray(y_h, dtype=theano.config.floatX), borrow=borrow)
    return T.cast(shared_y_h,  'int32')
