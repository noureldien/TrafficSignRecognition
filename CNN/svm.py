
import pickle
import gzip
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T

class OVASVMLayer(object):
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
        self.W = theano.shared(value=np.zeros((n_in, n_out),

        dtype=theano.config.floatX),
        name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,),

        dtype=theano.config.floatX),
        name='b', borrow=True)

        # parameters of the model
        self.params = [self.W, self.b]

        self.output = T.dot(input, self.W) + self.b

        self.y_pred = T.argmax(self.output, axis=1)

    def hinge(self, u):
        return

    def ova_svm_cost(self, y1):
        """ return the one-vs-all svm cost
        given ground-truth y in one-hot {-1, 1} form """
        y1_printed = theano.printing.Print('this is important')(T.max(y1))
        margin = y1 * self.output
        cost = T.maximum(0, 1 - margin).mean(axis=0).sum()
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

    def shared_dataset(self, data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch
        everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
        dtype=theano.config.floatX),
        borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
        dtype=theano.config.floatX),
        borrow=borrow)

        # one-hot encoded labels as {-1, 1}
        n_classes = len(np.unique(data_y)) # dangerous?
        y1 = -1 * np.ones((data_y.shape[0], n_classes))
        y1[np.arange(data_y.shape[0]), data_y] = 1
        shared_y1 = theano.shared(np.asarray(y1,
        dtype=theano.config.floatX),
        borrow=borrow)

        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32'), T.cast(shared_y1,'int32')

    def load_data(self, dataset):
        ''' Loads the dataset

        :type dataset: string
        :param dataset: the path to the dataset (here MNIST)
        '''

        #############
        # LOAD DATA #
        #############

        # Download the MNIST dataset if it is not present
        data_dir, data_file = os.path.split(dataset)
        if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
            import urllib
            origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            print('Downloading data from %s' % origin)
            urllib.urlretrieve(origin, dataset)
            print('... loading data')

        # Load the dataset
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()

        #train_set, valid_set, test_set format: tuple(input, target)
        #input is an numpy.ndarray of 2 dimensions (a matrix)
        #witch row's correspond to an example. target is a
        #numpy.ndarray of 1 dimensions (vector)) that have the same length as
        #the number of rows in the input. It should give the target
        #target to the example with the same index in the input.
        test_set_x, test_set_y, test_set_y1 = self.shared_dataset(test_set)
        valid_set_x, valid_set_y, valid_set_y1 = self.shared_dataset(valid_set)
        train_set_x, train_set_y, train_set_y1 = self.shared_dataset(train_set)

        rval = [(train_set_x, train_set_y, train_set_y1),
        (valid_set_x, valid_set_y, valid_set_y1),
        (test_set_x, test_set_y, test_set_y1)]
        return rval
