import os
import sys
import time

import cv2
import skimage
import skimage.transform

import numpy
import theano
import theano.tensor as T
import pickle
import CNN
import CNN.svm
import CNN.logit
import CNN.utils
import CNN.mlp
import CNN.conv
import CNN.enums
import CNN.recog
from CNN.mlp import HiddenLayer


def train_shallow(dataset_path, recognition_model_path, detection_model_path='', learning_rate=0.1, n_epochs=10, batch_size=500,
                  classifier=CNN.enums.ClassifierType.logit):
    datasets = CNN.utils.load_data(dataset_path)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches = int(n_train_batches / batch_size)
    n_valid_batches = int(n_valid_batches / batch_size)
    n_test_batches = int(n_test_batches / batch_size)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')
    y = T.imatrix('y')

    # load model and read it's parameters
    # the same weights of the convolutional layers will be used
    # in training the detector
    loaded_objects = CNN.utils.load_model(recognition_model_path)

    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    pool_size = loaded_objects[5]
    rng = numpy.random.RandomState(23455)

    # the number of layers in the MLP classifier to train is not optional
    # the input of the first MLP layer has to be compatible with the output of conv layers
    # while the output of the last MLP layer has to comprise the img_dim because at the end
    # of the day the MLP results is classes each of them represent a pixel
    # this is the regression fashion of the MLP, each class represents the pixel, i.e what
    # is the pixel of the predicted region
    mlp_layers = loaded_objects[4]
    mlp_layers = (mlp_layers[0], (img_dim + 1))

    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)

    # first, filter the given input images using the weights of the filters
    # of the given class_model_path
    # then, train a mlp as a regression model not classification
    # then save all of the cnn_model and the regression_model into a file 'det_model_path'

    layer0_img_dim = img_dim
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)

    # layer 0: Conv-Pool
    layer0_input = x.reshape((batch_size, 1, layer0_img_dim, layer0_img_dim))
    layer0 = CNN.conv.ConvPoolLayer_(
        input=layer0_input, W=layer0_W, b=layer0_b,
        image_shape=(batch_size, 1, layer0_img_dim, layer0_img_dim),
        filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
        poolsize=pool_size
    )

    # layer 1: Conv-Pool
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)  # = 12 in case of mnist
    layer1_kernel_dim = kernel_dim[1]
    layer1 = CNN.conv.ConvPoolLayer_(
        input=layer0.output, W=layer1_W, b=layer1_b,
        image_shape=(batch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
        filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
        poolsize=pool_size
    )

    # Layer 2: the HiddenLayer being fully-connected, it operates on 2D matrices
    layer2 = HiddenLayer(
        rng,
        input=layer1.output.flatten(2),
        n_in=nkerns[1] * layer2_img_dim * layer2_img_dim,
        n_out=mlp_layers[0],
        activation=T.tanh
    )

    # Layer 3: classify the values of the fully-connected sigmoidal layer
    layer3_n_outs = [mlp_layers[1]] * 4
    layer3 = CNN.logit.MultiLogisticRegression(input=layer2.output, n_in=mlp_layers[0], n_outs=layer3_n_outs)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # experimental, add L1, L2 regularization to the regressor
    # self.L1 = (
    #         abs(self.hiddenLayer.W).sum()
    #         + abs(self.logRegressionLayer.W).sum()
    #     )
    #
    #     # square of L2 norm ; one regularization option is to enforce
    #     # square of L2 norm to be small
    #     self.L2_sqr = (
    #         (self.hiddenLayer.W ** 2).sum()
    #         + (self.logRegressionLayer.W ** 2).sum()
    #     )

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y, mlp_layers[1]),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y, mlp_layers[1]),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatches before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):

        epoch += 1
        print("... epoch: %d" % epoch)

        for minibatch_index in range(int(n_train_batches)):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('... training @ iter = %.0f' % iter)

            # train the minibatch
            minibatch_avg_cost = train_model(minibatch_index)

            if (iter + 1) == validation_frequency:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(int(n_valid_batches))]
                this_validation_loss = numpy.mean(validation_losses)
                print('... epoch %d, minibatch %d/%d, validation error %.2f %%' % (
                    epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in range(int(n_test_batches))]
                    test_score = numpy.mean(test_losses)
                    print(('    epoch %i, minibatch %i/%i, test error of best model %.2f%%') % (
                        epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %.2f%% obtained at iteration %i with test performance %.2f%%' % (
        best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print(sys.stderr)

    if len(detection_model_path) == 0:
        return

    # serialize the params of the model
    # the -1 is for HIGHEST_PROTOCOL
    # this will overwrite current contents and it triggers much more efficient storage than numpy's default
    save_file = open(detection_model_path, 'wb')
    pickle.dump(dataset_path, save_file, -1)
    pickle.dump(img_dim, save_file, -1)
    pickle.dump(kernel_dim, save_file, -1)
    pickle.dump(nkerns, save_file, -1)
    pickle.dump(mlp_layers, save_file, -1)
    pickle.dump(pool_size, save_file, -1)
    pickle.dump(layer0.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer0.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer1.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer1.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer2.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer2.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer3.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer3.b.get_value(borrow=True), save_file, -1)
    save_file.close()


def train_deep(dataset_path, recognition_model_path, detection_model_path='', learning_rate=0.1, n_epochs=10, batch_size=10,
               mlp_layers=(1000, 81), classifier=CNN.enums.ClassifierType.logit):
    datasets = CNN.utils.load_data(dataset_path)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_valid_batches = int(valid_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_test_batches = int(test_set_x.get_value(borrow=True).shape[0] / batch_size)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')
    y = T.imatrix('y')

    # load model and read it's parameters
    # the same weights of the convolutional layers will be used
    # in training the detector
    loaded_objects = CNN.utils.load_model(recognition_model_path)

    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    pool_size = loaded_objects[5]
    rng = numpy.random.RandomState(23455)

    # the number of layers in the MLP classifier to train is not optional
    # the input of the first MLP layer has to be compatible with the output of conv layers
    # while the output of the last MLP layer has to comprise the img_dim because at the end
    # of the day the MLP results is classes each of them represent a pixel
    # this is the regression fashion of the MLP, each class represents the pixel, i.e what
    # is the pixel of the predicted region
    # mlp_layers = loaded_objects[4]
    # mlp_layers = (mlp_layers[0], (img_dim + 1))

    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)
    layer2_W = theano.shared(loaded_objects[10], borrow=True)
    layer2_b = theano.shared(loaded_objects[11], borrow=True)

    # first, filter the given input images using the weights of the filters
    # of the given class_model_path
    # then, train a mlp as a regression model not classification
    # then save all of the cnn_model and the regression_model into a file 'det_model_path'

    layer0_img_dim = img_dim
    layer0_kernel_dim = kernel_dim[0]
    layer0_input = x.reshape((batch_size, 1, layer0_img_dim, layer0_img_dim))

    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)
    layer1_kernel_dim = kernel_dim[1]

    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)
    layer2_kernel_dim = kernel_dim[2]

    layer3_img_dim = int((layer2_img_dim - layer2_kernel_dim + 1) / 2)

    # layer 0: Conv-Pool
    layer0 = CNN.conv.ConvPoolLayer_(
        input=layer0_input, W=layer0_W, b=layer0_b,
        image_shape=(batch_size, 1, layer0_img_dim, layer0_img_dim),
        filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
        poolsize=pool_size
    )

    # layer 1: Conv-Pool
    layer1 = CNN.conv.ConvPoolLayer_(
        input=layer0.output, W=layer1_W, b=layer1_b,
        image_shape=(batch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
        filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
        poolsize=pool_size
    )

    # layer 2: Conv-Pool
    layer2 = CNN.conv.ConvPoolLayer_(
        input=layer1.output, W=layer2_W, b=layer2_b,
        image_shape=(batch_size, nkerns[1], layer2_img_dim, layer2_img_dim),
        filter_shape=(nkerns[2], nkerns[1], layer2_kernel_dim, layer2_kernel_dim),
        poolsize=pool_size
    )

    # layer 3: the HiddenLayer being fully-connected, it operates on 2D matrices
    layer3 = HiddenLayer(
        rng,
        input=layer2.output.flatten(2),
        n_in=nkerns[2] * layer3_img_dim * layer3_img_dim,
        n_out=mlp_layers[0],
        activation=T.tanh
    )

    # layer 4: classify the values of the fully-connected sigmoidal layer
    layer4_n_outs = [mlp_layers[1]] * 4
    layer4 = CNN.logit.MultiLogisticRegression(input=layer3.output, n_in=mlp_layers[0], n_outs=layer4_n_outs)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # experimental, add L1, L2 regularization to the regressor
    # self.L1 = (
    #         abs(self.hiddenLayer.W).sum()
    #         + abs(self.logRegressionLayer.W).sum()
    #     )
    #
    #     # square of L2 norm ; one regularization option is to enforce
    #     # square of L2 norm to be small
    #     self.L2_sqr = (
    #         (self.hiddenLayer.W ** 2).sum()
    #         + (self.logRegressionLayer.W ** 2).sum()
    #     )

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y, mlp_layers[1]),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y, mlp_layers[1]),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatches before checking the network
    # on the validation set; in this case we
    # check every epoch

    print("... validation freq: %d" % validation_frequency)
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):

        epoch += 1
        print("... epoch: %d" % epoch)

        for minibatch_index in range(int(n_train_batches)):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('... training @ iter = %.0f' % iter)

            # train the minibatch
            minibatch_avg_cost = train_model(minibatch_index)

            if (iter + 1) == validation_frequency:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(int(n_valid_batches))]
                this_validation_loss = numpy.mean(validation_losses)
                print('... epoch %d, minibatch %d/%d, validation error %.2f %%' % (
                    epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in range(int(n_test_batches))]
                    test_score = numpy.mean(test_losses)
                    print(('    epoch %i, minibatch %i/%i, test error of best model %.2f%%') % (
                        epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %.2f%% obtained at iteration %i with test performance %.2f%%' % (
        best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print(sys.stderr)

    if len(detection_model_path) == 0:
        return

    # serialize the params of the model
    # the -1 is for HIGHEST_PROTOCOL
    # this will overwrite current contents and it triggers much more efficient storage than numpy's default
    save_file = open(detection_model_path, 'wb')
    pickle.dump(dataset_path, save_file, -1)
    pickle.dump(img_dim, save_file, -1)
    pickle.dump(kernel_dim, save_file, -1)
    pickle.dump(nkerns, save_file, -1)
    pickle.dump(mlp_layers, save_file, -1)
    pickle.dump(pool_size, save_file, -1)
    pickle.dump(layer0.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer0.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer1.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer1.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer2.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer2.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer3.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer3.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer4.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer4.b.get_value(borrow=True), save_file, -1)
    save_file.close()


def train_from_scatch(dataset_path, detection_model_path, learning_rate=0.1, n_epochs=10, batch_size=500,
                      nkerns=(40, 40 * 9), mlp_layers=(800, 29), kernel_dim=(5, 5), img_dim=28,
                      pool_size=(2, 2), classifier=CNN.enums.ClassifierType.logit):
    datasets = CNN.utils.load_data(dataset_path)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches = int(n_train_batches / batch_size)
    n_valid_batches = int(n_valid_batches / batch_size)
    n_test_batches = int(n_test_batches / batch_size)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')
    y = T.imatrix('y')
    rng = numpy.random.RandomState(23455)

    layer0_img_dim = img_dim
    layer0_kernel_dim = kernel_dim[0]
    layer0_input = x.reshape((batch_size, 1, layer0_img_dim, layer0_img_dim))
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)
    layer3_n_outs = [mlp_layers[1]] * 4

    # layer 0: Conv-Pool
    layer0 = CNN.conv.ConvPoolLayer(
        rng=rng,
        input=layer0_input,
        image_shape=(batch_size, 1, layer0_img_dim, layer0_img_dim),
        filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
        poolsize=pool_size
    )

    # layer 1: Conv-Pool
    layer1 = CNN.conv.ConvPoolLayer(
        rng=rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
        filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
        poolsize=pool_size
    )

    # Layer 2: the HiddenLayer being fully-connected, it operates on 2D matrices
    layer2 = HiddenLayer(
        rng,
        input=layer1.output.flatten(2),
        n_in=nkerns[1] * layer2_img_dim * layer2_img_dim,
        n_out=mlp_layers[0],
        activation=T.tanh
    )

    # Layer 3: classify the values of the fully-connected sigmoidal layer
    layer3 = CNN.logit.MultiLogisticRegression(input=layer2.output, n_in=mlp_layers[0], n_outs=layer3_n_outs)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # experimental, add L1, L2 regularization to the regressor
    # self.L1 = (
    #         abs(self.hiddenLayer.W).sum()
    #         + abs(self.logRegressionLayer.W).sum()
    #     )
    #
    #     # square of L2 norm ; one regularization option is to enforce
    #     # square of L2 norm to be small
    #     self.L2_sqr = (
    #         (self.hiddenLayer.W ** 2).sum()
    #         + (self.logRegressionLayer.W ** 2).sum()
    #     )

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y, mlp_layers[1]),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y, mlp_layers[1]),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatches before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):

        epoch += 1
        print("... epoch: %d" % epoch)

        for minibatch_index in range(int(n_train_batches)):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('... training @ iter = %.0f' % iter)

            # train the minibatch
            minibatch_avg_cost = train_model(minibatch_index)

            if (iter + 1) == validation_frequency:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(int(n_valid_batches))]
                this_validation_loss = numpy.mean(validation_losses)
                print('... epoch %d, minibatch %d/%d, validation error %.2f %%' % (
                    epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in range(int(n_test_batches))]
                    test_score = numpy.mean(test_losses)
                    print(('    epoch %i, minibatch %i/%i, test error of best model %.2f%%') % (
                        epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %.2f%% obtained at iteration %i with test performance %.2f%%' % (
        best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print(sys.stderr)

    if len(detection_model_path) == 0:
        return

    # serialize the params of the model
    # the -1 is for HIGHEST_PROTOCOL
    # this will overwrite current contents and it triggers much more efficient storage than numpy's default
    save_file = open(detection_model_path, 'wb')
    pickle.dump(dataset_path, save_file, -1)
    pickle.dump(img_dim, save_file, -1)
    pickle.dump(kernel_dim, save_file, -1)
    pickle.dump(nkerns, save_file, -1)
    pickle.dump(mlp_layers, save_file, -1)
    pickle.dump(pool_size, save_file, -1)
    pickle.dump(layer0.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer0.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer1.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer1.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer2.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer2.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer3.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer3.b.get_value(borrow=True), save_file, -1)
    save_file.close()


def train_fast(dataset_path, recognition_model_path, detection_model_path='', learning_rate=0.1, n_epochs=10, batch_size=500,
               classifier=CNN.enums.ClassifierType.logit):
    # do all the cov+pool computation so we can have the output ready
    # for the MLP classifier. This makes the computation faster because
    # we don't have to calcuate conv+pool every epoch

    # UPDATE: it didn't work as we couldn't do conv for 11K images in one batch

    print('... loading data')

    # Load the dataset
    f = open(dataset_path, 'rb')
    dataset = pickle.load(f)
    f.close()
    del f

    batch_of_images = numpy.concatenate((dataset[0][0], dataset[1][0]))
    batch_of_images = numpy.concatenate((batch_of_images, dataset[2][0]))
    del dataset

    # load model and read it's parameters
    # the same weights of the convolutional layers will be used
    # in training the detector
    loaded_objects = CNN.utils.load_model(recognition_model_path)

    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    pool_size = loaded_objects[5]
    rng = numpy.random.RandomState(23455)

    # the number of layers in the MLP classifier to train is not optional
    # the input of the first MLP layer has to be compatible with the output of conv layers
    # while the output of the last MLP layer has to comprise the img_dim because at the end
    # of the day the MLP results is classes each of them represent a pixel
    # this is the regression fashion of the MLP, each class represents the pixel, i.e what
    # is the pixel of the predicted region
    mlp_layers = loaded_objects[4]
    mlp_layers = (mlp_layers[0], (img_dim + 1))

    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)

    # first, filter the given input images using the weights of the filters
    # of the given class_model_path
    # then, train a mlp as a regression model not classification
    # then save all of the cnn_model and the regression_model into a file 'det_model_path'

    layer0_img_dim = img_dim
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)

    # layer 0: Conv-Pool
    conv_patch_size = batch_of_images.shape[0]
    layer0_input = T.tensor4('input')
    batch_of_images = batch_of_images.reshape((conv_patch_size, 1, layer0_img_dim, layer0_img_dim))
    layer0_output = CNN.conv.convpool_layer(
        input=layer0_input, W=layer0_W, b=layer0_b,
        image_shape=(conv_patch_size, 1, layer0_img_dim, layer0_img_dim),
        filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
        pool_size=pool_size
    )

    # layer 1: Conv-Pool
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)  # = 12 in case of mnist
    layer1_kernel_dim = kernel_dim[1]
    layer1_output = CNN.conv.convpool_layer(
        input=layer0_output, W=layer1_W, b=layer1_b,
        image_shape=(conv_patch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
        filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
        pool_size=pool_size
    )

    # do the filtering using 2 layers of Conv+Pool
    conv_function = theano.function([layer0_input], layer1_output)
    filters = conv_function(batch_of_images)

    #########################################
    # Train the classifier of the detector #
    ########################################

    train_cls = dataset[0][0]
    valid_cls = dataset[1][0]
    test_cls = dataset[2][0]
    n_train = train_cls.shape[0]
    n_valid = valid_cls.shape[0]
    n_test = test_cls.shape[0]
    n = n_train + n_valid + n_test

    train_set_x, train_set_y = CNN.utils.shared_dataset((filters[0: n_train], train_cls))
    valid_set_x, valid_set_y = CNN.utils.shared_dataset((filters[n_train: n_train + n_valid], valid_cls))
    test_set_x, test_set_y = CNN.utils.shared_dataset((filters[n_valid: n], test_cls))

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches = int(n_train_batches / batch_size)
    n_valid_batches = int(n_valid_batches / batch_size)
    n_test_batches = int(n_test_batches / batch_size)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')
    y = T.imatrix('y')

    # Layer 2: the HiddenLayer being fully-connected, it operates on 2D matrices
    layer2 = HiddenLayer(
        rng,
        input=layer1_output.flatten(2),
        n_in=nkerns[1] * layer2_img_dim * layer2_img_dim,
        n_out=mlp_layers[0],
        activation=T.tanh
    )

    # Layer 3: classify the values of the fully-connected sigmoidal layer
    layer3_n_outs = [mlp_layers[1]] * 4
    layer3 = CNN.logit.MultiLogisticRegression(input=layer2.output, n_in=mlp_layers[0], n_outs=layer3_n_outs)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # experimental, add L1, L2 regularization to the regressor
    # self.L1 = (
    #         abs(self.hiddenLayer.W).sum()
    #         + abs(self.logRegressionLayer.W).sum()
    #     )
    #
    #     # square of L2 norm ; one regularization option is to enforce
    #     # square of L2 norm to be small
    #     self.L2_sqr = (
    #         (self.hiddenLayer.W ** 2).sum()
    #         + (self.logRegressionLayer.W ** 2).sum()
    #     )

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y, mlp_layers[1]),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y, mlp_layers[1]),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatches before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):

        epoch += 1
        print("... epoch: %d" % epoch)

        for minibatch_index in range(int(n_train_batches)):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('... training @ iter = %.0f' % iter)

            # train the minibatch
            minibatch_avg_cost = train_model(minibatch_index)

            if (iter + 1) == validation_frequency:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(int(n_valid_batches))]
                this_validation_loss = numpy.mean(validation_losses)
                print('... epoch %d, minibatch %d/%d, validation error %.2f %%' % (
                    epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in range(int(n_test_batches))]
                    test_score = numpy.mean(test_losses)
                    print(('    epoch %i, minibatch %i/%i, test error of best model %.2f%%') % (
                        epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %.2f%% obtained at iteration %i with test performance %.2f%%' % (
        best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print(sys.stderr)

    if len(detection_model_path) == 0:
        return

    # serialize the params of the model
    # the -1 is for HIGHEST_PROTOCOL
    # this will overwrite current contents and it triggers much more efficient storage than numpy's default
    save_file = open(detection_model_path, 'wb')
    pickle.dump(dataset_path, save_file, -1)
    pickle.dump(img_dim, save_file, -1)
    pickle.dump(kernel_dim, save_file, -1)
    pickle.dump(nkerns, save_file, -1)
    pickle.dump(mlp_layers, save_file, -1)
    pickle.dump(pool_size, save_file, -1)
    pickle.dump(layer0_W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer0_b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer1_W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer1_b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer2.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer2.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer3.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer3.b.get_value(borrow=True), save_file, -1)
    save_file.close()


def detect_img_from_file(img_path, model_path, img_dim, model_type=CNN.enums.ModelType, classifier=CNN.enums.ClassifierType.logit):
    """
    detect a traffic sign form the given natural image
    detected signs depend on the given model, for example if it is a prohibitory detection model
    we'll only detect prohibitory traffic signs
    :param img_path:
    :param model_path:
    :param classifier:
    :param img_dim:
    :return:
    """

    # stride represents how dense to sample regions around the ground truth traffic signs
    # also down_scaling factor affects the sampling
    # initial dimension defines what is the biggest traffic sign to recognise
    # actually stride should be dynamic, i.e. smaller strides for smaller window size and vice versa

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(float) / 255.0

    # the biggest traffic sign to recognize is 400*400 in a 1360*800 image
    # that means, we'll start with a window with initial size 320*320
    # for each ground_truth boundary, extract regions in such that:
    # 1. each region fully covers the boundary
    # 2. the boundary must not be smaller than the 1/5 of the region
    # ie. according to initial window size, the boundary must not be smaller than 80*80
    # but sure we will recognize smaller ground truth because we down_scale the window every step
    # boundary is x1, y1, x2, y2 => (x1,y1) top left, (x2, y2) bottom right
    # don't forget that stride of sliding the window is dynamic

    down_scale_factor = 0.95
    window_dim = 110
    stride_factor = 5
    img_shape = img.shape
    img_width = img_shape[1]
    img_height = img_shape[0]

    regions = []
    locations = []
    s_count = 0

    # scale_down until you reach the min window
    # instead of scaling up the image itself, we scale down the sliding window
    while window_dim >= img_dim:

        # stride is dynamic, smaller strides for smaller scales
        # this means that stride is equivialant to 2 pixels
        # when the window is resized to the img_dim (required for CNN)
        r_factor = window_dim / img_dim
        stride = stride_factor * int(r_factor)

        s_count += 1
        r_count = 0

        # for the current scale of the window, extract regions, start from the
        # y_range = numpy.arange(start=0, stop=img_height, step=stride, dtype=int).tolist()
        # x_range = numpy.arange(start=0, stop=img_width, step=stride, dtype=int).tolist()
        y = 0
        x_count = 0
        y_count = 0
        region_shape = []
        while y <= img_height:
            x = 0
            x_count = 0
            while x <= img_width:
                # - add region to the region list
                # - adjust the position of the ground_truth to be relative to the window
                #   not relative to the image (i.e relative frame of reference)
                # - don't forget to re_scale the extracted/sampled region to be 28*28
                #   hence, multiply the relative position with this scaling accordingly
                # - also, the image needs to be preprocessed so it can be ready for the CNN
                region = img[y:y + window_dim, x:x + window_dim]
                region_shape = region.shape
                region = skimage.transform.resize(region, output_shape=(img_dim, img_dim))
                # we only need to store the region, it's top-left corner and sliding window dim
                regions.append(region)
                locations.append((x, y))

                r_count += 1

                # save region for experiemnt
                # filePathWrite = "D:\\_Dataset\\GTSDB\\Test_Regions\\%s_%s.png" % ("{0:03d}".format(s_count), "{0:03d}".format(r_count))
                # img_save = region * 255
                # img_save = img_save.astype(int)
                # cv2.imwrite(filePathWrite, img_save)

                x_count += 1
                x += stride
                if region_shape[1] < window_dim:
                    break

            y_count += 1
            if region_shape[0] < window_dim:
                break
            y += stride

        print("Scale: %d, window_dim: %d, regions: %d" % (s_count, window_dim, r_count))

        # now we want to re_scale, instead of down_scaling the whole image, we down_scale the window
        # don't forget to recalculate the window area
        window_dim = int(window_dim * down_scale_factor)

        # send the regions for the detector and convert the result to the probability map
        batch = numpy.asarray(regions)
        d_pred, d_duration = __detect_batch(batch, model_path, model_type, classifier)

        # now, after getting the predictions, construct the probability map
        # and show it
        __probability_map(d_pred, locations, window_dim, x_count, y_count, img_width, img_height, img_dim, s_count)

    x = 10


def __detect_batch(batch, model_path, model_type=CNN.enums.ModelType, classifier=CNN.enums.ClassifierType.logit):
    if model_type == CNN.enums.ModelType._01_conv2_mlp2:
        return __detect_batch_shallow_model(batch, model_path, classifier)
    elif model_type == CNN.enums.ModelType._02_conv3_mlp2:
        return __detect_batch_deep_model(batch, model_path, classifier)
    else:
        raise Exception("Unknown model type")


def __detect_batch_shallow_model(batch, model_path, classifier=CNN.enums.ClassifierType.logit):
    loaded_objects = CNN.utils.load_model(model_path)

    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    mlp_layers = loaded_objects[4]
    pool_size = loaded_objects[5]

    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)
    layer2_W = theano.shared(loaded_objects[10], borrow=True)
    layer2_b = theano.shared(loaded_objects[11], borrow=True)
    layer3_W = theano.shared(loaded_objects[12], borrow=True)
    layer3_b = theano.shared(loaded_objects[13], borrow=True)

    layer0_img_dim = img_dim  # = 28 in case of mnist
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)  # = 12 in case of mnist
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)  # = 4 in case of mnist

    # layer 0: Conv-Pool
    batch_size = batch.shape[0]
    filter_shape = (nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim)
    image_shape = (batch_size, 1, layer0_img_dim, layer0_img_dim)
    batch = batch.reshape(image_shape)

    layer0_input = T.tensor4(name='input')
    layer0_output = CNN.conv.convpool_layer(input=layer0_input, W=layer0_W, b=layer0_b, image_shape=image_shape,
                                            filter_shape=filter_shape, pool_size=pool_size)

    # layer 1: Conv-Pool
    filter_shape = (nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim)
    image_shape = (batch_size, nkerns[0], layer1_img_dim, layer1_img_dim)
    layer1_output = CNN.conv.convpool_layer(input=layer0_output, W=layer1_W, b=layer1_b,
                                            image_shape=image_shape, filter_shape=filter_shape,
                                            pool_size=pool_size)

    # layer 2: hidden layer
    hidden_n_in = nkerns[1] * layer2_img_dim * layer2_img_dim
    layer1_output_flattened = layer1_output.flatten(2)
    layer2 = CNN.mlp.HiddenLayer(input=layer1_output_flattened, W=layer2_W, b=layer2_b, n_in=hidden_n_in,
                                 n_out=mlp_layers[0], activation=T.tanh, rng=0)

    # layer 3: logit (logistic regression) or SVM
    layer3_n_outs = [mlp_layers[1]] * 4
    if classifier == CNN.enums.ClassifierType.logit:
        layer3_y, layer3_y_prob = CNN.logit.multi_logit_layer(input=layer2.output, W=layer3_W, b=layer3_b, n_outs=layer3_n_outs)
    elif classifier == CNN.enums.ClassifierType.svm:
        layer3_y, layer3_y_prob = CNN.svm.multi_svm_layer(input=layer2.output, W=layer3_W, b=layer3_b)
    else:
        raise TypeError('Unknown classifier type, should be either logit or svm', ('classifier:', classifier))

    start_time = time.clock()

    # two functions for calculating the result and confidence/probability per class
    f_pred = theano.function([layer0_input], layer3_y)
    d_pred = f_pred(batch)

    end_time = time.clock()
    d_duration = end_time - start_time

    return d_pred, d_duration


def __detect_batch_deep_model(batch, model_path, classifier=CNN.enums.ClassifierType.logit):
    loaded_objects = CNN.utils.load_model(model_path=model_path, model_type=CNN.enums.ModelType._02_conv3_mlp2)

    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    mlp_layers = loaded_objects[4]
    pool_size = loaded_objects[5]

    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)
    layer2_W = theano.shared(loaded_objects[10], borrow=True)
    layer2_b = theano.shared(loaded_objects[11], borrow=True)
    layer3_W = theano.shared(loaded_objects[12], borrow=True)
    layer3_b = theano.shared(loaded_objects[13], borrow=True)
    layer4_W = theano.shared(loaded_objects[14], borrow=True)
    layer4_b = theano.shared(loaded_objects[15], borrow=True)

    layer0_img_dim = img_dim  # = 28 in case of mnist
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)
    layer2_kernel_dim = kernel_dim[2]
    layer3_img_dim = int((layer2_img_dim - layer2_kernel_dim + 1) / 2)

    # layer 0: Conv-Pool
    batch_size = batch.shape[0]
    filter_shape = (nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim)
    image_shape = (batch_size, 1, layer0_img_dim, layer0_img_dim)
    batch = batch.reshape(image_shape)

    layer0_input = T.tensor4(name='input')
    layer0_output = CNN.conv.convpool_layer(input=layer0_input, W=layer0_W, b=layer0_b, image_shape=image_shape,
                                            filter_shape=filter_shape, pool_size=pool_size)

    # layer 1: Conv-Pool
    filter_shape = (nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim)
    image_shape = (batch_size, nkerns[0], layer1_img_dim, layer1_img_dim)
    layer1_output = CNN.conv.convpool_layer(input=layer0_output, W=layer1_W, b=layer1_b,
                                            image_shape=image_shape, filter_shape=filter_shape,
                                            pool_size=pool_size)

    # layer 2: Conv-Pool
    filter_shape = (nkerns[2], nkerns[1], layer2_kernel_dim, layer2_kernel_dim)
    image_shape = (batch_size, nkerns[1], layer2_img_dim, layer2_img_dim)
    layer2_output = CNN.conv.convpool_layer(input=layer1_output, W=layer2_W, b=layer2_b,
                                            image_shape=image_shape, filter_shape=filter_shape,
                                            pool_size=pool_size)

    # layer 3: hidden layer
    hidden_n_in = nkerns[2] * layer3_img_dim * layer3_img_dim
    layer2_output_flattened = layer2_output.flatten(2)
    layer3 = CNN.mlp.HiddenLayer(input=layer2_output_flattened, W=layer3_W, b=layer3_b, n_in=hidden_n_in,
                                 n_out=mlp_layers[0], activation=T.tanh, rng=0)

    # layer 4: logit (logistic regression) or SVM
    layer4_n_outs = [mlp_layers[1]] * 4
    if classifier == CNN.enums.ClassifierType.logit:
        layer4_y, layer3_y_prob = CNN.logit.multi_logit_layer(input=layer3.output, W=layer4_W, b=layer4_b, n_outs=layer4_n_outs)
    elif classifier == CNN.enums.ClassifierType.svm:
        layer4_y, layer3_y_prob = CNN.svm.multi_svm_layer(input=layer3.output, W=layer4_W, b=layer4_b)
    else:
        raise TypeError('Unknown classifier type, should be either logit or svm', ('classifier:', classifier))

    start_time = time.clock()

    # two functions for calculating the result and confidence/probability per class
    f_pred = theano.function([layer0_input], layer4_y)
    d_pred = f_pred(batch)

    end_time = time.clock()
    d_duration = end_time - start_time

    return d_pred, d_duration


def __detect_img(img4D, model_path, model_type=CNN.enums.ModelType, classifier=CNN.enums.ClassifierType.logit):
    if model_type == CNN.enums.ModelType._01_conv2_mlp2:
        return __detect_img_shallow_model(img4D, model_path, classifier)
    elif model_type == CNN.enums.ModelType._02_conv3_mlp2:
        return __detect_img_deep_model(img4D, model_path, classifier)
    else:
        raise Exception("Unknown model type")


def __detect_img_shallow_model(img4D, model_path, classifier=CNN.enums.ClassifierType.logit):
    loaded_objects = CNN.utils.load_model(model_path)

    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    mlp_layers = loaded_objects[4]
    pool_size = loaded_objects[5]

    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)
    layer2_W = theano.shared(loaded_objects[10], borrow=True)
    layer2_b = theano.shared(loaded_objects[11], borrow=True)
    layer3_W = theano.shared(loaded_objects[12], borrow=True)
    layer3_b = theano.shared(loaded_objects[13], borrow=True)

    layer0_img_dim = img_dim  # = 28 in case of mnist
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)  # = 12 in case of mnist
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)  # = 4 in case of mnist

    start_time = time.clock()

    # layer 0: Conv-Pool
    filter_shape = (nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim)
    image_shape = (1, 1, layer0_img_dim, layer0_img_dim)
    (layer0_filters, layer0_output) = CNN.conv.filter_image(img=img4D, W=layer0_W, b=layer0_b, image_shape=image_shape,
                                                            filter_shape=filter_shape, pool_size=pool_size)

    # layer 1: Conv-Pool
    filter_shape = (nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim)
    image_shape = (1, nkerns[0], layer1_img_dim, layer1_img_dim)
    (layer1_filters, layer1_output) = CNN.conv.filter_image(img=layer0_filters, W=layer1_W, b=layer1_b,
                                                            image_shape=image_shape, filter_shape=filter_shape,
                                                            pool_size=pool_size)

    # layer 2: hidden layer
    hidden_n_in = nkerns[1] * layer2_img_dim * layer2_img_dim
    layer1_output_flattened = layer1_output.flatten(2)
    hiddenLayer = CNN.mlp.HiddenLayer(input=layer1_output_flattened, W=layer2_W, b=layer2_b, n_in=hidden_n_in,
                                      n_out=mlp_layers[0], activation=T.tanh, rng=0)

    # layer 3: logit (logistic regression) or SVM
    c_result = []
    c_prob = []
    if classifier == CNN.enums.ClassifierType.logit:
        c_result, c_prob = CNN.logit.classify_images(input_flatten=layer1_output_flattened,
                                                     hidden_output=hiddenLayer.output,
                                                     filters=layer1_filters, W=layer3_W,
                                                     b=layer3_b)
    elif classifier == CNN.enums.ClassifierType.svm:
        c_result, c_prob = CNN.svm.classify_images(input_flatten=layer1_output_flattened,
                                                   hidden_output=hiddenLayer.output,
                                                   filters=layer1_filters, W=layer3_W,
                                                   b=layer3_b)
    else:
        raise TypeError('Unknown classifier type, should be either logit or svm', ('classifier:', classifier))

    end_time = time.clock()

    # that's because we only classified one image
    c_result = c_result[0]
    c_prob = c_prob[0]
    c_duration = end_time - start_time

    # __plot_filters_1(img4D, 1)
    # __plot_filters_1(layer0_filters, 2)
    # __plot_filters_1(layer1_filters, 3)
    # __plot_filters_2(loaded_objects[6], 4)
    # __plot_filters_2(loaded_objects[8], 5)

    print('Classification result: %d in %f sec.' % (c_result, c_duration))
    print('Classification confidence: %s' % (CNN.utils.numpy_to_string(c_prob)))

    return c_result, c_prob, c_duration


def __detect_img_deep_model(img4D, model_path, classifier=CNN.enums.ClassifierType.logit):
    loaded_objects = CNN.utils.load_model(model_path)

    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    mlp_layers = loaded_objects[4]
    pool_size = loaded_objects[5]

    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)
    layer2_W = theano.shared(loaded_objects[10], borrow=True)
    layer2_b = theano.shared(loaded_objects[11], borrow=True)
    layer3_W = theano.shared(loaded_objects[12], borrow=True)
    layer3_b = theano.shared(loaded_objects[13], borrow=True)

    layer0_img_dim = img_dim  # = 28 in case of mnist
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)  # = 12 in case of mnist
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)  # = 4 in case of mnist

    start_time = time.clock()

    # layer 0: Conv-Pool
    filter_shape = (nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim)
    image_shape = (1, 1, layer0_img_dim, layer0_img_dim)
    (layer0_filters, layer0_output) = CNN.conv.filter_image(img=img4D, W=layer0_W, b=layer0_b, image_shape=image_shape,
                                                            filter_shape=filter_shape, pool_size=pool_size)

    # layer 1: Conv-Pool
    filter_shape = (nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim)
    image_shape = (1, nkerns[0], layer1_img_dim, layer1_img_dim)
    (layer1_filters, layer1_output) = CNN.conv.filter_image(img=layer0_filters, W=layer1_W, b=layer1_b,
                                                            image_shape=image_shape, filter_shape=filter_shape,
                                                            pool_size=pool_size)

    # layer 2: hidden layer
    hidden_n_in = nkerns[1] * layer2_img_dim * layer2_img_dim
    layer1_output_flattened = layer1_output.flatten(2)
    hiddenLayer = CNN.mlp.HiddenLayer(input=layer1_output_flattened, W=layer2_W, b=layer2_b, n_in=hidden_n_in,
                                      n_out=mlp_layers[0], activation=T.tanh, rng=0)

    # layer 3: logit (logistic regression) or SVM
    c_result = []
    c_prob = []
    if classifier == CNN.enums.ClassifierType.logit:
        c_result, c_prob = CNN.logit.classify_images(input_flatten=layer1_output_flattened,
                                                     hidden_output=hiddenLayer.output,
                                                     filters=layer1_filters, W=layer3_W,
                                                     b=layer3_b)
    elif classifier == CNN.enums.ClassifierType.svm:
        c_result, c_prob = CNN.svm.classify_images(input_flatten=layer1_output_flattened,
                                                   hidden_output=hiddenLayer.output,
                                                   filters=layer1_filters, W=layer3_W,
                                                   b=layer3_b)
    else:
        raise TypeError('Unknown classifier type, should be either logit or svm', ('classifier:', classifier))

    end_time = time.clock()

    # that's because we only classified one image
    c_result = c_result[0]
    c_prob = c_prob[0]
    c_duration = end_time - start_time

    # __plot_filters_1(img4D, 1)
    # __plot_filters_1(layer0_filters, 2)
    # __plot_filters_1(layer1_filters, 3)
    # __plot_filters_2(loaded_objects[6], 4)
    # __plot_filters_2(loaded_objects[8], 5)

    print('Classification result: %d in %f sec.' % (c_result, c_duration))
    print('Classification confidence: %s' % (CNN.utils.numpy_to_string(c_prob)))

    return c_result, c_prob, c_duration


def __probability_map(predictions, locations, window_dim, x_count, y_count, img_width, img_height, img_dim, count):
    r_factor = window_dim / img_dim
    n = x_count * y_count
    locations = numpy.asarray(locations)
    predictions = numpy.asarray(predictions)
    shape = predictions.shape
    predictions = predictions.reshape((shape[1], shape[0]))

    # create an image
    img = numpy.zeros(shape=(img_height, img_width))
    for i in range(0, n):
        if predictions[i][2] <= predictions[i][0] or predictions[i][3] <= predictions[i][1]:
            continue
        newRegion = (predictions[i] * r_factor).astype(int)
        x1 = newRegion[0] + locations[i][0]
        y1 = newRegion[1] + locations[i][1]
        x2 = newRegion[2] + locations[i][0]
        y2 = newRegion[3] + locations[i][1]
        if y2 >= img_height:
            y2 = img_height - 1
        if y1 >= img_height:
            y1 = img_height - 1
        if x2 >= img_width:
            x2 = img_width - 1
        if x1 >= img_width:
            x1 = img_width - 1

        img[y1:y2, x1:x2] += 1

    # normalize image before saving
    img = img * 255 / (img.max() - img.min())
    cv2.imwrite("D:\\_Dataset\\GTSDB\\\Test_Regions\\" + "{0:05d}.png".format(count), img)

    # now, plot the image
    # plot original image and first and second components of output
    import matplotlib.pyplot as plt

    # plt.figure()
    # plt.gray()
    # plt.ion()
    # plt.axis('off')
    # plt.imshow(img, interpolation='nearest')
    # plt.show()
    # dummy_var = 10
