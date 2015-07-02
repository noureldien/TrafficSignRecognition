import os
import pickle
import gzip
import numpy
import theano
import theano.tensor
import matplotlib
import matplotlib.cm
import matplotlib.pyplot
import PIL
import PIL.Image

def unzip_load_data(dataset):

    f = gzip.open(dataset, 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()
    f.close()

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    print('... loading data')

    # Load the dataset
    f = open(dataset, 'rb')
    train_set, valid_set, test_set =  pickle.load(f)
    f.close()
    del f

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # witch row's correspond to an example. target is a
    # numpy.ndarray of 1 dimensions (vector)) that have the same length as
    # the number of rows in the input. It should give the target
    # target to the example with the same index in the input.

    test_set_x, test_set_y = __shared_dataset(test_set)
    valid_set_x, valid_set_y = __shared_dataset(valid_set)
    train_set_x, train_set_y = __shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def __shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        shared_y_casted = theano.tensor.cast(shared_y, 'int32')
        return shared_x, shared_y_casted

def serialize_gtsr():
    '''
    Read the preprocessed images (training and test) and save them on the disk
    Save them with the same format and data structure as the MNIST dataset

    :return:
    '''

    from os import listdir
    from os.path import isfile, join

    train_images = []
    train_classes = []
    test_images = []
    test_classes = []

    directoryTrain = "D:\\_Dataset\\GTSRB\\Final_Training_Preprocessed_28\\"
    directoryTest = "D:\\_Dataset\\GTSRB\\Final_Test_Preprocessed_28\\"
    csvFileName = "D:\\_Dataset\\GTSRB\\Final_Test\\GT-final_test.annotated.csv"

    # get the training data
    for i in range (0, 43):
        print(i)
        subDirectory = directoryTrain + "{0:05d}\\".format(i)
        onlyfiles = [ f for f in listdir(subDirectory) if isfile(join(subDirectory,f)) ]
        for file in onlyfiles:
            fileName = join(subDirectory,file)
            fileData = numpy.asarray(PIL.Image.open(fileName).getdata())
            train_images.append(fileData)
            train_classes.append(i)

    # get the ground truth of the test data
    import csv
    with open(csvFileName, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            if row[7] != "ClassId":
                test_classes.append(int(row[7]))

    # get the test data
    onlyfiles = [ f for f in listdir(directoryTest) if isfile(join(directoryTest,f)) ]
    for file in onlyfiles:
        fileName = join(directoryTest,file)
        fileData = numpy.asarray(PIL.Image.open(fileName).getdata())
        test_images.append(fileData)

    # now, save the training and data
    train_set = (train_images, train_classes)
    test_set = (test_images, test_classes)
    data = (train_set, test_set)
    pickle.dump(data, open('D:\\_Dataset\\GTSRB\\gtsrb.pkl', 'wb'))

    print("Finish Preparing Data")

def reduce_gtsr():
    '''
    Read the preprocessed images (training and test) and save them on the disk
    Save them with the same format and data structure as the MNIST dataset
    - only read training data of the first 10 classes only
    - read the test data corresponding to these 10 classes
    - split the training to train and valid testsets to have the same structure as MNIST dataset

    :return:
    '''

    data = pickle.load(open('D:\\_Dataset\\GTSRB\\gtsrb.pkl', 'rb'))
    tr_set = data[0]
    test_set = data[1]

    train_images = tr_set[0]
    train_classes = tr_set[1]
    test_images = test_set[0]
    test_classes = test_set[1]

    train_images_reduced = []
    train_classes_reduced = []
    test_images_reduced = []
    test_classes_reduced = []

    # reduce training
    for i in range(0, len(train_images)):
        if train_classes[i] > 9:
            break
        else:
            train_images_reduced.append(train_images[i])
            train_classes_reduced.append(train_classes[i])

    # reduce test
    for i in range(0, len(test_images)):
        if test_classes[i] < 10:
            test_images_reduced.append(test_images[i])
            test_classes_reduced.append(test_classes[i])

    print(len(train_images_reduced))
    print(len(train_classes_reduced))
    print(len(test_images_reduced))
    print(len(test_classes_reduced))

    # now, save the training and data
    train_set = (train_images_reduced, train_classes_reduced)
    test_set = (test_images_reduced, test_classes_reduced)
    data = (train_set, test_set)
    pickle.dump(data, open('D:\\_Dataset\\GTSRB\\gtsrb_reduced.pkl', 'wb'))

    print("Finish Preparing Data")

def organize_gtsr():
    """
    Read the reducted dataset (it contains only 10 classes out of 43)
    then split the training to training and validation, the save it on disk

    :return:
    """

    import random

    data = pickle.load(open('D:\\_Dataset\\GTSRB\\gtsrb_reduced.pkl', 'rb'))

    tr_set = data[0]

    tr_images = tr_set[0]
    tr_classes = tr_set[1]

    train_images = []
    train_classes = []
    valid_images = []
    valid_classes = []

    tr_images_reshaped = []
    for i in range(10):
        tr_images_reshaped.append([])

    for i in range(len(tr_images)):
        tr_images_reshaped[tr_classes[i]].append(tr_images[i])

    for i in range(10):
        n = tr_classes.count(i)
        nTrain = int(n * 3/4)
        nValid = n - nTrain

        # create shuffled indexes to suffle the train images and classes
        idx = numpy.arange(start=0, stop=n, dtype=int).tolist()
        random.shuffle(idx)

        # take the first nTrain items as train_set
        idxRange = numpy.arange(start=0, stop=nTrain, dtype=int).tolist()
        images = [ tr_images_reshaped[i][j] for j in idxRange ]
        classes = (numpy.ones(shape=(nTrain,), dtype=int) * i).tolist()
        train_images.extend(images)
        train_classes.extend(classes)

        # take the next nValid items as validation_set
        idxRange = numpy.arange(start=nTrain, stop=n, dtype=int).tolist()
        images = [tr_images_reshaped[i][j] for j in idxRange ]
        classes = (numpy.ones(shape=(nValid,), dtype=int) * i).tolist()
        valid_images.extend(images)
        valid_classes.extend(classes)

    # shuffle the train and valid dateset
    idx = numpy.arange(start=0, stop=len(train_classes), dtype=int).tolist()
    random.shuffle(idx)
    train_images_shuffled = [train_images[j] for j in idx]
    train_classes_shuffled = [train_classes[j] for j in idx]

    idx = numpy.arange(start=0, stop=len(valid_classes), dtype=int).tolist()
    random.shuffle(idx)
    valid_images_shuffled = [valid_images[j] for j in idx]
    valid_classes_shuffled = [valid_classes[j] for j in idx]

    # change array to numpy
    train_images = numpy.asarray(train_images_shuffled)
    train_classes = numpy.asarray(train_classes_shuffled)
    valid_images = numpy.asarray(valid_images_shuffled)
    valid_classes = numpy.asarray(valid_classes_shuffled)
    test_images = numpy.asarray(data[1][0])
    test_classes = numpy.asarray(data[1][1])

    # For all the images, cast the ndarray from int to float64 then normalize (divide by 255)
    train_images = train_images.astype(float) / 255.0
    valid_images = valid_images.astype(float) / 255.0
    test_images = test_images.astype(float) / 255.0

    # now, save the training and data
    data = ((train_images, train_classes), (valid_images, valid_classes), (test_images, test_classes))
    pickle.dump(data, open('D:\\_Dataset\\GTSRB\\gtsrb_normalized.pkl', 'wb'))

    print("Finish Preparing Data")

def serialize_belgiumTS():
    '''
    Read the preprocessed images (training and test) and save them on the disk
    Save them with the same format and data structure as the MNIST dataset

    :return:
    '''

    from os import listdir
    from os.path import isfile, join

    train_images = []
    train_classes = []
    test_images = []
    test_classes = []

    directoryTrain = "D:\\_Dataset\\BelgiumTS\\Training_Preprocessed_28\\"
    directoryTest = "D:\\_Dataset\\BelgiumTS\\Test_Preprocessed_28\\"

    # get the training data
    for i in range (0, 62):
        subDirectory = directoryTrain + "{0:05d}\\".format(i)
        onlyfiles = [f for f in listdir(subDirectory) if isfile(join(subDirectory,f))]
        for file in onlyfiles:
            fileName = join(subDirectory,file)
            fileData = numpy.asarray(PIL.Image.open(fileName).getdata())
            train_images.append(fileData)
            train_classes.append(i)

    # get the test data
    for i in range (0, 62):
        subDirectory = directoryTest + "{0:05d}\\".format(i)
        onlyfiles = [f for f in listdir(subDirectory) if isfile(join(subDirectory,f))]
        for file in onlyfiles:
            fileName = join(subDirectory,file)
            fileData = numpy.asarray(PIL.Image.open(fileName).getdata())
            test_images.append(fileData)
            test_classes.append(i)

    # now, save the training and data
    train_set = (train_images, train_classes)
    test_set = (test_images, test_classes)
    data = (train_set, test_set)

    pickle.dump(data, open('D:\\_Dataset\\BelgiumTS\\BelgiumTS.pkl', 'wb'))

    print("Finish Preparing Data")

def reduce_belgiumTS():
    '''
    Read the preprocessed images (training and test) and save them on the disk
    Save them with the same format and data structure as the MNIST dataset
    - only read training data of the first 10 classes only
    - read the test data corresponding to these 10 classes
    - split the training to train and valid testsets to have the same structure as MNIST dataset

    :return:
    '''

    data = pickle.load(open('D:\\_Dataset\\BelgiumTS\\BelgiumTS.pkl', 'rb'))

    tr_set = data[0]
    test_set = data[1]

    train_images = tr_set[0]
    train_classes = tr_set[1]
    test_images = test_set[0]
    test_classes = test_set[1]

    train_images_reduced = []
    train_classes_reduced = []
    test_images_reduced = []
    test_classes_reduced = []

    # reduce training
    for i in range(0, len(train_images)):
        if train_classes[i] > 9:
            break
        else:
            train_images_reduced.append(train_images[i])
            train_classes_reduced.append(train_classes[i])

    # reduce test
    for i in range(0, len(test_images)):
        if test_classes[i] > 9:
            break
        else:
            test_images_reduced.append(test_images[i])
            test_classes_reduced.append(test_classes[i])

    print(len(train_images_reduced))
    print(len(train_classes_reduced))
    print(len(test_images_reduced))
    print(len(test_classes_reduced))

    # now, save the training and data
    train_set = (train_images_reduced, train_classes_reduced)
    test_set = (test_images_reduced, test_classes_reduced)
    data = (train_set, test_set)
    pickle.dump(data, open('D:\\_Dataset\\BelgiumTS\\BelgiumTS_reduced.pkl', 'wb'))

    print("Finish Preparing Data")

def organize_belgiumTS():
    """
    Read the reducted dataset (it contains only 10 classes out of 43)
    then split the training to training and validation, the save it on disk

    :return:
    """

    import random

    data = pickle.load(open('D:\\_Dataset\\BelgiumTS\\BelgiumTS_reduced.pkl', 'rb'))

    tr_set = data[0]

    tr_images = tr_set[0]
    tr_classes = tr_set[1]

    train_images = []
    train_classes = []
    valid_images = []
    valid_classes = []
    test_images = data[1][0]
    test_classes = data[1][1]

    del data

    tr_images_reshaped = []
    for i in range(10):
        tr_images_reshaped.append([])

    for i in range(len(tr_images)):
        tr_images_reshaped[tr_classes[i]].append(tr_images[i])

    for i in range(10):
        n = tr_classes.count(i)
        nTrain = int(n * 3/4)
        nValid = n - nTrain

        # create shuffled indexes to suffle the train images and classes
        idx = numpy.arange(start=0, stop=n, dtype=int).tolist()
        random.shuffle(idx)

        # take the first nTrain items as train_set
        idxRange = numpy.arange(start=0, stop=nTrain, dtype=int).tolist()
        images = [ tr_images_reshaped[i][j] for j in idxRange ]
        classes = (numpy.ones(shape=(nTrain,), dtype=int) * i).tolist()
        train_images.extend(images)
        train_classes.extend(classes)

        # take the next nValid items as validation_set
        idxRange = numpy.arange(start=nTrain, stop=n, dtype=int).tolist()
        images = [tr_images_reshaped[i][j] for j in idxRange ]
        classes = (numpy.ones(shape=(nValid,), dtype=int) * i).tolist()
        valid_images.extend(images)
        valid_classes.extend(classes)

    # shuffle the train and valid dateset
    idx = numpy.arange(start=0, stop=len(train_classes), dtype=int).tolist()
    random.shuffle(idx)
    train_images_shuffled = [train_images[j] for j in idx]
    train_classes_shuffled = [train_classes[j] for j in idx]

    idx = numpy.arange(start=0, stop=len(valid_classes), dtype=int).tolist()
    random.shuffle(idx)
    valid_images_shuffled = [valid_images[j] for j in idx]
    valid_classes_shuffled = [valid_classes[j] for j in idx]

    idx = numpy.arange(start=0, stop=len(test_classes), dtype=int).tolist()
    random.shuffle(idx)
    test_images_shuffled = [test_images[j] for j in idx]
    test_classes_shuffled = [test_classes[j] for j in idx]

    # change array to numpy
    train_images = numpy.asarray(train_images_shuffled)
    train_classes = numpy.asarray(train_classes_shuffled)
    valid_images = numpy.asarray(valid_images_shuffled)
    valid_classes = numpy.asarray(valid_classes_shuffled)
    test_images = numpy.asarray(test_images_shuffled)
    test_classes = numpy.asarray(test_classes_shuffled)

    # For all the images, cast the ndarray from int to float64 then normalize (divide by 255)
    train_images = train_images.astype(float) / 255.0
    valid_images = valid_images.astype(float) / 255.0
    test_images = test_images.astype(float) / 255.0

    # now, save the training and data
    data = ((train_images, train_classes), (valid_images, valid_classes), (test_images, test_classes))
    pickle.dump(data, open('D:\\_Dataset\\BelgiumTS\\BelgiumTS_normalized.pkl', 'wb'))

    print("Finish Preparing Data")

def check_database_1():
    """
    Loop on random sample of the images and see if their values are correct or not
    :return:
    """

    data = pickle.load(open('D:\\_Dataset\\BelgiumTS\\BelgiumTS_normalized_28.pkl', 'rb'))
    #data = pickle.load(open('D:\\_Dataset\\mnist.pkl', 'rb'))

    images = data[0][0]
    classes = data[0][1]

    del data

    # get first column of the tuple (which represents the image, while second one represents the image class)
    # then get the first image and show it
    idx = numpy.arange(start=0, stop=(len(classes)), step= len(classes)/8, dtype=int).tolist()
    print(len(classes))
    for i in idx:
        photo = images[i]
        photoReshaped = photo.reshape((28, 28))
        matplotlib.pyplot.imshow(photoReshaped, cmap=matplotlib.cm.Greys_r)
        c = classes[i]
        print(c)

def check_database_2():

    data_1 = pickle.load(open('D:\\_Dataset\\GTSRB\\gtsrb_shuffled.pkl', 'rb'))
    data_2 = pickle.load(open("D:\\_Dataset\\mnist.pkl", 'rb'))

    img_1 = data_1[0][0]
    img_2 = data_2[0][0]

    del data_1
    del data_2

    x = 10

def check_database_3():
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (German Traffic Sign Recognition GTSR Dataset)
    '''

    #############
    # LOAD DATA #
    #############

    print('... loading data')

    # Load the dataset
    data = pickle.load(open('D:\\_Dataset\\BelgiumTS\\BelgiumTS_normalized_28.pkl', 'rb'))
    valid_imgs = data[1][0]
    del data

    # get first column of the tuple (which represents the image, while second one represents the image class)
    # then get the first image and show it
    photo = valid_imgs[0]

    photoReshaped = photo.reshape((28, 28))
    matplotlib.pyplot.imshow(photoReshaped, cmap=matplotlib.cm.Greys_r)

    #aPhoto = PIL.Image.open(".\data\\test-image.png")
    aPhoto = PIL.Image.open(".\data\\test_00014.ppm")
    data0 = aPhoto.getdata()
    data1 = list(data0)
    data2 = numpy.asarray(data0)
    data3 = numpy.asarray(aPhoto)
    matplotlib.pyplot.imshow(data3)
    hiThere = 10




