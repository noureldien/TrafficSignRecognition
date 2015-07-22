import os
from os import listdir
from os.path import isfile, join
import pickle
import gzip
import numpy
import theano
import theano.tensor
import PIL
import PIL.Image
import skimage
import skimage.transform
import skimage.exposure
import cv2
import CNN
import CNN.recog
import CNN.enums
import CNN.consts
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import random

# region Misc

def numpy_to_string(arr):
    return ", ".join(format(x, ".3f") for x in arr.tolist())


# endregion

# region Load Model/Data

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
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
    del f

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # witch row's correspond to an example. target is a
    # numpy.ndarray of 1 dimensions (vector)) that have the same length as
    # the number of rows in the input. It should give the target
    # target to the example with the same index in the input.

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval


def load_model(model_path):
    save_file = open(model_path, 'rb')
    loaded_objects = []
    for i in range(14):
        loaded_objects.append(pickle.load(save_file))
    save_file.close()
    return loaded_objects


def shared_dataset(data_xy, borrow=True):
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


# endregion

# region Pre-process

def preprocess_dataset_train():
    csvFileName = "D:\\_Dataset\\GTSRB\\Final_Test\\GT-final_test.annotated.csv"

    directory1 = "D:\\_Dataset\\GTSRB\\Final_Training_Cropped\\"
    directory2 = "D:\\_Dataset\\GTSRB\\Final_Training_Preprocessed_80\\"

    prohibitory_classes = CNN.consts.ClassesIDs.PROHIB_CLASSES
    for class_id in prohibitory_classes:
        folderName = "{0:05d}\\".format(class_id)
        subDirectory1 = directory1 + folderName
        subDirectory2 = directory2 + folderName
        files = [f for f in listdir(subDirectory1) if isfile(join(subDirectory1, f))]
        print(class_id)
        # create directory to save files in
        if not os.path.exists(subDirectory2):
            os.mkdir(subDirectory2)
        for file in files:
            # do the following steps : read -> Grayscale -> imadjust -> histeq
            # -> adapthisteq -> ContrastStretchNorm -> resize -> write
            filePathRead = join(subDirectory1, file)
            filePathWrite = join(subDirectory2, file)
            preprocess_image_(filePathRead=filePathRead, filePathWrite=filePathWrite,resize_dim=80)

        print('Finish Class: ' + folderName)


def preprocess_dataset_test():
    from os import listdir
    from os.path import isfile, join

    csvFileName = "D:\\_Dataset\\GTSRB\\Final_Test\\GT-final_test.annotated.csv"

    directory1 = "D:\\_Dataset\\GTSRB\\Final_Training_Scaled\\"
    directory2 = "D:\\_Dataset\\GTSRB\\Final_Training_Preprocessed_80\\"

    directory1 = "D:\\_Dataset\\GTSRB\\Final_Test_Scaled\\"
    directory2 = "D:\\_Dataset\\GTSRB\\Final_Test_Preprocessed_80\\"

    for i in range(0, 3):
        folderName = "{0:05d}\\".format(i)
        subDirectory1 = directory1 + folderName
        subDirectory2 = directory2 + folderName
        onlyfiles = [f for f in listdir(subDirectory1) if isfile(join(subDirectory1, f))]
        for file in onlyfiles:
            # do the following steps : read -> Grayscale -> imadjust -> histeq
            # -> adapthisteq -> ContrastStretchNorm -> resize -> write
            filePathRead = join(subDirectory1, file)
            filePathWrite = join(subDirectory2, file)
            preprocess_image(filePathRead, filePathWrite, 80)

        print('Finish Class: ' + folderName)


def preprocess_image_(filePathRead, filePathWrite, resize_dim=0):
    img = cv2.imread(filePathRead)
    img = preprocess_image(img, resize_dim)
    img *= 255
    img = img.astype(int)
    cv2.imwrite(filePathWrite, img)


def preprocess_image(img, resize_dim=0):
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_eq = skimage.exposure.equalize_hist(img_gs)
    img_adeq = skimage.exposure.equalize_adapthist(img_eq, clip_limit=0.2, kernel_size=(8, 8))
    img_int = skimage.exposure.rescale_intensity(img_adeq, in_range=(0.1, 0.8))
    if resize_dim > 0:
        img_int = skimage.transform.resize(img_int, output_shape=(resize_dim, resize_dim))
    return img_int

    plot_images = False

    if plot_images:
        # region Plot results
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 5))
        ax_img, ax_hist, ax_cdf = __plot_img_and_hist(img, axes[:, 0])
        ax_img.set_title('Low contrast image')
        y_min, y_max = ax_hist.get_ylim()
        ax_hist.set_ylabel('Number of pixels')
        ax_hist.set_yticks(numpy.linspace(0, y_max, 5))
        ax_img, ax_hist, ax_cdf = __plot_img_and_hist(img_eq, axes[:, 1])
        ax_img.set_title('Histogram equalization')
        ax_img, ax_hist, ax_cdf = __plot_img_and_hist(img_adeq, axes[:, 2])
        ax_img.set_title('Adaptive equalization')
        ax_img, ax_hist, ax_cdf = __plot_img_and_hist(img_int, axes[:, 3])
        ax_img.set_title('Contrast stretching')
        ax_cdf.set_ylabel('Fraction of total intensity')
        ax_cdf.set_yticks(numpy.linspace(0, 1, 5))
        # prevent overlap of y-axis labels
        fig.subplots_adjust(wspace=0.4)
        plt.show()

        return
        # endregion


def probability_map(img_path, model_path, classifier=CNN.enums.ClassifierType.logit, window_size=28):
    """
    For the given image, apply sliding window algorithm over different scales and return the heat-probability map
    :param img_path:
    :param model_path:
    :param classifier:
    :param img_dim:
    :return:
    """

    img = cv2.imread(img_path)
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scaling_factor = 0.75
    stride = int(window_size * 0.5)

    scales = -1
    d = min(img.shape[0], img.shape[1])
    while d >= window_size:
        d = int(d * scaling_factor)
        scales += 1

    batch_counts = []
    batch = []
    shapes = []

    # loop on different scales of the image
    for s in range(0, scales):

        # split the images to small images using overlapping sliding window
        # do classification of all of the small images as one batch
        # then output the classification result as probability map
        batch_count = 0
        y_range = numpy.arange(0, img_gs.shape[0] - window_size, stride)
        x_range = numpy.arange(0, img_gs.shape[1] - window_size, stride)
        for y in y_range:
            for x in x_range:
                roi = img_gs[y:y + window_size, x:x + window_size] / 255.0
                batch.append(roi.ravel())
                batch_count += 1

        batch_counts.append(batch_count)

        # re-scale the image to make it smaller
        shapes.append((len(y_range), len(x_range)))
        img_gs = skimage.transform.resize(img_gs, output_shape=(
            int(img_gs.shape[0] * scaling_factor), int(img_gs.shape[1] * scaling_factor)))

    # after we obained all the batches for all the image scales, classify the batches
    batch_np = numpy.asarray(batch, float)
    c_result, c_prob, c_duration = CNN.recog.classify_batch(batch_np, model_path, classifier)
    print('Classification of image batches in %f sec.' % (c_duration))

    # get the classification result and probability each scale
    return
    offset = 0
    c_prob[c_prob < 0.75] = 0
    for i in range(0, len(batch_counts)):
        results = c_result[offset: offset + batch_counts[i]]
        p_maps = numpy.asarray(c_prob[offset: offset + batch_counts[i]])
        p_maps = p_maps.reshape(p_maps.shape[1], p_maps.shape[0])
        __plot_prob_maps(p_maps, shapes[i], i + 1)
        # for p_map in p_maps:
        #    p_map = p_map.reshape(shapes[i])
        #    # display map
        offset += batch_counts[i]

        # generate image to show confidence maps for the 3 classes, each with probabilities
        # note that for each class, we have confidence map at each scale


def rgb_to_gs(path):
    img = cv2.imread(path)
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return img_gs


def rgb_to_gs_and_save(path):
    img = cv2.imread(path)
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    img_save = img_gs * 255
    img_save = img_save.astype(int)
    cv2.imwrite(path, img_save)


def ppm_to_png():
    from os import listdir
    from os.path import isfile, join

    csvFileName = "D:\\_Dataset\\GTSDB\\Train\\gt.txt"

    directory1 = "D:\\_Dataset\\GTSDB\\Train\\00\\"
    directory2 = "D:\\_Dataset\\GTSDB\\Train_PNG\\"

    for i in range(0, 4):
        file1 = "{0:05d}.ppm".format(i)
        file2 = "{0:05d}.png".format(i)
        filePathRead = join(directory1, file1)
        filePathWrite = join(directory1, file2)
        img = cv2.imread(filePathRead)
        cv2.imwrite(filePathWrite, img)


def __plot_img_and_hist(img, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.
    """
    import skimage
    import skimage.exposure
    import matplotlib.pyplot as plt

    img = skimage.img_as_float(img)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = skimage.exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def __plot_prob_maps(maps, shape, fig_num):
    plt.figure(fig_num)
    plt.gray()
    plt.ion()
    length = len(maps)
    for i in range(0, length):
        plt.subplot(1, length, i + 1)
        plt.axis('off')
        p_map = maps[i]
        p_map = p_map.reshape(shape)
        plt.imshow(p_map)
    plt.show()


# endregion

# region GTSR

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

    directoryTrain = "D:\\_Dataset\\GTSRB\\Final_Training_Preprocessed_80\\"
    directoryTest = "D:\\_Dataset\\GTSRB\\Final_Test_Preprocessed_80\\"
    csvFileName = "D:\\_Dataset\\GTSRB\\Final_Test\\GT-final_test.annotated.csv"

    # get the training data
    for i in range(0, 43):
        print(i)
        subDirectory = directoryTrain + "{0:05d}\\".format(i)
        onlyfiles = [f for f in listdir(subDirectory) if isfile(join(subDirectory, f))]
        for file in onlyfiles:
            fileName = join(subDirectory, file)
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
    onlyfiles = [f for f in listdir(directoryTest) if isfile(join(directoryTest, f))]
    for file in onlyfiles:
        fileName = join(directoryTest, file)
        fileData = numpy.asarray(PIL.Image.open(fileName).getdata())
        test_images.append(fileData)

    # now, save the training and data
    train_set = (train_images, train_classes)
    test_set = (test_images, test_classes)
    data = (train_set, test_set)
    pickle.dump(data, open('D:\\_Dataset\\GTSRB\\gtsrb_rerialized_80.pkl', 'wb'))

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
        nTrain = int(n * 3 / 4)
        nValid = n - nTrain

        # create shuffled indexes to shuffle the train images and classes
        idx = numpy.arange(start=0, stop=n, dtype=int).tolist()
        random.shuffle(idx)

        # take the first nTrain items as train_set
        idxRange = numpy.arange(start=0, stop=nTrain, dtype=int).tolist()
        images = [tr_images_reshaped[i][j] for j in idxRange]
        classes = (numpy.ones(shape=(nTrain,), dtype=int) * i).tolist()
        train_images.extend(images)
        train_classes.extend(classes)

        # take the next nValid items as validation_set
        idxRange = numpy.arange(start=nTrain, stop=n, dtype=int).tolist()
        images = [tr_images_reshaped[i][j] for j in idxRange]
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


# endregion

# region BelgiumTS

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
    for i in range(0, 62):
        subDirectory = directoryTrain + "{0:05d}\\".format(i)
        onlyfiles = [f for f in listdir(subDirectory) if isfile(join(subDirectory, f))]
        for file in onlyfiles:
            fileName = join(subDirectory, file)
            fileData = numpy.asarray(PIL.Image.open(fileName).getdata())
            train_images.append(fileData)
            train_classes.append(i)

    # get the test data
    for i in range(0, 62):
        subDirectory = directoryTest + "{0:05d}\\".format(i)
        onlyfiles = [f for f in listdir(subDirectory) if isfile(join(subDirectory, f))]
        for file in onlyfiles:
            fileName = join(subDirectory, file)
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
        nTrain = int(n * 3 / 4)
        nValid = n - nTrain

        # create shuffled indexes to shuffle the train images and classes
        idx = numpy.arange(start=0, stop=n, dtype=int).tolist()
        random.shuffle(idx)

        # take the first nTrain items as train_set
        idxRange = numpy.arange(start=0, stop=nTrain, dtype=int).tolist()
        images = [tr_images_reshaped[i][j] for j in idxRange]
        classes = (numpy.ones(shape=(nTrain,), dtype=int) * i).tolist()
        train_images.extend(images)
        train_classes.extend(classes)

        # take the next nValid items as validation_set
        idxRange = numpy.arange(start=nTrain, stop=n, dtype=int).tolist()
        images = [tr_images_reshaped[i][j] for j in idxRange]
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


# endregion

# region SuperClass

def serialize_SuperClass():
    '''
    Read the preprocessed images (training and test), then split the training to training and validation
    then save them on the disk, Save them with the same format and data structure as the MNIST dataset

    :return:
    '''

    from os import listdir
    from os.path import isfile, join
    import random

    tr_images = []
    tr_classes = []
    test_images = []
    test_classes = []

    directoryTrain = "D:\\_Dataset\\SuperClass\\Training_Preprocessed_Revised\\"
    directoryTest = "D:\\_Dataset\\SuperClass\\Test_Preprocessed_Revised\\"

    nClasses = 3

    # get the training data
    for i in range(0, nClasses):
        subDirectory = directoryTrain + "{0:05d}\\".format(i)
        onlyfiles = [f for f in listdir(subDirectory) if isfile(join(subDirectory, f))]
        for file in onlyfiles:
            fileName = join(subDirectory, file)
            fileData = numpy.asarray(PIL.Image.open(fileName).getdata())
            tr_images.append(fileData)
            tr_classes.append(i)

    # get the test data
    for i in range(0, nClasses):
        subDirectory = directoryTest + "{0:05d}\\".format(i)
        onlyfiles = [f for f in listdir(subDirectory) if isfile(join(subDirectory, f))]
        for file in onlyfiles:
            fileName = join(subDirectory, file)
            fileData = numpy.asarray(PIL.Image.open(fileName).getdata())
            test_images.append(fileData)
            test_classes.append(i)

    # split the tr to train and valid
    # normalize the images and save as double
    train_images = []
    train_classes = []
    valid_images = []
    valid_classes = []

    tr_images_reshaped = []
    for i in range(10):
        tr_images_reshaped.append([])

    for i in range(len(tr_images)):
        tr_images_reshaped[tr_classes[i]].append(tr_images[i])

    for i in range(nClasses):
        n = tr_classes.count(i)
        nTrain = int(n * 4 / 5)
        nValid = n - nTrain

        # create shuffled indexes to shuffle the train images and classes
        idx = numpy.arange(start=0, stop=n, dtype=int).tolist()
        random.shuffle(idx)

        # take the first nTrain items as train_set
        idxRange = numpy.arange(start=0, stop=nTrain, dtype=int).tolist()
        images = [tr_images_reshaped[i][j] for j in idxRange]
        classes = (numpy.ones(shape=(nTrain,), dtype=int) * i).tolist()
        train_images.extend(images)
        train_classes.extend(classes)

        # take the next nValid items as validation_set
        idxRange = numpy.arange(start=nTrain, stop=n, dtype=int).tolist()
        images = [tr_images_reshaped[i][j] for j in idxRange]
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
    pickle.dump(data, open('D:\\_Dataset\\SuperClass\\SuperClass_normalized.pkl', 'wb'))

    print("Finish Preparing Data")


# endregion

# region GTSD


def rerialize_gtsdb():
    # read the german traffic sign detection database
    # for each image, create multible scales
    # for each scale, create regions/batches around the ground truth boundary
    # all these created regions must comprise completely the ground truth
    # re-calculate the x,y of the ground truth, instead of the whole image
    # is the frame_of_reference, the region itself is the frame_of_reference

    from os import listdir
    from os.path import isfile, join
    import csv

    # get the ground truth of the test data

    csv_data = []
    csvFileName = "D:\\_Dataset\\GTSDB\\Ground_Truth\\gt.txt"
    with open(csvFileName, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            col_data = []
            for col in row:
                if len(col_data) == 0:
                    col_data.append(int(col[:-4]))
                else:
                    col_data.append(int(col))

            csv_data.append(col_data)

    # stride represents how dense to sample regions around the ground truth traffic signs
    # also up_scaling factor affects the sampling
    # initial dimension defines what is the biggest traffic sign to recognise
    # actually stride should be dynamic, i.e. smaller strides for smaller window size and vice versa
    # stride_factor = 2 gives more sampling than stride_factor = 1
    # don't use value smaller than 1 for stride factor
    up_scale_factor = 1.2
    biggest_area_factor = 3 ** 2
    min_region_dim = 18
    img_width = 1630
    img_height = 800
    resize_dim = 28
    stride_factor = 2
    regions = numpy.zeros(shape=(0, resize_dim * resize_dim), dtype=float)
    relative_boundaries = numpy.zeros(shape=(0, 4), dtype=int)

    directory = "D:\\_Dataset\\GTSDB\\Training_PNG\\"
    files = [f for f in listdir(directory) if isfile(join(directory, f))]

    for file in files:
        file_id = int(file[:-4])
        print(file_id)
        file_path = join(directory, file)
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # loop on each ground truth (i.e traffic sign)
        # for now, consider only the 'warning' for the time being
        # get the ground truth of 'warning' signs
        boundaries = __gtsr_get_boundaries(csv_data, file_id, CNN.enums.SuperclassType._01_Prohibitory)
        for boundary in boundaries:
            # the biggest traffic sign to recognize is 400*400 in a 1360*800 image
            # that means, we'll start with a window with initial size of the ground_truth
            # for each ground_truth boundary, extract regions in such that:
            # 1. each region fully covers the boundary
            # 2. the boundary must not be bigger than nth (biggest_area_factor) of the region
            # we will recognize smaller ground truth because we up_scale the window every step
            # also, we will recognize bigger ground_truth because the first region taken for
            # a ground_truth is almost it's size
            # boundary is x1, y1, x2, y2 => (x1,y1) top left, (x2, y2) bottom right
            # don't forget that stride of sliding the window is dynamic
            x1 = boundary[0]
            y1 = boundary[1]
            x2 = boundary[2]
            y2 = boundary[3]
            boundary_width = x2 - x1
            boundary_height = y2 - y1
            boundary_max_dim = max(boundary_width, boundary_height)
            window_dim = boundary_max_dim
            boundary_area = boundary_width * boundary_height
            while (window_dim ** 2 / boundary_area) <= biggest_area_factor \
                    and (boundary_max_dim * resize_dim / window_dim) >= min_region_dim \
                    and window_dim <= img_height and window_dim <= img_width:

                # for the current scale of the window, extract regions, start from the
                stride = int((boundary_max_dim * resize_dim / (stride_factor * window_dim)))
                y_range = numpy.arange(start=y2 - window_dim, stop=y1 + 1, step=stride, dtype=int).tolist()
                x_range = numpy.arange(start=x2 - window_dim, stop=x1 + 1, step=stride, dtype=int).tolist()

                # if last x in x_range don't make the sliding window reach the end of the region
                # then add one more x to the x_range to satisfy this
                x_r_len = len(x_range)
                y_r_len = len(y_range)
                if int((x1 - x_range[x_r_len - 1]) * boundary_width / window_dim) > 1:
                    x_range.append(x1)
                if int((y1 - y_range[y_r_len - 1]) * boundary_height / window_dim) > 1:
                    y_range.append(y1)

                # the factor used to rescale the region before saving it to the region array
                # note that we want to rescale to 28*28 to be compatible with our CNN recognition network
                r_factor = window_dim / resize_dim
                for y in y_range:
                    for x in x_range:

                        # make sure that the window in the current position (x, y) is within the image itself
                        if x < 0 or y < 0 or (x + window_dim) > img_width or (y + window_dim) > img_height:
                            continue

                        # TODO
                        # if using the extra-stride in the range makes the region get out of the window
                        # then rebound the window so we can take the last region before exiting the loop

                        # - add region to the region list
                        # - adjust the position of the ground_truth to be relative to the window
                        #   not relative to the image (i.e relative frame of reference)
                        # - don't forget to re_scale the extracted/sampled region to be 28*28
                        #   hence, multiply the relative position with this scaling accordingly
                        # - also, the image needs to be preprocessed so it can be ready for the CNN
                        # - reshape the region to 1-D vector to align with the structure of MNIST database

                        relative_boundary = (numpy.asarray([x1 - x, y1 - y, x2 - x, y2 - y]) / r_factor).astype(int)
                        relative_boundaries = numpy.vstack([relative_boundaries, relative_boundary])

                        region = numpy.copy(img[y:y + window_dim, x:x + window_dim])
                        region = skimage.transform.resize(region, output_shape=(resize_dim, resize_dim))
                        region = region.reshape((resize_dim * resize_dim,))
                        regions = numpy.vstack([regions, region])

                        # save region for experimenting/debugging
                        # filePathWrite = "D:\\_Dataset\\GTSDB\\Training_Regions\\" + file[:-4] + "_" + "{0:05d}.png".format(len(regions))
                        # cv2.imwrite(filePathWrite, region.reshape((resize_dim, resize_dim))* 255)

                # add some true negatives to increase variance of the machine
                # add only n images per scale per image, n = 2
                regions_negatives = __sample_true_negatives(img, window_dim, resize_dim, boundaries, 2)
                regions = numpy.vstack([regions, regions_negatives])
                relative_boundaries = numpy.vstack([relative_boundaries, numpy.zeros(shape=(regions_negatives.shape[0], 4), dtype=int)])

                # also add relative boundaries for them as the ground truth
                # which should only be zeros

                # saving images for experimenting/debugging
                # ss_count = 0
                # for s in regions_negative:
                # ss_count += 1
                # filePathWrite = "D:\\_Dataset\\GTSDB\\Training_Regions\\" + file[:-4] + "_" + "{0:03d}".format(len(regions)) + "{0:02d}.png".format(ss_count)
                # cv2.imwrite(filePathWrite, s.reshape((resize_dim, resize_dim)) * 255)

                # now we up_scale the window
                window_dim = int(window_dim * up_scale_factor)

            dummy_variable_end_file_loop = 10

    # dump the regions into a pickle file
    data = (regions, relative_boundaries)
    pickle.dump(data, open('D:\\_Dataset\\GTSDB\\gtsdb_prohibitory_serialized.pkl', 'wb'))

    print("Finish sampling regions for detector")


def organize_gtsdb():
    data = pickle.load(open('D:\\_Dataset\\GTSDB\\gtsdb_prohibitory_serialized.pkl', 'rb'))
    regions = data[0]
    boundaries = data[1]
    del data

    n = len(boundaries)
    nTrain = int(n * 2 / 4)
    nValid = int(n * 1 / 4)

    # may be try to shuffle the data before dividing
    train_images = regions[0:nTrain]
    train_classes = boundaries[0:nTrain]
    valid_images = regions[nTrain:nTrain + nValid]
    valid_classes = boundaries[nTrain:nTrain + nValid]
    test_images = regions[nTrain + nValid:n]
    test_classes = boundaries[nTrain + nValid:n]

    # shuffle the train, valid and test dateset
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

    # now, save the training and data
    data = ((train_images, train_classes), (valid_images, valid_classes), (test_images, test_classes))
    pickle.dump(data, open('D:\\_Dataset\\GTSDB\\gtsdb_prohibitory_organized.pkl', 'wb'))

    print("Finish Preparing Data")


def __sample_true_negatives(img, window_dim, resize_dim, boundaries, count):
    img_h = img.shape[0]
    img_w = img.shape[1]
    n_boundary = len(boundaries)

    regions = numpy.zeros(shape=(0, resize_dim * resize_dim), dtype=float)
    while regions.shape[0] < count:
        # create a region in a random/stochastic way
        # the region be with the same window_dim
        # and does not intersect with any of the boundaries
        y1 = random.randint(0, img_h - window_dim)
        y2 = y1 + window_dim
        x1 = random.randint(0, img_w - window_dim)
        x2 = x1 + window_dim

        # now, check if any of the boundaries intersect
        # with the region
        n_conditions = 0
        for boundary in boundaries:
            b_x1 = boundary[0]
            b_y1 = boundary[1]
            b_x2 = boundary[2]
            b_y2 = boundary[3]
            if x2 <= b_x1 or x1 >= b_x2 or y2 <= b_y1 or y1 >= b_y2:
                n_conditions += 1
            else:
                break
        if n_conditions == n_boundary:
            region = numpy.copy(img[y1:y2, x1:x2])
            region = skimage.transform.resize(region, output_shape=(resize_dim, resize_dim))
            region = region.reshape((resize_dim * resize_dim,))
            regions = numpy.vstack([regions, region])

    return regions


def __gtsr_get_boundaries(gt_data, image_id, superclass_type=CNN.enums.SuperclassType._00_All):
    '''
    Get the list of ground truth (boundary of a traffic sign) in the image with the given id
    If superclass is provided, then get the ground truth for only this superclass
    Else, get them all
    :param image_id:
    :param superclass_type:
    :return:
    '''

    prohib_classes = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
    warning_classes = [11, 12, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    info_classes = [33, 34, 35, 36, 37, 38, 39, 40]
    superclasses = [prohib_classes, warning_classes, info_classes]
    type_id = superclass_type.value - 1

    if type_id == -1:
        items = [f[1:5] for f in gt_data if f[0] == image_id]
    else:
        items = [f[1:5] for f in gt_data if f[0] == image_id and f[5] in superclasses[type_id]]

    return items


# endregion

# region Check Database

def check_database_1():
    """
    Loop on random sample of the images and see if their values are correct or not
    :return:
    """

    import matplotlib.pyplot as plt

    # data = pickle.load(open('D:\\_Dataset\\mnist.pkl', 'rb'))
    # data = pickle.load(open('D:\\_Dataset\\BelgiumTS\\BelgiumTS_normalized_28.pkl', 'rb'))
    data = pickle.load(open('D:\\_Dataset\\SuperClass\\SuperClass_normalized.pkl', 'rb'))

    images = data[1][0]
    classes = data[1][1]

    del data

    # get first column of the tuple (which represents the image, while second one represents the image class)
    # then get the first image and show it
    idx = numpy.arange(start=0, stop=(len(classes)), step=len(classes) / 12, dtype=int).tolist()
    print(len(classes))
    plt.figure(1)
    plt.ion()
    plt.gray()
    plt.axis('off')
    for i in idx:
        photo = images[i]
        photoReshaped = photo.reshape((28, 28))
        c = classes[i]
        print(c)
        plt.imshow(photoReshaped)
        plt.show()
        x = 10


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

    # aPhoto = PIL.Image.open(".\data\\test-image.png")
    aPhoto = PIL.Image.open(".\data\\test_00014.ppm")
    data0 = aPhoto.getdata()
    data1 = list(data0)
    data2 = numpy.asarray(data0)
    data3 = numpy.asarray(aPhoto)
    matplotlib.pyplot.imshow(data3)
    hiThere = 10

# endregion
