import CNN.recog
import CNN.detec
import CNN.enums
import CNN.consts
import CNN.utils

import numpy
import pickle

import lasagne
import lasagne.layers
import nolearn
import nolearn.lasagne


def classify_testset():

    dataset_path = "D:\\_Dataset\GTSRB\\Final_Test_Preprocessed\\superclass_organized_28.pkl"
    sc_model_path = "D:\\_Dataset\\SuperClass\\cnn_model_las_28.pkl"
    prohib_model_path = "D:\\_Dataset\\GTSRB\\cnn_model_las_p_28.pkl"
    warning_model_path = "D:\\_Dataset\\GTSRB\\cnn_model_las_w_28.pkl"
    mandat_model_path = "D:\\_Dataset\\GTSRB\\cnn_model_las_m_28.pkl"
    odd_model_path = "D:\\_Dataset\\GTSRB\\cnn_model_las_o_28.pkl"

    train_images = []
    train_classes = []

    # load data
    print('... loading data')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # load models
    with open(sc_model_path, 'rb') as f:
        sc_model = pickle.load(f)

    with open(sc_model_path, 'rb') as f:
        prohib_model = pickle.load(f)





    sc_model =