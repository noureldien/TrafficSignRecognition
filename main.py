__author__ = 'Noureldien'

from builtins import print

import os
import matplotlib.pyplot as plt
import learn_classes
import learn_types
import learn_scope
import learn_generators
import read_minst_db
import cnn
import utils
#import convolutional_mlp

print('Hello World')
#convolutional_mlp.evaluate_lenet5(learning_rate=0.1, n_epochs=200, nkerns=[20, 50], batch_size=500)
#convolutional_mlp.evaluate_lenet5(learning_rate=0.1, n_epochs=1, nkerns=[20, 50], batch_size=100)
#convolutional_mlp.evaluate_lenet5(learning_rate=0.1, n_epochs=1, nkerns=[10, 20], batch_size=100) # 4.22%
#convolutional_mlp.evaluate_lenet5(learning_rate=0.2, n_epochs=5, nkerns=[10, 20], batch_size=50) # 1.53%
#convolutional_mlp.evaluate_lenet5(learning_rate=0.05, n_epochs=5, nkerns=[10, 20], batch_size=50) # 1.64%
#convolutional_mlp.evaluate_lenet5(learning_rate=0.2, n_epochs=5, nkerns=[20, 50], batch_size=50) # 1.13%
#cnn.evaluate_lenet5(img_dim=mnist_dim, dataset=mnist_dataset, learning_rate=0.2, n_epochs=1, kernel_dim=[5, 5],
#                    nkerns=[100, 200], mpl_layers=[500, 10], batch_size=50) # 1.98%

gtsrb_dataset = "D:\\_Dataset\\GTSRB\\gtsrb_normalized_28.pkl"
belgiumTS_dataset = "D:\\_Dataset\\\BelgiumTS\\BelgiumTS_normalized_28.pkl"
mnist_dataset = "D:\\_Dataset\\mnist.pkl"

gtsrb_dim = 28
mnist_dim = 28

# train model on mnist database
# cnn.train(img_dim=mnist_dim, dataset_path=mnist_dataset, learning_rate=0.2, n_epochs=5, nkerns=[20, 50], batch_size=50)

# train model on gtsrb database
#cnn.train(img_dim=mnist_dim, dataset_path=mnist_dataset, learning_rate=0.2, n_epochs=5, kernel_dim=(5, 5), nkerns=(100, 200), mpl_layers=(500, 10), batch_size=50)
#cnn.train(img_dim=gtsrb_dim, dataset_path=gtsrb_dataset, learning_rate=0.1, n_epochs=5, nkerns=(20, 50), batch_size=50) # 3.73%
#cnn.train(img_dim=gtsrb_dim, dataset_path=gtsrb_dataset, learning_rate=0.2, n_epochs=1, nkerns=(20, 50), batch_size=50) # 3.35%
#cnn.train(img_dim=gtsrb_dim, dataset_path=gtsrb_dataset, learning_rate=0.1, n_epochs=5, nkerns=(40, 100), batch_size=50) # 3.06%
#cnn.train(img_dim=gtsrb_dim, dataset_path=gtsrb_dataset, learning_rate=0.2, n_epochs=5, nkerns=(40, 100), batch_size=50) # 2.77%
#cnn.train(img_dim=gtsrb_dim, dataset_path=gtsrb_dataset, learning_rate=0.2, n_epochs=5, nkerns=(40, 100), batch_size=50, mlp_layers=(800, 10)) # 3.12%
#cnn.train(img_dim=gtsrb_dim, dataset_path=gtsrb_dataset, learning_rate=0.2, n_epochs=1, nkerns=(20, 50), batch_size=50, mlp_layers=(500, 10)) # 4.77%
#cnn.train(img_dim=gtsrb_dim, dataset_path=gtsrb_dataset, learning_rate=0.2, n_epochs=1, nkerns=(20, 50), batch_size=50, mlp_layers=(100, 10)) # 5.06%
#cnn.train(img_dim=gtsrb_dim, dataset_path=gtsrb_dataset, learning_rate=0.2, n_epochs=1, nkerns=[20, 50], batch_size=50, mlp_layers=(200, 10)) # 4.31%
#cnn.train(img_dim=gtsrb_dim, dataset_path=gtsrb_dataset, learning_rate=0.2, n_epochs=1, nkerns=[4, 10], batch_size=50, mlp_layers=(50, 10)) # 8.00%

# test model on specific image
cnn.classify_img_from_file("D:\\_Dataset\\GTSRB\\Final_Test_Preprocessed_28\\01860.png")

# test model on gtsrb database
#cnn.evaluate_lenet5(img_dim=gtsrb_dim, dataset=gtsrb_dataset, learning_rate=0.1, n_epochs=5, kernel_dim=[5, 5],
#                    nkerns=[100, 200], mpl_layers=[500, 10], batch_size=50)



