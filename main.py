from builtins import print

import CNN
import CNN.utils
import CNN.recog
import CNN.detec
import CNN.enums
import CNN.prop
import CNN.stview
import numpy

print('Traffic Sign Recognition')

# region Recognition (Model 28)

# img_dim_28 = 80

# mnist_dataset = "D://_Dataset//MNIST//mnist.pkl"
# CNN.recog.train_linear_classifier(dataset_path=mnist_dataset, learning_rate=0.1, n_epochs=10, mlp_layers=(500, 1), nkerns=(10, 20), batch_size=100)

# gtsrb_dataset_28 = "D://_Dataset//GTSRB//gtsrb_organized_28.pkl"
# belgiumTS_dataset_28 = "D://_Dataset///BelgiumTS//BelgiumTS_normalized_28.pkl"
# superclass_dataset_28 = "D://_Dataset//SuperClass//SuperClass_normalized.pkl"
#
# gtsrb_model_28 = 'D://_Dataset//GTSRB//cnn_model.pkl'
# superclass_model_28 = 'D://_Dataset//SuperClass//cnn_model.pkl'
# superclass_model_28_svm = 'D://_Dataset//SuperClass//cnn_model_svm.pkl'

# train model on mnist database
# CNN.recog.train(dataset_path=mnist_dataset ,learning_rate=0.1, n_epochs=200, nkerns=(20, 50), batch_size=500)
# CNN.recog.train(dataset_path=mnist_dataset ,learning_rate=0.1, n_epochs=1, nkerns=(20, 50), batch_size=100)
# CNN.recog.train(dataset_path=mnist_dataset ,learning_rate=0.1, n_epochs=1, nkerns=(10, 20), batch_size=100) # 4.22%
# CNN.recog.train(dataset_path=mnist_dataset ,learning_rate=0.2, n_epochs=5, nkerns=(10, 20), batch_size=50) # 1.53%
# CNN.recog.train(dataset_path=mnist_dataset ,learning_rate=0.05, n_epochs=5, nkerns=(10, 20), batch_size=50) # 1.64%
# CNN.recog.train(dataset_path=mnist_dataset ,learning_rate=0.2, n_epochs=5, nkerns=(20, 50), batch_size=50) # 1.13%
# CNN.recog.train(dataset_path=mnist_dataset, learning_rate=0.2, n_epochs=1, kernel_dim=(5, 5), nkerns=(100, 200), mpl_layers=(500, 10), batch_size=50) # 1.98%
# CNN.recog.train(img_dim_28=img_dim_28, dataset_path=mnist_dataset, learning_rate=0.2, n_epochs=5, nkerns=[20, 50], batch_size=50)

# train model on gtsrb database
# CNN.recog.train(dataset_path=gtsrb_dataset_28, learning_rate=0.2, n_epochs=5, kernel_dim=(5, 5), nkerns=(100, 200), mpl_layers=(500, 10), batch_size=50)
# CNN.recog.train(dataset_path=gtsrb_dataset_28, learning_rate=0.1, n_epochs=5, nkerns=(20, 50), mlp_layers=(500, 12), batch_size=50)  # 3.73%
# CNN.recog.train(dataset_path=gtsrb_dataset_28, learning_rate=0.2, n_epochs=1, nkerns=(20, 50), mlp_layers=(500, 12), batch_size=50) # 3.35%
# CNN.recog.train(dataset_path=gtsrb_dataset_28, learning_rate=0.1, n_epochs=5, nkerns=(40, 100), mlp_layers=(500, 12), batch_size=50) # 3.06%
# CNN.recog.train(dataset_path=gtsrb_dataset_28, learning_rate=0.2, n_epochs=5, nkerns=(40, 100), mlp_layers=(500, 12), batch_size=50) # 2.77%
# CNN.recog.train(dataset_path=gtsrb_dataset_28, learning_rate=0.2, n_epochs=5, nkerns=(40, 100), mlp_layers=(500, 12), batch_size=50, mlp_layers=(800, 10)) # 3.12%
# CNN.recog.train(dataset_path=gtsrb_dataset_28, learning_rate=0.2, n_epochs=1, nkerns=(20, 50), mlp_layers=(500, 12), batch_size=50, mlp_layers=(500, 10)) # 4.77%
# CNN.recog.train(dataset_path=gtsrb_dataset_28, learning_rate=0.2, n_epochs=1, nkerns=(20, 50), mlp_layers=(500, 12), batch_size=50, mlp_layers=(100, 10)) # 5.06%
# CNN.recog.train(dataset_path=gtsrb_dataset_28, learning_rate=0.2, n_epochs=1, nkerns=(20, 50), mlp_layers=(500, 12), batch_size=50, mlp_layers=(200, 10)) # 4.31%
# CNN.recog.train(dataset_path=gtsrb_dataset_28, learning_rate=0.2, n_epochs=1, nkerns=(4, 10), mlp_layers=(500, 12), batch_size=50, mlp_layers=(50, 10)) # 8.00%

# test model on specific image
# CNN.recog.classify_img_from_file("D://_Dataset//GTSRB//Final_Test_Preprocessed_28//00412.png", gtsrb_model_28)
# CNN.recog.classify_img_from_file("D://_Dataset//BelgiumTS//Test_Scaled//00031//02656_00000.png", gtsrb_model_28, is_rgb=True)

# train model on gtsrb database
# cnn.evaluate_lenet5(img_dim_28=img_dim_28, dataset=mnist_dataset_28, learning_rate=0.1, n_epochs=5, kernel_dim=[5, 5], nkerns=[100, 200], mpl_layers=[500, 10], batch_size=50)

# endregion

# region Recognition (Prohibitory 80)

# img_dim_80 = 80
# gtsrb_model_80 = 'D://_Dataset//GTSRB//lsn_model_p_80.pkl'
# gtsrb_dataset_80 = 'D://_Dataset//GTSRB//gtsrb_organized_p_80.pkl'

# CNN.utils.preprocess_dataset_train(img_dim_80)
# CNN.utils.preprocess_dataset_test(img_dim_80)
# CNN.utils.serialize_gtsr(img_dim_28)
# CNN.utils.organize_gtsr(img_dim_28)
# CNN.utils.remap_class_ids(img_dim_80)
# CNN.utils.check_database_4()

# train model on GTSRB 80 database
# CNN.recog.train_deep(dataset_path=gtsrb_dataset_80, model_path=gtsrb_model_80, learning_rate=0.05, n_epochs=2, kernel_dim=(13, 5, 4),
#                     nkerns=(10, 50, 200), mlp_layers=(500, 12), batch_size=10)

# just make sure that the train_deep network is working
# CNN.recog.train_deep(dataset_path=d, n_epochs=5, batch_size=10, img_dim=28, learning_rate=0.01, kernel_dim=(5, 3, 3), nkerns=(10, 80, 200), mlp_layers=(500, 12), pool_size=(2, 2)) # 40.0%
# CNN.recog.train_deep(dataset_path=d, n_epochs=5, batch_size=5, img_dim=28, learning_rate=0.01, kernel_dim=(5, 3, 3), nkerns=(10, 80, 200), mlp_layers=(500, 12), pool_size=(2, 2)) # 9.0%
# CNN.recog.train_deep(dataset_path=d, n_epochs=5, batch_size=5, img_dim=28, learning_rate=0.1, kernel_dim=(5, 3, 3), nkerns=(10, 80, 200), mlp_layers=(500, 12), pool_size=(2, 2)) # 4.0%
# CNN.recog.train_deep(dataset_path=d, n_epochs=5, batch_size=20, img_dim=28, learning_rate=0.1, kernel_dim=(5, 3, 3), nkerns=(20, 160, 400), mlp_layers=(800, 12), pool_size=(2, 2)) # 3.4%

# test the recognition
# p = "D://_Dataset/GTSDB//Training_Regions//00473_04450.png"
# CNN.recog.classify_img_from_file(img_path=p, model_path=gtsrb_model_80, img_dim=img_dim_80, model_type=CNN.enums.ModelType._02_conv3_mlp2)

# endregion

# region Recognition (Mandatory 80)

# img_dim_80 = 80
# gtsrb_model_80 = 'D://_Dataset//GTSRB//cnn_model_m_80.pkl'
# gtsrb_dataset_80 = 'D://_Dataset//GTSRB//gtsrb_organized_m_80.pkl'

# CNN.utils.serialize_gtsr(img_dim_80, CNN.enums.SuperclassType._03_Mandatory)
# CNN.utils.organize_gtsr(img_dim_80, CNN.enums.SuperclassType._03_Mandatory)
# CNN.utils.map_class_ids(img_dim_80, CNN.enums.SuperclassType._03_Mandatory)
# CNN.utils.check_database_4()

# train model on GTSRB 80 database
# CNN.recog.train_deep(dataset_path=gtsrb_dataset_80, model_path=gtsrb_model_80, learning_rate=0.05, n_epochs=5, kernel_dim=(9, 7, 4),
#                      nkerns=(10, 50, 200), mlp_layers=(500, 8), batch_size=10)

# test the recognition
# p = "D://_Dataset/GTSDB//Training_Regions//00473_04450.png"
# CNN.recog.classify_img_from_file(img_path=p, model_path=gtsrb_model_80, img_dim=img_dim_80, model_type=CNN.enums.ModelType._02_conv3_mlp2)

# endregion

# region Recognition (Warning 80)

# img_dim_80 = 80
# gtsrb_model_80 = 'D://_Dataset//GTSRB//cnn_model_w_80.pkl'
# gtsrb_dataset_80 = 'D://_Dataset//GTSRB//gtsrb_organized_w_80.pkl'

# CNN.utils.serialize_gtsr(img_dim_80, CNN.enums.SuperclassType._02_Warning)
# CNN.utils.organize_gtsr(img_dim_80, CNN.enums.SuperclassType._02_Warning)
# CNN.utils.map_class_ids(img_dim_80, CNN.enums.SuperclassType._02_Warning)
# CNN.utils.check_database_4()

# train model on GTSRB 80 database
# CNN.recog.train_deep(dataset_path=gtsrb_dataset_80, model_path=gtsrb_model_80, learning_rate=0.04, n_epochs=5, kernel_dim=(9, 7, 4),
#                     nkerns=(10, 50, 200), mlp_layers=(600, 15), batch_size=10) # 9.5% increase the MLP capacity to get better results!

# test the recognition
# p = "D://_Dataset/GTSDB//Training_Regions//00473_04450.png"
# CNN.recog.classify_img_from_file(img_path=p, model_path=gtsrb_model_80, img_dim=img_dim_80, model_type=CNN.enums.ModelType._02_conv3_mlp2)

# endregion

# region Recognition (Superclass 80)

# img_dim_80 = 80
# img_dim_28 = 28
#
# superclass_model_80 = 'D://_Dataset//SuperClass//cnn_model_80.pkl'
# superclass_dataset_80 = 'D://_Dataset//SuperClass//cnn_model_80.pkl'
#
# superclass_model_28 = 'D://_Dataset//SuperClass//cnn_model_28_lasagne.pkl'
# superclass_dataset_28 = 'D://_Dataset//SuperClass//superclass_normalized_28.pkl'

# train super-class model on 28 pixels
# CNN.recog.train_superclass_classifier_shallow(dataset_path=superclass_dataset_28, model_path=superclass_model_28, n_epochs=30)

# train super-class model on 80 pixels
# CNN.recog.train_superclass_classifier_deep(dataset_path=superclass_dataset_80, model_path=superclass_model_80, n_epochs=30)

# test super-class classifier model
# CNN.recog.classify_superclass_from_database(model_path=superclass_model_28, img_dim=img_dim_28)

# endregion

# region Detection (Model 28)

# gtsrb_model_28 = 'D://_Dataset//GTSRB//cnn_model_28.pkl'
# gtsdb_dataset_28 = 'D://_Dataset//GTSDB//gtsdb_prohibitory_organized_28.pkl'
# gtsdb_model_28 = 'D://_Dataset//GTSDB//cnn_model_28.pkl'

# extract region images to train the detector
# CNN.utils.serialize_gtsdb()
# CNN.utils.organize_gtsdb()

# train the detector
# CNN.detec.train_shallow(dataset_path=gtsdb_dataset_28, recognition_model_path=gtsrb_model_28, detection_model_path=gtsdb_model_28, batch_size=500, learning_rate=0.01, n_epochs=1) # 10.01%
# CNN.detec.train_shallow(dataset_path=gtsdb_dataset_28, recognition_model_path=gtsrb_model_28, detection_model_path=gtsdb_model_28, batch_size=10, learning_rate=0.005, n_epochs=1) # 7.48%
# CNN.detec.train_shallow(dataset_path=gtsdb_dataset_28, recognition_model_path=gtsrb_model_28, detection_model_path=gtsdb_model_28, batch_size=10, learning_rate=0.1, n_epochs=1) # 29.5 %
# CNN.detec.train_shallow(dataset_path=gtsdb_dataset_28, recognition_model_path=gtsrb_model_28, batch_size=10, learning_rate=0.1, n_epochs=1)

# train the detector from scratch
# CNN.detec.train_from_scatch(dataset_path=gtsdb_dataset_28, detection_model_path=gtsdb_model_28, batch_size=10, learning_rate=0.1, n_epochs=1) # 29.5 %
# CNN.detec.train_from_scatch(dataset_path=gtsdb_dataset_28, detection_model_path=gtsdb_model_28, batch_size=10, learning_rate=0.1, n_epochs=1,
#                            nkerns=(40, 40 * 9), mlp_layers=(800, 29), kernel_dim = (5, 5))  # 29.5 %

# test the detector
# CNN.detec.detect_img_from_file(img_path="D://_Dataset//GTSDB//Test_PNG//00025.png", model_path=gtsdb_model_28)
# CNN.detec.detect_img_from_file(img_path="D://_Dataset//GTSDB//Test_PNG//_img2.png", model_path=gtsdb_model_28)


# endregion

# region Detection (Prohibitory 80)

# img_dim_80 = 80
# gtsrb_model_80 = 'D:\\_Dataset\\GTSRB\\cnn_model_80.pkl'
# gtsdb_model_80 = 'D:\\_Dataset\\GTSDB\\las_model_p_80.pkl'
# gtsdb_model_bin_80 = 'D:\\_Dataset\\GTSDB\\las_model_p_80_binary.pkl'
# gtsdb_dataset_80 = 'D:\\_Dataset\\GTSDB\\gtsdb_organized_p_80.pkl'
# gtsdb_dataset_conv_80 = 'D:\\_Dataset\\GTSDB\\gtsdb_convolved_p_80.pkl'
# gtsdb_dataset_conv_bin_80 = 'D:\\_Dataset\\GTSDB\\gtsdb_convolved_p_80_binary.pkl'

# extract region images to train the detector
# CNN.utils.serialize_gtsdb(img_dim_80, CNN.enums.SuperclassType._01_Prohibitory, True, True)
# CNN.utils.organize_gtsdb(img_dim_80, CNN.enums.SuperclassType._01_Prohibitory)
# CNN.utils.convolve_gtsdb(gtsrb_model_80, CNN.enums.SuperclassType._01_Prohibitory)
# CNN.utils.change_target_to_binary(img_dim_80, CNN.enums.SuperclassType._01_Prohibitory)

# detection proposals
# CNN.prop.detection_proposal_and_save(img_path="D://_Dataset//GTSDB//Test_PNG//00028.png", min_dim=16, max_dim=160)

# train the detector (detector will convolve the images each epoch)
# CNN.detec.train_deep(dataset_path=gtsdb_dataset_80, recognition_model_path=gtsrb_model_80, detection_model_path=gtsdb_model_80,
#                     mlp_layers=(7200, 4), batch_size=500, n_epochs=20, learning_rate=0.01, momentum=0.9)


# train only the regressor (images already convolved/filtered)
# CNN.detec.train_binary_detector(dataset_path=gtsdb_dataset_conv_bin_80, detection_model_path=gtsdb_model_bin_80, n_epochs=100)

# test the detector
# CNN.detec.detect_from_dataset(dataset_path=gtsdb_dataset_80, recognition_model_path=gtsrb_model_80, detection_model_path=gtsdb_model_80)

# test the detector
# CNN.detec.binary_detect_from_file_fast(img_path="D://_Dataset//GTSDB//Test_PNG//_img16.png", model_type=CNN.enums.ModelType._02_conv3_mlp2,
#                                  recognition_model_path=gtsrb_model_80, detection_model_path=gtsdb_model_bin_80, img_dim=img_dim_80)

# img = "D://_Dataset//GTSDB//Test_PNG//00061.png"
# CNN.prop.detection_proposal_and_save(img, 10, 160)

# endregion

# region Detection (Mandatory 80)

# img_dim_80 = 80
#
# gtsrb_model_80 = 'D:\\_Dataset\\GTSRB\\cnn_model_m_80.pkl'
# gtsdb_model_bin_80 = 'D:\\_Dataset\\GTSDB\\las_model_m_80_binary.pkl'
# gtsdb_model_bin_from_scratch_80 = 'D:\\_Dataset\\GTSDB\\las_model_m_80_binary_scratch.pkl'
#
# gtsdb_dataset_80 = 'D:\\_Dataset\\GTSDB\\gtsdb_organized_m_80.pkl'
# gtsdb_dataset_bin_80 = 'D:\\_Dataset\\GTSDB\\gtsdb_organized_m_80_binary.pkl'
# gtsdb_dataset_conv_bin_80 = 'D:\\_Dataset\\GTSDB\\gtsdb_convolved_m_80_binary.pkl'

# extract region images to train the detector
# CNN.utils.serialize_gtsdb(img_dim_80, CNN.enums.SuperclassType._03_Mandatory, True, True)
# CNN.utils.organize_gtsdb(img_dim_80, CNN.enums.SuperclassType._03_Mandatory)
# CNN.utils.convolve_gtsdb(gtsrb_model_80, CNN.enums.SuperclassType._03_Mandatory)
# CNN.utils.change_target_to_binary(img_dim_80, CNN.enums.SuperclassType._03_Mandatory, True)
# CNN.utils.change_target_to_binary(img_dim_80, CNN.enums.SuperclassType._03_Mandatory, False)
# CNN.utils.check_database_detector(img_dim_80, CNN.enums.SuperclassType._03_Mandatory)

# train binary detector from scratch
# CNN.detec.train_from_scatch_binary_detector(dataset_path=gtsdb_dataset_bin_80, model_path='gtsdb_model_bin_from_scratch_80')

# train only binary detector (images already convolved/filtered)
# CNN.detec.train_binary_detector(dataset_path=gtsdb_dataset_conv_bin_80, detection_model_path=gtsdb_model_bin_80, n_epochs=5)

# test the detector
# CNN.detec.binary_detect_from_file_fast(img_path="D://_Dataset//GTSDB//Test_PNG//_img16.png",
#                                       recognition_model_path=gtsrb_model_80, detection_model_path=gtsdb_model_bin_80, img_dim=img_dim_80)

# img = "D://_Dataset//GTSDB//Test_PNG//00061.png"
# CNN.prop.detection_proposal_and_save(img, 10, 160)

# endregion

# region Detection (Warning 80)

# img_dim_80 = 80
# gtsrb_model_80 = 'D:\\_Dataset\\GTSRB\\cnn_model_w_80.pkl'
# gtsdb_model_bin_80 = 'D:\\_Dataset\\GTSDB\\las_model_w_80_binary.pkl'
# gtsdb_dataset_80 = 'D:\\_Dataset\\GTSDB\\gtsdb_organized_w_80.pkl'
# gtsdb_dataset_conv_bin_80 = 'D:\\_Dataset\\GTSDB\\gtsdb_convolved_w_80_binary.pkl'

# extract region images to train the detector
# CNN.utils.serialize_gtsdb(img_dim_80, CNN.enums.SuperclassType._02_Warning, True, True)
# CNN.utils.organize_gtsdb(img_dim_80, CNN.enums.SuperclassType._02_Warning)
# CNN.utils.convolve_gtsdb(gtsrb_model_80, CNN.enums.SuperclassType._02_Warning)
# CNN.utils.change_target_to_binary(img_dim_80, CNN.enums.SuperclassType._02_Warning)

# train only the regressor (images already convolved/filtered)
# CNN.detec.train_binary_detector(dataset_path=gtsdb_dataset_conv_bin_80, detection_model_path=gtsdb_model_bin_80, n_epochs=100)

# test the detector
# CNN.detec.binary_detect_from_file_fast(img_path="D://_Dataset//GTSDB//Test_PNG//_img16.png",
#                                       recognition_model_path=gtsrb_model_80, detection_model_path=gtsdb_model_bin_80, img_dim=img_dim_80)

# img = "D://_Dataset//GTSDB//Test_PNG//00061.png"
# CNN.prop.detection_proposal_and_save(img, 10, 160)

# endregion

# region Google Street View

street_view = CNN.stview.StreetViewSpan(True)

# for i in range(100, 200):
#     img_id = "{0:05d}".format(i)
#     print("image: %s" % (img_id))
#     img_path = "D:\\_Dataset\\GTSDB\\Test_PNG\\%s.png" % (img_id)
#     street_view.process_image_and_save(img_path, i)

# endregion
