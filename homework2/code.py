'''
Based on Wide Residual Networks (Zagoruyko et. al.). My implementation has 16 convolution layers, 3 of which are skip connections.
The network can be divided into 5 group: conv1, conv2, conv3, conv4 and prediction. conv1 is simple convolution layer next to input. Each of conv2-4 group have two basic residual block (i.e. two consecutive convolution). The prediction group consists of max pool, flatten and dense layers.
Each convolution layer has batch normalization and relu activation preceeding it (including the consecutive convolution in residual block but excluding the first convolution).
Validation set accuracy: 81.3%
'''
from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

from six.moves import range
import os

import logging
logging.basicConfig(level=logging.DEBUG)

import sys
sys.stdout = sys.stderr
# Prevent reaching to maximum recursion depth in `theano.tensor.grad`
sys.setrecursionlimit(2 ** 20)

import numpy as np
np.random.seed(2 ** 10)

from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from keras import backend as K

import tensorflow as tf
#tf.python.control_flow_ops = tf

#from utils import mk_dir

import pickle
  
import cv2

# ================================================
# NETWORK/TRAINING CONFIGURATION:
logging.debug("Loading network/training configuration...")

dropout_probability = .3
weight_decay = 0.0005   
batch_size = 128        

# Regarding nb_epochs, lr_schedule and sgd, see bottom page 10:
nb_epochs = 200
lr_schedule = [60, 120, 160] # epoch_step
def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.1
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.02 # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.004
    return 0.0008
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

# Other config from code; throughtout all layer:
use_bias = False        # following functions 'FCinit(model)' and 'DisableBias(model)' in utils.lua
weight_init="he_normal" # follows the 'MSRinit(model)' function in utils.lua

# Use theano image_dim_ordering
logging.debug("image_dim_ordering = 'th'")
channel_axis = 1
input_shape = (3, 32, 32)

# ================================================
# DATA CONFIGURATION:
logging.debug("Loading data...")

def getDataset(feature_file, label_file):
  f = open(feature_file, "rb")
  feat = pickle.load(f)
  f.close()

  f = open(label_file, "rb")
  lab = pickle.load(f)
  f.close()

  return (feat, lab)

X_train, y_train =  getDataset("../Homework2_data/train_feat.pickle", "../Homework2_data/train_lab.pickle")
X_validation, y_validation = getDataset("../Homework2_data/validation_feat.pickle", "../Homework2_data/validation_lab.pickle")
X_test, y_test = getDataset("../Homework2_data/test_feat.pickle", "../Homework2_data/test_lab.pickle")

X_train = X_train.astype('float32')
X_validation = X_validation.astype('float32')
X_test = X_test.astype('float32')

X_train = np.transpose(X_train, [0, 3, 2, 1])
X_validation = np.transpose(X_validation, [0, 3, 2, 1])
X_test = np.transpose(X_test, [0, 3, 2, 1])


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_validation = np_utils.to_categorical(y_validation, 10)
Y_test = np_utils.to_categorical(y_test, 10) # not really required

# ================================================
# OUTPUT CONFIGURATION:
print_model_summary = True
save_model_and_weights = True
save_model_plot = False

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/')
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')

# ========================================================
# The architecture
# Wide residual network http://arxiv.org/abs/1605.07146

inputs = Input(shape=input_shape)
# "One conv at the beginning (spatial size: 32x32x16)"
model = Convolution2D(
                        nb_filter=16, nb_row=3, nb_col=3,
                        subsample=(1, 1),
                        border_mode="same",
                        init=weight_init,
                        W_regularizer=l2(weight_decay),
                        bias=use_bias
                    )(inputs) 

# -----------------------------------------------------
# GROUP CONV2 START
model = BatchNormalization(axis=channel_axis)(model)
model = Activation("relu")(model)
shortcut_start = model

# conv2[0][0] (spatial size 32x32x32)
model = Convolution2D(
                        nb_filter=32, nb_col=3, nb_row=3,
                        subsample=(1,1),
                        border_mode='same',
                        init=weight_init,
                        W_regularizer=l2(weight_decay),
                        bias=use_bias
                    )(model)

model = BatchNormalization(axis=channel_axis)(model)
model = Activation("relu")(model)

# conv2[0][1]
model = Convolution2D(
                        nb_filter=32, nb_col=3, nb_row=3,
                        subsample=(1,1),
                        border_mode='same',
                        init=weight_init,
                        W_regularizer=l2(weight_decay),
                        bias=use_bias
                    )(model)

# shortcut connection (a 1x1x32 convolution)
shortcut = Convolution2D(
                            32, nb_col=1, nb_row=1,
                            subsample=(1,1),
                            border_mode="same",
                            init=weight_init,
                            W_regularizer=l2(weight_decay),
                            bias=use_bias
                        )(shortcut_start)

# merge layer for conv2[0]
model = merge([model, shortcut], mode="sum")

# conv2[1] start
shortcut = model

model = BatchNormalization(axis=channel_axis)(model)
model = Activation("relu")(model)

# conv2[1][0] (spatial size 32x32x32)
model = Convolution2D(
                        nb_filter=32, nb_col=3, nb_row=3,
                        subsample=(1,1),
                        border_mode='same',
                        init=weight_init,
                        W_regularizer=l2(weight_decay),
                        bias=use_bias
                    )(model)

model = BatchNormalization(axis=channel_axis)(model)
model = Activation("relu")(model)

# conv2[1][1] (spatial size 32x32x32)
model = Convolution2D(
                        nb_filter=32, nb_col=3, nb_row=3,
                        subsample=(1,1),
                        border_mode='same',
                        init=weight_init,
                        W_regularizer=l2(weight_decay),
                        bias=use_bias
                    )(model)

# merge layer for conv2[1]
model = merge([model, shortcut], mode="sum")

# -----------------------------------------------------
# GROUP CONV3 START
model = BatchNormalization(axis=channel_axis)(model)
model = Activation("relu")(model)
shortcut_start = model

# conv3[0][0] (spatial size 16x16x64)
model = Convolution2D(
                        nb_filter=64, nb_col=3, nb_row=3,
                        subsample=(2,2),
                        border_mode='same',
                        init=weight_init,
                        W_regularizer=l2(weight_decay),
                        bias=use_bias
                    )(model)

model = BatchNormalization(axis=channel_axis)(model)
model = Activation("relu")(model)

# conv3[0][1]
model = Convolution2D(
                        nb_filter=64, nb_col=3, nb_row=3,
                        subsample=(1,1),
                        border_mode='same',
                        init=weight_init,
                        W_regularizer=l2(weight_decay),
                        bias=use_bias
                    )(model)

# shortcut connection (a 1x1x32 convolution)
shortcut = Convolution2D(
                            64, nb_col=1, nb_row=1,
                            subsample=(2,2),
                            border_mode="same",
                            init=weight_init,
                            W_regularizer=l2(weight_decay),
                            bias=use_bias
                        )(shortcut_start)

# merge layer for conv3[0]
model = merge([model, shortcut], mode="sum")

# conv3[1] start
shortcut = model

model = BatchNormalization(axis=channel_axis)(model)
model = Activation("relu")(model)

# conv3[1][0] (spatial size 16x16x64)
model = Convolution2D(
                        nb_filter=64, nb_col=3, nb_row=3,
                        subsample=(1,1),
                        border_mode='same',
                        init=weight_init,
                        W_regularizer=l2(weight_decay),
                        bias=use_bias
                    )(model)

model = BatchNormalization(axis=channel_axis)(model)
model = Activation("relu")(model)

# conv3[1][1] (spatial size 16x16x64)
model = Convolution2D(
                        nb_filter=64, nb_col=3, nb_row=3,
                        subsample=(1,1),
                        border_mode='same',
                        init=weight_init,
                        W_regularizer=l2(weight_decay),
                        bias=use_bias
                    )(model)

# merge layer for conv3[1]
model = merge([model, shortcut], mode="sum")

# -----------------------------------------------------
# GROUP CONV4 START
model = BatchNormalization(axis=channel_axis)(model)
model = Activation("relu")(model)
shortcut_start = model

# conv4[0][0] (spatial size 8x8x128)
model = Convolution2D(
                        nb_filter=128, nb_col=3, nb_row=3,
                        subsample=(2,2),
                        border_mode='same',
                        init=weight_init,
                        W_regularizer=l2(weight_decay),
                        bias=use_bias
                    )(model)

model = BatchNormalization(axis=channel_axis)(model)
model = Activation("relu")(model)

# conv4[0][1]
model = Convolution2D(
                        nb_filter=128, nb_col=3, nb_row=3,
                        subsample=(1,1),
                        border_mode='same',
                        init=weight_init,
                        W_regularizer=l2(weight_decay),
                        bias=use_bias
                    )(model)

# shortcut connection (a 1x1x128 convolution)
shortcut = Convolution2D(
                            128, nb_col=1, nb_row=1,
                            subsample=(2,2),
                            border_mode="same",
                            init=weight_init,
                            W_regularizer=l2(weight_decay),
                            bias=use_bias
                        )(shortcut_start)

# merge layer for conv4[0]
model = merge([model, shortcut], mode="sum")

# conv4[1] start
shortcut = model

model = BatchNormalization(axis=channel_axis)(model)
model = Activation("relu")(model)

# conv4[1][0] (spatial size 8x8x128)
model = Convolution2D(
                        nb_filter=128, nb_col=3, nb_row=3,
                        subsample=(1,1),
                        border_mode='same',
                        init=weight_init,
                        W_regularizer=l2(weight_decay),
                        bias=use_bias
                    )(model)

model = BatchNormalization(axis=channel_axis)(model)
model = Activation("relu")(model)

# conv4[1][1] (spatial size 16x16x64)
model = Convolution2D(
                        nb_filter=128, nb_col=3, nb_row=3,
                        subsample=(1,1),
                        border_mode='same',
                        init=weight_init,
                        W_regularizer=l2(weight_decay),
                        bias=use_bias
                    )(model)

# merge layer for conv4[1]
model = merge([model, shortcut], mode="sum")

model = BatchNormalization(axis=channel_axis)(model)
model = Activation("relu")(model)

# Classifier block
model = AveragePooling2D(
                            pool_size=(8, 8), 
                            strides=(1, 1), 
                            border_mode="same"
                        )(model)

model = Flatten()(model)
model = Dense(
                output_dim=10, 
                init=weight_init, 
                bias=use_bias,
                W_regularizer=l2(weight_decay), 
                activation="softmax"
            )(model)

model = Model(input=inputs, output=model)

# Some output
model.summary()
plot(model, to_file="architecture.png", show_shapes=True)

if __name__ == '__main__':
    # output predicted labels in separate folder for easy viewing
    for i in range(10):
        os.system("mkdir -p predicted_images/" + str(i))

    # select which data you want to evaluate on (validation or testing)
    X_evaluation = X_test
    Y_evaluation = Y_test
    y_evaluation = y_test

    with tf.device('/cpu:0'):
        model.load_weights( os.path.join(MODEL_PATH, 'WRN-16-2-own-81accuracy.h5') )
        model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])
        
        validation_datagen = ImageDataGenerator(
                                                    featurewise_center=True,
                                                    featurewise_std_normalization=True,
                                                    zca_whitening=True
                                                )
        validation_datagen.fit(X_train)
        generator = validation_datagen.flow(X_evaluation, Y_evaluation, batch_size=1, shuffle=False);
        
        total_correct = 0
        for sample_idx in range(X_evaluation.shape[0]):
            (X, y) = generator.next()
            preds = model.predict(X)
            predicted_label = np.argmax(preds)
            actual_label = y_evaluation[sample_idx] # np.argmax(y) # 

            # output the predicted label/image
            y_evaluation[sample_idx] = predicted_label
            cv2.imwrite(
                            "predicted_images/" + str(predicted_label) + "/" + str(sample_idx) + ".png",
                            np.transpose((X_evaluation[sample_idx]+1.)/2.*255., [1, 2, 0])
                       )

            # if predicted_label == actual_label:
            #     total_correct += 1
                  
            # print('Status: ', predicted_label == actual_label, \
            #       'Predicted: ', predicted_label, 'Actual: ', actual_label)

        # print('Accuracy: ', float(total_correct) / X_evaluation.shape[0])

        f = open('test_lab.pickle', 'wb')
        pickle.dump(y_evaluation, f)
        f.close()

