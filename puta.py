import pandas as pd
import yaml
import sys
import os
import datetime
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random
import dill
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy, Precision, Recall, AUC
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, MaxPool2D, Conv2D, Flatten, LeakyReLU, BatchNormalization, \
    Activation, concatenate, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import Constant
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2
from tensorflow.keras.utils import multi_gpu_model
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

model = load_model('small_model.h5')

##test_img_gen = ImageDataGenerator(preprocessing_function=None,
##                samplewise_std_normalization=True, samplewise_center=True)

##data = {}
##img_shape = tuple([224,224])
##data['TEST'] = pd.read_csv('test.csv')

##test_generator = test_img_gen.flow_from_dataframe(dataframe=data['TEST'],
##        directory='C:\\Users\\invicto\\Desktop\\',
##        x_col="filename", y_col = 'label_str', target_size=img_shape,
##        batch_size=32,
##        class_mode='binary', validate_filenames=False, shuffle=False)
#print(test_generator)
lol = sys.argv[1]
#print(lol)
img = cv2.imread(lol)
img = cv2.resize(img,(224,224))
img = np.expand_dims(img, axis=0)
img = tf.cast(img, tf.float32)
#img = np.reshape(img,[1,320,240,3])


classes = model.predict_classes(img)
encoder = LabelEncoder()
train = pd.read_csv('small_train.csv')
y = train['labels'].values
encoder.fit(y)
label_name = encoder.inverse_transform(classes)
print(label_name)
