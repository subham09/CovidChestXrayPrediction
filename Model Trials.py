import pandas as pd
import yaml
import os
import datetime

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
import dill
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from math import ceil
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy, Precision, Recall, AUC
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.optimizers import Adam, SGD
#from models.models import *
#from visualization.visualize import *
#from custom.metrics import F1Score
#from data.preprocess import remove_text


from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, MaxPool2D, Conv2D, Flatten, LeakyReLU, BatchNormalization, \
    Activation, concatenate, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import Constant
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2
from tensorflow.keras.utils import multi_gpu_model




train_img_gen = ImageDataGenerator(rotation_range=10, preprocessing_function=None,
                            samplewise_std_normalization=True, samplewise_center=True)
val_img_gen = ImageDataGenerator(preprocessing_function=None,
                            samplewise_std_normalization=True, samplewise_center=True)
test_img_gen = ImageDataGenerator(preprocessing_function=None,
                            samplewise_std_normalization=True, samplewise_center=True)

print(train_img_gen)

img_shape = tuple([224,224])
y_col = 'label_str'
class_mode = 'categorical'
    
data = {}

data['TRAIN'] = pd.read_csv('train_set.csv')
data['VAL'] = pd.read_csv('val_set.csv')
data['TEST'] = pd.read_csv('test_set.csv')

#data['TRAIN'] = random_minority_oversample(data['TRAIN']) #new line added
train_set = data['TRAIN']

X_train = train_set[[x for x in train_set.columns if x != 'label']].to_numpy()
if X_train.shape[1] == 1:
    X_train = np.expand_dims(X_train, axis=-1)
Y_train = train_set['label'].to_numpy()
sampler = RandomOverSampler(random_state=np.random.randint(0, high=1000))
X_resampled, Y_resampled = sampler.fit_resample(X_train, Y_train)
filenames = X_resampled[:, 1]     # Filename is in second column
label_strs = X_resampled[:, 2]    # Class name is in second column
print("Train set shape before oversampling: ", X_train.shape, " Train set shape after resampling: ", X_resampled.shape)
train_set_resampled = pd.DataFrame({'filename': filenames, 'label': Y_resampled, 'label_str': label_strs})
    
data['TRAIN'] = train_set_resampled


train_generator = train_img_gen.flow_from_dataframe(dataframe=data['TRAIN'],
        directory='D:\\notes\\covid-cxr-master\\RAW_DATA\\',
        x_col="filename", y_col=y_col, target_size=img_shape,
        batch_size=32,
        class_mode=class_mode, validate_filenames=False)

val_generator = val_img_gen.flow_from_dataframe(dataframe=data['VAL'],
        directory='D:\\notes\\covid-cxr-master\\RAW_DATA\\',
        x_col="filename", y_col=y_col, target_size=img_shape,
        batch_size=32,
        class_mode=class_mode, validate_filenames=False)

test_generator = test_img_gen.flow_from_dataframe(dataframe=data['TEST'],
        directory='D:\\notes\\covid-cxr-master\\RAW_DATA\\',
        x_col="filename", y_col=y_col, target_size=img_shape,
        batch_size=32,
        class_mode=class_mode, validate_filenames=False, shuffle=False)

#dill.dump(test_generator.class_indices, open('output_class_indices.pkl','wb'))

histogram = np.bincount(np.array(train_generator.labels).astype(int))
class_weight = None
class_multiplier = [0.15, 1.0]
class_multiplier = [class_multiplier[['non-COVID-19', 'COVID-19'].index(c)] for c in test_generator.class_indices]

weights = [None] * len(histogram)
for i in range(len(histogram)):
    weights[i] = (1.0 / len(histogram)) * sum(histogram) / histogram[i]
class_weight = {i: weights[i] for i in range(len(histogram))}
if class_multiplier is not None:
    class_weight = [class_weight[i] * class_multiplier[i] for i in range(len(histogram))]
print("Class weights: ", class_weight)


covid_class_idx = test_generator.class_indices['COVID-19'] 
thresholds = 1.0 / len(['non-COVID-19', 'COVID-19'])    
metrics = ['accuracy', CategoricalAccuracy(name='accuracy'),
            Precision(name='precision', thresholds=thresholds, class_id=covid_class_idx),
            Recall(name='recall', thresholds=thresholds, class_id=covid_class_idx),
            AUC(name='auc')
            ]

input_shape = [224,224] + [3]
num_gpus = 1
gpus = 1


#model start
nodes_dense0 = 128
lr = 0.00001
dropout = 0.4
l2_lambda = 0.0001
optimizer = Adam(learning_rate=lr)
init_filters = 16
filter_exp_base = 3
conv_blocks = 3
kernel_size = (3,3)
max_pool_size = (2,2)
strides = (1,1)



X_input = Input(input_shape)
X = X_input

for i in range(conv_blocks):
    X_res = X
    X = Conv2D(init_filters * (filter_exp_base ** i), kernel_size,
            strides=strides, padding='same',
            kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda),
            name='conv' + str(i) + '_0')(X)
    X = BatchNormalization()(X)
    X = LeakyReLU()(X)
    X = Conv2D(init_filters * (filter_exp_base ** i), kernel_size,
            strides=strides, padding='same',
            kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda),
            name='conv' + str(i) + '_1')(X)
    X = concatenate([X, X_res], name='concat' + str(i))
    X = BatchNormalization()(X)
    X = LeakyReLU()(X)
    X = MaxPool2D(max_pool_size, padding='same')(X)

X = Flatten()(X)
X = Dropout(dropout)(X)
X = Dense(nodes_dense0, kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda))(X)
X = LeakyReLU()(X)
X = Dense(2)(X)
Y = Activation('softmax', dtype='float32', name='output')(X)


model = Model(inputs=X_input, outputs=Y)
model.summary()
if gpus >= 2:
    model = multi_gpu_model(model, gpus=gpus)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)




steps_per_epoch = ceil(train_generator.n / train_generator.batch_size)
val_steps = ceil(val_generator.n / val_generator.batch_size)




history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
            epochs=25,
            validation_data=val_generator,
            validation_steps=val_steps,
            verbose=1)



test_results = model.evaluate_generator(test_generator, verbose=1)
test_metrics = {}
test_summary_str = [['**Metric**', '**Value**']]
for metric, value in zip(model.metrics_names, test_results):
    test_metrics[metric] = value
    print(metric, ' = ', value)
    test_summary_str.append([metric, str(value)])


model.save('my_model25.h5')
