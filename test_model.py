import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from sklearn.preprocessing import LabelEncoder
import cv2

train = pd.read_csv('final_my_images_test.csv')

train_image = []
count=0
for i in range(train.shape[0]):#changing loop size to 10, train.shape[0]
    img = cv2.imread("D:\\notes\\model\\images\\"+train['imageid'][i])
    count+=1
    print(count)
    #print(img)
    #img = img.resize((224,224))
    img = np.asarray(img)
    img = np.ndarray(shape = [224,224,3])
    
    #img = img/255
    print(img.dtype)
    
    train_image.append(img)
X = np.array(train_image)

y = train['newlabels'].values
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
y = to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(13, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save('small_model.h5')
