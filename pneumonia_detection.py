import pandas as pd
import numpy as np
import os
import cv2
from skimage.io import imread,imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import keras as K
import tensorflow as tf
from PIL import Image, ImageOps
from keras.preprocessing.image import ImageDataGenerator
tf.random.set_seed(7) # fixes some parameters to start with.

df=pd.read_csv('../input/chest-xrays-bacterial-viral-pneumonia-normal/labels_train.csv')
df.head()

names=list(df['file_name'])
y_train=list(df['class_id'])

y_train=np.asarray(y_train)

x_train=np.zeros((len(names),256,256,3),dtype=np.float32)

for i in range(len(names)):
    img=imread('../input/chest-xrays-bacterial-viral-pneumonia-normal/train_images/train_images/'+str(names[i]))
    img=resize(img,(256,256,3),mode='constant',preserve_range=True)
    x_train[i]=img
    if i%1000==0:
        print('Done')

x_train=x_train/255

x_train.shape

y_train=tf.keras.utils.to_categorical(y_train, num_classes=None, dtype='float32')

y_train

model = tf.keras.applications.InceptionResNetV2(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=[256,256,3],
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
)

x2=tf.keras.layers.Flatten()(model.output)
x = tf.keras.layers.Dense(3, activation="softmax", name="dense_final")(x2)
model = tf.keras.Model(inputs=model.input, outputs=x)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

history=model.fit(x_train,y_train,batch_size=8,epochs=20,shuffle=True)

xx=model.evaluate(x_train,y_train,verbose=1)

