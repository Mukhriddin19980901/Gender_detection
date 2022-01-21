#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import cv2
import cvlib as cv
import glob
import os
tf.config.list_physical_devices()


# In[23]:


# step 1
def dataset(path):
    data = []
    x_train = []
    y_train = []
    for  i in range(2):
        labels = ['female','male']
        path_new = path + labels[i] + '\\*'
        for image in glob.glob(path_new):
            image = cv2.imread(image)
            with open('size.txt','a+') as f:
                f.write(f"|{image.shape}|")
            image = cv2.resize(image,(96,96))
            x_train.append(image)
            if i==0:
                y_train.append([1,0])
            else:
                y_train.append([0,1])
    return x_train,y_train
train_dir = r'C:\\Users\\USER\\datasets2021\\Genders\\train\\'
test_dir = r'C:\\Users\\USER\\datasets2021\\Genders\\test\\'

#step 2
x_train = []
y_train = []
x_train,y_train= dataset(train_dir)
x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
x_test = []
y_test = []
x_test,y_test= dataset(test_dir)
x_test = np.array(x_test)
y_test = np.array(y_test)
print(x_test.shape)


# In[24]:


y_train.shape


# In[25]:


model = keras.Sequential([
     keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(96,96,3)),
     keras.layers.MaxPooling2D((2,2)),
     keras.layers.Conv2D(64,(3,3),activation='relu'),
     keras.layers.MaxPooling2D((2,2)),
     keras.layers.Conv2D(128,(3,3),activation='relu'),
     keras.layers.MaxPooling2D((2,2)),
     keras.layers.Flatten(),
     keras.layers.Dense(64,activation='relu'),
     keras.layers.Dense(32,activation='relu'),
     keras.layers.Dense(16,activation='relu'),
     keras.layers.Dense(2,activation='sigmoid'),
    ])
tf.keras.layers.BatchNormalization(axis=-1,momentum=0.9)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) 
with tf.device('/GPU:0'):
    model_gpu = model
    model_gpu.fit(x_train,y_train,epochs=5)


# In[26]:


with tf.device('/GPU:0'):
    model_gpu.evaluate(x_test,y_test)
    model_gpu.summary()


# In[27]:


# saving model
model_gpu.save('gender_gpu1.model',save_format='h5')


# In[2]:


# importing model to use
model = keras.models.load_model('gender_gpu1.model')


# In[14]:


# check through video using OpenCV
video = cv2.VideoCapture(r"videos/teazel1.mp4")
genders = ['ayol','erkak']
e=0; a=0
import time
start = time.time()
while video.isOpened():
    _,image = video.read()
    face,confidence = cv.detect_face(image)
    for (x,y,w,h) in face:
        yuz_np = np.copy(image[y:h,x:w])
        if yuz_np.shape[0]<10 or yuz_np.shape[1]<10:
            continue
        yuz_np = cv2.resize(yuz_np,(96,96))
        yuz_np = np.expand_dims(yuz_np,0)
        bashorat = model.predict(yuz_np)
        index = np.argmax(bashorat)
        gender = genders[index]
        
        if gender=="erkak":
            color = (0,255,0)
            e+=1
        else:
            color = (0,0,255)
            a+=1
        gender = f'{gender} :  {np.around(bashorat[0][index]*100,2)} %'
        if y-10>10:
            Y=y-10
        else:
            Y=y+10        
        cv2.rectangle(image,(x,y),(w,h),color,2)
        cv2.putText(image,gender,(x,Y),cv2.FONT_HERSHEY_COMPLEX,0.7,color,2)
    cv2.imshow("gender detection media",image)
    end = time.time()
    if end-start >= 5:
        start=time.time()
        with open('size2.txt','a+') as f:
            f.write(f"|{e} erkak | {a} ayol\n")
        e=0; a=0
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video.release()
cv2.destroyAllWindows()

