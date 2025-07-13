import numpy as np
import cv2 as cv
import pandas as pd
import keras
import tensorflow as tf
import os
import random
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from sklearn import metrics

# REFER TO README FOR DESCRIPTION OF ALL FILES

# setting seeds to ensure reproducibility
# following the top answer on this post:
# https://stackoverflow.com/questions/50659482/why-cant-i-get-reproducible-results-in-keras-even-though-i-set-the-random-seeds?rq=2
# seed_value = 16
# os.environ['PYTHONHASHSEED']=str(seed_value)
# random.seed(seed_value)
# np.random.seed(seed_value)
# tf.random.set_seed(seed_value)
# however, unfortunately it might be difficult to ensure
# that the same results are observed in multiple environments.
# the results my local device might not be the same as the 
# marker's
# if this is the case, i'm honestly not sure how to fix it

# addendum: via post #440 on ed, cnn variability is expected
# since the global seeds were not working anyways, i decided
# to comment them out

metadataDF = pd.read_csv("./train/train_metadata.csv")
temp = list(metadataDF[metadataDF["image_path"] == "img_000001.jpg"]["ClassId"])
print(temp)

imgs = []
labels = []

# Reading in image files
for i in range(1, 5489):
    imgNum = str(i)
    imgNumLen = len(imgNum)
    for j in range(6 - imgNumLen):
        imgNum = "0" + imgNum
    imgName = "img_" + imgNum + ".jpg"
    img = cv.imread("./train/" + imgName)

    resized = cv.resize(img, (25, 25))
    imgs.append(resized)

    label = list(metadataDF[metadataDF["image_path"] == imgName]["ClassId"])
    labels.append(label[0])

print(labels)
print(imgs[0].shape)

imgs = np.array(imgs) / 255.0

XTrain, XValid, yTrain, yValid = train_test_split(imgs, labels, test_size=0.2, random_state=16)
# print(XTrain)
# print(yTrain)
# print(XValid)
# print(yValid)

# Model
model = Sequential([
  Conv2D(25, (3, 3), activation='relu', input_shape=(25,25,3)),
  MaxPooling2D(2, 2),
  Flatten(),
  Dense(43, activation='softmax'),
])

model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# change epochs number variable here to see different results
model.fit(
  XTrain,
  to_categorical(yTrain),
  epochs=25,
  validation_data=(XValid, to_categorical(yValid)),
)

# uncomment corresponding lines to see different results of epochs
# cnn.weights.h5 is generated from epochs=10
# model.save_weights('./cnn/cnn.weights.h5')
# cnn.weights.h5 is generated from epochs=20
# model.save_weights('./cnn/cnn2.weights.h5')
# cnn.weights.h5 is generated from epochs=25
model.save_weights('./cnn/cnn3.weights.h5')

# Not much accuracy increase beyond epoch ~15, 16

yPred = model.predict(XValid)
print(yPred)
print(np.argmax(yPred, axis=1)) 
accuracy = metrics.accuracy_score(yValid, np.argmax(yPred, axis=1)) 
print("Accuracy: ", accuracy)
