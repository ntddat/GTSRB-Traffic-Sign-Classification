import numpy as np
import cv2 as cv
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# REFER TO README FOR DESCRIPTION OF ALL FILES

metadataDF = pd.read_csv("./test/test_metadata.csv")
print(metadataDF)
metadataDF = metadataDF.drop(columns=["ClassId"])

imgNames = []
imgs = []
labels = []

# Reading in image files
for i in range(5489, 7842):
    imgNum = str(i)
    imgNumLen = len(imgNum)
    for j in range(6 - imgNumLen):
        imgNum = "0" + imgNum
    # print(imgNum)
    imgName = "img_" + imgNum + ".jpg"
    imgNames.append(imgName)
    img = cv.imread("./test/" + imgName)

    resized = cv.resize(img, (25, 25))
    imgs.append(resized)

# print(labels)
print(imgs[0].shape)

imgs = np.array(imgs) / 255.0

# Model
model = Sequential([
  Conv2D(25, (3, 3), activation='relu', input_shape=(25,25,3)),
  MaxPooling2D(2, 2),
  Flatten(),
  Dense(43, activation='softmax'),
])

# Loading in generated weights
# model.load_weights('./cnn/cnn.weights.h5')
model.load_weights('./cnn/cnn2.weights.h5')

yPred = model.predict(imgs)
print(yPred)
print(np.argmax(yPred, axis=1)) 

# Saving results to csv file
predicted = {"image_path": imgNames, "ClassId": np.argmax(yPred, axis=1)}
predictedDF = pd.DataFrame(predicted)
finalDF = pd.concat([metadataDF.set_index('image_path'), predictedDF.set_index('image_path')], axis=1, join='inner').reset_index()
finalDF = finalDF.drop(columns=["image_path"])
print(finalDF)

# finalDF.to_csv("./cnn/cnn10epochs.csv", index=False)
finalDF.to_csv("./cnn/cnn20epochs.csv", index=False)
