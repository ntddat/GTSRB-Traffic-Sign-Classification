import numpy as np
import cv2 as cv
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

metadataDF = pd.read_csv("./test/test_metadata.csv")
print(metadataDF)
temp = list(metadataDF[metadataDF["image_path"] == "img_000001.jpg"]["ClassId"])
print(temp)
metadataDF = metadataDF.drop(columns=["ClassId"])

imgNames = []
imgs = []
labels = []

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

print(imgs[0].shape)

imgs = np.array(imgs) / 255.0

model = Sequential([
  Conv2D(25, (3, 3), activation='relu', input_shape=(25,25,3)),
  MaxPooling2D(2, 2),
  Conv2D(50, (3, 3), activation='relu'),
  MaxPooling2D(2, 2),
  Flatten(),
  Dense(100, activation='relu'),
  Dropout(0.5),
  Dense(43, activation='softmax'),
])

# uncomment corresponding lines to see different results of epochs
# model.load_weights('./cnn/cnnMoreLayersDD.weights.h5')
# model.load_weights('./cnn/cnnMoreLayersDD2.weights.h5')
# model.load_weights('./cnn/cnnMoreLayersDD3.weights.h5')
model.load_weights('./cnn/cnnMoreLayersDD4.weights.h5')
# model.load_weights('./cnn/cnnMoreLayersDD5.weights.h5')


yPred = model.predict(imgs)
print(yPred)
print(np.argmax(yPred, axis=1)) 

predicted = {"image_path": imgNames, "ClassId": np.argmax(yPred, axis=1)}
predictedDF = pd.DataFrame(predicted)
finalDF = pd.concat([metadataDF.set_index('image_path'), predictedDF.set_index('image_path')], axis=1, join='inner').reset_index()
finalDF = finalDF.drop(columns=["image_path"])
print(finalDF)

# finalDF.to_csv("./cnn/cnnMoreLayers10epochs.csv", index=False)
# finalDF.to_csv("./cnn/cnnMoreLayers20epochs.csv", index=False)
finalDF.to_csv("./cnn/cnnMoreLayersDD30epochs.csv", index=False)
