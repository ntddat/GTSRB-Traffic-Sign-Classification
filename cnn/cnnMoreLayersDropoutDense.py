import numpy as np
import cv2 as cv
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from sklearn import metrics

metadataDF = pd.read_csv("./train/train_metadata.csv")
temp = list(metadataDF[metadataDF["image_path"] == "img_000001.jpg"]["ClassId"])
print(temp)

imgs = []
labels = []

for i in range(1, 5489):
    imgNum = str(i)
    imgNumLen = len(imgNum)
    for j in range(6 - imgNumLen):
        imgNum = "0" + imgNum
    # print(imgNum)
    imgName = "img_" + imgNum + ".jpg"
    # imgNames.append(imgName)
    img = cv.imread("./train/" + imgName)

    resized = cv.resize(img, (25, 25))
    # imgData = img_to_array(resized)
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

model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# change epochs number variable here to see different results
model.fit(
  XTrain,
  to_categorical(yTrain),
  epochs=35,
  validation_data=(XValid, to_categorical(yValid)),
)

# uncomment corresponding lines to see different results of epochs
# cnnMoreLayersDD.weights.h5 is generated from epochs=10
# model.save_weights('./cnn/cnnMoreLayersDD.weights.h5')
# cnnMoreLayersDD2.weights.h5 is generated from epochs=20
# model.save_weights('./cnn/cnnMoreLayersDD2.weights.h5')
# cnnMoreLayersDD3.weights.h5 is generated from epochs=25
# model.save_weights('./cnn/cnnMoreLayersDD3.weights.h5')
# cnnMoreLayersDD4.weights.h5 is generated from epochs=30
# model.save_weights('./cnn/cnnMoreLayersDD4.weights.h5')
# cnnMoreLayersDD5.weights.h5 is generated from epochs=35
model.save_weights('./cnn/cnnMoreLayersDD5.weights.h5')

# Not much accuracy increase beyond epoch ~28, 29
# Mostly fluctuates

yPred = model.predict(XValid)
print(yPred)
print(np.argmax(yPred, axis=1)) 
accuracy = metrics.accuracy_score(yValid, np.argmax(yPred, axis=1)) 
print("Accuracy: ", accuracy)
