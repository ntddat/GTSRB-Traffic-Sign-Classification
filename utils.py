import cv2 as cv
import imutils
import pytesseract
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# drawing contours onto images to see 
# contour example in report can be found here
# press 0 to move to next image
for i in range(1, 5489):
    if i == 30:
        break
    imgNum = str(i)
    imgNumLen = len(imgNum)
    for j in range(6 - imgNumLen):
        imgNum = "0" + imgNum
    print(imgNum)
    imgName = "img_" + imgNum + ".jpg"
    img = cv.imread("./train/" + imgName)

    # image preprocessing
    resized = imutils.resize(img, width=25)
    ratio = img.shape[0] / float(resized.shape[0])
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(gray, 60, 255, cv.THRESH_OTSU)[1]

    # find contours in the thresholded image and initialize the
    # shape detector
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) == 0:
        cv.imshow("no contours", img)
        k = cv.waitKey(0)
        continue
    maxContour = max(contours, key=cv.contourArea)
    cv.drawContours(resized, contours, -1, (0, 255, 0), 2)
    cv.imshow("yes contour", resized)
    k = cv.waitKey(0)

metadataDF = pd.read_csv("./train/train_metadata.csv")

counts = metadataDF["ClassId"].value_counts()
# counts = counts.to_frame()
print(counts)

plt.bar(counts.index, counts.values)
plt.title('Class labels count')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig("classCount.png", format="png")
plt.clf()
