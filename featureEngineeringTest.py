import cv2 as cv
import imutils
import pytesseract
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

imgNames = []
aspectRatios = []
extents = []
solidities = []
equiDiameters = []
orientations = []
allPixelpoints = []
minVals = []
maxVals = []
minLocs = []
maxLocs = []
meanVals = []
leftmosts = []
rightmosts = []
topmosts = []
bottommosts = []
hu1 = []
hu2 = []
hu3 = []
hu4 = []
hu5 = []
hu6 = []
hu7 = []


# Code for OpenCV shape/contour detection adapted from this guide
# https://pyimagesearch.com/2016/02/08/opencv-shape-detection/
for i in range(5489, 7842):
    """
    if i == 30:
        break
    """
    imgNum = str(i)
    imgNumLen = len(imgNum)
    for j in range(6 - imgNumLen):
        imgNum = "0" + imgNum
    print(imgNum)
    imgName = "img_" + imgNum + ".jpg"
    imgNames.append(imgName)
    img = cv.imread("./test/" + imgName)

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

    # Hu Moments Guide:
    # https://learnopencv.com/shape-matching-using-hu-moments-c-python/
    huGray = cv.imread("./test/" + imgName, cv.IMREAD_GRAYSCALE)
    _, huFinal = cv.threshold(huGray, 128, 255, cv.THRESH_BINARY)
    moments = cv.moments(huFinal)
    huMoments = cv.HuMoments(moments)
    print(huMoments)
    hu1.append(huMoments[0][0])
    hu2.append(huMoments[1][0])
    hu3.append(huMoments[2][0])
    hu4.append(huMoments[3][0])
    hu5.append(huMoments[4][0])
    hu6.append(huMoments[5][0])
    hu7.append(huMoments[6][0])

    if len(contours) == 0:
        """
        cv.imshow("no contours", img)
        k = cv.waitKey(0)
        """
        aspectRatios.append(0)
        extents.append(0)
        solidities.append(0)
        equiDiameters.append(0)
        orientations.append(0)
        leftmosts.append(0)
        rightmosts.append(0)
        topmosts.append(0)
        bottommosts.append(0)
        continue
    maxContour = max(contours, key=cv.contourArea)
    """
    cv.drawContours(resized, contours, -1, (0, 255, 0), 2)
    cv.imshow("yes contour", resized)
    k = cv.waitKey(0)
    """

    # Features extracted from contours
    # Contours might not be very accurate however
    # Ideas for features from this OpenCV page: 
    # https://docs.opencv.org/3.4/d1/d32/tutorial_py_contour_properties.html
    area = cv.contourArea(maxContour)
    x,y,w,h = cv.boundingRect(maxContour)

    # Aspect Ratio
    aspectRatio = float(w)/h
    print("Aspect ratio: " + str(aspectRatio))
    aspectRatios.append(aspectRatio)

    # Extent
    rectArea = w*h
    extent = 0
    if rectArea != 0:
        extent = float(area)/rectArea
    print("Extent: " + str(extent))
    extents.append(extent)

    # Solidity
    hull = cv.convexHull(maxContour)
    hullArea = cv.contourArea(hull)
    solidity = 0
    if hullArea != 0:
        solidity = float(area)/hullArea
    print("Solidity: " + str(solidity))
    solidities.append(solidity)

    # Equivalent Diameter
    equiDiameter = np.sqrt(4*area/np.pi)
    print("Equi Diameters: " + str(equiDiameter))
    equiDiameters.append(equiDiameter)

    # Orientation
    # can only be done if the contour has at least 5 points
    angle = 0
    if (len(maxContour) >= 5):
        (x,y),(MA,ma),angle = cv.fitEllipse(maxContour)
    print("Orientation: " + str(angle))
    orientations.append(angle)


metadataDF = pd.read_csv("./test/test_metadata.csv")
metadataDF = metadataDF.sort_values(by="id")
print(metadataDF)


generatedFeatures = {"image_path": imgNames,
                       "aspect_ratio": aspectRatios,
                       "extent": extents,
                       "solidity": solidities,
                       "equivalent_diameter": equiDiameters,
                       "orientation": orientations,
                       "hu_1": hu1, "hu_2": hu2,
                       "hu_3": hu3, "hu_4": hu4,
                       "hu_5": hu5, "hu_6": hu6,
                       "hu_7": hu7}
generatedFeaturesDF = pd.DataFrame(generatedFeatures)
# print(generatedFeaturesDF)
                       
generatedFeaturesDF = pd.concat([metadataDF.set_index('image_path'), generatedFeaturesDF.set_index('image_path')], axis=1, join='inner').reset_index()
print(generatedFeaturesDF)
# generatedFeaturesDF = generatedFeaturesDF.sort_values(by="id")
# print(generatedFeaturesDF)
generatedFeaturesDF = generatedFeaturesDF.sort_values(by="image_path")
print(generatedFeaturesDF)

generatedFeaturesDF.to_csv("engineered_features_test.csv")

additionalFeaturesDF = pd.read_csv("./test/Features/additional_features.csv")
colorHistogramDF = pd.read_csv("./test/Features/color_histogram.csv")
hogPCADF = pd.read_csv("./test/Features/hog_pca.csv")

featuresDF = pd.concat([generatedFeaturesDF.set_index('image_path'), additionalFeaturesDF.set_index('image_path')], axis=1, join='inner').reset_index()
featuresDF = pd.concat([featuresDF.set_index('image_path'), colorHistogramDF.set_index('image_path')], axis=1, join='inner').reset_index()
featuresDF = pd.concat([featuresDF.set_index('image_path'), hogPCADF.set_index('image_path')], axis=1, join='inner').reset_index()
# featuresDF = featuresDF.drop(columns=["Unnamed: 0"])

print(featuresDF)

featuresDF.to_csv("all_features_test.csv")

