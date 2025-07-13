import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
top10 = ["hog_pca_0", "hog_pca_1", "hog_pca_2", "hog_pca_3", "hog_pca_4",
         "hog_pca_5", "hog_pca_7", "hog_pca_8", "hog_pca_9", "hog_pca_12"] 

featuresTrainDF = pd.read_csv("./all_features.csv")
featuresTrainDF = featuresTrainDF.drop(columns=["Unnamed: 0"])

featuresTestDF = pd.read_csv("./all_features_test.csv")
featuresTestDF = featuresTestDF.drop(columns=["Unnamed: 0"])

print(featuresTrainDF)
print(featuresTestDF)

yTrain = featuresTrainDF["ClassId"]
print(yTrain)

XTrain = featuresTrainDF.drop(columns=["image_path", "id", "ClassId"])
XTrain = XTrain[top10]
print(XTrain)

XTest = featuresTestDF.drop(columns=["image_path", "id", "ClassId"])
XTest = XTest[top10]
print(XTest)

scaler = StandardScaler().fit(XTrain)
XTrain = scaler.transform(XTrain)
XTest = scaler.transform(XTest)
# print(XTrain)
# print(XTest)

classifier = svm.SVC()
classifier.fit(XTrain, yTrain)
yPred = classifier.predict(XTest)

final = {"id": featuresTestDF["id"], "ClassId": yPred}
finalDF = pd.DataFrame(final)
print(finalDF)

finalDF.to_csv("./svm/svmANOVA10features.csv", index=False)
