import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

featuresTrainDF = pd.read_csv("./all_features.csv")
featuresTrainDF = featuresTrainDF.drop(columns=["Unnamed: 0"])

featuresTestDF = pd.read_csv("./all_features_test.csv")
featuresTestDF = featuresTestDF.drop(columns=["Unnamed: 0"])

print(featuresTrainDF)
print(featuresTestDF)

yTrain = featuresTrainDF["ClassId"]
print(yTrain)

# all features performed the best so we select all
XTrain = featuresTrainDF.drop(columns=["image_path", "id", "ClassId"])
print(XTrain)

XTest = featuresTestDF.drop(columns=["image_path", "id", "ClassId"])
print(XTest)

# print(XTrain)
# print(XTest)

classifier = RandomForestClassifier(random_state=16)
classifier.fit(XTrain, yTrain)
yPred = classifier.predict(XTest)

final = {"id": featuresTestDF["id"], "ClassId": yPred}
finalDF = pd.DataFrame(final)
print(finalDF)

finalDF.to_csv("./rf/rfallfeatures.csv", index=False)
