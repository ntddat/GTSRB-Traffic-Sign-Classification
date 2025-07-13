import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import svm, metrics

featuresDF = pd.read_csv("./all_features.csv")
featuresDF = featuresDF.drop(columns=["Unnamed: 0"])

labelsDF = featuresDF["ClassId"]
print(labelsDF)

featuresDF = featuresDF.drop(columns=["image_path", "id", "ClassId"])
print(featuresDF)

accuracies = []
kValues = range(10, 132, 10)

kFoldIndices = KFold(n_splits=5, shuffle=True, random_state=16)

for train, test in kFoldIndices.split(featuresDF):
    XTrain, XValid = featuresDF.iloc[train], featuresDF.iloc[test]
    yTrain, yValid = labelsDF.iloc[train], labelsDF.iloc[test]
    # print(XTrain)
    print(XValid)

    scaler = StandardScaler().fit(XTrain)
    XTrain = scaler.transform(XTrain)
    XValid = scaler.transform(XValid)

    classifier = svm.SVC()
    selector = SequentialFeatureSelector(classifier, n_features_to_select="auto", tol=0.01,
                                         scoring="accuracy", n_jobs=-1)
    selector.fit(XTrain, yTrain)
    XTrainNew = selector.transform(XTrain)

    selected = featuresDF.columns[selector.get_support(indices=True)].tolist()
    print(selected)

    XValid = featuresDF[selected].iloc[test]
    print(XValid)

    classifier = svm.SVC()
    classifier.fit(XTrainNew, yTrain)
    yPred = classifier.predict(XValid)

    accuracy = metrics.accuracy_score(yValid, yPred) 
    print("Accuracy: ", accuracy)
    print("Precision: ", metrics.precision_score(yValid, yPred, average='micro'))
    print("Recall: ", metrics.recall_score(yValid, yPred, average='micro'))

    accuracies.append(accuracy)

print(accuracies)
print(np.mean(accuracies))


