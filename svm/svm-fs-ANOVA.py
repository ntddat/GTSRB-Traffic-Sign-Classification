import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif
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

selectedK10 = []

# there are 132 features
for k in kValues:
    if k == 130:
        k = 132

    foldAccuracies = []

    kFoldIndices = KFold(n_splits=5, shuffle=True, random_state=16)
    for train, test in kFoldIndices.split(featuresDF):
        XTrain, XValid = featuresDF.iloc[train], featuresDF.iloc[test]
        yTrain, yValid = labelsDF.iloc[train], labelsDF.iloc[test]
        # print(XTrain)
        print(XValid)

        scaler = StandardScaler().fit(XTrain)
        XTrain = scaler.transform(XTrain)
        XValid = scaler.transform(XValid)

        selector = SelectKBest(f_classif, k=k)
        selector.fit(XTrain, yTrain)
        XTrainNew = selector.transform(XTrain)
        selected = featuresDF.columns[selector.get_support(indices=True)].tolist()
        print(selected)

        # accuracy is highest when fewer features
        # so we collect all features across different folds to see which ones should be used
        if k == 10:
            selectedK10.append(selected)

        XValid = featuresDF[selected].iloc[test]
        print(XValid)

        classifier = svm.SVC()
        classifier.fit(XTrainNew, yTrain)
        yPred = classifier.predict(XValid)

        accuracy = metrics.accuracy_score(yValid, yPred) 
        print("Accuracy: ", accuracy)
        print("Precision: ", metrics.precision_score(yValid, yPred, average='micro'))
        print("Recall: ", metrics.recall_score(yValid, yPred, average='micro'))

        foldAccuracies.append(accuracy)

    accuracies.append(np.mean(foldAccuracies))

plt.plot(kValues, accuracies)
plt.xlabel("k Values")
plt.ylabel("Accuracy")
plt.savefig("./svm/ANOVA F-values SVM.png", format="png")
plt.clf()

print("10 features across different folds: ")
print(selectedK10)


