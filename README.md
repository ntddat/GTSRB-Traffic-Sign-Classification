# GTSRB Traffic Sign Classification

Several machine learning techniques implemented to classify images of Traffic Signs, into 42 different classes (types of traffic signs). The training and test datasets were obtained from the ![German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) (GTSRB). Implemented techniques include Support Vector Machine (SVM), Random Forest (RF), and Convoluted Neural Network (CNN). More information, and analysis of each technique can be found in ![report.pdf](/report.pdf).

This was done as part of Assignment 2, in the University of Melbourne's Machine Learning course (COMP30027, Semester 1, 2025).

## Dependencies and Execution

### Dependencies

Libraries used for this project:
- NumPy: https://numpy.org/install/
- Pandas: https://pandas.pydata.org/docs/getting\_started/install.html
- MatPlotLib: https://matplotlib.org/stable/install/index.html
- sklearn: https://scikit-learn.org/stable/install.html
- OpenCV: https://opencv.org/get-started/ 
- Ensure tensorflow: https://www.tensorflow.org/install/pip, which should include keras, is installed (sometimes this can be buggy not sure why). I used this command:
```
pip3 install tensorflow[and-cuda]
```
(might be different on your environment)
- pytesseract: https://pypi.org/project/pytesseract/ (not actually used to obtain results in the report, but experimented with)

### Running the Models

NOTICE: ALL FILES SHOULD BE RUN IN THE MAIN DIRECTORY
Depending on your environment, run one of the following commands to run each file
```
python <path>/<filename>.py
```
```
python3 <path>/<filename>.py
```
For example,
```
python3 ./cnn/cnn.py
```

## Directory Explanations

### General Descriptions:
`train/` and `test/` directories contain the training and test data provided with the assignment.
Feature engineering files should be run first, then feature select files.
Files that contain `-fs` are files that has to do with feature selection
Files that end with `-eval` are the final versions of each model type predicting the test set, and
thus they should be run after every file for that model.
Results for each model can be found in their respective directories.

### Specific Descriptions:
- `svm-fs-ANOVA.py`
Testing feature selection with ANOVA F-score (`f\_classif`) metric using SVM.
- `svm-fs-mi.py`
Testing feature selection with Mutual Information metric using SVM.
- `svm-fs-fwdSelect.py`
Testing feature selection with Sequential Forward Selection metric using SVM.
This file took 2h30m to finish running + no accuracy increase observed.
- `rf-fs-ANOVA.py`
Testing feature selection with ANOVA F-score (`f\_classif`) metric using RF.
- `rf-fs-mi.py`
Testing feature selection with Mutual Information metric using RF.
- `cnn.py`
Basic CNN with basic layers.
- `cnnMoreLayers.py`
Same as previous CNN, but with an additional Conv2D and MaxPooling layer.
- `cnnMoreLayersDropout.py`
Same as previous CNN, but with an additional Dropout layer.
No real increase on validation set observed, so no `-eval` file created.
- `cnnMoreLayersDropoutDense.py`
Same as previous CNN, but with an additional Dense layer.
- `cnnMoreLayers2DropoutDense.py`
Same as previous CNN, but with an additional Conv2D and MaxPooling layer.
No real increase on validation set observed, so no `-eval` file created.
After this, no more layer addition was experimented with due to time + overfitting concern.
- `featureEngineering.py`
Feature engineering and preprocessing for the training dataset.
- `featureEngineeringTest.py`
Feature engineering and preprocessing for the test dataset.
- `utils.py`
File to get results for the report that can't really fit anywhere else.
