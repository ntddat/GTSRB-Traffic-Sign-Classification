TRAFFIC SIGN CLASSIFICATION – COMP30027 PROJECT 2 (2025)
=========================================================

!!!NOTICE: ALL FILES SHOULD BE RUN IN THE MAIN DIRECTORY (2025_A2/)
Depending on your environment, run one of the following commands to run each file
python <path>/<filename>.py
python3 <path>/<filename>.py
For example,
python3 ./cnn/cnn.py

!!Also ensure tensorflow is installed (sometimes this can be buggy not sure why)
I used this command
pip3 install tensorflow[and-cuda]
(might be different on your environment)

Directory Structure:
--------------------
The project is organized as follows:

2025_A2/
|-- train/
|-- test/
|-- svm/
|   |-- svm-fs-ANOVA.py
|   |-- svm-fs-mi.py
|   |-- svm-fs-fwdSelect.py
|   |-- svm-eval.py
|   |-- other csv/png files generated from the py files
|-- rf/
|   |-- rf-fs-ANOVA.py
|   |-- rf-fs-mi.py
|   |-- rf-eval.py
|   |-- other csv/png files generated from the py files
|-- cnn/
|   |-- cnn.py
|   |-- cnn-eval.py
|   |-- cnnMoreLayers.py
|   |-- cnnMoreLayers-eval.py
|   |-- cnnMoreLayersDropout.py
|   |-- cnnMoreLayersDropoutDense.py
|   |-- cnnMoreLayersDropoutDense-eval.py
|   |-- cnnMoreLayers2DropoutDense.py
|   |-- other csv/weights files generated from the py files
|-- featureEngineering.py
|-- featureEngineeringTest.py
|-- utils.py
|-- Readme.txt (this file)
|-- other csv/png files generated from the py files

File Description:
--------------------

- General Descriptions:
"train/" and "test/" directories contain the training and test data provided with the assignment.
Feature engineering files should be run first, then feature select files.
Files that contain "-fs" are files that has to do with feature selection
Files that end with "-eval" are the final versions of each model type predicting the test set, and
thus they should be run after every file for that model.
Results for each model can be found in their respective directories.

- Specific Descriptions:
  - svm-fs-ANOVA.py
    Testing feature selection with ANOVA F-score (f_classif) metric using SVM.
  - svm-fs-mi.py
    Testing feature selection with Mutual Information metric using SVM.
  - svm-fs-fwdSelect.py
    Testing feature selection with Sequential Forward Selection metric using SVM.
    This file took me 2h30m to finish running + no accuracy increase observed.
  - rf-fs-ANOVA.py
    Testing feature selection with ANOVA F-score (f_classif) metric using RF.
  - rf-fs-mi.py
    Testing feature selection with Mutual Information metric using RF.
  - cnn.py
    Basic CNN with basic layers.
  - cnnMoreLayers.py
    Same as previous CNN, but with an additional Conv2D and MaxPooling layer.
  - cnnMoreLayersDropout.py
    Same as previous CNN, but with an additional Dropout layer.
    No real increase on validation set observed, so no "-eval" file created.
  - cnnMoreLayersDropoutDense.py
    Same as previous CNN, but with an additional Dense layer.
  - cnnMoreLayers2DropoutDense.py
    Same as previous CNN, but with an additional Conv2D and MaxPooling layer.
    No real increase on validation set observed, so no "-eval" file created.
    After this, no more layer addition was experimented with due to time + overfitting concern.
  - featureEngineering.py
    Feature engineering and preprocessing for the training dataset.
  - featureEngineeringTest.py
    Feature engineering and preprocessing for the test dataset.
  - utils.py
    File to get results for the report that can't really fit anywhere else.
