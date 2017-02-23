### Real-time human activity recognition from accelerometer data using Convolutional Neural Networks

---

#### 1. Overview

This code implements a convolutional neural network architecture for real-time user-independent human activity recognition.

Among its main features are very short recognition intervals of size up to 1 second, no manual feature engineering or data 
preprocessing, and domain-independent architecture that with a minimal amount of modifications can be successfully applied 
to different types of datasets.

---

#### 2. Dependencies

###### Data segmentation:

- matlab or octave

###### Activity classification

- python 2.7+
- scikit-learn
- numpy
- tesnorflow

---

#### 3. Experiments

The system was evaluated on two commonly used [WISDM] and [UCI] datasets that contain labeled accelerometer data from 
36 and 30 users respectively and can be freely downloaded from the corresponding websites.

##### Data Segmentation

To perform a segmentation of the initial time series and generate datasets for testing the model, unzip file  
"data_processing/datasets.zip"  and run matlab scripts  "run_WISDM.m"  or "run_UCI.m" for WISDM and UCI 
datasets respectively. The parameters of segmentation are specified in the header of these scripts.

##### Baseline HAR techniques

- To test an approach based on Random Forest + hand-crafted features run "wisdm_random_forest.py".
- To test an approach based on Random Forest + PCA features set parameter 'use_pca_features' to true in "run_WISDM.m" 
and after the data is generated run "wisdm_random_forest.py".
- To test an approach based on the classification of raw accelerometer time series using K-nearest neighbor algorithm, 
run "wisdm_knn.py".


##### CNN Model

The proposed CNN-based model is implemented using ***tensorflow*** machine learning library.

To apply CNN to WISDM dataset, generate traing data and run "cnn_wisdm.py". For UCI dataset run "cnn_uci.py". 
The parameters of the CNN are specified in the header of these scripts.

---

#### 4. System performance


When using UCI dataset and segments (recognition intervals) of size 128, CNN should achieve the accuracy score of about 97%.
For segments of size 50 this value should be about 94%.

When using WISDM dataset and segments (recognition intervals) of size 200, CNN should achieve the accuracy score of about 93%.
For segments of size 50 this value should be about 90%.


 [WISDM]: <http://www.cis.fordham.edu/wisdm/dataset.php>
 [UCI]: <https://archive.ics.uci.edu/ml/machine-learning-databases/00240>
