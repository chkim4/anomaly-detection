# Anomaly-detection with machine learning
This repo is network anomaly detection models trained and tested by the [UNSW-NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/) dataset. Benchmarking with [KDD-NSL](https://www.unb.ca/cic/datasets/nsl.html) is included too.
It created because of section IV of the report for information security class. The paper is available [here](https://www.overleaf.com/read/bvcffqqhdkfr). This repo consists of following machine learning models in each folder: 

* Decision Tree
* Naive Bayes 
* Neural Network
* SOM(Self-Organizing Map)

## Installation 

* [python3](https://www.python.org/downloads/release) 
* [pandas](https://pypi.org/project/pandas/) 
* [numpy](https://pypi.org/project/numpy/) 
* [scikit-learn](https://pypi.org/project/scikit-learn/)
* [tensorflow2.0](https://pypi.org/project/tensorflow/) 
* [keras](https://pypi.org/project/Keras/) 
* [somber](https://github.com/stephantul/somber) 
* [joblib](https://joblib.readthedocs.io/en/latest/)

## Usage 

* benchmark.py includes benchmarking the NSL-KDD dataset with models trained by UNSW-NB15.
* To test the UNSW-NB15 dataset and create each models for benchmark, execute python files in each folder.  
* The models calculate TP(True Positive), FN(False Negative), FP(False Positive) and TN(True Negative) from the training dataset
