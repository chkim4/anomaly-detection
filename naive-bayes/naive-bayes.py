# Anomaly-detection with Naive Bayes and performance test 

import pandas as pd  
import numpy as np
from sklearn.naive_bayes import GaussianNB 
import time
import sys  
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import utils # my module 

def main():  
    # start measuring execution time
    start = time.time() 
    # Load training & testing datasets
    X_train, y_train, X_test, y_test = utils.load_dataset()
    
    #To reshape y_train into 1d array
    y_train = y_train['label'] 
    
    #Create a Gaussian Classifier
    model = GaussianNB()

    #Train the model using the training sets
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)  
    print("execution time: ", time.time() - start)
    utils.cal_accuracy(y_test, y_pred) 

if __name__=="__main__": 
    main() 