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
    utils.cal_accuracy(y_test, y_pred, 175341)  

    #######Create models for benchmarking NSL-KDD####### 

    X_train = pd.read_csv('../dataset/unsw-nb15/nsl-kdd-ver/train-set.csv') 
    
    model = GaussianNB() 
    model.fit(X_train,y_train)  

    # Save the models
    utils.save_model(model,'./model/nsl-kdd-model.sav')
 
if __name__=="__main__": 
    main() 