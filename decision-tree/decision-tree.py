# Anomaly-detection with decision tree and performance test using test dataset from the same dataset
# Finally, save a model for '../benchmark.py'

import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier  
import time
import sys  
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # to use utils.py
import utils # my module
        
# Calculate impurity in the terminal nodes with gini index.
# Calculating gini index is usually faster than entropy but the result can be more biased    
def train_with_index(index, X_train, y_train): 
  
    # Create a classifier object 
    # criterion: index name to use (gini or entropy)
    # random_state: To limit the number of applying heuristic algorithms to build a decision tree
    # max_depth: limit the depth of the decision tree to avoid overfitting 
    # min_samples_leaf: minimum number of the samples that each terminal nodes should include
    classfier = DecisionTreeClassifier(criterion = index, 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
    # Train and return the model
    classfier.fit(X_train, y_train) 
    return classfier 
        
# Prediction of classifier object(clf_object) based on X-test
# and return the prediction values
def prediction(X_test, clf_object): 
    y_pred = clf_object.predict(X_test) 
    return y_pred 

def main(): 
    # start measuring execution time
    start = time.time() 
    # Load training & testing datasets
    X_train, y_train, X_test, y_test = utils.load_dataset()

    gini_classifier = train_with_index('gini', X_train, y_train)  

    print("Results Using Gini Index: \n")
    
    # Prediction using gini 
    y_pred_gini = prediction(X_test, gini_classifier) 
    print("execution time of classification with gini index: ", time.time() - start)
    utils.cal_accuracy(y_test, y_pred_gini, 175341)  
    
    start = time.time()
    entropy_classifier = train_with_index('entropy', X_train, y_train)

    print("Results Using Entropy: \n") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, entropy_classifier) 
    print("execution time of classification with entropy index: ", time.time() - start)
    utils.cal_accuracy(y_test, y_pred_entropy, 175341)    


    #######Create models for testing the NSL-KDD dataset#######   

    X_train = pd.read_csv('../dataset/unsw-nb15/nsl-kdd-ver/train-set.csv') 
    
    gini_classifier = train_with_index('gini', X_train, y_train)
    entropy_classifier = train_with_index('entropy', X_train, y_train) 

    # Save the models
    utils.save_model(gini_classifier,'./model/nsl-kdd-model-gini.sav')
    utils.save_model(entropy_classifier,'./model/nsl-kdd-model-entropy.sav') 

# Calling main function 
if __name__=="__main__": 
    main()