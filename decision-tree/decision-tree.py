# Anomaly-detection with decision tree and performance test

import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
        
# Calculate impurity in the terminal nodes with gini index.
# Calculating gini index is usually faster than entropy but the result can be more biased    
def train_with_index(index, X_train, X_test, y_train): 
  
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
      
# Calculate accuracy comparing test values and prediction values
def cal_accuracy(y_test, y_pred): 

    # the number of the label. (The number of values in 'raw_data/test-set-labels.csv')
    total_labels = 175341
    
    # labels: Change the order of the result 
    # to not be confused with the order of 
    # TP(True Positive), FN(False Negative), FP(False Positive) and TN(True Negative)    
    # TP: predict the data as an anomaly and it is anomaly
    # FN: predict the data as a normal but it is anomaly
    # FP: predict the data as an anomaly but it is normal
    # TN: predict the data as a normal and it is normal 

    con_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])  

    # Calculate the ratio of TP, FN, FP, and TN to total_labels
    tp = round(con_matrix[0][0]/total_labels,2) * 100
    fn = round(con_matrix[0][1]/total_labels,2) * 100
    fp = round(con_matrix[1][0]/total_labels,2) * 100 
    tn = round(con_matrix[1][1]/total_labels,2) * 100 

    print("True Positive: ",tp,"%")
    print("False Negative: ",fn,"%")
    print("False Positive: ",fp,"%")
    print("True Negative: ",tn,"%") 
    print()
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
    print()  
    print("Report : ")  
    # Summary of the precision, recall, F1 score for each class. 
    print(classification_report(y_test, y_pred, target_names=['normal', 'anomaly'])) 
    print("\n") 
  
def main(): 
      
    X_train = pd.read_csv('../dataset/train-set.csv') 
    y_train = pd.read_csv('../dataset/train-set-label.csv') 

    X_test = pd.read_csv('../dataset/test-set.csv') 
    y_test = pd.read_csv('../dataset/test-set-label.csv') 

    gini_classifier = train_with_index('gini', X_train, X_test, y_train) 
    entropy_classifier = train_with_index('entropy', X_train, X_test, y_train) 
       
    print("Results Using Gini Index: \n")
      
    # Prediction using gini 
    y_pred_gini = prediction(X_test, gini_classifier) 
    cal_accuracy(y_test, y_pred_gini) 
      
    print("Results Using Entropy: \n") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, entropy_classifier) 
    cal_accuracy(y_test, y_pred_entropy) 
      
# Calling main function 
if __name__=="__main__": 
    main()