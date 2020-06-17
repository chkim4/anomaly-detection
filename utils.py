# modules used in several files.

import pandas as pd 
import numpy as np
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import joblib 

# load datasets. 
# Note that the path of 'read_csv' must match the location of the file executing this function. 
def load_dataset():
    X_train = pd.read_csv('../dataset/unsw-nb15/unsw-nb15-train-test/train-set.csv') 
    y_train = pd.read_csv('../dataset/unsw-nb15/unsw-nb15-train-test/train-set-label.csv') 
    X_test = pd.read_csv('../dataset/unsw-nb15/unsw-nb15-train-test/test-set.csv') 
    y_test = pd.read_csv('../dataset/unsw-nb15/unsw-nb15-train-test/test-set-label.csv')

    return X_train, y_train, X_test, y_test  

#save the model 
def save_model(model, name):
    joblib.dump(model, name)

# Calculate accuracy(tp,fn,fp,tn) comparing test values and prediction values
def cal_accuracy(y_test, y_pred, total_labels): 

    # the number of the label. (The number of values in 'raw_data/test-set-labels.csv')
    #total_labels = 175341
    
    # labels: Change the order of the result 
    # to not be confused with the order of 
    # TP(True Positive), FN(False Negative), FP(False Positive) and TN(True Negative)    
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

# Calculate accuracy(tp,fn,fp,tn) of neural-network model 
def cal_accuracy_neural_network(results):
    tp = results[1] 
    tn = results[2] 
    fp = results[3] 
    fn = results[4]

    total_labels = tp + tn + fp + fn 

    tp_rate = round(tp/total_labels,2) * 100
    fn_rate = round(fn/total_labels,2) * 100
    fp_rate = round(fp/total_labels,2) * 100
    tn_rate = round(tn/total_labels,2) * 100

    print("True Positive: ",tp_rate,"%") 
    print("False Negative: ",fn_rate,"%")
    print("False Positive: ",fp_rate,"%")
    print("True Negative: ",tn_rate,"%")  


# Convert y_pred of SOM (which consists of unit numbers) into 0 (normal) or 1 (anomaly) 
# Assuming that some units consist of features of the anomaly. 
# (The ratio of the units to the whole units is decided by 'percent' of 'total_labels' and 'upper'.)  
# 
# y_pred: The original y_pred value of SOM 
# total_labels: Indicates the number of elements in y_pred. 
# percent: Indicates the proportion of the anomaly in the test dataset 
# upper: In a combination with 'percent', decides the highest proportion of the anomaly in the test dataset   
def create_binary_prediction_som(y_pred,total_labels, percent, upper): 
    
    # Select units that are likely to be an attack. (percent ~ percent+upper % from the dataset)  
    # Create a dictionary: (unit number(0 ~ 24 for each), # element in the unit) 
    unique, counts = np.unique(y_pred, return_counts=True)  
    y_pred_dic = dict(zip(unique, counts))  

    # Change the dictionary into a list which contains tuples: 
    # (unit number(0 ~ 24 for each), # element in the unit) 
    # Then, sort the list by # element in the unit in ascending order   
    y_pred_tuple = sorted(y_pred_dic.items(), key=(lambda x: x[1]))  
    
    # Iterate the list and save the sum of # element of each unit to 'sum'
    # Then, append the unit number to 'units'. 
    # If 'sum' percent ~ (percent+upper)% of the test dataset, stop the iteration 
    sum = 0 
    units = [] 
    
    for tuples in y_pred_tuple:
        units.append(tuples[0])
        sum = tuples[1] + sum 
        if sum > total_labels*percent and sum < total_labels*(percent+upper): 
            break     
   
   # Change the values of y_pred which are in the 'units' to 1(attack) 
   # Change it 0 if the unit does not in 'units' 
    for index, value in enumerate(y_pred):  
        if value in units:
            y_pred[index] = 1 
        else: 
            y_pred[index] = 0  
    
    return y_pred
