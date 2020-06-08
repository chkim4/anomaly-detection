# modules used in several files.
import pandas as pd 
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

# load datasets. 
# Note that the path of 'read_csv' must match the location of the file executing this function. 
def load_dataset():
    X_train = pd.read_csv('../dataset/train-set.csv') 
    y_train = pd.read_csv('../dataset/train-set-label.csv') 
    X_test = pd.read_csv('../dataset/test-set.csv') 
    y_test = pd.read_csv('../dataset/test-set-label.csv')

    return X_train, y_train, X_test, y_test 

# Calculate accuracy(tp,fn,fp,tn) comparing test values and prediction values
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