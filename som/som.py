# Anomaly-detection with SOM(Self Organization Map) and performance test 

import pandas as pd
from somber import Som 
import numpy as np

def main(): 

    # Load training & testing datasets 
    X_train = pd.read_csv('../dataset/train-set.csv')  
    X_test = pd.read_csv('../dataset/test-set.csv') 
    y_test = pd.read_csv('../dataset/test-set-label.csv')  

    # Make a SOM whose units(clusters) are from 0 to 24 and learning from X_train   
    s = Som((5, 5), 42, learning_rate=0.3)
    s.fit(X_train, num_epochs=10, updates_epoch=10) 

    # Choose the proper unit for each row in the testing dataset
    test_predictions = s.predict(X_test) 

    # Select units that are likely to be an attack. (5 ~ 10% from the dataset)  
    # Create a dictionary: (unit number(0 ~ 24 for each), # element in the unit) 
    unique, counts = np.unique(test_predictions, return_counts=True)  
    test_predictions_dic = dict(zip(unique, counts))  

    # Change the dictionary into a list which contains tuples: 
    # (unit number(0 ~ 24 for each), # element in the unit) 
    # Then, sort the list by # element in the unit in ascending order   
    test_predictions_tuple = sorted(test_predictions_dic.items(), key=(lambda x: x[1]))  
    
    # Iterate the list and save the sum of # element of each unit to 'sum'
    # Then, add the unit number to 'units'. 
    # If 'sum' reaches 5 ~ 10% (4116 ~ 8233) of the test dataset, stop the iteration 
    sum = 0 
    units = []
    for tuples in test_predictions_tuple:
        units.append(tuples[0])
        sum = tuples[1] + sum 
        if sum > 4116 and sum < 8233: 
            break     
   
   # Change the values of test_predictions which are in the 'units' to 1(attack) 
   # Change it 0 if the unit does not in 'units' 
    for index, value in enumerate(test_predictions):  
        if value in units:
            test_predictions[index] = 1 
        else: 
            test_predictions[index] = 0      
    tp = 0
    fn = 0
    fp = 0 
    tn = 0 
    total_labels = 175341

    # Calculate TP, FP, FN and TN
    for index, value in enumerate(test_predictions):  
        if value == 1: 
            if(y_test['label'][index] == 1): 
                tp = tp+1 
            else: 
                fp = fp+1 
        else: 
            if(y_test['label'][index] == 0):
                tn = tn+1
            else: 
                fn = fn+1

    tp_rate = round(tp/total_labels,2) * 100
    fn_rate = round(fn/total_labels,2) * 100
    fp_rate = round(fp/total_labels,2) * 100
    tn_rate = round(tn/total_labels,2) * 100

    print("True Positive: ",tp_rate,"%") 
    print("False Negative: ",fn_rate,"%")
    print("False Positive: ",fp_rate,"%")
    print("True Negative: ",tn_rate,"%") 

# Calling main function 
if __name__=="__main__": 
    main()
