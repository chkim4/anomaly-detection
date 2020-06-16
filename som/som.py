# Anomaly-detection with SOM(Self Organization Map) and performance test 

import pandas as pd
from somber import Som 
import numpy as np 
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
    # Then, append the unit number to 'units'. 
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
            
    print("execution time: ", time.time() - start)
    utils.cal_accuracy(y_test, test_predictions)   
     
# Calling main function 
if __name__=="__main__": 
    main()
