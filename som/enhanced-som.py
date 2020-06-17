# Anomaly-detection with enhanced SOM(Self Organization Map) and performance test using test dataset from the same dataset
# Asuume that 65 ~ 70 % clusters include abnormal traffics 
# 
# The structure of this model is exactly the same as one of 'som.py'. 
# So, creating a model for the NSL-KDD is not included here. 

import pandas as pd
from somber import Som 
import numpy as np 
import time 
import sys  
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # to use utils.py
import utils # my module

def main(): 
    # start measuring execution time
    start = time.time()
    # Load training & testing datasets 
    X_train, y_train, X_test, y_test = utils.load_dataset()  

    # Make a SOM whose units(clusters) are from 0 to 24 and learning from X_train   
    model = Som((5, 5), 42, learning_rate=0.2)
    model.fit(X_train, num_epochs=1, updates_epoch=1) 

    # Choose the proper unit for each row in the testing dataset
    y_pred = model.predict(X_test) 
    y_pred = utils.create_binary_prediction_som(y_pred, 175341, 0.65, 0.05) 
            
    print("execution time: ", time.time() - start)
    utils.cal_accuracy(y_test, y_pred, 175341)   
     
# Calling main function 
if __name__=="__main__": 
    main()
