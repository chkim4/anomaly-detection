# Anomaly-detection with SOM(Self Organization Map) and performance test using test dataset from the same dataset 
# Finally, save a model for '../benchmark.py' 
# Asuume that 5 ~ 10 % clusters include abnormal traffics

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
    model.fit(X_train, num_epochs=10, updates_epoch=10) 

    # Choose the proper unit for each row in the testing dataset
    y_pred = model.predict(X_test) 
    y_pred = utils.create_binary_prediction_som(y_pred, 175341, 0.05, 0.1)
  
    print("execution time: ", time.time() - start)
    utils.cal_accuracy(y_test, y_pred, 175341)    

    #######Create models for testing the NSL-KDD dataset####### 

    X_train = pd.read_csv('../dataset/unsw-nb15/nsl-kdd-ver/train-set.csv') 
    
    model = Som((5, 5), 5, learning_rate=0.2) 
    model.fit(X_train, num_epochs=1, updates_epoch=1) 

    # Save the model
    utils.save_model(model,'./model/nsl-kdd-model.sav')
 
     
# Calling main function 
if __name__=="__main__": 
    main()
