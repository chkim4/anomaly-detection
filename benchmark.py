import joblib
import pandas as pd  
import numpy as np
import tensorflow as tf
from keras import models, layers, optimizers, losses, metrics   
from keras.models import model_from_json
import time
import utils # my module 


def main(): 

    X_test = pd.read_csv('./dataset/nsl-kdd/test-set.csv')
    y_test = pd.read_csv('./dataset/nsl-kdd/test-set-label.csv')

    # benchmark using gini index of decision tree
    """
    start = time.time()
    decision_tree_gini_model = joblib.load('./decision-tree/model/nsl-kdd-model-gini.sav')
    y_pred = decision_tree_gini_model.predict(X_test)
    print("execution time of classification with gini index: ", time.time() - start) 
    utils.cal_accuracy(y_test, y_pred, 11850) 
    """

    # benchmark using entropy index of decision tree
    """
    start = time.time()
    decision_tree_entropy_model = joblib.load('./decision-tree/model/nsl-kdd-model-entropy.sav')
    y_pred = decision_tree_entropy_model.predict(X_test)
    print("execution time of classification with entropy index: ", time.time() - start) 
    utils.cal_accuracy(y_test, y_pred, 11850) 
    """
    
    # benchmark using naive bayes
    """
    start = time.time()
    naive_bayes_model = joblib.load('./naive-bayes/model/nsl-kdd-model.sav')
    y_pred = naive_bayes_model.predict(X_test)
    print("execution time of naive-bayes: ", time.time() - start) 
    utils.cal_accuracy(y_test, y_pred, 11850)
    """
    
    # benchmark using neural-network 
    
    start = time.time()
    
    # load json and create model
    neural_network_model_info = open('./neural-network/model/nsl-kdd-model.json', 'r')
    neural_network_model_json = neural_network_model_info.read()
    neural_network_model_info.close()
    neural_network_model = model_from_json(neural_network_model_json)
    
    # load weights into new model 
    neural_network_model.load_weights("./neural-network/model/nsl-kdd-model.h5")
    

    neural_network_model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                loss=losses.binary_crossentropy,
                 metrics=[tf.keras.metrics.TruePositives()
                        ,tf.keras.metrics.FalsePositives()
                          ,tf.keras.metrics.FalseNegatives()
                          ,tf.keras.metrics.TrueNegatives()
                          ])
    results = neural_network_model.evaluate(X_test, y_test, verbose=0) 

    print("execution time of neural-network: ", time.time() - start)

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
    

    # benchmark using som 
    """
    start = time.time()
    som_model = joblib.load('./som/model/nsl-kdd-model.sav')
    y_pred = som_model.predict(X_test)
    
    # Choose the proper unit for each row in the testing dataset
    test_predictions = som_model.predict(X_test) 

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
    total_labels = 11850 
    percent = 0.12
    upper = 0.01
    for tuples in test_predictions_tuple:
        units.append(tuples[0])
        sum = tuples[1] + sum 
        if sum > total_labels*percent and sum < total_labels*(percent+upper): 
            break     
   
   # Change the values of test_predictions which are in the 'units' to 1(attack) 
   # Change it 0 if the unit does not in 'units' 
    for index, value in enumerate(test_predictions):  
        if value in units:
            test_predictions[index] = 1 
        else: 
            test_predictions[index] = 0

    print("execution time of som: ", time.time() - start) 
    utils.cal_accuracy(y_test, test_predictions, 11850) 
   """

if __name__=="__main__": 
    main() 