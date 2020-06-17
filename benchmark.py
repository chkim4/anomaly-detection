# Performance test of the models that are trained with the UNSW-NB15 using the NSL-KDD dataset.
# To execute each models, you need to eliminate the """ comment of each model.
# 
# Since the number of features of train and test datasets should be equal, 
# the models are trained with the UNSW-NB15 dataset('./dataset/unsw-nb15/nsl-kdd-ver/train-set.csv') 
# whose features are compatible with the NSL-KDD dataset('./dataset/nsl-kdd/test-set.csv')
# The processes of creating each models are written in each model's files which are in each model folders.    

import joblib
import pandas as pd  
import numpy as np
import tensorflow as tf
from keras import models, layers, optimizers, losses, metrics   
from keras.models import model_from_json
import time
import utils # my module 

def main(): 

    # Load dataset
    X_test = pd.read_csv('./dataset/nsl-kdd/test-set.csv')
    y_test = pd.read_csv('./dataset/nsl-kdd/test-set-label.csv')

    # Test using gini index of decision tree
    """
    start = time.time()
    decision_tree_gini_model = joblib.load('./decision-tree/model/nsl-kdd-model-gini.sav')
    y_pred = decision_tree_gini_model.predict(X_test)
    print("execution time of classification with gini index: ", time.time() - start) 
    utils.cal_accuracy(y_test, y_pred, 11850) 
    """

    # Test using entropy index of decision tree
    """
    start = time.time()
    decision_tree_entropy_model = joblib.load('./decision-tree/model/nsl-kdd-model-entropy.sav')
    y_pred = decision_tree_entropy_model.predict(X_test)
    print("execution time of classification with entropy index: ", time.time() - start) 
    utils.cal_accuracy(y_test, y_pred, 11850) 
    """
    
    # Test using naive bayes
    """
    start = time.time()
    naive_bayes_model = joblib.load('./naive-bayes/model/nsl-kdd-model.sav')
    y_pred = naive_bayes_model.predict(X_test)
    print("execution time of naive-bayes: ", time.time() - start) 
    utils.cal_accuracy(y_test, y_pred, 11850)
    """
    
    # Test using neural-network 
    """
    start = time.time()
    
    # Load the model's information (JSON file) and create model
    neural_network_model_info = open('./neural-network/model/nsl-kdd-model.json', 'r')
    neural_network_model_json = neural_network_model_info.read()
    neural_network_model_info.close()
    neural_network_model = model_from_json(neural_network_model_json)
    
    # Assign weights into the new model 
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

    utils.cal_accuracy_neural_network(results)
    """
    # Test using SOM
    """
    start = time.time()
    som_model = joblib.load('./som/model/nsl-kdd-model.sav')
    
    # Choose the proper unit for each row in the testing dataset
    y_pred = som_model.predict(X_test) 
    y_pred = utils.create_binary_prediction_som(y_pred, 11850, 0.82, 0.01) 

    print("execution time of som: ", time.time() - start) 
    utils.cal_accuracy(y_test, y_pred, 11850) 
    """

if __name__=="__main__": 
    main() 