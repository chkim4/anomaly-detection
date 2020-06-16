# Anomaly-detection with neural network and performance test 

import tensorflow as tf
import pandas as pd
import numpy as np
from keras import models, layers, optimizers, losses, metrics  
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

    # Change data to float type data for the model
    X_train = np.asarray(X_train).astype("float32")
    y_train = np.asarray(y_train).astype("float32") 
    X_test = np.asarray(X_train).astype("float32")
    y_test = np.asarray(y_train).astype("float32")

    # Initialize model
    model = models.Sequential()

    # Add layers to model(# of the neuron, activation function, input size)
    # neuron: a unit that calculates the weight for each feature and send the result to the next neurons of the next layer
    # activation function: determines the value of a combination of weight and each feature   
    # relu: to prevent vanishing gradient problem of sigmoid 
    # sigmod: outputs between 0 and 1
    model.add(layers.Dense(64, activation="relu", input_shape=(42,)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    
    # optimizer: optimization function to be applied to decrease errors
    # RMSprop reflects new inputs more than the old ones
    # loss: loss function that calculates differences between y_train and the model's prediction value. 
    # binary_crossentropy is used when y_train consists of two categories (0 and 1)
    # metrics: measures tp, fp, fn and tn for each epochs 
    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                loss=losses.binary_crossentropy,
                metrics=[tf.keras.metrics.TruePositives(name='true_positives')
                        ,tf.keras.metrics.FalsePositives(name='false_positives')
                          ,tf.keras.metrics.FalseNegatives(name='false_negatives')
                          ,tf.keras.metrics.TrueNegatives(name='true_negatives')
                          ])
    
    # epochs: the number of calculating weights(training phase)
    model.fit(X_train, y_train, epochs=20, batch_size=1024)  

    results = model.evaluate(X_test, y_test, batch_size=128) 
    print("execution time: ", time.time() - start) 

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
 
# Calling main function 
if __name__=="__main__": 
    main()

