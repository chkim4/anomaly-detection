# Anomaly-detection with neural network and performance test 
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import models, layers, optimizers, losses, metrics  
import time

def main():    
    start = time.time()
    # Load training & testing datasets 
    X_train = pd.read_csv('../dataset/train-set.csv')  
    y_train = pd.read_csv('../dataset/train-set-label.csv') 
    X_test = pd.read_csv('../dataset/test-set.csv') 
    y_test = pd.read_csv('../dataset/test-set-label.csv')

    # Change data to float type data 
    X_train = np.asarray(X_train).astype("float32")
    y_train = np.asarray(y_train).astype("float32") 
    X_test = np.asarray(X_train).astype("float32")
    y_test = np.asarray(y_train).astype("float32")

    # Initialize model
    model = models.Sequential()

    # Add layers to model(# of the neuron, activation function, input size)  
    # relu: to prevent vanishing gradient problem of sigmoid 
    # sigmod: outputs between 0 and 1
    model.add(layers.Dense(64, activation="relu", input_shape=(42,)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    
    # Set method of learning and evaluation (Gradient descent, error function, evaluation indicator)
    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                loss=losses.binary_crossentropy,
                metrics=[tf.keras.metrics.TruePositives(name='true_positives')
                        ,tf.keras.metrics.FalsePositives(name='false_positives')
                          ,tf.keras.metrics.FalseNegatives(name='false_negatives')
                          ,tf.keras.metrics.TrueNegatives(name='true_negatives')
                          ])
    
    model.fit(X_train, y_train, epochs=20, batch_size=1024)  

    results = model.evaluate(X_test, y_test, batch_size=128) 
    
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
    print("execution time: ", time.time() - start)
 
# Calling main function 
if __name__=="__main__": 
    main()