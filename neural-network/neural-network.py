# Anomaly-detection with neural network and performance test using test dataset from the same dataset
# Finally, save a model for '../benchmark.py' 

import tensorflow as tf
import pandas as pd
import numpy as np
from keras import models, layers, optimizers, losses, metrics  
import time 
import sys  
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # to use utils.py
import utils # my module

# Create a keras sequential model which is trained by X_train and y_train and 
# whose input_shpe is input_features
def create_model(X_train, y_train, input_features):
    
    # Initialize model
    model = models.Sequential()

    # Add layers to model(# of the neuron, activation function, input size)
    # neuron: a unit that calculates the weight for each feature and send the result to the next neurons of the next layer
    # activation function: determines the value of a combination of weight and each feature   
    # relu: to prevent vanishing gradient problem of sigmoid 
    # sigmod: outputs between 0 and 1 
    # input_shape: The number of features of the train dataset
    model.add(layers.Dense(25, activation="relu", input_shape=(input_features,)))
    model.add(layers.Dense(1, activation="sigmoid"))
    
    # optimizer: optimization function to be applied to decrease errors
    # RMSprop reflects new inputs more than the old ones
    # loss: loss function that calculates differences between y_train and the model's prediction value. 
    # binary_crossentropy is used when y_train consists of two categories (0 and 1)
    # metrics: measures tp, fp, fn and tn for each epochs 
    
    model.compile(optimizer=optimizers.RMSprop(lr=0.2),
                loss=losses.binary_crossentropy,
                 metrics=[tf.keras.metrics.TruePositives()
                        ,tf.keras.metrics.FalsePositives()
                          ,tf.keras.metrics.FalseNegatives()
                          ,tf.keras.metrics.TrueNegatives()
                          ])
    
    # epochs: the number of calculating weights(training phase)
    model.fit(X_train, y_train, epochs=20, batch_size=1024)  

    return model

def main():    
    # start measuring execution time
    start = time.time()
    # Load training & testing the NSL-KDD datasets
    X_train, y_train, X_test, y_test = utils.load_dataset() 

    # Change data to float type data for the model
    X_train = np.asarray(X_train).astype("float32")
    y_train = np.asarray(y_train).astype("float32") 
    X_test = np.asarray(X_train).astype("float32")
    y_test = np.asarray(y_train).astype("float32") 

    model = create_model(X_train, y_train, 42)

    results = model.evaluate(X_test, y_test, batch_size=128) 
    print("execution time: ", time.time() - start) 
    
    utils.cal_accuracy_neural_network(results)  

    #######Create model for testing the NSL-KDD dataset#######  

    X_train = pd.read_csv('../dataset/unsw-nb15/nsl-kdd-ver/train-set.csv') 
    
    model = create_model(X_train, y_train,5)  

    # Create a JSON file that saves the model's information. Ex. class name(Sequential), layers(relu and sigmoid) and so on.
    model_json = model.to_json()
    with open("./model/nsl-kdd-model.json", "w") as json_file:
        json_file.write(model_json)
    
    # Serialize weights to HDF5
    model.save_weights("./model/nsl-kdd-model.h5")
 
# Calling main function 
if __name__=="__main__": 
    main()

