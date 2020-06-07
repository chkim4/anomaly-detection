# Anomaly-detection with neural network and performance test 

import pandas as pd
import numpy as np
from keras import models, layers, optimizers, losses, metrics

def main():  
    
    # Load training & testing datasets 
    X_train = pd.read_csv('../dataset/train-set.csv')  
    y_train = pd.read_csv('../dataset/train-set-label.csv') 
    X_test = pd.read_csv('../dataset/test-set.csv') 
    y_test = pd.read_csv('../dataset/test-set-label.csv')

    # Change x and y to float type data
    X_train = np.asarray(X_train).astype("float32")
    y_train = np.asarray(y_train).astype("float32") 
    X_test = np.asarray(X_train).astype("float32")
    y_test = np.asarray(y_train).astype("float32")

    # Initialize model
    model = models.Sequential()
    # Add layers to model(Output size, activation function, input size)
    model.add(layers.Dense(64, activation="relu", input_shape=(500,)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    
    # Set method of learning and evaluation (Gradient descent, error function, evaluation indicator)
    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

    # Start learning (Input, output, number of trial, size of input at once
    model.fit(X_train, y_train, epochs=20, batch_size=1024) 

    # Evaluate model with test dataset
    result = model.evaluate(X_test, y_test)
    test_accuracy = result[1] * 100
    print("Accuracy is {:.2f}%".format(test_accuracy))

# Calling main function 
if __name__=="__main__": 
    main()