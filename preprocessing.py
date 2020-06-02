# The original datasets ('./dataset/raw_data/train-set-raw.csv' and './dataset/raw_data/test-set-raw.csv')
# are mixed with features and labels. So, they are needed to be divided into four files:    
# 'dataset/test-set.csv', 'dataset/test-set-label.csv', 'dataset/train-set.csv' and 'dataset/train-set-label.csv' 
# Some values in the train dataset and test dataset are characters that cannot be trained by models. 
# So, converting them into number is required (preprocessing)  
# After preprocessing, save them as csv files.

import pandas as pd
def main(): 
    ######### Preprocessing for features #########
    # Load the raw train dataset and test dataset 
    raw_data_train = pd.read_csv('./dataset/raw_data/train-set-raw.csv')  
    raw_data_test = pd.read_csv('./dataset/raw_data/test-set-raw.csv')

    # Column[0] of the train and test dataset are id, a meaningless sequence number.  
    # also, column[44] of the both datasets is the status information ex. normal, Exploits, etc. 
    # So, both columns are excluded in the new data set.   
    X_raw_data_train = raw_data_train.iloc[:,1:43]   
    X_raw_data_test = raw_data_test.iloc[:,1:43]
    
    # Collect every kinds of 'proto', 'service' and 'state' which includes string data from train dataset without duplication 
    s_proto = set(X_raw_data_train['proto']) 
    s_service = set(X_raw_data_train['service'])
    s_state = set(X_raw_data_train['state']) 
    # There are some values that only in the 'test-set-raw.csv'
    # So, the threee sets are needed to be updated. 
    s_proto.update(X_raw_data_test['proto'])
    s_service.update(X_raw_data_test['service'])
    s_state.update(X_raw_data_test['state'])
   
    # Convert the sets into the list. So that each value can be converted into its index
    l_proto = list(s_proto)
    l_service = list(s_service)
    l_state = list(s_state)

    # Change the character values in the train dataset into index value of the list
    for index, row in X_raw_data_train.iterrows():
        X_raw_data_train.iat[index, 1] = l_proto.index(X_raw_data_train.iat[index, 1])
        X_raw_data_train.iat[index, 2] = l_service.index(X_raw_data_train.iat[index, 2]) 
        X_raw_data_train.iat[index, 3] = l_state.index(X_raw_data_train.iat[index, 3])  
    
    # Change the character values in the test dataset into index value of the list
    for index, row in X_raw_data_test.iterrows():
        X_raw_data_test.iat[index, 1] = l_proto.index(X_raw_data_test.iat[index, 1])
        X_raw_data_test.iat[index, 2] = l_service.index(X_raw_data_test.iat[index, 2]) 
        X_raw_data_test.iat[index, 3] = l_state.index(X_raw_data_test.iat[index, 3])

    # Save them as csv file.
    
    X_raw_data_train.to_csv('./dataset/train-set.csv', index=False) 
    X_raw_data_test.to_csv('./dataset/test-set.csv', index=False) 
    
    ######### Preprocessing for labels ######### 
    raw_data_train_labels = pd.DataFrame({'label': raw_data_train['label']})
    raw_data_test_labels = pd.DataFrame({'label': raw_data_test['label']})
    
    raw_data_train_labels.to_csv('./dataset/train-set-label.csv', index=False)
    raw_data_test_labels.to_csv('./dataset/test-set-label.csv', index=False)

if __name__=="__main__": 
    main() 