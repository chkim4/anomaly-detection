# Preprocessing of the UNSW-NB15 and the NSL-KDD dataset
# 
# The UNSW-NB15 dataset includes its train and test dataset but some values of them are string. 
# So, they need to be converted into numbers. Also, they are mixed with features and labels. 
# So, they need to be divided into four files:    
# 'dataset/unsw-nb15/unsw-nb15-train-test/test-set.csv', 'dataset/unsw-nb15/unsw-nb15-train-test/test-set-label.csv', 
# 'dataset/unsw-nb15/unsw-nb15-train-test/train-set.csv' and 'dataset/unsw-nb15/unsw-nb15-train-test/train-set-label.csv'  
# Since the NSL-KDD dataset has the same issues with the UNSW-NB15 dataset and it has only 5 features that are compatible with those of the UNSW-NB15,
# it needs to be extracted some featues which are: duration, protocol_type, service, src_bytes, and dst_bytes.  
# Also, it needs to be divided into two files which are: 
# 'dataset/nsl-kdd/test-set.csv' and 'dataset/nsl-kdd/test-set-label.csv'

import pandas as pd
def main(): 

    ######### Preprocessing for UNSW-NB15 #########
    
    # Load the raw train dataset and test dataset 
    raw_data_train = pd.read_csv('./dataset/unsw-nb15/raw-data/train-set-raw.csv')  
    raw_data_test = pd.read_csv('./dataset/unsw-nb15/raw-data/test-set-raw.csv')

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

    # Save X_train and X_test as csv file.
    X_raw_data_train.to_csv('./dataset/unsw-nb15/unsw-nb15-train-test/train-set.csv', index=False) 
    X_raw_data_test.to_csv('./dataset/unsw-nb15/unsw-nb15-train-test/test-set.csv', index=False) 
    
    ######### Preprocessing for labels ######### 
    raw_data_train_label = pd.DataFrame({'label': raw_data_train['label']})
    raw_data_test_label = pd.DataFrame({'label': raw_data_test['label']}) 
    
    # Save y_train and y_test (labels) as csv file.
    raw_data_train_label.to_csv('./dataset/unsw-nb15/unsw-nb15-train-test/train-set-label.csv', index=False)
    raw_data_test_label.to_csv('./dataset/unsw-nb15/unsw-nb15-train-test/test-set-label.csv', index=False)    

    ######### Preprocessing for NSL-KDD #########
    
    # Load the dataset
    nsl_kdd = pd.read_csv('./dataset/nsl-kdd/raw-data/KDDTest-21.csv')
    
    # Extract features that are compatible with the UNSW-NB15
    nsl_kdd_test = nsl_kdd.iloc[:,1:7] 
    nsl_kdd_test.drop(nsl_kdd_test.columns[[3]], axis=1, inplace=True) 
    
    # Change the character values into index value of the list 
    # The NSL-KDD includes services(nsl_kdd_test.iat[index, 2]) that are not included in the UNSW-NB15. 
    # So, they are changed into the length of the 'l_service'  
    for index, row in nsl_kdd_test.iterrows(): 
        nsl_kdd_test.iat[index, 1] = l_proto.index(nsl_kdd_test.iat[index, 1]) 
        
        try:
            nsl_kdd_test.iat[index, 2] = l_service.index(nsl_kdd_test.iat[index, 2]) 
        except ValueError:
            nsl_kdd_test.iat[index, 2] = len(l_service) 
            continue 
    
    # Since UNSW-NB15 and NSL-KDD use different label names, convert those of NSL-KDD.
    nsl_kdd_test.rename(columns={'duration': 'dur', 'protocol_type':'proto', 
        'src_bytes': 'sbytes', 'dst_bytes': 'dbytes'}, inplace = True)

    # Extract labels to build y_test 
    nsl_kdd_test_labels = pd.DataFrame({'label': nsl_kdd['label']})

    # Since the original dataset's label is either 'normal' or 'anomaly', 
    # They need to be converted into 1(anomaly) and 0(normal).
    for index, row in nsl_kdd_test_labels.iterrows():     
        if(nsl_kdd_test_labels.iat[index, 0] == 'anomaly'):
            nsl_kdd_test_labels.iat[index, 0] = 1 
        else: 
            nsl_kdd_test_labels.iat[index, 0] = 0 
    
    # Save X_test as a csv file.
    nsl_kdd_test.to_csv('./dataset/nsl-kdd/test-set.csv', index=False)  
    # Save y_test as a csv file.
    nsl_kdd_test_labels.to_csv('./dataset/nsl-kdd/test-set-label.csv', index=False)
    
if __name__=="__main__": 
    main() 