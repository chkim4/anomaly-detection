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
    raw_data_train = pd.read_csv('./dataset/unsw-nb15/raw-data/unsw-nb15-train-test/train-set-raw.csv')  
    raw_data_test = pd.read_csv('./dataset/unsw-nb15/raw-data/unsw-nb15-train-test/test-set-raw.csv')

    # Column[0] of the train and test dataset are id, a meaningless sequence number.  
    # also, column[44] of the both datasets is the status information ex. normal, Exploits, etc. 
    # So, both columns are excluded in the new data set.   
    X_raw_data_train = raw_data_train.iloc[:,1:43]   
    X_raw_data_test = raw_data_test.iloc[:,1:43]
    
    # Collect every kinds of 'proto', 'service' and 'state' which includes string data from train dataset without duplication 
    s_proto = set(X_raw_data_train['proto']) 
    s_service = set(X_raw_data_train['service'])
    s_state = set(X_raw_data_train['state'])
    #s_new_label = set(X_raw_data_train['attack_cat']) 

    # There are some values that only in the 'test-set-raw.csv'
    # So, the threee sets are needed to be updated. 
    s_proto.update(X_raw_data_test['proto'])
    s_service.update(X_raw_data_test['service'])
    s_state.update(X_raw_data_test['state'])
   
    # Convert the sets into the list. So that each value can be converted into its index
    l_proto = list(s_proto)
    l_service = list(s_service)
    l_state = list(s_state) 
    #l_new_label = list(s_new_label) 

    # Change the character values in the train dataset into index value of the list
    for index, row in X_raw_data_train.iterrows():
    #    X_raw_data_train.iat[index, 46] = l_new_label.index(X_raw_data_train.iat[index, 44])
        X_raw_data_train.iat[index, 1] = l_proto.index(X_raw_data_train.iat[index, 1])
        X_raw_data_train.iat[index, 2] = l_service.index(X_raw_data_train.iat[index, 2]) 
        X_raw_data_train.iat[index, 3] = l_state.index(X_raw_data_train.iat[index, 3])

    
    # Change the character values in the test dataset into index value of the list
    for index, row in X_raw_data_test.iterrows(): 
    #    X_raw_data_test.iat[index, 46] = l_new_label.index(X_raw_data_test.iat[index, 44])
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
    # Load nsl_kdd test dataset whose feature includes 'label'. 
    # So, I need to divide two files which are: 
    # test-set.csv: every features without label and 
    # test-set-label.csv: includes label only

    nsl_kdd = pd.read_csv('./dataset/nsl-kdd/raw-data/KDDTest-21.csv')
    nsl_kdd_test = nsl_kdd.iloc[:,1:7] 
    nsl_kdd_test.drop(nsl_kdd_test.columns[[3]], axis=1, inplace=True) 

    unsw_headers = list(X_raw_data_test.columns.values)
    
    # Change the character values in the test dataset into index value of the list
    for index, row in nsl_kdd_test.iterrows(): 

        nsl_kdd_test.iat[index, 1] = l_proto.index(nsl_kdd_test.iat[index, 1]) 
        
        try:
            nsl_kdd_test.iat[index, 2] = l_service.index(nsl_kdd_test.iat[index, 2])

        except ValueError:
            nsl_kdd_test.iat[index, 2] = len(l_service) 
            continue 
    
    nsl_kdd_test.rename(columns={'duration': 'dur', 'protocol_type':'proto', 
        'src_bytes': 'sbytes', 'dst_bytes': 'dbytes'}, inplace = True)

    nsl_kdd_headers = list(nsl_kdd_test) 
    index_num = 5

    for header in unsw_headers: 
        if(header not in nsl_kdd_headers):
            nsl_kdd_test.insert(index_num,header,-1) 
            index_num = index_num+1
        
        
    # Save X_test as a csv file.
    nsl_kdd_test.to_csv('./dataset/nsl-kdd/test-set.csv', index=False)  

    nsl_kdd_test_labels = pd.DataFrame({'label': nsl_kdd['label']})

    # Since the original dataset's label is either 'normal' or 'anomaly', it needs to be converted into 1(anomaly) and 0(normal).
    for index, row in nsl_kdd_test_labels.iterrows():     
        if(nsl_kdd_test_labels.iat[index, 0] == 'anomaly'):
            nsl_kdd_test_labels.iat[index, 0] = 1 
        else: 
            nsl_kdd_test_labels.iat[index, 0] = 0

    # Save y_test as a csv file.
    nsl_kdd_test_labels.to_csv('./dataset/nsl-kdd/test-set-label.csv', index=False)
    
if __name__=="__main__": 
    main() 