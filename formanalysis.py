import numpy as np
import os
import re
from pathlib import Path
from random import sample
import pandas as pd
from dtaidistance import dtw

def count_rows(filepath):
    """
    Count the number of rows in a CSV file.
    Returns: Integer count of the number of rows
    """
    res = 0
    f = open(filepath, 'r')
    res = len(f.readlines())
    f.close()
    return res

def test_training_data(input_train_data, input_train_labels):
    permuted_idx = np.random.permutation(input_train_data.shape[0])
    train_data = input_train_data[permuted_idx[0:23]]
    train_labels = input_train_labels[permuted_idx[0:23]]
    test_data = input_train_data[permuted_idx[23:]]
    test_labels = input_train_labels[permuted_idx[23:]]
    #Ensure that test set has each type of class
    while not all(x in test_labels for x in [0,1,2,3]):
        permuted_idx = np.random.permutation(input_train_data.shape[0])
        train_data = input_train_data[permuted_idx[0:23]]
        train_labels = input_train_labels[permuted_idx[0:23]]
        test_data = input_train_data[permuted_idx[23:]]
        test_labels = input_train_labels[permuted_idx[23:]]
    return train_data, train_labels, test_data, test_labels

def form_to_num(exercise:str,form:str):
    """
    Used to convert the form file path name to integer form id. Takes in exercise name and form name.
    Returns: Integer of form
    """
    if exercise == "BP":
        if form == "Good":
            return 0
        elif form == "Elbow_flare":
            return 1
        elif form == "Locking_elbows":
            return 2
        elif form == "Arms_too_low":
            return 3

def sample_testing_data(exercise:str):
    # GOOD, GOOD, GOOD, LOW, LOCK, LOW, FLARE, FLARE
    df_list = []
    src_path = 'reference/testdata/' + exercise + '/'
    regex = re.compile('.*accel.*\.csv$')
    pathlist = []
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if regex.match(file):
                pathlist.append(root+file)
    for path in pathlist:
        input_df = pd.read_csv(path)
        df_list.append(input_df)
    return df_list

def get_max_row_count(input_dfs):
    """
    Determines the largest number of rows in a set of dataframes.
    Returns: Integer value of the most rows in any of the dataframes.
    """
    max_count = 0
    for df in input_dfs:
        df_count = len(df.index)
        if df_count > max_count:
            max_count = df_count
    return max_count

def read_training_data(exercise:str, max_count:int):
    """
    Reads in training data for a given exercise.
    Returns: two numpy arrays, one for training data and one for labels
    """
    input_data_list = []
    label_list = []
    max_c = max_count
    src_path = 'reference/traindata/' + exercise + '/'
    regex = re.compile('.*accel.*\.csv$')
    subfolders = []
    for path in Path(src_path).iterdir():
        if path.is_dir():
            subfolders.append(str(path) + '/')
    print(subfolders)
    # Find max cell count in training data CSVs
    for folder in subfolders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if regex.match(file):
                    count = count_rows(str(root+file))
                    if count > max_c:
                        max_c = count
    print(max_c)
    # Iterate through each form type in the respective exercise folder
    for folder in subfolders:
        pathlist = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if regex.match(file):
                    pathlist.append(root+file)
        form_name = folder.split("/")[-2]
        form_index = form_to_num(exercise,form_name)

        # Iterate through each CSV file
        for path in pathlist:
            input_df = pd.read_csv(path)
            input_df['time'] = input_df['time'].str[14:].astype(float)
            df_length = input_df.shape[0]
            # Padding
            for i in range(max_c-input_df.shape[0]):
                new_time = (df_length + i)/100
                input_df = input_df.append({'time':new_time,'x':0,'y':0,'z':0}, ignore_index=True)
            input_data_list.append(input_df.to_numpy())
            label_list.append(form_index)
    train_data = np.array(input_data_list)
    train_labels = np.array(label_list)
    return train_data, train_labels

def read_test_data(data : pd.DataFrame, max_c : int):
    """
    Reads in a single dataframe of testing data and a maximum cell count and performs padding.
    Returns: One Pandas dataframe containing padded test data
    """
    #data['time'] = data['time'].str[14:].astype(float)
    df_length = data.shape[0]
    for i in range(max_c-data.shape[0]):
        new_time = (df_length + i)/100
        data = data.append({'time':new_time,'x':0,'y':0,'z':0}, ignore_index=True)
    return data

def test_dfs_to_list(df_list):
    """
    Merges a list of test dfs into a single list.
    Returns: One numpy array containing test data for each rep
    """
    input_data_list = []
    for df in df_list:
        input_data_list.append(df.to_numpy())
    return np.array(input_data_list)

def classifyNN(k:int, idx:int, train_set, train_lbls, test_set) -> str:
    idxs=range(0,train_set.shape[0])
    n=train_set.shape[0]
    distances_x=[]
    distances_y=[]
    distances_z=[]
    counters={}
    c=1
    max_value=0
    for r in range(n):
        # X
        distances_x.append(dtw.distance(test_set[idx][:,1], train_set[idxs[r]][:,1],window=15,use_pruning=True))
        NN_x=sorted(range(len(distances_x)), key=lambda i: distances_x[i], reverse=False)[:k]
        
        # Y
        distances_y.append(dtw.distance(test_set[idx][:,2], train_set[idxs[r]][:,2],window=15,use_pruning=True))
        NN_y=sorted(range(len(distances_y)), key=lambda i: distances_y[i], reverse=False)[:k]

        # Z
        distances_z.append(dtw.distance(test_set[idx][:,3], train_set[idxs[r]][:,3],window=15,use_pruning=True))
        NN_z=sorted(range(len(distances_z)), key=lambda i: distances_z[i], reverse=False)[:k]
    for l in list(set(train_lbls)):
        counters[l]=0
    for r in NN_x:
        l=train_lbls[r]
        counters[l]+=1
        if (counters[l])>max_value:
            max_value=counters[l]
        c+=1
    for r in NN_y:
        l=train_lbls[r]
        counters[l]+=1
        if (counters[l])>max_value:
            max_value=counters[l]
        c+=1
    for r in NN_z:
        l=train_lbls[r]
        counters[l]+=1
        if (counters[l])>max_value:
            max_value=counters[l]
        c+=1
    
    # find the label(s) with the highest frequency
    keys = [k for k in counters if counters[k] == max_value]

    # in case of a tie, return one at random
    output = (sample(keys,1)[0])
    return output

def compute_confusion_matrix(true, pred):
    """
    Function for generating confusion matrix
    """
    K = len(np.unique(true)) # Number of classes
    result = np.zeros((K, K))
    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
    return result

def analyze_train_data():
    train_data, train_labels = read_training_data('BP', 0)
    train_data_split, train_labels_split, test_data_split, test_labels_split = test_training_data(train_data, train_labels)
    true_labels = []
    pred_labels = []
    for idx,val in enumerate(test_labels_split):
        rep_class = classifyNN(4,idx,train_data_split,train_labels_split,test_data_split)
        pred_labels.append(rep_class)
    print(pred_labels)
    for idx,val in enumerate(test_labels_split):
        true_labels.append(test_labels_split[idx])
    conf_matrix = compute_confusion_matrix(true_labels,pred_labels)
    print(conf_matrix)

def form_int_to_str(form:int):
    if form == 0:
        return "good"
    elif form == 1:
        return "arm flair"
    elif form == 2:
        return "locking elbows"
    elif form == 3:
        return "arms too low"

def analyze_test_data(df_list):
    max_c = get_max_row_count(df_list)
    print(max_c)
    train_data, train_labels = read_training_data('BP', max_c)
    print(train_labels)
    df_list_padded = []
    for df in df_list:
        df_list_padded.append(read_test_data(df,max_c))
    test_data = test_dfs_to_list(df_list_padded)
    outputs = []
    for rep in range(test_data.shape[0]):
        rep_class = form_int_to_str(classifyNN(4,rep,train_data,train_labels,test_data))
        outputs.append(rep_class)
    return outputs

# df_list = sample_testing_data('BP')
# max_c = get_max_row_count(df_list)
# train_data, train_labels = read_training_data('BP', max_c)
# df_list_padded = []
# for df in df_list:
#     df_list_padded.append(read_test_data(df,max_c))
# test_data = test_dfs_to_list(df_list_padded)
# outputs = []
# for rep in range(test_data.shape[0]):
#     rep_class = classifyNN(4,rep,train_data,train_labels,test_data)
#     outputs.append(rep_class)
# print(outputs)
