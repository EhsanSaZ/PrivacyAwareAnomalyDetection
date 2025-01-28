from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

import os

import numpy as np
import pandas as pd
import xgboost as xgb


def process_and_prepare_dmatrix(args, remove_labels=None, features=None, filenames=None):
    if remove_labels is None:
        remove_labels = [17, 21, 25, 29]
    if features is None:
        features = ['sender_avg_rtt_value', 'sender_retrans', 'sender_segs_in', 'sender_tcp_snd_buffer_max','sender_nic_send_bytes', 'sender_nic_receive_bytes',
                    'receiver_seg_out', 'receiver_tcp_rcv_buffer_max', 'receiver_nic_send_bytes', 'receiver_nic_receive_bytes', 'sender_remote_ost_read_bytes', 'receiver_remote_ost_write_bytes']
    if filenames is None:
        filenames = {
            "wisconsin_ssd_merged": "./ds/v3/selected_cols_merged/wisconsin-220g2-10Gbps_ssd_merged_V3.csv",
            # "wisconsin_ssd_unmerged": "./ds/v3/selected_cols/wisconsin-220g2-10Gbps_ssd_unmerged_V3.csv",

            "wisconsin_hdd_merged": "./ds/v3/selected_cols_merged/wisconsin-220g2-10Gbps_hdd_merged_V3.csv",
            # "wisconsin_hdd_unmerged": "./ds/v3/selected_cols/wisconsin-220g2-10Gbps_hdd_unmerged_V3.csv",
        }

    clients_data_loaders = {}
    client_test_loaders = {}
    combined_X_test, combined_y_test = [], []
    test_data_dict = {}

    for client_name, file_path in filenames.items():
        # Step 1: Load the dataset and Label encoding and scaling
        df = pd.read_csv(file_path)
        
        # Step 2: Remove specified labels
        for lbl in remove_labels:
            df = df.drop(df[df.label_value == lbl].index)
        
        # Normalize for transfer learning 
        # df = normalize_df(df)

        X = df.drop(columns="label_value")[features]
        y = df.label_value

        encoder = LabelEncoder()
        scaler = StandardScaler()
        
        y = encoder.fit_transform(y)
        # X = scaler.fit_transform(X)

        # Step 3: Split into train and test sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=args.seed)
        
       
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        # Step 4: Apply oversampling to training data
        X_train, y_train = RandomOverSampler(sampling_strategy="all").fit_resample(X_train, y_train)

        X_train = X_train.to_numpy() if not isinstance(X_train, np.ndarray) else X_train
        X_test = X_test.to_numpy() if not isinstance(X_test, np.ndarray) else X_test
        y_train = y_train.to_numpy() if not isinstance(y_train, np.ndarray) else y_train
        y_test = y_test.to_numpy() if not isinstance(y_test, np.ndarray) else y_test

        clients_data_loaders[client_name] = xgb.DMatrix(X_train, label=y_train)

        # Combine test data for unified test dataset
        combined_X_test.append(X_test)
        combined_y_test.append(y_test)

        client_test_loaders[client_name] = xgb.DMatrix(X_test, label=y_test)

        
    # Combine all test data
    combined_X_test = np.vstack(combined_X_test)
    combined_y_test = np.hstack(combined_y_test)
    total_classes = len(np.unique(combined_y_test))

    global_test_loader = xgb.DMatrix(combined_X_test, label=combined_y_test)


    args.input_size = len(features)
    args.output_size = total_classes

    return clients_data_loaders, client_test_loaders, global_test_loader, total_classes, args 


def set_log_path(args):
    import datetime
    path =  './log/' + args.log_path+ '/'
    if not os.path.exists(path):
        os.makedirs(path)
    path_log = os.path.join(path)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return path_log + '_' + str(timestamp)

def summarize_dataset(dataset):
    print("=== Dataset Summary ===")
    # Dataset length
    dataset_size = dataset.num_row()
    print(f"Total samples: {dataset_size}")
    
    # Inspect the data
    data_shape = dataset.feature_names
    print(f"Data shape: {data_shape}")
    
    # Inspect the labels
    labels = dataset.get_label()
    labels_shape = labels.shape
    labels_type = labels.dtype
    print(f"Labels shape: {labels_shape}, Labels type: {labels_type}")
    
    print("===========================")


def load_datasets(partition_id: int, data_loaders, args):
    clients_data_loaders, client_test_loaders = data_loaders

    client_name = list(args.filenames.keys())[int(partition_id)]
    
    trainloader = clients_data_loaders[client_name]
    testloader = client_test_loaders[client_name]
    valloader = testloader

    return trainloader, valloader, testloader


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict