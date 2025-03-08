
import copy, os

import numpy as np
import torch, random
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.tensorboard import SummaryWriter

import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from collections import Counter

from imblearn.over_sampling import RandomOverSampler


def normalize_df(df):
    df['sender_avg_rtt_value'] = df['sender_avg_rtt_value'] / df[df.label_value == 0].sender_avg_rtt_value.mean()
    df['sender_retrans'] = df['sender_retrans'] / df[df.label_value == 0].sender_seg_out.mean()
    # df["sender_avg_send_value"] = df["sender_avg_send_value"] / df[df.label_value == 0].sender_avg_send_value.mean()
    df["sender_segs_in"] = df["sender_segs_in"] / df[df.label_value == 0].sender_segs_in.mean()

    # df["sender_ost_read"] = df["sender_ost_read"] / df[df.label_value == 0].sender_ost_read.mean()

    df["sender_nic_send_bytes"] = df["sender_nic_send_bytes"] / df[df.label_value == 0].sender_nic_send_bytes.mean()
    df["sender_nic_receive_bytes"] = df["sender_nic_receive_bytes"] / df[df.label_value == 0].sender_nic_receive_bytes.mean()

    df["sender_remote_ost_read_bytes"] =df["sender_remote_ost_read_bytes"] / df[df.label_value == 0].sender_remote_ost_read_bytes.mean()

    # df["receiver_segs_in"] = df["receiver_segs_in"] / df[df.label_value == 0].receiver_segs_in.mean()
    df["receiver_seg_out"] = df["receiver_seg_out"] / df[df.label_value == 0].receiver_seg_out.mean()

    # df["receiver_write_bytes"] = df["receiver_write_bytes"] / df[df.label_value == 0].receiver_write_bytes.mean()
    # df["receiver_ost_write"] = df["receiver_ost_write"] / df[df.label_value == 0].receiver_ost_write.mean()

    df["receiver_nic_send_bytes"] = df["receiver_nic_send_bytes"] / df[df.label_value == 0].receiver_nic_send_bytes.mean()
    df["receiver_nic_receive_bytes"] = df["receiver_nic_receive_bytes"] / df[df.label_value == 0].receiver_nic_receive_bytes.mean()

    df["receiver_remote_ost_write_bytes"] = df["receiver_remote_ost_write_bytes"] / df[df.label_value == 0].receiver_remote_ost_write_bytes.mean()

    df["sender_tcp_snd_buffer_max"] = df["sender_tcp_snd_buffer_max"] / df[df.label_value == 0].sender_tcp_snd_buffer_max.mean()
    df["receiver_tcp_rcv_buffer_max"] = df["receiver_tcp_rcv_buffer_max"] / df[df.label_value == 0].receiver_tcp_rcv_buffer_max.mean()

    # df["sender_write_bytes_io"] = df["sender_write_bytes_io"] / df[df.label_value == 0].sender_write_bytes_io.mean()
    # df["sender_read_bytes_io"] = df["sender_read_bytes_io"] / df[df.label_value == 0].sender_read_bytes_io.mean()
    #
    # df["receiver_read_bytes_io"] = df["receiver_read_bytes_io"] / df[df.label_value == 0].receiver_read_bytes_io.mean()
    # df["receiver_write_bytes_io"] = df["receiver_write_bytes_io"] / df[df.label_value == 0].receiver_write_bytes_io.mean()

    #---------------
    # df["sender_ssthresh_value"] = df.sender_ssthresh_value / df.sender_cwnd_rate
    # df["sender_req_active"] = df["sender_req_active"] / df[df.label_value == 0].sender_req_active.mean()
    df["sender_cwnd_rate"] = df["sender_cwnd_rate"] / df[df.label_value == 0].sender_cwnd_rate.mean()
    return df

def process_and_prepare_loaders(args, remove_labels=None, features=None, filenames=None, save_dir="data_loaders"):
    print("=" * 25 + " CREATING DATA LOADERS " + "=" * 25)
    if remove_labels is None:
        remove_labels = [17, 21, 25, 29]
    if features is None:
        features = ['sender_cwnd_rate', 'sender_avg_rtt_value', 'sender_retrans', 'sender_segs_in', 'sender_tcp_snd_buffer_max','sender_nic_send_bytes', 'sender_nic_receive_bytes',
                    'receiver_seg_out', 'receiver_tcp_rcv_buffer_max', 'receiver_nic_send_bytes', 'receiver_nic_receive_bytes', 'sender_remote_ost_read_bytes', 'receiver_remote_ost_write_bytes']
    if filenames is None:
        filenames = {
            "wisconsin_ssd_merged": "../ds/v3/selected_cols_merged/wisconsin-220g2-10Gbps_ssd_merged_V3.csv",
            # "wisconsin_ssd_unmerged": "./ds/v3/selected_cols/wisconsin-220g2-10Gbps_ssd_unmerged_V3.csv",

            "wisconsin_hdd_merged": "../ds/v3/selected_cols_merged/wisconsin-220g2-10Gbps_hdd_merged_V3.csv",
            # "wisconsin_hdd_unmerged": "./ds/v3/selected_cols/wisconsin-220g2-10Gbps_hdd_unmerged_V3.csv",
        }

    os.makedirs(save_dir, exist_ok=True)
    dataset_name = "all_testbeds"
    dataset_file = os.path.join(save_dir, f"data_loaders_{dataset_name}_seed_{args.seed}.npz")
    if os.path.exists(dataset_file):
        print("Loading saved dataset...")
        dataset = np.load(dataset_file, allow_pickle=True)
        clients_data_loaders, client_test_loaders = {}, {}
        combined_X_train, combined_y_train = [], []

        for client in dataset['train'].item():
            train_data = dataset['train'].item()[client]['data']
            train_label = dataset['train'].item()[client]['label']
            test_data = dataset['test'].item()[client]['data']
            test_label = dataset['test'].item()[client]['label']

            train_dataset = TensorDataset(
                torch.tensor(train_data, dtype=torch.float32),
                torch.tensor(train_label, dtype=torch.long)
            )
            test_dataset = TensorDataset(
                torch.tensor(test_data, dtype=torch.float32),
                torch.tensor(test_label, dtype=torch.long)
            )
            clients_data_loaders[client] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
            client_test_loaders[client] = DataLoader(test_dataset, batch_size=args.batch_size)
            combined_X_train.append(train_data)
            combined_y_train.append(train_label)

        combined_X_train = np.vstack(combined_X_train)
        combined_y_train = np.hstack(combined_y_train)
        global_train_loader = DataLoader(TensorDataset(torch.tensor(combined_X_train, dtype=torch.float32),
                                                    torch.tensor(combined_y_train, dtype=torch.long))
                                        , batch_size=args.batch_size, shuffle=True)

        combined_X_test = np.vstack([dataset['test'].item()[client]['data'] for client in dataset['test'].item()])
        combined_y_test = np.hstack([dataset['test'].item()[client]['label'] for client in dataset['test'].item()])
        global_test_loader = DataLoader(TensorDataset(torch.tensor(combined_X_test, dtype=torch.float32),
                                                    torch.tensor(combined_y_test, dtype=torch.long))
                                        , batch_size=args.batch_size, shuffle=False)
        total_classes = len(np.unique(combined_y_test))
        args.input_size = len(features)
        args.output_size = total_classes
        return clients_data_loaders, client_test_loaders, global_test_loader, global_train_loader, total_classes, args
    
    print("Saved dataset not found. Processing and creating dataset...")
    clients_data, test_data = {}, {}
    combined_X_test, combined_y_test = [], []
    for client_name, file_path in filenames.items():
        # Step 1: Load the dataset and Label encoding and scaling
        df = pd.read_csv(file_path)
        
        # Step 2: Remove specified labels
        for lbl in remove_labels:
            df = df.drop(df[df.label_value == lbl].index)
        
        # Normalize for transfer learning 
        if args.TR_enabled:
            df = normalize_df(df)

        X = df.drop(columns="label_value")[features]
        y = df.label_value

        encoder = LabelEncoder()
        scaler = StandardScaler()
        
        y = encoder.fit_transform(y)
        # X = scaler.fit_transform(X)

        # Step 3: Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=args.seed)
        
       
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        # Step 4: Apply oversampling to training data
        # X_train, y_train = RandomOverSampler(sampling_strategy="all", random_state=args.seed).fit_resample(X_train, y_train)
        X_train, y_train = RandomOverSampler(sampling_strategy="all").fit_resample(X_train, y_train)

        X_train = X_train.to_numpy() if not isinstance(X_train, np.ndarray) else X_train
        X_test = X_test.to_numpy() if not isinstance(X_test, np.ndarray) else X_test
        y_train = y_train.to_numpy() if not isinstance(y_train, np.ndarray) else y_train
        y_test = y_test.to_numpy() if not isinstance(y_test, np.ndarray) else y_test

        clients_data[client_name] = {
                'data': X_train,
                'label': y_train
        }
        test_data[client_name] = {
                'data': X_test,
                'label': y_test
        }
    # Save dataset
    print("Saving dataset...")
    np.savez_compressed(
        dataset_file,
        train=clients_data,
        test=test_data
    )

    # Recreate combined test data
    combined_X_test = np.vstack([test_data[client]['data'] for client in clients_data])
    combined_y_test = np.hstack([test_data[client]['label'] for client in test_data])

    clients_data_loaders = {}
    client_test_loaders = {}

    for client in clients_data:
        train_dataset = TensorDataset(
            torch.tensor(clients_data[client]['data'], dtype=torch.float32),
            torch.tensor(clients_data[client]['label'], dtype=torch.long)
        )
        ldr_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        clients_data_loaders[client] = ldr_train

        test_dataset = TensorDataset(
            torch.tensor(test_data[client]['data'], dtype=torch.float32),
            torch.tensor(test_data[client]['label'], dtype=torch.long)
        )
        client_test_loaders[client] = DataLoader(test_dataset, batch_size=args.batch_size)
        

    total_classes = len(np.unique(combined_y_test))
     # Create combined test DataLoader
    global_test_loader = DataLoader(TensorDataset(torch.tensor(combined_X_test, dtype=torch.float32),
                                                   torch.tensor(combined_y_test, dtype=torch.long))
                                    , batch_size=args.batch_size, shuffle=False)

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

def summarize_dataloader(dataloader):
    print("=== DataLoader Summary ===")
    # Dataset length
    dataset_size = len(dataloader.dataset)
    print(f"Total samples: {dataset_size}")
    
    # Batch size
    batch_size = dataloader.batch_size
    print(f"Batch size: {batch_size}")
    
    # Number of batches
    num_batches = len(dataloader)
    print(f"Number of batches: {num_batches}")
    
    # Inspect a single batch
    for i, batch in enumerate(dataloader):
        print(f"Inspecting Batch {i+1}:")
        if isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, (list, tuple)):
                    print(f"  {key}: List/Tuple of length {len(value)}")
                else:
                    print(f"  {key}: Shape {value.shape}, Type {value.dtype}")
        elif isinstance(batch, (list, tuple)):
            for idx, value in enumerate(batch):
                if isinstance(value, torch.Tensor):
                    print(f"  Element {idx}: Shape {value.shape}, Type {value.dtype}")
                else:
                    print(f"  Element {idx}: Type {type(value)}")
        else:
            print("  Batch is not a dict, list, or tuple. Unexpected format.")
        # Only inspect the first batch
        break

    print("===========================")


def load_datasets(partition_id: int, data_loaders: tuple[dict[str, DataLoader], dict[str, DataLoader]], args=None):
    clients_data_loaders, client_test_loaders = data_loaders
    client_name = list(args.filenames.keys())[int(partition_id)]
    
    trainloader = clients_data_loaders[client_name]
    testloader = client_test_loaders[client_name]
    valloader = testloader

    return trainloader, valloader, testloader


def train(net, ldr_train, epochs: int, device, verbose=False, local_lr=0.001):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=local_lr)
    # optimizer = optim.Adam(net.parameters(), lr=local_lr, betas=(0.9, 0.999), eps=1e-08)
    epochs_losses = []
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for _, (batch_X, labels) in enumerate(ldr_train):
            batch_X, labels = batch_X.to(device), labels.to(device)
            net.zero_grad()
            # optimizer.zero_grad()
            log_probs = net.forward(batch_X)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epochs_losses.append(loss.item())
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(log_probs.data, 1)[1] == labels).sum().item()
        if verbose:
            epoch_acc = correct / total
            epoch_loss /= len(ldr_train.dataset)
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    w_new = copy.deepcopy(net.state_dict())
    return w_new, sum(epochs_losses) / len(epochs_losses)


def test(net, ldr_test, device, complete_results=False):
    
    net = copy.deepcopy(net).to(device)
    loss_func = nn.CrossEntropyLoss()
    net.eval()
    correct, total, test_loss = 0, 0, 0.0
    
    all_preds, all_targets = [], []

    with torch.no_grad():
        for index, (data, target) in enumerate(ldr_test):
             data, target = data.to(device), target.to(device)
             log_probs = net.forward(data)
             test_loss += loss_func(log_probs, target).item()
             _, predicted = torch.max(log_probs, -1) # TODO CHECK FOR GET -1 pr 1 is correct
             
             total += target.size(0)
             correct += predicted.eq(target).sum()
             all_preds.extend(predicted.cpu().numpy())
             all_targets.extend(target.cpu().numpy())
    test_loss /= len(ldr_test.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted') 
    results = { 
            "f1_score": f1,
            "accuracy": accuracy,          
        }
    if complete_results: 
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        # report = classification_report(all_targets, all_preds)
        # confusion = confusion_matrix(all_targets, all_preds)
        results.update({
            "precision": precision,
            "recall": recall,
            # "report": report,
            # "confusion_matrix": confusion
        })
       
    
    return test_loss, results