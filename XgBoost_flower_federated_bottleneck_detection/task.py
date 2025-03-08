
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from experiment_config import args


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



class SingletonDataLoader:
    _instance = None

    def __init__(self):
        if SingletonDataLoader._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.data_loaders = None
            SingletonDataLoader._instance = self

    @staticmethod
    def get_instance():
        if SingletonDataLoader._instance is None:
            SingletonDataLoader._instance = SingletonDataLoader()
        return SingletonDataLoader._instance

    def get_data_loaders(self, remove_labels=args.remove_labels,  features=args.features, filenames=args.filenames, seed=args.seed, save_dir=args.save_dir):
        if self.data_loaders is None:
            self.data_loaders = self._create_data_loaders(remove_labels, features, filenames, seed, save_dir)
            
        return self.data_loaders

    def _create_data_loaders(self, remove_labels, features, filenames, seed, save_dir):
        
        print("=" * 25 + " CREATING DATA LOADERS " + "=" * 25)
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Filename for saving dataset
        dataset_file = os.path.join(save_dir, f"data_loaders_seed_{seed}.npz")
        # Create a unique dataset name using all filenames
        # dataset_name = "_".join(filenames.keys())
        dataset_name = "all_testbeds"
        dataset_file = os.path.join(save_dir, f"data_loaders_{dataset_name}_seed_{seed}.npz")

        if os.path.exists(dataset_file):
            print("Loading saved dataset...")
            # Load saved dataset
            dataset = np.load(dataset_file, allow_pickle=True)

            clients_data_loaders = {
                client: xgb.DMatrix(dataset["train"].item()[client]['data'], 
                                    label=dataset["train"].item()[client]['label'])
                for client in dataset['train'].item()
            }

            client_test_loaders = {
                client: xgb.DMatrix(dataset["test"].item()[client]['data'], 
                                    label=dataset["test"].item()[client]['label'])
                for client in dataset['test'].item()
            }

            # Recreate combined test data from client_test_loaders
            conbined_X_train = np.vstack([dataset['train'].item()[client]['data'] for client in dataset['train'].item()])
            combined_y_train = np.hstack([dataset['train'].item()[client]['label'] for client in dataset['train'].item()])
            global_train_loader = xgb.DMatrix(combined_X_train, label=combined_y_train)

            combined_X_test = np.vstack([dataset['test'].item()[client]['data'] for client in dataset['test'].item()])
            combined_y_test = np.hstack([dataset['test'].item()[client]['label'] for client in dataset['test'].item()])
            global_test_loader = xgb.DMatrix(combined_X_test, label=combined_y_test)

            total_classes = len(np.unique(combined_y_test))

            return clients_data_loaders, client_test_loaders, global_test_loader, global_train_loader, total_classes, list(filenames.keys())

        print("Saved dataset not found. Processing and creating dataset...")
        
        clients_data_loaders = {}
        client_test_loaders = {}

        for client_name, file_path in filenames.items():
            df = pd.read_csv(file_path)

            # Remove specified labels
            for lbl in remove_labels:
                df = df.drop(df[df.label_value == lbl].index)

            if args.TR_enabled:
                df = normalize_df(df)

            X = df[features]
            y = df.label_value

            encoder = LabelEncoder()
            scaler = StandardScaler()

            y = encoder.fit_transform(y)
            # X = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

            # Apply oversampling to training data
            X_train, y_train = RandomOverSampler(sampling_strategy="all").fit_resample(X_train, y_train)

            # Convert data to numpy arrays
            X_train = X_train.to_numpy() if not isinstance(X_train, np.ndarray) else X_train
            X_test = X_test.to_numpy() if not isinstance(X_test, np.ndarray) else X_test
            y_train = y_train.to_numpy() if not isinstance(y_train, np.ndarray) else y_train
            y_test = y_test.to_numpy() if not isinstance(y_test, np.ndarray) else y_test

            clients_data_loaders[client_name] = {
                'data': X_train,
                'label': y_train
            }
            client_test_loaders[client_name] = {
                'data': X_test,
                'label': y_test
            }

        # Save dataset
        print("Saving dataset...")
        np.savez_compressed(
            dataset_file,
            train=clients_data_loaders,
            test=client_test_loaders
        )

        # Recreate combined test data
        combined_X_train = np.vstack([clients_data_loaders[client]['data'] for client in clients_data_loaders])
        combined_y_train = np.hstack([clients_data_loaders[client]['label'] for client in clients_data_loaders])
        combined_X_test = np.vstack([client_test_loaders[client]['data'] for client in client_test_loaders])
        combined_y_test = np.hstack([client_test_loaders[client]['label'] for client in client_test_loaders])

        global_train_loader = xgb.DMatrix(combined_X_train, label=combined_y_train)
        global_test_loader = xgb.DMatrix(combined_X_test, label=combined_y_test)

        total_classes = len(np.unique(combined_y_test))

        # Convert to DMatrix for return
        clients_data_loaders = {
            client: xgb.DMatrix(clients_data_loaders[client]['data'], label=clients_data_loaders[client]['label'])
            for client in clients_data_loaders
        }

        client_test_loaders = {
            client: xgb.DMatrix(client_test_loaders[client]['data'], label=client_test_loaders[client]['label'])
            for client in client_test_loaders
        }

        return clients_data_loaders, client_test_loaders, global_test_loader, global_train_loader, total_classes, list(clients_data_loaders.keys())


def set_log_path(args):
    import datetime
    path =  './log/' + args.log_path+ '/'
    if not os.path.exists(path):
        os.makedirs(path)
    path_log = os.path.join(path)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return path_log + '_' + str(timestamp)


def load_datasets(partition_id: int, data_loaders, filename_list):
    clients_data_loaders, client_test_loaders = data_loaders

    client_name = filename_list[int(partition_id)]
    
    trainloader = clients_data_loaders[client_name]
    testloader = client_test_loaders[client_name]
    valloader = testloader

    return trainloader, valloader, testloader


def summarize_dataset(dataset):
    """Summarize the dataset details."""
    print("=== Dataset Summary ===")
    dataset_size = dataset.num_row()
    print(f"Total samples: {dataset_size}")
    labels = dataset.get_label()
    print(f"Labels: {len(labels)}, Type: {labels.dtype}")
    print("==========================")

def replace_keys(input_dict, match="-", target="_"):
    """Replace specific characters in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict