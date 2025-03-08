# config.py
import argparse
import torch
import random
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--gpu',
                    type=int,
                    default=0,
                    help="GPU ID, -1 for CPU")
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    help="seed")
parser.add_argument('--repeat', type=int, default=1, help='repeat index')
meta_args = parser.parse_args("")

# # DEVICE = torch.device("cpu")
# print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
# disable_progress_bar()
meta_args.device = torch.device('cuda:{}'.format(meta_args.gpu) if torch.cuda.is_available() and meta_args.gpu != -1 else 'cpu')

meta_args.TR_enabled = False
meta_args.save_dir = "data_loaders"
meta_args.log_path = "results/MLP_fed_avg_WTR"

meta_args.model = "mlp"

meta_args.round = 100  # 50
meta_args.epoch_iterations = 20
# meta_args.batch_size = 64
meta_args.batch_size = 150

# meta_args.local_lr = 0.001 # with min lr 0.001
# meta_args.min_local_lr = 0.001
# meta_args.decay_weight = 1.0
# meta_args.data_type = ""
# meta_args.epoch_iterations = 20

# meta_args.local_lr = 0.003 
# meta_args.decay_weight = 0.85
# meta_args.epoch_iterations = 20

# meta_args.local_lr = 0.003 # with min lr 0.001
# meta_args.min_local_lr = 0.001
# meta_args.decay_weight = 0.85
# meta_args.data_type = ""
# meta_args.epoch_iterations = 20

# meta_args.local_lr = 0.003 # with min lr 0.001
# meta_args.min_local_lr = 0.0001 
# meta_args.data_type = ""
# meta_args.epoch_iterations = 20


# meta_args.local_lr = 0.003 # with simple adam no betas  gets 81% at round 40 and gets ~89% but train loss also start increasing ar round 38
# meta_args.min_local_lr = 1e-08
# meta_args.decay_weight = 0.85
# meta_args.data_type = ""
# meta_args.epoch_iterations = 20

# meta_args.local_lr = 0.003 # with simple adam no betas 
# meta_args.min_local_lr = 0.001
# meta_args.decay_weight = 0.85
# meta_args.data_type = ""
# meta_args.epoch_iterations = 20

# meta_args.local_lr = 0.003 # with simple adam no betas 
# meta_args.min_local_lr = 0.0001
# meta_args.decay_weight = 0.85
# meta_args.data_type = ""
# meta_args.epoch_iterations = 20

meta_args.local_lr = 0.003 # with simple adam no betas gets 81% at round 29 and gets ~87% but train lossalso much lower than above case
meta_args.min_local_lr = 1e-05
meta_args.decay_weight = 0.85
meta_args.data_type = ""
meta_args.epoch_iterations = 20

# meta_args.local_lr = 0.003 # with simple adam no betas gets 81% at round ?? and gets ~87% but train lossalso much lower than above case
# meta_args.min_local_lr = 1e-05
# meta_args.decay_weight = 0.88
# meta_args.data_type = ""
# meta_args.epoch_iterations = 20

# meta_args.local_lr = 0.002 # with simple adam no betas 81% at round 31 and gets ~87% but train lossalso much lower than above case
# meta_args.min_local_lr = 1e-05
# meta_args.decay_weight = 0.9
# meta_args.data_type = ""
# meta_args.epoch_iterations = 20

# *************************************************************** TR Settings  *******************************************************************
# meta_args.TR_enabled = True 
# meta_args.save_dir = "data_loaders_TR"
# meta_args.log_path = "results/MLP_fed_avg_TR"

# meta_args.local_lr = 0.001 # with simple adam no betas gets 81% at round 29 and gets ~87% but train lossalso much lower than above case
# meta_args.min_local_lr = 1e-08
# meta_args.decay_weight = 0.6
# meta_args.data_type = ""
# meta_args.epoch_iterations = 1

# meta_args.local_lr = 0.003 # with simple adam no betas gets 81% at round 29 and gets ~87% but train lossalso much lower than above case
# meta_args.min_local_lr = 1e-08
# meta_args.decay_weight = 0.85
# meta_args.data_type = ""
# meta_args.epoch_iterations = 1


meta_args.remove_labels = [17, 21, 25, 29]
meta_args.features = ['sender_cwnd_rate', 'sender_avg_rtt_value', 'sender_retrans', 'sender_segs_in', 'sender_tcp_snd_buffer_max',
                      'sender_nic_send_bytes', 'sender_nic_receive_bytes',
                      'receiver_seg_out', 'receiver_tcp_rcv_buffer_max', 'receiver_nic_send_bytes',
                      'receiver_nic_receive_bytes', 'sender_remote_ost_read_bytes', 'receiver_remote_ost_write_bytes']
meta_args.filenames = {
    "wisconsin_ssd_merged": "./ds/v3/selected_cols_merged/wisconsin-220g2-10Gbps_ssd_merged_V3.csv",
    "wisconsin_hdd_merged": "./ds/v3/selected_cols_merged/wisconsin-220g2-10Gbps_hdd_merged_V3.csv",
    "wisconsin_hdd_ssd_merged": "./ds/v3/selected_cols_merged/wisconsin-220g2-hdd-ssd_merged_V3.csv",
    "wisconsin_ssd_delay_10ms_merged": "./ds/v3/selected_cols_merged/wisconsin-220g2-ssd-delayed-10ms_merged_V3.csv",
    
    "wisconsin_hdd_delay_10ms_merged":"./ds/v3/selected_cols_merged/wisconsin-220g2-hdd-delayed-10ms_merged_V3.csv",
    "utah_ssd_merged": "./ds/v3/selected_cols_merged/utah-6525-25g-25Gbps_ssd_merged.csv",
    "utah_ssd_delay_30ms_merged":"./ds/v3/selected_cols_merged/utah-6525-25-ssd-delayed-30ms_merged_V3.csv",
    "utah_ssd_delay_10ms_merged":"./ds/v3/selected_cols_merged/utah-6525-25-ssd-delayed-10ms_merged_V3.csv",

}

meta_args.input_size = len(meta_args.features)
meta_args.output_size = 9
args = meta_args



def set_global_seed(seed):
    """Set the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)