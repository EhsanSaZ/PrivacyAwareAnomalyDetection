from collections import OrderedDict
from typing import List, Tuple, Optional, Union
import copy

# import matplotlib.pyplot as plt
import numpy as np
import torch

# import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
# from flwr.server import ServerApp, ServerConfig, ServerAppComponents
# from flwr.server.strategy import FedAvg
# from flwr.simulation import run_simulation
# from flwr_datasets import FederatedDataset
# from flwr.server.client_proxy import ClientProxy
# from flwr.common import (
#     FitRes,
#     Parameters,
#     Scalar,
# )


# import argparse
# import pandas as pd
# from sklearn.calibration import LabelEncoder
# from sklearn.discriminant_analysis import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score

# from imblearn.over_sampling import RandomOverSampler
from experiment_config import args, set_global_seed
from model import MLPClassifier_torch
from task import train, test, load_datasets


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, partition_id, args):
        self.p_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.args = args
        

    # def get_parameters(self, config):
    #     return get_parameters(self.net)
    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        # Exclude parameters of BN layers when using FedBN
        return [
            val.cpu().numpy()
            for name, val in self.net.state_dict().items()
            if "bn" not in name
        ]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        keys = [k for k in self.net.state_dict().keys() if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        # print(f"Client {self.p_id} starting fit with learning rate {config['lr']}")
        # self.set_parameters(self.net, parameters)
        self.set_parameters(parameters)
        _, train_loss = train(self.net, self.trainloader, epochs=args.epoch_iterations, device=args.device, verbose=False ,local_lr=config['lr'])
        loss, accuracy, f1_score = test(self.net, self.trainloader, device=args.device) 
        # return self.get_parameters(self.net), len(self.trainloader), {"loss": loss, "accuracy":  float(accuracy), "f1_score": f1_score, "train_local_loss": train_loss}
        return self.get_parameters(config), len(self.trainloader), {"loss": loss, "accuracy":  float(accuracy), "f1_score": f1_score, "train_local_loss": train_loss}


    def evaluate(self, parameters, config):
        # self.set_parameters(self.net, parameters)
        self.set_parameters(parameters)
        loss, accuracy, f1_score = test(self.net, self.valloader, args.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy), "loss": float(loss), "f1_score": float(f1_score)}
        # return float(loss), len(self.valloader), {"accuracy": float(accuracy), "loss": float(loss), "f1_score": float(f1_score)}
    
def create_client(args, data_loaders) -> ClientApp:
    
    def client_fn(context: Context) -> Client:
        """Create a Flower client representing a single organization."""
        # Load model
        net = MLPClassifier_torch(input_size=args.input_size, output_size=args.output_size, hidden_layer_sizes=(250,)).to(args.device)

        # Load data (CIFAR-10)
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data partition
        # Read the node_config to fetch data partition associated to this node
        partition_id = context.node_config["partition-id"]
        trainloader, valloader, _ = load_datasets(partition_id=partition_id, data_loaders=data_loaders, args=args)

        # Create a single Flower client representing a single organization
        # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
        # to convert it to a subclass of `flwr.client.Client`
        return FlowerClient(net, trainloader, valloader, partition_id, args).to_client()

    # Create the ClientApp
    client = ClientApp(client_fn=client_fn)

    return client
