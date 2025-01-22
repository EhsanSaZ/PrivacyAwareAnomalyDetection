from collections import OrderedDict
from typing import List, Tuple, Optional, Union
import copy, os

# import matplotlib.pyplot as plt
# import numpy as np
import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from datasets.utils.logging import disable_progress_bar
# from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


# import flwr
# from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
# from flwr.simulation import run_simulation
# from flwr_datasets import FederatedDataset
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)


# import argparse
# import pandas as pd
# from sklearn.calibration import LabelEncoder
# from sklearn.discriminant_analysis import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score

# from imblearn.over_sampling import RandomOverSampler

from config import args
from model import MLPClassifier_torch
from task import test, set_log_path


def get_evaluate_fn(testloader):
    """Return a function that can be called to do global evaluation."""

    def evaluate_fn(server_round: int, parameters, config):
        """Evaluate global model on the whole test set."""

        model = MLPClassifier_torch(input_size=args.input_size, output_size=args.output_size, hidden_layer_sizes=(200,)).to(args.device)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # set parameters to the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # call test (evaluate model as in centralised setting)
        loss, accuracy, f1_score = test(model, testloader, args.device)
        # print(f"Round {server_round} - Evaluation: loss {loss}, accuracy {accuracy}, f1_score {f1_score}")
        return loss, {"accuracy": accuracy, "loss": loss, "f1_score": f1_score}

    return evaluate_fn

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # print("\nPrinting metrics in weighted_average function \n {} \n".format(metrics))

    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # print("\nPrinting metrics in fit_metrics_aggregation_fn function \n {} \n".format(metrics))

    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    train_loss = [m["train_local_loss"] for _, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, m in metrics]
    # return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples), "f1_score": sum(f1_scores) / sum(examples), "train_loss": sum(train_loss) / len(train_loss)}
    return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(train_loss) / len(train_loss), "f1_score": sum(f1_scores) / sum(examples), "train_loss": sum(train_loss) / len(train_loss)}


class FedAvgCustom(FedAvg):
    def __init__(self, meta_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Run simulation
        print("{:<50}".format("-" * 15 + " log path " + "-" * 50)[0:60])
        log_path = set_log_path(meta_args)
        print(log_path)
        self.writer = SummaryWriter(log_path)
    
    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]], failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],):
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        # print(f"Round {server_round} - Aggregated fit: {metrics_aggregated}")
        self.writer.add_scalar("train_loss", metrics_aggregated["train_loss"], server_round)
        return parameters_aggregated, metrics_aggregated

    def evaluate(self, server_round: int, parameters: Parameters):
        loss, metrics = super().evaluate(server_round, parameters)
        print(f"Round {server_round} - Evaluation: {metrics}")
        self.writer.add_scalar("test_accuracy", metrics["accuracy"], server_round)
        self.writer.add_scalar("test_loss", loss, server_round)
        self.writer.add_scalar("test_f1_score", metrics["f1_score"], server_round)


def create_server(global_test_loader, args, num_rounds=2):
    """Create and return the server instance."""
    strategy = FedAvgCustom(
        meta_args=args,
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=8,  # Never sample less than 10 clients for training
        min_evaluate_clients=8,  # Never sample less than 5 clients for evaluation
        min_available_clients=8,
        evaluate_metrics_aggregation_fn=weighted_average, # callback defined earlier
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,  # callback defined earlier
        evaluate_fn=get_evaluate_fn(
            global_test_loader, 
        ),  # Wait until all 3 clients are available
    )

    def server_fn(context: Context) -> ServerAppComponents:
        """Construct components that set the ServerApp behaviour.

        You can use the settings in `context.run_config` to parameterize the
        construction of all elements (e.g the strategy or the number of rounds)
        wrapped in the returned ServerAppComponents object.
        """

        # Configure the server for 5 rounds of training
        # config = ServerConfig(num_rounds=args.round)
        config = ServerConfig(num_rounds=num_rounds)
        
        return ServerAppComponents(strategy=strategy, config=config)

    # Create the ServerApp
    server = ServerApp(server_fn=server_fn)

    return server





