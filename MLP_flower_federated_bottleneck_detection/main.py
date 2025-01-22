import torch
from datasets.utils.logging import disable_progress_bar
import flwr
from flwr.simulation import run_simulation


from task import summarize_dataloader, process_and_prepare_loaders

from config import args

from client import create_client
from server import create_server

print(f"Training on {args.device}")

clients_data_loaders, client_test_loaders, global_test_loader, total_classes, args = process_and_prepare_loaders(args, remove_labels=args.remove_labels, features=args.features, filenames=args.filenames)
# print(clients_data_loaders, "\n")
# print(client_test_loaders, "\n")
summarize_dataloader(client_test_loaders["wisconsin_ssd_merged"])


client = create_client(args=args)
server = create_server(global_test_loader=global_test_loader, args=args, num_rounds=args.round)

run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=8,
    backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.25}}
)