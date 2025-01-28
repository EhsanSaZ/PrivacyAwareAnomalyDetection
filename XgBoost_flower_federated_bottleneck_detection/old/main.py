from flwr.simulation import run_simulation
import flwr as fl

from experiment_config import set_global_seed, args

set_global_seed(args.seed)
print(f"Training on {args.device}")

from task import summarize_dataset, process_and_prepare_dmatrix
from client import create_client
from server import create_server


if __name__ == "__main__":
    args.repeat = 1
    for r in range(args.repeat):
        args.seed = args.seed + r + 42
        set_global_seed(args.seed)

        clients_data_loaders, client_test_loaders, global_test_loader, total_classes, args = process_and_prepare_dmatrix(args, remove_labels=args.remove_labels, features=args.features, filenames=args.filenames)

        # print(clients_data_loaders, "\n")
        # print(client_test_loaders, "\n")
        summarize_dataset(client_test_loaders["wisconsin_ssd_merged"])


        # trainloader, valloader, testloader = load_datasets(partition_id=0, data_loaders=(clients_data_loaders, client_test_loaders), args=args)

        client_run_config = {
        "local-epochs": 1,
        "params": {
            "objective": "multi:softprob",
            # "objective": "multi:softmax",
            "num_class": args.output_size,
            "eta": 0.1,
            "max-depth": 8,
            "eval_metric": "mlogloss",
            # "eval_metric": "auc",
            "nthread": 16,
            "num-parallel-tree": 1,
            "subsample": 1,
            "tree-method": "hist"
        }
        }

        client = create_client(args=args, data_loaders=(clients_data_loaders, client_test_loaders), run_config=client_run_config)


        server_run_config = {
            "num-server-rounds": 3,
            "fraction-fit": 0.1,
            "fraction-evaluate": 0.1
        }

        server = create_server(global_test_loader=global_test_loader, args=args, run_config=server_run_config)


        run_simulation(
            server_app=server,
            client_app=client,
            num_supernodes=2,
            backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
        )