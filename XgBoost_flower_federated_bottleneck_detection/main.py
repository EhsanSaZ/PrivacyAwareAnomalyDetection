from flwr.simulation import run_simulation

from experiment_config import set_global_seed, args
from task import SingletonDataLoader

set_global_seed(args.seed)
print(f"Training on {args.device}")

from task import summarize_dataset
from client import get_client_app
from server import get_server_app


if __name__ == "__main__":
    args.repeat = 1
    for r in range(args.repeat):
        # args.seed = args.seed + r + 42
        args.seed = args.seed + r + 42 
        set_global_seed(args.seed)

        loader = SingletonDataLoader.get_instance()
        clients_data_loaders, client_test_loaders, _ , total_classes, filenames = loader.get_data_loaders(remove_labels=args.remove_labels,  features=args.features, filenames=args.filenames, seed=args.seed)
        # print(clients_data_loaders, "\n")
        # print(client_test_loaders, "\n")
        # summarize_dataset(client_test_loaders["wisconsin_ssd_merged"])
        summarize_dataset(client_test_loaders["wisconsin_hdd_merged"])

        # server_app = get_server_app(args, num_rounds=args.round)
        server_app = get_server_app(args, num_rounds=10)
        client_app = get_client_app(args)
        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=8,
            backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
        )