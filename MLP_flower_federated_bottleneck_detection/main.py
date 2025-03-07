# from datasets.utils.logging import disable_progress_bar
from flwr.simulation import run_simulation
from experiment_config import set_global_seed, args

set_global_seed(args.seed)
print(f"Training on {args.device}")


from task import summarize_dataloader, process_and_prepare_loaders

from client import create_client
from server import create_server

if __name__ == "__main__":
    args.repeat = 3
    for r in range(args.repeat):
        # r = r + 1
        args.seed = args.seed + r + 42
        # args.seed = args.seed + r + 5
        set_global_seed(args.seed)
        clients_data_loaders, client_test_loaders, global_test_loader, total_classes, args = process_and_prepare_loaders(args, remove_labels=args.remove_labels, features=args.features, filenames=args.filenames, save_dir=args.save_dir)
        # print(clients_data_loaders, "\n")
        # print(client_test_loaders, "\n")
        summarize_dataloader(client_test_loaders["wisconsin_ssd_merged"])
        
        client = create_client(args=args, data_loaders=(clients_data_loaders, client_test_loaders))
        # server = create_server(global_test_loader=global_test_loader, args=args, num_rounds=args.round)
        # server = create_server(global_test_loader=global_test_loader, args=args, num_rounds=2) 
        server = create_server(global_test_loader=global_test_loader, args=args, num_rounds=100) 

        run_simulation(
            server_app=server,
            client_app=client,
            num_supernodes=8,
            backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
        )