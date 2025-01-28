from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedXgbBagging

from flwr.common import (
    Code,
    FitRes,
    FitIns,
    EvaluateIns,
    EvaluateRes,
    Parameters,
    Scalar,
    Status,
)

def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (mlogloss) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    # Weighted average of mlogloss by number of examples
    mlogloss_aggregated = sum([
        metrics["mlogloss"] * num for num, metrics in eval_metrics
    ]) / total_num if total_num > 0 else 0.0

    metrics_aggregated = {"mlogloss": mlogloss_aggregated}
    return metrics_aggregated


def config_func(rnd: int) -> dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def create_server(global_test_loader, args, run_config):


    def server_fn(context: Context):
        # Read from config
        num_rounds = run_config["num-server-rounds"]
        fraction_fit = run_config["fraction-fit"]
        fraction_evaluate = run_config["fraction-evaluate"]

        # Optionally set initial global model (here, empty)
        parameters = Parameters(tensor_type="", tensors=[])

        # Define strategy
        strategy = FedXgbBagging(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            initial_parameters=parameters,
        )
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)

    # Build the ServerApp
    server_app = ServerApp(server_fn=server_fn)

    return server_app