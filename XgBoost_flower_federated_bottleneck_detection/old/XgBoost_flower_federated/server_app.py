from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedXgbBagging
from flwr.common import Context, Parameters

def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (mlogloss) for evaluation."""
    examples = [num_examples for num_examples, _ in eval_metrics]
    # Multiply accuracy of each client by number of examples used
    mlogloss_aggregated = [num_examples * m["mlogloss"] for num_examples, m in eval_metrics]
    accuracies = [num_examples * m["accuracy"] for num_examples, m in eval_metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in eval_metrics]
    metrics_aggregated = {"mlogloss": sum(mlogloss_aggregated)/sum(examples), "accuracy": sum(accuracies) / sum(examples), "f1_score": sum(f1_scores) / sum(examples)}
    return metrics_aggregated

def config_func(rnd: int) -> dict[str, str]:
    """Return a configuration with global epochs."""
    return {"global_round": str(rnd)}

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

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

app = ServerApp(server_fn=server_fn)
