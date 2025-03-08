import xgboost as xgb
import json, copy

from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedXgbBagging
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from flwr.common import (
    Scalar,
    Parameters,
    Context,
)

from torch.utils.tensorboard import SummaryWriter

from task import SingletonDataLoader, set_log_path, replace_keys



def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (mlogloss) for evaluation."""
    examples = [num_examples for num_examples, _ in eval_metrics]
    # Multiply accuracy of each client by number of examples used
    mlogloss_aggregated = [num_examples * m["mlogloss"] for num_examples, m in eval_metrics]
    accuracies = [num_examples * m["accuracy"] for num_examples, m in eval_metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in eval_metrics]
    precision_scores = [num_examples * m["precision"] for num_examples, m in eval_metrics]
    recall_scores = [num_examples * m["recall"] for num_examples, m in eval_metrics]
    metrics_aggregated = {"agg_eval_mlogloss": sum(mlogloss_aggregated)/sum(examples), "agg_eval_accuracy": sum(accuracies) / sum(examples),
                           "agg_eval_f1_score": sum(f1_scores) / sum(examples), "agg_eval_precision": sum(precision_scores) / sum(examples), 
                           "agg_eval_recall": sum(recall_scores) / sum(examples)}
    return metrics_aggregated

def config_func(rnd: int) -> dict[str, str]:
    """Return a configuration with global epochs."""
    return {"global_round": str(rnd)}


def get_evaluate_fn(test_data, params):
    """Return a function for centralised evaluation."""

    def evaluate_fn(
        server_round: int, parameters: Parameters, config: dict[str, Scalar]
    ):
        # If at the first round, skip the evaluation
        if server_round == 0:
            return 0, {"accuracy": 0, "mlogloss_value": 0, "f1_score": 0, "precision": 0, "recall": 0}

        else:
            bst = xgb.Booster(params=params)
            for para in parameters.tensors:
                para_b = bytearray(para)

            # Load global model
            bst.load_model(para_b)
            # Run evaluation

            preds = bst.predict(test_data)
            labels = test_data.get_label()
            accuracy = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="weighted")
            precision = precision_score(labels, preds, average="weighted", zero_division=0)
            recall = recall_score(labels, preds, average="weighted", zero_division=0)


            eval_results = bst.eval_set(
                evals=[(test_data, "valid")],
                iteration=bst.num_boosted_rounds() - 1,
            )
                    # parse out the mlogloss
            items = eval_results.split("\t")
            # We expect something like ["[0]", "valid-mlogloss:0.5"]
            mlogloss_value = 0.0
            for it in items:
                if "valid-mlogloss:" in it:
                    mlogloss_value = float(it.split(":")[1])
                    break
            result = {"accuracy": accuracy, "mlogloss_value": mlogloss_value, "f1_score": f1, "precision": precision, "recall": recall}
            return mlogloss_value, result
    return evaluate_fn

class FedXgbBaggingCustom(FedXgbBagging):
    def __init__(self, meta_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set_global_seed(meta_args.seed)
        print("{:<50}".format("-" * 15 + " log path " + "-" * 50)[0:60])
        log_path = set_log_path(meta_args)
        print(log_path)
        self.writer = SummaryWriter(log_path)
        args_dict = vars(copy.deepcopy(meta_args))
        del args_dict["device"]
        json_string = json.dumps(args_dict, indent=4).replace("\n", "<br>").replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
        self.writer.add_text("Experiment Details", json_string)
        
    def evaluate(self, server_round: int, parameters: Parameters):
        loss, metrics = super().evaluate(server_round, parameters)

        print(f"Round {server_round} - Evaluation: {metrics}")
        self.writer.add_scalar("Server_Test_Accuracy", metrics["accuracy"], server_round)
        self.writer.add_scalar("Server_Test_Precision", metrics["precision"], server_round)
        self.writer.add_scalar("Server_Test_Recall", metrics["recall"], server_round)
        self.writer.add_scalar("Server_Test_F1_Score", metrics["f1_score"], server_round)
        self.writer.add_scalar("Server_Test_Loss", loss, server_round)

def get_server_app(args, num_rounds=5):    

    def server_fn(context: Context):
        # Read from config
        params = replace_keys(args.params)
        # Optionally set initial global model (here, empty)
        parameters = Parameters(tensor_type="", tensors=[])
        loader = SingletonDataLoader.get_instance()
        _, _, global_test_loader, total_classes, filenames = loader.get_data_loaders(remove_labels=args.remove_labels,  features=args.features, filenames=args.filenames, seed=args.seed)

        # Define strategy
        strategy = FedXgbBaggingCustom(
            meta_args=args,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            initial_parameters=parameters,
            evaluate_function=get_evaluate_fn(
                global_test_loader, params
            ), 
        )
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)

    server_app = ServerApp(server_fn=server_fn)
    return server_app