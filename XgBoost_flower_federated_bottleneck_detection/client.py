import xgboost as xgb
from flwr.client import Client, ClientApp
from flwr.common import FitIns, FitRes, EvaluateIns, EvaluateRes, Parameters, Status, Code
from flwr.common.config import unflatten_dict
from task import replace_keys, load_datasets, SingletonDataLoader

from sklearn.metrics import f1_score, accuracy_score

class FlowerClient(Client):
    def __init__(self, train_dmatrix, valid_dmatrix, num_local_rounds, params):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_local_rounds = num_local_rounds
        self.params = params

    def _local_boost(self, bst_input):
        # Update trees based on local training data.
        for _ in range(self.num_local_rounds):
            bst_input.update(
                self.train_dmatrix, bst_input.num_boosted_rounds()
            )
        # Bagging: extract the last N=num_local_rounds trees for server aggregation
        bst = bst_input[
            bst_input.num_boosted_rounds() - self.num_local_rounds : bst_input.num_boosted_rounds()
        ]
        return bst
    
    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])

        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_rounds,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            bst = xgb.Booster(params=self.params)
            global_model = bytearray(ins.parameters.tensors[0])

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.train_dmatrix.num_row(),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        preds = bst.predict(self.valid_dmatrix)
        # preds = preds.argmax(axis=1)
        labels = self.valid_dmatrix.get_label()
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")

        # Run evaluation at the final iteration
        eval_result = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        print(eval_result)
        # eval_result might look like: "[round]\tvalid-mlogloss:0.xxx"
        # parse out the mlogloss
        items = eval_result.split("\t")
        # We expect something like ["[0]", "valid-mlogloss:0.5"]
        mlogloss_value = 0.0
        for it in items:
            if "valid-mlogloss:" in it:
                mlogloss_value = float(it.split(":")[1])
                break

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=mlogloss_value,
            num_examples=self.valid_dmatrix.num_row(),
            metrics={"accuracy": accuracy, "f1_score": f1, "mlogloss": mlogloss_value},
        )

def get_client_app(client_run_config):
    def client_fn(context):
        # Use SingletonDataLoader to get data loaders
        # data_loader = SingletonDataLoader.get_instance()
        # print(data_loader)
        # print(client_run_config)
        args = client_run_config["args"]
        loader = SingletonDataLoader.get_instance()
        clients_data_loaders, client_test_loaders, _, total_classes, filenames = loader.get_data_loaders(remove_labels=args.remove_labels,  features=args.features, filenames=args.filenames, seed=args.seed)

        partition_id = context.node_config["partition-id"]
                
        num_local_rounds = client_run_config["local-epochs"]
            
        params = replace_keys(client_run_config["params"])
        params["num_class"] = total_classes
        train_dmatrix, valid_dmatrix, _ = load_datasets(partition_id=partition_id, 
                                                        data_loaders=(clients_data_loaders, client_test_loaders),
                                                        filename_list=filenames)
        return FlowerClient(train_dmatrix, valid_dmatrix, num_local_rounds, params)

    client_app = ClientApp(client_fn)
    return client_app