
import flwr
from flwr.client import Client, ClientApp
from flwr.common import Metrics, Context
from flwr.common.typing import UserConfig
from flwr.common.config import unflatten_dict

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

from task import load_datasets, replace_keys

class FlowerClient(Client):
    def __init__(self, train_dmatrix, valid_dmatrix, num_train, num_val, 
                    num_local_round, params, partition_id):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params
        self.p_id = partition_id
    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
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
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        # auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
        
        # eval_result might look like: "[round]\tvalid-mlogloss:0.xxx"
        # parse out the mlogloss
        items = eval_results.split("\t")
        # We expect something like ["[0]", "valid-mlogloss:0.5"]
        mlogloss_value = 0.0
        for it in items:
            if "valid-mlogloss:" in it:
                mlogloss_value = float(it.split(":")[1])
                break

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=mlogloss_value,
            num_examples=self.num_val,
            metrics={"mlogloss": mlogloss_value},
        )
        


def create_client(args, data_loaders, run_config: UserConfig) -> ClientApp:
    
    def client_fn(context: Context) -> Client:
        # Load model and data
        partition_id = context.node_config["partition-id"]
        # num_partitions = context.node_config["num-partitions"]
        # train_dmatrix, valid_dmatrix, num_train, num_val = load_datasets(
        #     partition_id, num_partitions
        # )
        train_dmatrix, valid_dmatrix, test_dmatrix = load_datasets(partition_id=partition_id, 
                                                                   data_loaders=data_loaders,
                                                                   args=args)

        cfg = replace_keys(unflatten_dict(run_config))
        num_local_round = cfg["local_epochs"]

        # Return Client instance
        return FlowerClient(
            train_dmatrix,
            valid_dmatrix,
            train_dmatrix.num_row(),
            valid_dmatrix.num_row(),
            num_local_round,
            cfg["params"],
        )
    # Create the ClientApp
    client = ClientApp(client_fn=client_fn)

    return client