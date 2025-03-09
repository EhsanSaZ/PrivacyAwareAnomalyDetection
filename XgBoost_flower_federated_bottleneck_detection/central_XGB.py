import xgboost as xgb
import numpy as np 

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from experiment_config import set_global_seed, args
from task import SingletonDataLoader, replace_keys

from xgboost import XGBClassifier

if __name__ == "__main__":
    args.repeat = 5
    losses, accuracies, precision_scores, recall_scores, f1_scores = [], [], [], [], []

    for r in range(args.repeat):
        
        # args.seed = args.seed + r + 42
        args.seed = args.seed + r + 42 
        set_global_seed(args.seed)
        loader = SingletonDataLoader.get_instance()
        loader.data_loaders = None

        _, _, test_dmatrix, train_dmatrix, total_classes, filenames = loader.get_data_loaders(remove_labels=args.remove_labels,  features=args.features, filenames=args.filenames, seed=args.seed)
        params = replace_keys(args.params)
        params["num_class"] = total_classes

        bst = xgb.train(params, train_dmatrix)

        preds = bst.predict(test_dmatrix)
        labels = test_dmatrix.get_label()

        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        precision = precision_score(labels, preds, average="weighted", zero_division=0)
        recall = recall_score(labels, preds, average="weighted", zero_division=0)

        eval_results = bst.eval_set(
                evals=[(test_dmatrix, "valid")],
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
        
        print(f"Round {r}: Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}, Mlogloss: {mlogloss_value}")
        
        accuracies.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        losses.append(mlogloss_value)
    # Print per run values and average values
    # for run in range(args.repeat):
    #     print(f"Run {run + 1}: loss {losses[run]}, accuracy {accuracies[run]}, f1_score {f1_scores[run]}, precision {precision_scores[run]}, recall {recall_scores[run]}")  
    
    print(f"Average: loss {np.mean(losses)}, accuracy {np.mean(accuracies)}, precision {np.mean(precision_scores)}, recall {np.mean(recall_scores)}, f1_score {np.mean(f1_scores)}")
        