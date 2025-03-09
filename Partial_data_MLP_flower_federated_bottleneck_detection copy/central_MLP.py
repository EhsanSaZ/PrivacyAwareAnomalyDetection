import copy
import torch
import numpy as np 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torch import nn, optim

from experiment_config import set_global_seed, args
from model import MLPClassifier_torch

set_global_seed(args.seed)
print(f"Training on {args.device}")

from task import summarize_dataloader, process_and_prepare_loaders


def central_train(net, ldr_train, epochs: int, device, verbose=False, local_lr=0.001):
    # optimizer = optim.Adam(net.parameters(), lr=local_lr, betas=(0.9, 0.999), eps=1e-08)
    epochs_losses = []
    net.train()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=local_lr)
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for _, (batch_X, labels) in enumerate(ldr_train):
            batch_X, labels = batch_X.to(device), labels.to(device)
            net.zero_grad()
            # optimizer.zero_grad()
            log_probs = net.forward(batch_X)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epochs_losses.append(loss.item())
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(log_probs.data, 1)[1] == labels).sum().item()
        if verbose and epoch % 10 == 0:
            epoch_acc = correct / total
            epoch_loss /= len(ldr_train.dataset)
            print(f"\tEpoch {epoch}: train loss {epoch_loss}, accuracy {epoch_acc}")
    w_new = copy.deepcopy(net.state_dict())
    return w_new, net, sum(epochs_losses) / len(epochs_losses)


def central_test(net, ldr_test, device, complete_results=False):

    net = copy.deepcopy(net).to(device)
    loss_func = nn.CrossEntropyLoss()
    net.eval()
    correct, total, test_loss = 0, 0, 0.0
    
    all_preds, all_targets = [], []

    with torch.no_grad():
        for index, (data, target) in enumerate(ldr_test):
             data, target = data.to(device), target.to(device)
             log_probs = net.forward(data)
             test_loss += loss_func(log_probs, target).item()
             _, predicted = torch.max(log_probs, -1) # TODO CHECK FOR GET -1 pr 1 is correct
             
             total += target.size(0)
             correct += predicted.eq(target).sum()
             all_preds.extend(predicted.cpu().numpy())
             all_targets.extend(target.cpu().numpy())
    test_loss /= len(ldr_test.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted') 
    results = { 
            "f1_score": f1,
            "accuracy": accuracy,          
        }
    if complete_results: 
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        # report = classification_report(all_targets, all_preds)
        # confusion = confusion_matrix(all_targets, all_preds)
        results.update({
            "precision": precision,
            "recall": recall,
            # "report": report,
            # "confusion_matrix": confusion
        })
    
    return test_loss, results

if __name__ == "__main__":
    args.repeat = 3
    losses, accuracies, precision_scores, recall_scores, f1_scores = [], [], [], [], []
    for r in range(args.repeat):
        print(f"Run {r + 1}/{args.repeat}")

        args.seed = args.seed + r + 42
        args.round = 1
        set_global_seed(args.seed)
        _, _, global_test_loader, global_train_loader, total_classes, args = process_and_prepare_loaders(args, remove_labels=args.remove_labels, features=args.features, filenames=args.filenames, save_dir=args.save_dir)

            # Reset the model and other variables if necessary
        clf = MLPClassifier_torch(input_size=args.input_size, output_size=args.output_size, hidden_layer_sizes=(250,)).to(args.device)
        lr = args.local_lr
        for r in range(args.round):
            lr = lr * args.decay_weight if args.decay_weight < 1.0 and lr > args.min_local_lr else args.min_local_lr
            print(f"Round: {r}, learning rate {lr}")
            _, _, loss = central_train(clf, global_train_loader, epochs=args.epoch_iterations, device=args.device, verbose=False, local_lr=lr)
            loss, results = central_test(clf, global_test_loader, device=args.device, complete_results=True)
            accuracy = results["accuracy"]
            precision = results["precision"]
            recall = results["recall"]
            f1 = results["f1_score"]
            print(f"Round {r}: loss {loss}, accuracy {accuracy}, precision {precision}, recall {recall}, f1 {f1}")
        
            accuracies.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            losses.append(loss)

    # Print per run values and average values
    for run in range(args.repeat):
        print(f"Run {run + 1}: loss {losses[run]}, accuracy {accuracies[run]}, f1_score {f1_scores[run]}")

    print(f"Average: loss {np.mean(losses)}, accuracy {np.mean(accuracies)}, precision {np.mean(precision_scores)}, recall {np.mean(recall_scores)}, f1_score {np.mean(f1_scores)}")