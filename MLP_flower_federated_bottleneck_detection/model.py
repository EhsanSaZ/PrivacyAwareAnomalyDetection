import torch
import torch.nn as nn

class MLPClassifier_torch(nn.Module):
    def __init__(self, input_size, output_size=2, hidden_layer_sizes=(100,),
                 learning_rate=0.001, max_iter=200, tol=1e-4, random_state=None):
        super(MLPClassifier_torch, self).__init__()

        if random_state is not None:
            torch.manual_seed(random_state)

        # Create the network architecture
        layers = []
        prev_size = input_size
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        # layers.append(nn.Softmax(dim=1))  # Softmax for multi-class classification

        self.model = nn.Sequential(*layers)
        # self.learning_rate = learning_rate
        # self.max_iter = max_iter
        # self.tol = tol
        self.optimizer = None
        # self.criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class log loss
        self.criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class log loss

    def forward(self, x):
        return self.model(x)