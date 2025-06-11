from library import feature_cols, activation_map
import torch.nn as nn

class FinalANN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.fc1 = nn.Linear(len(feature_cols), params['n_hidden1'])
        self.act1 = activation_map[params['activation']]
        self.dropout1 = nn.Dropout(params['dropout'])
        self.fc2 = nn.Linear(params['n_hidden1'], params['n_hidden2'])
        self.act2 = activation_map[params['activation']]
        self.dropout2 = nn.Dropout(params['dropout'])
        self.output = nn.Linear(params['n_hidden2'], 3)

    def forward(self, x):
        x = self.dropout1(self.act1(self.fc1(x)))
        x = self.dropout2(self.act2(self.fc2(x)))
        return self.output(x)
