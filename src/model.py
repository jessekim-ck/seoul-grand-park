import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=20):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(0.1)
        self.layer4 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        _x = nn.ReLU()(self.bn1(self.layer1(x)))
        x = nn.ReLU()(self.bn2(self.layer2(_x)))
        x = nn.ReLU()(self.bn3(self.layer3(x))) + _x
        # x = self.dropout(x)
        x = self.layer4(x)
        return x
