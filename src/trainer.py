import os

import numpy as np

import torch
import torch.nn as nn

from .data import get_loader
from .model import MLP


class Trainer:
    def __init__(self, model_weight=None):
        """Intialize class variables."""
        self.model = MLP(input_dim=12, hidden_dim=64)
        if model_weight is not None:
            self.model.load_state_dict(torch.load(model_weight))
        self.model.cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0003, weight_decay=0.0001)
        _, self.train_loader = get_loader("data/seoul_grand_park/train.csv",
                                          batch_size=128,
                                          shuffle=True)
        _, self.test_loader = get_loader("data/seoul_grand_park/test.csv", 
                                         batch_size=256, 
                                         shuffle=False)
        self.best_rmse = 100000

    def train(self):
        """Run main training loop."""
        for epoch_idx in range(2000):
            print(f"\nEpoch {epoch_idx + 1}")
            self.train_epoch()
            flag = self.evaluate()
            if flag:
                self.save()

    def train_epoch(self):
        """Train model an epoch."""
        self.model.train()
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = torch.tensor(list(zip(*x))).float()
            y_pred = self.model(x.cuda())
            cost = torch.mean(torch.abs(y.cuda().unsqueeze(1) - y_pred))  # Mean absolute error
            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()
        print(f"Batch {batch_idx + 1}/{len(self.train_loader)} | Cost: {cost.item()}")

    def evaluate(self):
        """Evaluate model by test data."""
        flag = False
        self.model.eval()
        y_list = list()
        y_pred_list = list()
        for x, y in self.test_loader:
            with torch.no_grad():
                x = torch.tensor(list(zip(*x))).float()
                y_pred = self.model(x.cuda()).unsqueeze(1).cpu().numpy()
            y_list.extend(y)
            y_pred_list.extend(y_pred)
        mae = np.mean(np.abs(np.array(y_list) - np.array(y_pred_list)))
        rmse = np.sqrt(np.mean((np.array(y_list) - np.array(y_pred_list))**2))
        if rmse < self.best_rmse:
            self.best_rmse = rmse
            flag = True
        print(f"Evaluation || MAE: {mae:.2f} | RMSE: {rmse:.2f} | Best RMSE: {self.best_rmse:.2f}")
        return flag

    def save(self):
        if not os.path.exists("results/weights"):
            os.makedirs("results/weights")
        torch.save(self.model.state_dict(), "results/weights/model_weight.pt")
        print("Saved best model!")
