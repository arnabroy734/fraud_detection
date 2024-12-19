import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from models.model_architecture import FraudClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
from torch.optim.lr_scheduler import StepLR
from time import time

class ScoringNet(nn.Module):
    def __init__(self, input_size):
        '''2 hidden layers, 64 and 32 neurons and output is a real number'''
        super(ScoringNet, self).__init__()
        self.l1 = nn.Linear(input_size,64) # 64 units
        self.l1_ac = nn.ReLU()
        self.l2 = nn.Linear(64 , 32) # 32 Units
        self.l2_ac = nn.ReLU()
        self.l3 = nn.Linear(32, 16) # 16 Units
        self.l3_ac = nn.ReLU()
        self.l4 = nn.Linear(16, 1)
    def forward(self,x):
        out = self.l1(x)
        out = self.l1_ac(out)
        out = self.l2(out)
        out = self.l2_ac(out)
        out = self.l3(out)
        out = self.l3_ac(out)
        out = self.l4(out)
        return out
    
class DeviationLoss(nn.Module):
    def __init__(self):
       super(DeviationLoss, self).__init__()
    def forward(self,y_pred,y_true):
        """
        1. Randomly sample l=5000 N(0,1) points. Calculate mu and sigma
        2. dev_i = (y_pred[i] - mu)/sigma
        3. cost_i = (1-y_true_i)*|dev_i| + y_true_i*max(0, a-dev_i), here a = 3
        4. Loss = sum of cost_i
        """
        prior_scores = np.random.normal(size=5000)
        mu, sigma = np.mean(prior_scores), np.std(prior_scores)
        dev = (y_pred - mu)/sigma
        zeros = torch.tensor(np.zeros(dev.shape)).type(torch.float32).to('cuda')
        cost = (1-y_true)*torch.abs(dev) + y_true*torch.max(zeros,3-dev)
        loss = torch.mean(cost)
        return loss

class CustomDataset(Dataset):
    def __init__(self, X_train, y_train):
        mask_0 = np.where(y_train==0)
        mask_1 = np.where(y_train==1)
        X_0, X_1, y_0, y_1 = X_train[mask_0], X_train[mask_1], y_train[mask_0], y_train[mask_1]
        self.X_0 = torch.tensor(X_0).type(torch.float32)
        self.X_1 = torch.tensor(X_1).type(torch.float32)
        self.y_0 = torch.tensor(y_0.reshape((-1,1))).type(torch.float32)
        self.y_1 = torch.tensor(y_1.reshape((-1,1))).type(torch.float32)
        self.n_samples = min(self.X_1.shape[0], self.X_0.shape[0])
    def __len__(self):
        return self.n_samples
    def __getitem__(self, index):
        return (self.X_0[index], self.y_0[index],self.X_1[index], self.y_1[index])

class FraudClassifierDEV(FraudClassifier):
    def __init__(self):
        self.name = "DEV"
        self.id = int(time())
        self.model = None
        self.recall = None # score on test data
        self.precision = None
        self.f1 = None
        self.accuracy = None
    
    def train(self,X_train,y_train):
        sm = SMOTE(random_state=100)
        # X_train_os, y_train_os = sm.fit_resample(X_train, y_train)
        X_train_os, y_train_os = X_train, y_train
        device_gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        batch_size = 256
        input_size = X_train_os.shape[1]
        lr = 0.003
        model = ScoringNet(input_size)
        model.to(device_gpu)
        loss_fn = DeviationLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        # scheduler = StepLR(optimizer, 30, 0.1)
        epochs = 200
        dataset = CustomDataset(X_train_os, y_train_os)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        total_steps = len(dataloader)
        for i in range(epochs):
            for j, (X_0, y_0, X_1, y_1) in enumerate(dataloader):
                # forward pass
                X = torch.vstack((X_0, X_1)).to(device_gpu)
                y = torch.vstack((y_0, y_1)).to(device_gpu)
                outputs = model(X)
                loss = loss_fn(outputs, y)
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (j+1)%10 == 0:
                    print(f"At epoch {i+1}, step {j+1}/{total_steps}, deviation loss = {loss.item():0.5f}")
            # scheduler.step()
            
        self.model = model.to('cpu')
    
    @torch.no_grad()
    def predict(self, X):
        X = torch.tensor(X).type(torch.float32)
        scores = self.model(X)
        scores = scores.detach().numpy()
        pred_fn = lambda x: 1 if x >= 1.96 else 0
        pred_fn = np.vectorize(pred_fn)
        y_pred = pred_fn(scores)
        return y_pred