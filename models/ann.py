import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from models.model_architecture import FraudClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
from models.ml_models import FraudClassifierRF

class CustomDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.X = torch.tensor(X_train).type(torch.float32)
        self.y = torch.tensor(y_train.reshape((-1,1))).type(torch.float32)
        self.n_samples = self.X.shape[0]
    def __len__(self):
        # lenght of the data
        return self.n_samples 
    def __getitem__(self, index):
        return (self.X[index],self.y[index])

class ANNClassifier(nn.Module):
    def __init__(self, input_size):
        super(ANNClassifier, self).__init__()
        self.l1 = nn.Linear(input_size,64) # 64 units
        self.l1_ac = nn.ReLU()
        self.l2 = nn.Linear(64, 32) # 32 Units
        self.l2_ac = nn.ReLU()
        self.l3 = nn.Linear(32, 1) 
        self.l3_ac = nn.Sigmoid()
        
    def forward(self, x):
        out = self.l1(x)
        out = self.l1_ac(out)
        out = self.l2(out)
        out = self.l2_ac(out)
        out = self.l3(out)
        out = self.l3_ac(out)
        
        return out
    
class FraudClassifierANN(FraudClassifier):
    def __init__(self):
        self.name = "ANN"
        self.model = None
        self.recall = None # score on test data
        self.precision = None
        self.f1 = None
        self.accuracy = None
    
    def train(self, X_train, y_train):
        sm = SMOTE(random_state=100)
        X_train_os, y_train_os = sm.fit_resample(X_train, y_train)
        device_gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        batch_size = 2000
        input_size = X_train_os.shape[1]
        lr = 0.01
        model = ANNClassifier(input_size)
        model.to(device_gpu)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=10**(-3))
        epochs = 2
        dataset = CustomDataset(X_train_os, y_train_os)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        total_steps = len(dataloader)
        for i in range(epochs):
            for j, (X, y) in enumerate(dataloader):
                # forward pass
                X = X.to(device_gpu)
                y = y.to(device_gpu)
                outputs = model(X)
                loss = loss_fn(outputs, y)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (j+1)%100 == 0:
                    print(f"At epoch {i+1}, step {j}/{total_steps}, loss = {loss.item():0.4f}")
        self.model = model.to('cpu') # during inferencing model will in in CPU
    
    @torch.no_grad()
    def predict(self, X):
        X = torch.tensor(X).type(torch.float32)
        pred = self.model(X)
        pred = pred.detach().numpy()
        pred_fn = lambda x: 1 if x >= 0.5 else 0
        pred_fn = np.vectorize(pred_fn)
        pred_labels = pred_fn(pred)    
        return pred_labels

    @torch.no_grad()
    def predict_proba(self, X):
        X = torch.tensor(X).type(torch.float32)
        pred = self.model(X)
        pred = pred.detach().numpy()
        return pred
    
class FraudClassifierMixed(FraudClassifier):
    def __init__(self, model1: FraudClassifierRF, model2: FraudClassifierANN, wt1: float, wt2: float):
        self.name = f"MIXED_{wt1}_{wt2}"
        self.model1, self.model2 = model1, model2
        self.wt1, self.wt2 = wt1, wt2
        self.recall = None # score on test data
        self.precision = None
        self.f1 = None
        self.accuracy = None
    
    def predict(self, X):
        # get probability scores from both the models
        model_1_scores = self.model1.predict_proba(X)
        model_2_scores = self.model2.predict_proba(X).flatten()
        comb_scores = self.wt1*model_1_scores + self.wt2*model_2_scores
        pred_fn = lambda x: 1 if x >= 0.5 else 0
        pred_fn = np.vectorize(pred_fn)
        pred_labels = pred_fn(comb_scores)    
        return pred_labels
