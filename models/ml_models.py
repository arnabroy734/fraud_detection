from models.model_architecture import FraudClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, f1_score
from time import time
from log import logging
from paths.setup_path import Paths

class FraudClassifierRF(FraudClassifier):
    def __init__(self):
        self.name = "RF"
        self.id = int(time())
        self.model = None
        self.recall = None # score on test data
        self.precision = None
        self.f1 = None
        self.accuracy = None
    
    def train(self, X_train, y_train):
        """Tune hyperparameter and find the best model"""
        params = {
            'n_estimators' : [100, 150],
        }
        clf = GridSearchCV(
                estimator = RandomForestClassifier(n_jobs=-1),
                param_grid=params, 
                cv=5,
                verbose=3,
                scoring="recall"
            )
        # sm = SMOTE(random_state=100)
        # X_train_os, y_train_os = sm.fit_resample(X_train, y_train)
        X_train_os, y_train_os = X_train, y_train
        clf.fit(X_train_os, y_train_os)
        self.model = clf.best_estimator_
        description = f"id={self.id} n_estimators={self.model.n_estimators} max_depth={self.model.max_depth}"
        logging.log_model_description(Paths.description_rf(), description)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]
    
