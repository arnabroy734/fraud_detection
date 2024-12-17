from models.model_architecture import FraudClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE


class FraudClassifierRF(FraudClassifier):
    def __init__(self):
        self.name = "RF"
        self.model = None
        self.recall = None # score on test data
        self.precision = None
        self.f1 = None
        self.accuracy = None
    
    def train(self, X_train, y_train):
        """Tune hyperparameter and find the best model"""
        params = {
            'n_estimators' : [2],
            'max_depth' : [3],
            'min_samples_split' : [2]
        }
        clf = GridSearchCV(
                estimator = RandomForestClassifier(n_jobs=-1),
                param_grid=params, 
                cv=2,
                verbose=3,
                scoring="recall"
            )
        sm = SMOTE(random_state=100)
        X_train_os, y_train_os = sm.fit_resample(X_train, y_train)
        clf.fit(X_train_os, y_train_os)
        self.model = clf.best_estimator_
        print(f"Best Model: {self.model}, best validation recall: {clf.best_score_}")
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]
    
