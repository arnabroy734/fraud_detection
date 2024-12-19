from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

class FraudClassifier(ABC):
    """
    This class provides a common interface for all ML/DL models
    """
    @abstractmethod 
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict a batch input

        Parameter:
            X(np.ndarray): inputs of size (N x dim)
        
        Returns:
            results: size N x 1 (predictions)
        """
        raise NotImplementedError
    
    def evaluate(self, X_test, y_test):
        """Measure performance of the model on test data"""
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        precision = cm[1,1]/(cm[0,1]+cm[1,1])
        recall = cm[1,1]/(cm[1,0]+cm[1,1])
        accuracy = (cm[1,1]+cm[0,0])/(np.sum(cm.flatten()))
        f1= f1_score(y_test, y_pred)
        metrics={
            "recall": float(recall),
            "precision": float(precision),
            "accuracy" : float(accuracy),
            "f1" : float(f1)
        }
        self.recall = float(recall)
        self.precision = float(precision)
        self.f1 = float(f1)
        self.accuracy = float(accuracy)
        print(cm)
        return metrics

