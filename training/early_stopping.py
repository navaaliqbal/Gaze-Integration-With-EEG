"""
Early stopping implementation
"""
import torch
import numpy as np

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, delta=0, path='best_model.pth', verbose=True):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric = -np.inf
    
    def __call__(self, metric_value, model, save_best_acc=True):
        score = metric_value
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric_value, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric_value, model)
            self.counter = 0
    
    def save_checkpoint(self, metric_value, model):
        """Save model checkpoint"""
        if self.verbose:
            prev = '-inf' if self.best_metric == -np.inf else f'{self.best_metric:.6f}'
            print(f'Validation metric improved ({prev} --> {metric_value:.6f}). Saving model ...')
        
        torch.save(model.state_dict(), self.path)
        self.best_metric = metric_value