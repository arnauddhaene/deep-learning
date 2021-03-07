import time
import torch
from torch import Tensor
import utils as prologue

def nearest_classification(train_input, train_target, x):
    """
    A function that gets a training set and a test sample and returns the label of the training point
    the closest to the latter.

    param train_input: is a 2d float tensor of dimension n × d containing the training vectors
    param train_target: is a 1d long tensor of dimension n containing the training labels
    param x: is 1d float tensor of dimension d containing the test vector
    
    returns class of the train sample closest to x using the L^2 norm
    """
    return train_target[torch.argmin(torch.sum(torch.pow(train_input - x, 2), dim=1))]

def compute_nb_errors(train_input, train_target, test_input, test_target, mean = None, proj = None):
    """
    Function that subtracts mean (if it is not None) from the vectors of both train_input
    and test_input, apply the operator proj (if it is not None) to both.
    
    param train_input: is a 2d float tensor of dimension n × d containing the train vectors
    param train_target: is a 1d long tensor of dimension n containing the train labels
    param test_input: is a 2d float tensor of dimension m × d containing the test vectors
    param test_target: is a 1d long tensor of dimension m containing the test labels
    param mean: is either None or a 1d float tensor of dimension d
    param proj: is either None or a 2d float tensor of dimension c × d
    
    returns the number of classification errors using the 1-NN rule on the resulting data.
    """
    if mean is not None:
        train_input = train_input - mean
        test_input  = test_input  - mean

    if proj is not None:
        train_input = train_input @ proj.T 
        test_input  = test_input  @ proj.T
    
    classification_errors = 0
    
    for example in range(test_input.size()[0]):
        classification_errors += int(test_target[example] != nearest_classification(train_input, train_target, test_input[example, :]))
    
    return classification_errors

def pca(x):
    """
    Principal Component Analysis
    
    param x: is a 2d float tensor of dimension n × d
    
    returns a pair composed of the 1d mean vector of dimension d and
            the PCA basis, ranked in decreasing order of the eigen-values, as a 2d tensor d x d
    """
    eig_vals, eig_vecs = torch.eig(x.T @ x, eigenvectors=True)
    
    _, indices = torch.sort(eig_vals[:, 0], descending=True)
    
    return torch.mean(x, dim=0), eig_vecs[:, indices]
