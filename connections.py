import mne
import json
import numpy as np
from pathlib import Path

class GMatrixCalculator:
    def __init__(self):
        """Initializes the GMatrixCalculator class."""
        pass
    @staticmethod
    def _compute_G_matrix(trial_data):
        """Computes the Pearson correlation matrix G for each trial and applies thresholding."""
        num_channels = trial_data.shape[0]
        G = np.corrcoef(trial_data)
        G[G <= 0] = 0  # Apply thresholding: set negative or zero values to 0
        return G
    
    def compute_G_matrices(self, dataset):
        """Computes G matrices for all trials in a dataset.
        :param dataset: Numpy array of shape (samples, 12, time_points)
        :return: Numpy array of shape (samples, 12, 12)
        """
        return np.array([self._compute_G_matrix(trial) for trial in dataset])