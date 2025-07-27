import mne
import json
import numpy as np
import torch
from pathlib import Path
from scipy import signal
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
import warnings

class GMatrixCalculator:
    def __init__(self, method='pearson', fs=256):
        """
        Initializes the GMatrixCalculator class with optimized implementations.
        
        Parameters:
        -----------
        method : str
            Connectivity method to use. Options:
            - 'pearson': Pearson correlation (default)
            - 'coherence': Magnitude-squared coherence
            - 'pli': Phase Lag Index
            - 'wpli': Weighted Phase Lag Index
            - 'mutual_info': Mutual information
            - 'plv': Phase Locking Value
        fs : int
            Sampling frequency in Hz (default: 256)
        """
        self.method = method
        self.fs = fs
        self.supported_methods = ['pearson', 'coherence', 'pli', 'wpli', 'mutual_info', 'plv']
        if method not in self.supported_methods:
            raise ValueError(f"Method '{method}' not supported. Choose from: {self.supported_methods}")
    
    @staticmethod
    def _compute_G_matrix_pearson(trial_data):
        
        G = np.corrcoef(trial_data)
        G = np.maximum(G, 0)
        return G
    
    def _compute_G_matrix_coherence(self, trial_data, nperseg=256, freq_range=(1, 40)):
        """
        Computes the magnitude-squared coherence matrix with vectorized operations.
        
        Parameters:
        -----------
        trial_data : ndarray
            EEG data of shape (channels, time_points)
        nperseg : int
            Length of each segment for computing coherence
        freq_range : tuple
            Frequency range of interest (low, high) in Hz
            
        Returns:
        --------
        G : ndarray
            Coherence matrix of shape (channels, channels)
        """
        n_channels = trial_data.shape[0]
        G = np.eye(n_channels)  # Initialize with 1s on diagonal
        
        # Precompute FFTs for all channels
        # We'll use Welch's method for better frequency resolution
        f, all_psds = signal.welch(trial_data, fs=self.fs, nperseg=nperseg, axis=1)
        
        # Extract frequency band of interest
        freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        f_band = f[freq_mask]
        
        # For upper triangular part of matrix
        for i in range(n_channels):
            # We can compute cross-spectral density for channel i with all others at once
            for j in range(i+1, n_channels):
                # Calculate cross-spectral density
                f, Cxy = signal.csd(trial_data[i], trial_data[j], self.fs, nperseg=nperseg)
                
                # Calculate coherence: |Cxy|^2 / (Pxx * Pyy)
                Pxx = all_psds[i]
                Pyy = all_psds[j]
                coherence = np.abs(Cxy)**2 / (Pxx * Pyy)
                
                # Average over frequency band
                avg_coherence = np.mean(coherence[freq_mask])
                
                # Store in both upper and lower triangular parts (symmetric)
                G[i, j] = G[j, i] = avg_coherence
        
        return G
    
    def _compute_phase_signals(self, trial_data, freq_range=(8, 12)):
        """
        Efficiently compute instantaneous phase for all channels.
        
        Parameters:
        -----------
        trial_data : ndarray
            EEG data of shape (channels, time_points)
        freq_range : tuple
            Frequency range for bandpass filter (low, high) in Hz
            
        Returns:
        --------
        phases : ndarray
            Complex phase signals of shape (channels, time_points)
        """
        n_channels, n_samples = trial_data.shape
        
        # Apply bandpass filter to all channels at once
        nyquist = self.fs / 2
        low, high = freq_range[0] / nyquist, freq_range[1] / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Zero-phase filtering
        filtered_data = signal.filtfilt(b, a, trial_data, axis=1)
        
        # Apply Hilbert transform to all channels at once
        analytic_signal = signal.hilbert(filtered_data, axis=1)
        
        # Convert to unit-phase vectors (complex values on unit circle)
        phases = np.exp(1j * np.angle(analytic_signal))
        
        return phases
    
    def _compute_G_matrix_pli(self, trial_data, freq_range=(8, 12)):
        """
        Compute Phase Lag Index (PLI) matrix efficiently.
        PLI = |mean(sign(imag(cross-spectrum)))|
        
        Parameters:
        -----------
        trial_data : ndarray
            EEG data of shape (channels, time_points)
        freq_range : tuple
            Frequency range for bandpass filter (low, high) in Hz
            
        Returns:
        --------
        G : ndarray
            PLI matrix of shape (channels, channels)
        """
        n_channels = trial_data.shape[0]
        phases = self._compute_phase_signals(trial_data, freq_range)
        
        # Initialize matrix with ones on diagonal
        G = np.eye(n_channels)
        
        # Calculate PLI for upper triangular part
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                # Compute cross-spectrum
                cross_spectrum = phases[i] * np.conjugate(phases[j])
                
                # Calculate PLI
                pli = np.abs(np.mean(np.sign(np.imag(cross_spectrum))))
                
                # Store in both upper and lower triangular parts (symmetric)
                G[i, j] = G[j, i] = pli
        
        return G
    
    def _compute_G_matrix_wpli(self, trial_data, freq_range=(8, 12)):
        """
        Compute Weighted Phase Lag Index (wPLI) matrix efficiently.
        wPLI = |mean(imag(cross-spectrum))| / mean(|imag(cross-spectrum)|)
        
        Parameters:
        -----------
        trial_data : ndarray
            EEG data of shape (channels, time_points)
        freq_range : tuple
            Frequency range for bandpass filter (low, high) in Hz
            
        Returns:
        --------
        G : ndarray
            wPLI matrix of shape (channels, channels)
        """
        n_channels = trial_data.shape[0]
        phases = self._compute_phase_signals(trial_data, freq_range)
        
        # Initialize matrix with ones on diagonal
        G = np.eye(n_channels)
        
        # Calculate wPLI for upper triangular part
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                # Compute cross-spectrum
                cross_spectrum = phases[i] * np.conjugate(phases[j])
                
                # Calculate numerator and denominator for wPLI
                imag_cs = np.imag(cross_spectrum)
                num = np.abs(np.mean(imag_cs))
                denom = np.mean(np.abs(imag_cs))
                
                # Calculate wPLI with check for zero denominator
                wpli = num / denom if denom > 0 else 0
                
                # Store in both upper and lower triangular parts (symmetric)
                G[i, j] = G[j, i] = wpli
        
        return G
    
    def _compute_G_matrix_plv(self, trial_data, freq_range=(8, 12)):
        """
        Compute Phase Locking Value (PLV) matrix efficiently.
        PLV = |mean(exp(i*phase_diff))|
        
        Parameters:
        -----------
        trial_data : ndarray
            EEG data of shape (channels, time_points)
        freq_range : tuple
            Frequency range for bandpass filter (low, high) in Hz
            
        Returns:
        --------
        G : ndarray
            PLV matrix of shape (channels, channels)
        """
        n_channels = trial_data.shape[0]
        phases = self._compute_phase_signals(trial_data, freq_range)
        
        # Initialize matrix with ones on diagonal
        G = np.eye(n_channels)
        
        # Calculate PLV for upper triangular part
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                # Compute phase difference
                phase_diff = phases[i] * np.conjugate(phases[j])
                
                # Calculate PLV
                plv = np.abs(np.mean(phase_diff))
                
                # Store in both upper and lower triangular parts (symmetric)
                G[i, j] = G[j, i] = plv
        
        return G
    
    def _compute_G_matrix_mutual_info(self, trial_data, bins=20):
        """
        Compute mutual information matrix efficiently using vectorized binning.
        
        Parameters:
        -----------
        trial_data : ndarray
            EEG data of shape (channels, time_points)
        bins : int
            Number of bins for calculating mutual information
            
        Returns:
        --------
        G : ndarray
            Mutual information matrix of shape (channels, channels)
        """
        n_channels = trial_data.shape[0]
        
        # Initialize matrix with ones on diagonal
        G = np.eye(n_channels)
        
        # Precompute binned data and individual entropies
        binned_data = np.empty((n_channels, trial_data.shape[1]), dtype=np.int32)
        channel_entropies = np.empty(n_channels)
        
        # Bin data efficiently
        for i in range(n_channels):
            # Use uniform binning across the range
            hist, bin_edges = np.histogram(trial_data[i], bins=bins)
            binned_data[i] = np.digitize(trial_data[i], bin_edges[1:-1])
            
            # Calculate entropy for this channel
            p = hist / np.sum(hist)
            p = p[p > 0]  # Remove zeros
            channel_entropies[i] = -np.sum(p * np.log2(p))
        
        # Calculate mutual information for upper triangular part
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                # Calculate mutual information
                c_xy = np.histogram2d(binned_data[i], binned_data[j], bins=bins)[0]
                c_xy = c_xy / np.sum(c_xy)  # Normalize to create a joint probability
                
                # Calculate marginal probabilities
                c_x = np.sum(c_xy, axis=1)
                c_y = np.sum(c_xy, axis=0)
                
                # Calculate mutual information
                mi = 0
                for ii in range(bins):
                    for jj in range(bins):
                        if c_xy[ii, jj] > 0 and c_x[ii] > 0 and c_y[jj] > 0:
                            mi += c_xy[ii, jj] * np.log2(c_xy[ii, jj] / (c_x[ii] * c_y[jj]))
                
                # Normalize MI by minimum entropy
                min_entropy = min(channel_entropies[i], channel_entropies[j])
                norm_mi = mi / min_entropy if min_entropy > 0 else 0
                
                # Store in both upper and lower triangular parts (symmetric)
                G[i, j] = G[j, i] = norm_mi
        
        return G
    
    def compute_G_matrix(self, trial_data):
        """
        Compute G matrix based on the selected method.
        
        Parameters:
        -----------
        trial_data : ndarray
            EEG data of shape (channels, time_points)
            
        Returns:
        --------
        G : ndarray
            Connectivity matrix of shape (channels, channels)
        """
        if self.method == 'pearson':
            return self._compute_G_matrix_pearson(trial_data)
        elif self.method == 'coherence':
            return self._compute_G_matrix_coherence(trial_data)
        elif self.method == 'pli':
            return self._compute_G_matrix_pli(trial_data)
        elif self.method == 'wpli':
            return self._compute_G_matrix_wpli(trial_data)
        elif self.method == 'mutual_info':
            return self._compute_G_matrix_mutual_info(trial_data)
        elif self.method == 'plv':
            return self._compute_G_matrix_plv(trial_data)
        else:
            raise ValueError(f"Method '{self.method}' not implemented.")
    
    def compute_G_matrices(self, dataset):
        """
        Computes G matrices for all trials in a dataset.
        
        Parameters:
        -----------
        dataset : ndarray
            EEG data of shape (samples, channels, time_points)
            
        Returns:
        --------
        G_matrices : ndarray
            Connectivity matrices of shape (samples, channels, channels)
        """
        return np.array([self.compute_G_matrix(trial) for trial in dataset])