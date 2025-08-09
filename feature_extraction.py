import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats, signal
from scipy.spatial.distance import cdist
import pywt
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
from numba import jit, prange
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')


# Configuration
@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    fs: int = 500
    n_features: int = 150
    resample_length: int = 100
    use_parallel: bool = True
    n_workers: Optional[int] = None

class FeatureExtractor:
    """Modular feature extractor for EEG signals"""
    
    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.config = config
        self.fs = config.fs
        
    # === TIME-DOMAIN FEATURES ===
    
    def extract_statistical_features(self, signal: np.ndarray) -> List[float]:
        """Extract basic statistical features (10 features)"""
        return [
            np.mean(signal),
            np.var(signal),
            np.std(signal),
            stats.skew(signal),
            stats.kurtosis(signal),
            np.sqrt(np.mean(signal**2)),  # RMS
            np.median(signal),
            np.min(signal),
            np.max(signal),
            np.ptp(signal),  # peak-to-peak
        ]
    
    def extract_percentile_features(self, signal: np.ndarray) -> List[float]:
        """Extract percentile-based features (7 features)"""
        features = []
        for p in [5, 10, 25, 75, 90, 95]:
            features.append(np.percentile(signal, p))
        features.append(np.percentile(signal, 75) - np.percentile(signal, 25))  # IQR
        return features
    
    def extract_morphological_features(self, signal: np.ndarray) -> List[float]:
        """Extract morphological features (8 features)"""
        N = len(signal)
        features = []
        
        # Zero crossings
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        features.extend([zero_crossings, zero_crossings / N])
        
        # Slope sign changes
        diff_signal = np.diff(signal)
        ssc = np.sum(np.diff(np.sign(diff_signal)) != 0)
        features.extend([ssc, ssc / (N-2)])
        
        # Waveform length
        wl = np.sum(np.abs(diff_signal))
        features.extend([wl, wl / (N-1)])
        
        # Mean absolute value slope & variance of absolute gradient
        features.append(np.mean(np.abs(diff_signal)))
        features.append(np.var(np.abs(np.gradient(signal))))
        
        return features
    
    def extract_hjorth_parameters(self, signal: np.ndarray) -> List[float]:
        """Extract Hjorth parameters (3 features)"""
        activity = np.var(signal)
        diff1 = np.diff(signal)
        diff2 = np.diff(diff1)
        
        mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
        complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility if mobility > 0 and np.var(diff1) > 0 else 0
        
        return [activity, mobility, complexity]
    
    # === FREQUENCY-DOMAIN FEATURES ===
    
    def extract_band_power_features(self, signal: np.ndarray) -> Tuple[List[float], Dict[str, float]]:
        """Extract frequency band power features (27 features)"""
        N = len(signal)
        fft_vals = np.abs(np.fft.rfft(signal))
        fft_freq = np.fft.rfftfreq(N, d=1/self.fs)
        
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha1': (8, 10),
            'alpha2': (10, 13),
            'beta1': (13, 20),
            'beta2': (20, 30),
            'gamma1': (30, 50),
            'gamma2': (50, 80),
            'gamma3': (80, 100)
        }
        
        band_powers = {}
        total_power = np.sum(fft_vals**2)
        features = []
        
        for band_name, (low, high) in bands.items():
            band_mask = (fft_freq >= low) & (fft_freq < high)
            band_power = np.sum(fft_vals[band_mask]**2)
            band_powers[band_name] = band_power
            
            features.extend([
                band_power,  # Absolute power
                band_power / total_power if total_power > 0 else 0,  # Relative power
                np.log(band_power + 1e-10)  # Log power
            ])
        
        return features, band_powers, fft_vals, fft_freq, total_power
    
    def extract_band_ratios(self, band_powers: Dict[str, float], total_power: float) -> List[float]:
        """Extract band power ratios (15 features)"""
        eps = 1e-10
        return [
            band_powers['theta'] / (band_powers['alpha1'] + band_powers['alpha2'] + eps),
            band_powers['alpha1'] / (band_powers['alpha2'] + eps),
            band_powers['beta1'] / (band_powers['beta2'] + eps),
            band_powers['gamma1'] / (band_powers['gamma2'] + eps),
            band_powers['gamma2'] / (band_powers['gamma3'] + eps),
            (band_powers['theta'] + band_powers['alpha1']) / (band_powers['beta1'] + band_powers['beta2'] + eps),
            band_powers['delta'] / (band_powers['theta'] + eps),
            (band_powers['gamma1'] + band_powers['gamma2'] + band_powers['gamma3']) / (total_power + eps),
            band_powers['alpha1'] / (band_powers['beta1'] + eps),
            band_powers['alpha2'] / (band_powers['beta2'] + eps),
            (band_powers['beta1'] + band_powers['beta2']) / (band_powers['gamma1'] + band_powers['gamma2'] + eps),
            band_powers['theta'] / (band_powers['gamma1'] + eps),
            (band_powers['alpha1'] + band_powers['alpha2']) / (total_power + eps),
            (band_powers['beta1'] + band_powers['beta2']) / (total_power + eps),
            (band_powers['delta'] + band_powers['theta']) / (band_powers['alpha1'] + band_powers['alpha2'] + eps),
        ]
    
    def extract_spectral_features(self, fft_vals: np.ndarray, fft_freq: np.ndarray, 
                                 total_power: float) -> List[float]:
        """Extract spectral features (18 features)"""
        features = []
        eps = 1e-10
        
        # Spectral centroid and spread
        spectral_centroid = np.sum(fft_freq * fft_vals**2) / (total_power + eps)
        features.append(spectral_centroid)
        
        spectral_spread = np.sqrt(np.sum(((fft_freq - spectral_centroid)**2) * fft_vals**2) / (total_power + eps))
        features.append(spectral_spread)
        
        # Spectral edge frequencies
        cumsum_power = np.cumsum(fft_vals**2)
        cumsum_power = cumsum_power / cumsum_power[-1] if cumsum_power[-1] > 0 else cumsum_power
        
        for percentile in [50, 75, 80, 85, 90, 95]:
            idx = np.where(cumsum_power >= percentile/100)[0]
            features.append(fft_freq[idx[0]] if len(idx) > 0 else 0)
        
        # Other spectral features
        features.append(np.mean(np.diff(fft_vals)))  # Spectral flux
        
        # Spectral rolloff
        rolloff_idx = np.where(np.cumsum(fft_vals) >= 0.85 * np.sum(fft_vals))[0]
        features.append(fft_freq[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0)
        
        # Peak frequency and magnitude
        peak_idx = np.argmax(fft_vals)
        features.extend([fft_freq[peak_idx], fft_vals[peak_idx]])
        
        # Spectral flatness
        geometric_mean = np.exp(np.mean(np.log(fft_vals + eps)))
        arithmetic_mean = np.mean(fft_vals)
        features.append(geometric_mean / (arithmetic_mean + eps))
        
        # Spectral entropy
        psd_norm = fft_vals / np.sum(fft_vals)
        features.append(-np.sum(psd_norm * np.log(psd_norm + eps)))
        
        # Spectral skewness and kurtosis
        features.extend([stats.skew(fft_vals), stats.kurtosis(fft_vals)])
        
        return features
    
    # === ENTROPY FEATURES ===
    
    @staticmethod
    @jit(nopython=True)
    def _approx_entropy_core(signal, m, r):
        """Numba-optimized approximate entropy calculation"""
        N = len(signal)
        def _maxdist(xi, xj, m):
            return np.max(np.abs(xi[:m] - xj[:m]))
        
        def _phi(m):
            C = np.zeros(N - m + 1)
            for i in range(N - m + 1):
                template = signal[i:i + m]
                matching = 0
                for j in range(N - m + 1):
                    if _maxdist(template, signal[j:j + m], m) <= r:
                        matching += 1
                C[i] = matching / (N - m + 1)
            return np.mean(np.log(C + 1e-10))
        
        return _phi(m) - _phi(m + 1)
    
    def extract_entropy_features(self, signal: np.ndarray) -> List[float]:
        """Extract various entropy measures (8 features)"""
        features = []
        
        # Shannon entropy
        hist, _ = np.histogram(signal, bins=20)
        hist = hist / np.sum(hist)
        features.append(-np.sum(hist * np.log(hist + 1e-10)))
        
        # Approximate entropy
        r = 0.15 * np.std(signal)
        features.append(self._approx_entropy_core(signal, 2, r))
        
        # Permutation entropy
        features.append(self._permutation_entropy(signal, order=3))
        
        # Renyi entropy
        features.append(-np.log(np.sum(hist**2) + 1e-10))
        
        # Tsallis entropy
        features.append((1 - np.sum(hist**2)) / (2 - 1))
        
        # Log energy entropy
        log_energy = np.log(signal**2 + 1e-10)
        log_energy_sum = np.sum(log_energy)
        if log_energy_sum != 0:
            log_energy_norm = log_energy / log_energy_sum
            features.append(-np.sum(log_energy_norm * np.log(log_energy_norm + 1e-10)))
        else:
            features.append(0)
        
        # SVD entropy
        features.append(self._svd_entropy(signal))
        
        # Sample entropy (approximation)
        features.append(self._approx_entropy_core(signal, 2, r) * 0.9)
        
        return features
    
    @staticmethod
    def _permutation_entropy(signal: np.ndarray, order: int = 3) -> float:
        """Calculate permutation entropy"""
        n = len(signal)
        if n < order:
            return 0
        
        perms = {}
        for i in range(n - order + 1):
            sorted_indices = tuple(np.argsort(signal[i:i+order]))
            perms[sorted_indices] = perms.get(sorted_indices, 0) + 1
        
        total = sum(perms.values())
        probs = [count/total for count in perms.values()]
        return -sum(p * np.log(p) for p in probs if p > 0)
    
    def _svd_entropy(self, signal: np.ndarray) -> float:
        """Calculate SVD entropy"""
        N = len(signal)
        try:
            embed_dim = min(10, N//10)
            if embed_dim > 2:
                embed_matrix = np.array([signal[i:i+embed_dim] for i in range(N-embed_dim+1)])
                _, S, _ = np.linalg.svd(embed_matrix, compute_uv=False)
                S = S / np.sum(S)
                return -np.sum(S * np.log(S + 1e-10))
        except:
            pass
        return 0
    
    # === WAVELET FEATURES ===
    
    def extract_wavelet_features(self, signal: np.ndarray) -> List[float]:
        """Extract wavelet features (30 features)"""
        features = []
        N = len(signal)
        wavelet = 'db4'
        
        try:
            max_level = pywt.dwt_max_level(N, pywt.Wavelet(wavelet).dec_len)
            coeffs = pywt.wavedec(signal, wavelet, level=min(max_level, 5))
            
            total_wavelet_energy = sum(np.sum(c**2) for c in coeffs)
            
            for i, coeff in enumerate(coeffs[:6]):
                energy = np.sum(coeff**2)
                features.extend([
                    energy,
                    np.log(energy + 1e-10),
                    energy / (total_wavelet_energy + 1e-10)
                ])
                
                # Wavelet entropy
                if energy > 0:
                    coeff_norm = coeff**2 / energy
                    coeff_norm = coeff_norm[coeff_norm > 0]
                    wavelet_entropy = -np.sum(coeff_norm * np.log(coeff_norm))
                else:
                    wavelet_entropy = 0
                features.append(wavelet_entropy)
                
                features.append(np.var(coeff))
            
            # Pad if less than 6 levels
            while len(features) < 30:
                features.append(0)
                
        except:
            features = [0] * 30
            
        return features[:30]
    
    # === NONLINEAR FEATURES ===
    
    def extract_nonlinear_features(self, signal: np.ndarray) -> List[float]:
        """Extract nonlinear features (7 features)"""
        features = []
        N = len(signal)
        
        # Higuchi Fractal Dimension
        features.append(self._higuchi_fd(signal))
        
        # Petrosian Fractal Dimension
        diff = np.diff(signal)
        N_delta = np.sum(np.diff(np.sign(diff)) != 0)
        petrosian_fd = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * N_delta)))
        features.append(petrosian_fd)
        
        # Katz Fractal Dimension
        L = np.sum(np.sqrt(1 + np.diff(signal)**2))
        d = np.sqrt((N-1)**2 + (signal[-1] - signal[0])**2)
        features.append(np.log10(L) / np.log10(d) if d > 0 else 1)
        
        # DFA
        features.append(self._dfa_fast(signal))
        
        # Hurst exponent
        features.append(self._hurst_fast(signal))
        
        # Lyapunov exponent
        features.append(np.mean(np.log(np.abs(np.diff(signal)) + 1e-10)))
        
        # Correlation dimension (placeholder)
        features.append(2.0)
        
        return features
    
    @staticmethod
    def _higuchi_fd(signal: np.ndarray, k_max: int = 10) -> float:
        """Calculate Higuchi fractal dimension"""
        N = len(signal)
        L = []
        for k in range(1, min(k_max, N//2)):
            Lk = []
            for m in range(k):
                Lm = 0
                for i in range(1, int((N-m)/k)):
                    Lm += abs(signal[m+i*k] - signal[m+(i-1)*k])
                Lm = Lm * (N-1) / (k * int((N-m)/k))
                Lk.append(Lm)
            L.append(np.mean(Lk))
        
        if len(L) > 1:
            x = np.log(range(1, len(L)+1))
            y = np.log(L)
            fd = np.polyfit(x, y, 1)[0]
        else:
            fd = 1
        return fd
    
    @staticmethod
    def _dfa_fast(signal: np.ndarray) -> float:
        """Fast detrended fluctuation analysis"""
        N = len(signal)
        if N < 20:
            return 0.5
            
        y = np.cumsum(signal - np.mean(signal))
        scales = [10, 20, 40] if N > 40 else [10]
        F = []
        
        for scale in scales:
            if scale > N//4:
                continue
            n_segments = N // scale
            rms = []
            
            for i in range(n_segments):
                segment = y[i*scale:(i+1)*scale]
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                fit = np.polyval(coeffs, x)
                rms.append(np.sqrt(np.mean((segment - fit)**2)))
            
            if rms:
                F.append(np.mean(rms))
        
        if len(F) > 1:
            return np.polyfit(np.log(scales[:len(F)]), np.log(F), 1)[0]
        return 0.5
    
    @staticmethod
    def _hurst_fast(signal: np.ndarray) -> float:
        """Fast Hurst exponent calculation"""
        N = len(signal)
        if N < 20:
            return 0.5
        
        lags = [1, 2, 4, 8, 16]
        tau = []
        
        for lag in lags:
            if lag >= N:
                break
            tau.append(np.std(np.diff(signal, n=lag)))
        
        if len(tau) > 1:
            return np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)[0] * 0.5
        return 0.5
    
    # === PHASE SPACE FEATURES ===
    
    def extract_phase_features(self, signal: np.ndarray) -> List[float]:
        """Extract phase space features (8 features)"""
        from scipy import signal as scipy_signal
        
        features = []
        N = len(signal)
        
        # Teager-Kaiser Energy
        if N > 2:
            tke = np.mean(signal[1:-1]**2 - signal[:-2] * signal[2:])
        else:
            tke = 0
        features.append(tke)
        
        # Phase synchronization features
        analytic_signal = scipy_signal.hilbert(signal)
        instantaneous_phase = np.angle(analytic_signal)
        
        # Phase coherence
        phase_diff = np.diff(instantaneous_phase)
        phase_coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
        features.append(phase_coherence)
        
        # Instantaneous frequency stats
        inst_freq = np.diff(instantaneous_phase) * self.fs / (2 * np.pi)
        inst_freq = inst_freq[np.isfinite(inst_freq)]
        
        if len(inst_freq) > 0:
            features.extend([
                np.mean(inst_freq),
                np.std(inst_freq),
                np.median(inst_freq),
                stats.skew(inst_freq),
                stats.kurtosis(inst_freq)
            ])
        else:
            features.extend([0] * 5)
        
        # Mean frequency
        features.append(np.mean(np.abs(inst_freq)) if len(inst_freq) > 0 else 0)
        
        return features
    
    # === TIME-FREQUENCY COUPLING ===
    
    def extract_pac_features(self, signal: np.ndarray) -> List[float]:
        """Extract phase-amplitude coupling features (12 features)"""
        from scipy import signal as scipy_signal
        
        features = []
        
        # Extract phase from low frequencies
        theta_band = scipy_signal.butter(4, [4, 8], btype='band', fs=self.fs, output='sos')
        theta_signal = scipy_signal.sosfilt(theta_band, signal)
        theta_phase = np.angle(scipy_signal.hilbert(theta_signal))
        
        alpha_band = scipy_signal.butter(4, [8, 13], btype='band', fs=self.fs, output='sos')
        alpha_signal = scipy_signal.sosfilt(alpha_band, signal)
        alpha_phase = np.angle(scipy_signal.hilbert(alpha_signal))
        
        # Extract amplitude from high frequencies
        gamma_band = scipy_signal.butter(4, [30, 100], btype='band', fs=self.fs, output='sos')
        gamma_signal = scipy_signal.sosfilt(gamma_band, signal)
        gamma_amplitude = np.abs(scipy_signal.hilbert(gamma_signal))
        
        # PAC metrics
        features.append(np.abs(np.mean(gamma_amplitude * np.exp(1j * theta_phase))))
        features.append(np.abs(np.mean(gamma_amplitude * np.exp(1j * alpha_phase))))
        
        # Phase-locking values
        for n_phases in [4, 6, 8]:
            phase_bins = np.linspace(-np.pi, np.pi, n_phases + 1)
            theta_binned = np.digitize(theta_phase, phase_bins)
            
            pac_values = []
            for i in range(1, n_phases + 1):
                mask = theta_binned == i
                if np.any(mask):
                    pac_values.append(np.mean(gamma_amplitude[mask]))
                else:
                    pac_values.append(0)
            
            pac_values = np.array(pac_values)
            if np.sum(pac_values) > 0:
                pac_norm = pac_values / np.sum(pac_values)
                mi = (np.log(n_phases) + np.sum(pac_norm * np.log(pac_norm + 1e-10))) / np.log(n_phases)
            else:
                mi = 0
            features.append(mi)
        
        # Cross-frequency ratios
        eps = 1e-10
        features.extend([
            np.mean(gamma_amplitude) / (np.mean(np.abs(theta_signal)) + eps),
            np.mean(gamma_amplitude) / (np.mean(np.abs(alpha_signal)) + eps),
            np.std(gamma_amplitude) / (np.std(theta_signal) + eps),
            np.std(gamma_amplitude) / (np.std(alpha_signal) + eps),
            np.max(gamma_amplitude) / (np.max(np.abs(theta_signal)) + eps),
            np.max(gamma_amplitude) / (np.max(np.abs(alpha_signal)) + eps),
        ])
        
        features.append(np.mean(gamma_amplitude))
        
        return features
    
    def extract_all_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract all features from a signal"""
        features = []
        
        # Time domain (28 features)
        features.extend(self.extract_statistical_features(signal))
        features.extend(self.extract_percentile_features(signal))
        features.extend(self.extract_morphological_features(signal))
        features.extend(self.extract_hjorth_parameters(signal))
        
        # Frequency domain (60 features)
        band_features, band_powers, fft_vals, fft_freq, total_power = self.extract_band_power_features(signal)
        features.extend(band_features)
        features.extend(self.extract_band_ratios(band_powers, total_power))
        features.extend(self.extract_spectral_features(fft_vals, fft_freq, total_power))
        
        # Entropy (8 features)
        features.extend(self.extract_entropy_features(signal))
        
        # Wavelets (30 features)
        features.extend(self.extract_wavelet_features(signal))
        
        # Nonlinear (7 features)
        features.extend(self.extract_nonlinear_features(signal))
        
        # Phase space (8 features)
        features.extend(self.extract_phase_features(signal))
        
        # PAC (12 features)
        features.extend(self.extract_pac_features(signal))
        
        # Ensure exactly 150 features
        features = features[:150]
        while len(features) < 150:
            features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def extract_hybrid_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract features + intelligently resampled signal"""
        # Extract features
        features = self.extract_all_features(signal)
        
        # Intelligent resampling to fixed length
        resampled = self.intelligent_resample(signal, self.config.resample_length)
        
        # Combine
        return np.concatenate([features, resampled])
    
    def intelligent_resample(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """
        Intelligently resample signal to target length using appropriate method
        """
        from scipy import signal as scipy_signal
        from scipy.interpolate import interp1d
        
        current_length = len(signal)
        
        if current_length == target_length:
            return signal
        
        # For upsampling: use cubic spline interpolation
        if current_length < target_length:
            # Create interpolation function
            x_old = np.linspace(0, 1, current_length)
            x_new = np.linspace(0, 1, target_length)
            
            # Use cubic interpolation for smooth upsampling
            f = interp1d(x_old, signal, kind='cubic', fill_value='extrapolate')
            resampled = f(x_new)
            
        # For downsampling: use anti-aliasing filter
        else:
            # Calculate downsampling factor
            factor = current_length / target_length
            
            # Apply anti-aliasing filter before downsampling
            # Cutoff frequency should be Nyquist / factor
            cutoff = 0.5 / factor
            
            # Design filter
            b, a = scipy_signal.butter(8, cutoff, btype='low')
            
            # Apply filter
            filtered = scipy_signal.filtfilt(b, a, signal)
            
            # Downsample using scipy's resample (includes anti-aliasing)
            resampled = scipy_signal.resample(filtered, target_length)
        
        return resampled
    
    def intelligent_resample_(self, eeg_trial: np.ndarray, target_length: int) -> np.ndarray:
        """
        Intelligently resample EEG trial to target length using appropriate method
        
        Args:
            eeg_trial: EEG data with shape [channels, timepoints]
            target_length: Target number of timepoints
            
        Returns:
            Resampled EEG trial with shape [channels, target_length]
        """
        from scipy import signal as scipy_signal
        from scipy.interpolate import interp1d
        
        n_channels, current_length = eeg_trial.shape
        
        if current_length == target_length:
            return eeg_trial
        
        # For upsampling: use cubic spline interpolation
        if current_length < target_length:
            # Create interpolation points
            x_old = np.linspace(0, 1, current_length)
            x_new = np.linspace(0, 1, target_length)
            
            # Use cubic interpolation for smooth upsampling
            # interp1d can handle 2D arrays along axis=1 (timepoints)
            f = interp1d(x_old, eeg_trial, kind='cubic', axis=1, fill_value='extrapolate')
            resampled = f(x_new)
            
        # For downsampling: use anti-aliasing filter
        else:
            # Calculate downsampling factor
            factor = current_length / target_length
            
            
            cutoff = 0.5 / factor
            
            # Design filter
            b, a = scipy_signal.butter(8, cutoff, btype='low')
            
            # Apply filter to all channels simultaneously
            # filtfilt works along the last axis by default, so we need axis=1
            filtered = scipy_signal.filtfilt(b, a, eeg_trial, axis=1)
            
            # Downsample using scipy's resample (includes anti-aliasing)
            # resample works along the last axis by default, so we need axis=1
            resampled = scipy_signal.resample(filtered, target_length, axis=1)
        
        return resampled
        

