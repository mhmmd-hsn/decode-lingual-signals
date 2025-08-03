
import numpy as np

from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count

import os
from typing import List, Tuple, Optional
warnings.filterwarnings('ignore')

from feature_extraction import *
from utils import *

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class OptimizedDataProcessor:
    
    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.config = config
        self.feature_extractor = FeatureExtractor(config)
        
    def process_single_trial(self, args: Tuple) -> Tuple:
        """Process a single trial - optimized for parallel processing"""
        trial_eeg, word_label = args
        n_channels, n_timepoints = trial_eeg.shape
        
        # Pre-allocate array for efficiency
        channel_features = np.zeros((n_channels, 250), dtype=np.float32)
        
        # Extract features for each channel
        for ch_idx in range(n_channels):
            channel_features[ch_idx] = self.feature_extractor.extract_hybrid_features(trial_eeg[ch_idx])
        
        # Replace NaN and inf values
        channel_features = np.nan_to_num(channel_features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Compute correlation matrix efficiently (only on subset of features)
        # Using only first 10 features for speed
        corr_matrix = np.corrcoef(channel_features[:, :10])
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Ensure correlation matrix is the right size
        if corr_matrix.shape != (n_channels, n_channels):
            corr_matrix = np.eye(n_channels, dtype=np.float32)
        
        return channel_features, word_label, corr_matrix
    
    def process_batch(self, batch_data: List[Tuple], n_workers: Optional[int] = None) -> Tuple:
        """Process a batch of trials in parallel"""
        if n_workers is None:
            n_workers = self.config.n_workers or max(1, cpu_count() - 1)
        
        with Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(self.process_single_trial, batch_data, chunksize=100),
                total=len(batch_data),
                desc="Processing batch"
            ))
        
        # Unpack results
        features = np.array([r[0] for r in results], dtype=np.float32)
        labels = np.array([r[1] for r in results], dtype=np.int32)
        connections = np.array([r[2] for r in results], dtype=np.float32)
        
        return features, labels, connections
    


def get_feature_names() -> List[str]:
    """Get list of all feature names"""
    feature_names = []
    
    # Time domain features (28)
    feature_names.extend([
        'mean', 'variance', 'std_dev', 'skewness', 'kurtosis',
        'rms', 'median', 'min', 'max', 'peak_to_peak',
        'percentile_5', 'percentile_10', 'percentile_25', 
        'percentile_75', 'percentile_90', 'percentile_95', 'iqr',
        'zero_crossings', 'zero_crossing_rate', 'slope_sign_changes',
        'slope_sign_change_rate', 'waveform_length', 'waveform_length_norm',
        'mean_abs_slope', 'var_abs_gradient',
        'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'
    ])
    
    # Frequency domain features (60)
    bands = ['delta', 'theta', 'alpha1', 'alpha2', 'beta1', 'beta2', 'gamma1', 'gamma2', 'gamma3']
    for band in bands:
        feature_names.extend([f'{band}_power', f'{band}_relative_power', f'{band}_log_power'])
    
    feature_names.extend([
        'theta_alpha_ratio', 'alpha1_alpha2_ratio', 'beta1_beta2_ratio',
        'gamma1_gamma2_ratio', 'gamma2_gamma3_ratio', 'theta_alpha_beta_ratio',
        'delta_theta_ratio', 'gamma_total_ratio', 'alpha1_beta1_ratio',
        'alpha2_beta2_ratio', 'beta_gamma_ratio', 'theta_gamma1_ratio',
        'alpha_total_ratio', 'beta_total_ratio', 'delta_theta_alpha_ratio',
        'spectral_centroid', 'spectral_spread',
        'spectral_edge_50', 'spectral_edge_75', 'spectral_edge_80',
        'spectral_edge_85', 'spectral_edge_90', 'spectral_edge_95',
        'spectral_flux', 'spectral_rolloff', 'peak_frequency',
        'peak_magnitude', 'spectral_flatness', 'spectral_entropy',
        'spectral_skewness', 'spectral_kurtosis'
    ])
    
    # Entropy features (8)
    feature_names.extend([
        'shannon_entropy', 'approx_entropy', 'permutation_entropy',
        'renyi_entropy', 'tsallis_entropy', 'log_energy_entropy',
        'svd_entropy', 'sample_entropy'
    ])
    
    # Wavelet features (30)
    for i in range(6):
        feature_names.extend([
            f'wavelet_level{i}_energy', f'wavelet_level{i}_log_energy',
            f'wavelet_level{i}_relative_energy', f'wavelet_level{i}_entropy',
            f'wavelet_level{i}_variance'
        ])
    
    # Nonlinear features (7)
    feature_names.extend([
        'higuchi_fd', 'petrosian_fd', 'katz_fd', 'dfa_exponent',
        'hurst_exponent', 'lyapunov_exponent', 'correlation_dimension'
    ])
    
    # Phase space features (8)
    feature_names.extend([
        'teager_kaiser_energy', 'phase_coherence',
        'inst_freq_mean', 'inst_freq_std', 'inst_freq_median',
        'inst_freq_skewness', 'inst_freq_kurtosis', 'mean_frequency'
    ])
    
    # PAC features (12)
    feature_names.extend([
        'theta_gamma_pac', 'alpha_gamma_pac',
        'theta_gamma_mi_4bins', 'theta_gamma_mi_6bins', 'theta_gamma_mi_8bins',
        'gamma_theta_amp_ratio', 'gamma_alpha_amp_ratio',
        'gamma_theta_std_ratio', 'gamma_alpha_std_ratio',
        'gamma_theta_max_ratio', 'gamma_alpha_max_ratio',
        'gamma_mean_amplitude'
    ])
    
    # Downsampled signal features (100)
    for i in range(100):
        feature_names.append(f'signal_sample_{i}')
    
    return feature_names


def save_final_results(
    output_dir: Path,
    features: np.ndarray,
    labels: np.ndarray,
    connections: np.ndarray,
    subject
):
    """Save final results"""
    np.save(output_dir / subject / f'{subject}_features.npy', features)
    np.save(output_dir / subject / f'{subject}_labels.npy', labels)
    np.save(output_dir /subject / f'{subject}_connections.npy', connections)

def save_intermediate_results(
    output_dir: Path,
    features: List[np.ndarray],
    labels: List[np.ndarray],
    connections: List[np.ndarray],
    batch_idx: int
):
    
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    np.save(checkpoint_dir / f'features_checkpoint_{batch_idx}.npy', np.vstack(features))
    np.save(checkpoint_dir / f'labels_checkpoint_{batch_idx}.npy', np.hstack(labels))
    np.save(checkpoint_dir / f'connections_checkpoint_{batch_idx}.npy', np.vstack(connections))

def save_metadata(
    output_dir: Path,
    word_to_label: Dict[str, int],
    channel_indices: Optional[List[int]],
    use_reduced_channels: bool,
    recommended_channels: Dict[str, int],
    n_channels: int
):
    """Save metadata files"""
    # Save word-label mapping
    with open(output_dir / 'word_to_label.json', 'w') as f:
        json.dump(word_to_label, f, indent=2)
    
    # Save channel information
    if use_reduced_channels:
        channel_info = {
            'recommended_channels': recommended_channels,
            'channel_indices': channel_indices,
            'n_channels': n_channels
        }
        with open(output_dir / 'channel_info.json', 'w') as f:
            json.dump(channel_info, f, indent=2)

def is_valid_eeg(eeg):
    """Check if EEG data has valid shape (not (2,) which indicates no data)"""
    if len(eeg.shape) != 2:
        return False
    # Additional check: ensure it's not the invalid (2,) shape
    if eeg.shape == (2,):
        return False
    if eeg.shape[0] < 50:
        return False
    return True


def has_valid_eeg_trials(f, eeg_data) -> bool:
  
    if hasattr(eeg_data, 'shape') and eeg_data.shape == (2,):
        return False
    

    if hasattr(eeg_data, 'dtype') and eeg_data.dtype == 'object':
        if len(eeg_data.shape) == 2:
            n_trials = eeg_data.shape[0]
        elif len(eeg_data.shape) == 1:
            n_trials = eeg_data.shape[0]
        else:
            n_trials = 1
 
        for trial_idx in range(n_trials):
            if len(eeg_data.shape) == 2:
                trial_ref = eeg_data[trial_idx, 0]
            else:
                trial_ref = eeg_data[trial_idx]
            
            if trial_ref:
                try:
                    actual_eeg = f[trial_ref][:]
                    if is_valid_eeg(actual_eeg):
                        return True
                except:
                    continue
    else:
        # Direct EEG data
        try:
            actual_eeg = eeg_data[:]
            return is_valid_eeg(actual_eeg)
        except:
            return False
    
    return False