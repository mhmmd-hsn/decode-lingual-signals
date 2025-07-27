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
import numba
from numba import jit, prange
warnings.filterwarnings('ignore')


import numpy as np
from scipy import stats, signal
from scipy.spatial.distance import cdist
import pywt
from numba import jit
import warnings
warnings.filterwarnings('ignore')

def extract_channel_features_enhanced(channel_signal, fs=500):
    """
    Enhanced feature extraction optimized for language decoding.
    Extracts ~150 features per channel focusing on language-relevant characteristics.
    """
    features = []
    N = len(channel_signal)
    
    # === TIME-DOMAIN FEATURES (25 features) ===
    
    # Basic statistics (10 features)
    features.extend([
        np.mean(channel_signal),
        np.var(channel_signal),
        np.std(channel_signal),
        stats.skew(channel_signal),
        stats.kurtosis(channel_signal),
        np.sqrt(np.mean(channel_signal**2)),  # RMS
        np.median(channel_signal),
        np.min(channel_signal),
        np.max(channel_signal),
        np.ptp(channel_signal),  # peak-to-peak
    ])
    
    # Percentiles (7 features)
    for p in [5, 10, 25, 75, 90, 95]:
        features.append(np.percentile(channel_signal, p))
    features.append(np.percentile(channel_signal, 75) - np.percentile(channel_signal, 25))  # IQR
    
    # Morphological features (8 features)
    # Zero crossings
    zero_crossings = np.sum(np.diff(np.sign(channel_signal)) != 0)
    features.append(zero_crossings)
    features.append(zero_crossings / N)  # ZC rate
    
    # Slope sign changes
    diff_signal = np.diff(channel_signal)
    ssc = np.sum(np.diff(np.sign(diff_signal)) != 0)
    features.append(ssc)
    features.append(ssc / (N-2))  # SSC rate
    
    # Waveform length
    wl = np.sum(np.abs(np.diff(channel_signal)))
    features.append(wl)
    features.append(wl / (N-1))  # normalized WL
    
    # Mean absolute value slope
    mavs = np.mean(np.abs(np.diff(channel_signal)))
    features.append(mavs)
    
    # Variance of absolute gradient
    vag = np.var(np.abs(np.gradient(channel_signal)))
    features.append(vag)
    
    # === HJORTH PARAMETERS (3 features) ===
    activity = np.var(channel_signal)
    diff1 = np.diff(channel_signal)
    diff2 = np.diff(diff1)
    mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
    complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility if mobility > 0 and np.var(diff1) > 0 else 0
    features.extend([activity, mobility, complexity])
    
    # === FREQUENCY-DOMAIN FEATURES (60 features) ===
    
    # FFT
    fft_vals = np.abs(np.fft.rfft(channel_signal))
    fft_freq = np.fft.rfftfreq(N, d=1/fs)
    
    # Extended frequency bands for language processing
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha1': (8, 10),  # Lower alpha
        'alpha2': (10, 13), # Upper alpha
        'beta1': (13, 20),  # Lower beta
        'beta2': (20, 30),  # Upper beta
        'gamma1': (30, 50), # Lower gamma
        'gamma2': (50, 80), # Mid gamma
        'gamma3': (80, 100) # High gamma (important for language)
    }
    
    band_powers = {}
    total_power = np.sum(fft_vals**2)
    
    # For each band: absolute power, relative power, log power (27 features)
    for band_name, (low, high) in bands.items():
        band_mask = (fft_freq >= low) & (fft_freq < high)
        band_power = np.sum(fft_vals[band_mask]**2)
        band_powers[band_name] = band_power
        
        features.append(band_power)  # Absolute power
        features.append(band_power / total_power if total_power > 0 else 0)  # Relative power
        features.append(np.log(band_power + 1e-10))  # Log power
    
    # Band power ratios important for language (15 features)
    features.extend([
        band_powers['theta'] / (band_powers['alpha1'] + band_powers['alpha2'] + 1e-10),
        band_powers['alpha1'] / (band_powers['alpha2'] + 1e-10),
        band_powers['beta1'] / (band_powers['beta2'] + 1e-10),
        band_powers['gamma1'] / (band_powers['gamma2'] + 1e-10),
        band_powers['gamma2'] / (band_powers['gamma3'] + 1e-10),
        (band_powers['theta'] + band_powers['alpha1']) / (band_powers['beta1'] + band_powers['beta2'] + 1e-10),
        band_powers['delta'] / (band_powers['theta'] + 1e-10),
        (band_powers['gamma1'] + band_powers['gamma2'] + band_powers['gamma3']) / (total_power + 1e-10),
        band_powers['alpha1'] / (band_powers['beta1'] + 1e-10),
        band_powers['alpha2'] / (band_powers['beta2'] + 1e-10),
        (band_powers['beta1'] + band_powers['beta2']) / (band_powers['gamma1'] + band_powers['gamma2'] + 1e-10),
        band_powers['theta'] / (band_powers['gamma1'] + 1e-10),
        (band_powers['alpha1'] + band_powers['alpha2']) / (total_power + 1e-10),
        (band_powers['beta1'] + band_powers['beta2']) / (total_power + 1e-10),
        (band_powers['delta'] + band_powers['theta']) / (band_powers['alpha1'] + band_powers['alpha2'] + 1e-10),
    ])
    
    # Spectral features (18 features)
    # Spectral centroid
    spectral_centroid = np.sum(fft_freq * fft_vals**2) / (total_power + 1e-10)
    features.append(spectral_centroid)
    
    # Spectral spread
    spectral_spread = np.sqrt(np.sum(((fft_freq - spectral_centroid)**2) * fft_vals**2) / (total_power + 1e-10))
    features.append(spectral_spread)
    
    # Spectral edge frequencies (important for speech/language)
    for percentile in [50, 75, 80, 85, 90, 95]:
        cumsum_power = np.cumsum(fft_vals**2)
        cumsum_power = cumsum_power / cumsum_power[-1]
        idx = np.where(cumsum_power >= percentile/100)[0]
        if len(idx) > 0:
            features.append(fft_freq[idx[0]])
        else:
            features.append(0)
    
    # Spectral flux
    spectral_flux = np.mean(np.diff(fft_vals))
    features.append(spectral_flux)
    
    # Spectral rolloff
    rolloff_idx = np.where(np.cumsum(fft_vals) >= 0.85 * np.sum(fft_vals))[0]
    spectral_rolloff = fft_freq[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
    features.append(spectral_rolloff)
    
    # Peak frequency and magnitude
    peak_idx = np.argmax(fft_vals)
    features.append(fft_freq[peak_idx])
    features.append(fft_vals[peak_idx])
    
    # Spectral flatness and entropy
    geometric_mean = np.exp(np.mean(np.log(fft_vals + 1e-10)))
    arithmetic_mean = np.mean(fft_vals)
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
    features.append(spectral_flatness)
    
    # Spectral entropy
    psd_norm = fft_vals / np.sum(fft_vals)
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
    features.append(spectral_entropy)
    
    # Spectral skewness and kurtosis
    features.extend([
        stats.skew(fft_vals),
        stats.kurtosis(fft_vals)
    ])
    
    # === ENTROPY MEASURES (8 features) ===
    
    # Shannon entropy
    hist, _ = np.histogram(channel_signal, bins=20)
    hist = hist / np.sum(hist)
    shannon_entropy = -np.sum(hist * np.log(hist + 1e-10))
    features.append(shannon_entropy)
    
    # Approximate entropy (simplified)
    def approx_entropy_fast(signal, m=2, r=None):
        if r is None:
            r = 0.15 * np.std(signal)
        N = len(signal)
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def _phi(m):
            patterns = np.array([signal[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            for i in range(N - m + 1):
                matching = 0
                for j in range(N - m + 1):
                    if _maxdist(patterns[i], patterns[j]) <= r:
                        matching += 1
                C[i] = matching / (N - m + 1)
            return np.mean(np.log(C))
        
        return _phi(m) - _phi(m + 1)
    
    features.append(approx_entropy_fast(channel_signal))
    
    # Permutation entropy (order 3)
    def perm_entropy_fast(signal, order=3):
        n = len(signal)
        if n < order:
            return 0
        
        # Get all permutations
        perms = {}
        for i in range(n - order + 1):
            sorted_indices = tuple(np.argsort(signal[i:i+order]))
            perms[sorted_indices] = perms.get(sorted_indices, 0) + 1
        
        # Calculate entropy
        total = sum(perms.values())
        probs = [count/total for count in perms.values()]
        return -sum(p * np.log(p) for p in probs if p > 0)
    
    features.append(perm_entropy_fast(channel_signal))
    
    # Renyi entropy (order 2)
    renyi_entropy = -np.log(np.sum(hist**2) + 1e-10)
    features.append(renyi_entropy)
    
    # Tsallis entropy (q=2)
    tsallis_entropy = (1 - np.sum(hist**2)) / (2 - 1)
    features.append(tsallis_entropy)
    
    # Log energy entropy
    log_energy = np.log(channel_signal**2 + 1e-10)
    log_energy_entropy = -np.sum(log_energy / np.sum(log_energy) * np.log(log_energy / np.sum(log_energy) + 1e-10))
    features.append(log_energy_entropy)
    
    # SVD entropy
    try:
        # Create embedding matrix
        embed_dim = min(10, N//10)
        if embed_dim > 2:
            embed_matrix = np.array([channel_signal[i:i+embed_dim] for i in range(N-embed_dim+1)])
            _, S, _ = np.linalg.svd(embed_matrix, compute_uv=False)
            S = S / np.sum(S)
            svd_entropy = -np.sum(S * np.log(S + 1e-10))
            features.append(svd_entropy)
        else:
            features.append(0)
    except:
        features.append(0)
    
    # Sample entropy (simplified)
    sample_entropy = approx_entropy_fast(channel_signal, m=2) * 0.9  # Approximation
    features.append(sample_entropy)
    
    # === WAVELET FEATURES (30 features) ===
    
    # Multi-level DWT
    wavelet = 'db4'
    try:
        max_level = pywt.dwt_max_level(N, pywt.Wavelet(wavelet).dec_len)
        coeffs = pywt.wavedec(channel_signal, wavelet, level=min(max_level, 5))
        
        # For each level: energy, log energy, relative energy, entropy, variance
        total_wavelet_energy = sum(np.sum(c**2) for c in coeffs)
        
        for i, coeff in enumerate(coeffs[:6]):  # Max 6 levels
            energy = np.sum(coeff**2)
            features.append(energy)
            features.append(np.log(energy + 1e-10))
            features.append(energy / (total_wavelet_energy + 1e-10))
            
            # Wavelet entropy
            if energy > 0:
                coeff_norm = coeff**2 / energy
                coeff_norm = coeff_norm[coeff_norm > 0]
                wavelet_entropy = -np.sum(coeff_norm * np.log(coeff_norm))
            else:
                wavelet_entropy = 0
            features.append(wavelet_entropy)
            
            # Variance of coefficients
            features.append(np.var(coeff))
        
        # Pad if less than 6 levels
        while len(features) < 88 + 30:  # Previous features + 30 wavelet features
            features.append(0)
            
    except:
        features.extend([0] * 30)
    
    # === NONLINEAR FEATURES (15 features) ===
    
    # Higuchi Fractal Dimension
    def higuchi_fd(signal, k_max=10):
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
    
    features.append(higuchi_fd(channel_signal))
    
    # Petrosian Fractal Dimension
    diff = np.diff(channel_signal)
    N_delta = np.sum(np.diff(np.sign(diff)) != 0)
    petrosian_fd = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * N_delta)))
    features.append(petrosian_fd)
    
    # Katz Fractal Dimension
    L = np.sum(np.sqrt(1 + np.diff(channel_signal)**2))
    d = np.sqrt((N-1)**2 + (channel_signal[-1] - channel_signal[0])**2)
    katz_fd = np.log10(L) / np.log10(d)
    features.append(katz_fd)
    
    # Detrended Fluctuation Analysis
    def dfa_fast(signal):
        N = len(signal)
        if N < 20:
            return 0.5
            
        # Integrate
        y = np.cumsum(signal - np.mean(signal))
        
        # Calculate for a few scales
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
        else:
            return 0.5
    
    features.append(dfa_fast(channel_signal))
    
    # Hurst exponent
    def hurst_fast(signal):
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
        else:
            return 0.5
    
    features.append(hurst_fast(channel_signal))
    
    # Lyapunov exponent (simplified)
    lyap = np.mean(np.log(np.abs(np.diff(channel_signal)) + 1e-10))
    features.append(lyap)
    
    # Correlation dimension (simplified)
    corr_dim = 2.0  # Placeholder - full calculation is expensive
    features.append(corr_dim)
    
    # === PHASE SPACE FEATURES (8 features) ===
    
    # Teager-Kaiser Energy
    if N > 2:
        tke = np.mean(channel_signal[1:-1]**2 - channel_signal[:-2] * channel_signal[2:])
    else:
        tke = 0
    features.append(tke)
    
    # Phase synchronization features
    analytic_signal = signal.hilbert(channel_signal)
    instantaneous_phase = np.angle(analytic_signal)
    
    # Phase coherence
    phase_diff = np.diff(instantaneous_phase)
    phase_coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
    features.append(phase_coherence)
    
    # Instantaneous frequency stats
    inst_freq = np.diff(instantaneous_phase) * fs / (2 * np.pi)
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
    mean_freq = np.mean(np.abs(inst_freq)) if len(inst_freq) > 0 else 0
    features.append(mean_freq)
    
    # === TIME-FREQUENCY COUPLING (12 features) ===
    
    # Phase-Amplitude Coupling indicators
    # Low frequency phases (theta, alpha) vs high frequency amplitudes (gamma)
    
    # Extract phase from low frequencies
    theta_band = signal.butter(4, [4, 8], btype='band', fs=fs, output='sos')
    theta_signal = signal.sosfilt(theta_band, channel_signal)
    theta_phase = np.angle(signal.hilbert(theta_signal))
    
    alpha_band = signal.butter(4, [8, 13], btype='band', fs=fs, output='sos')
    alpha_signal = signal.sosfilt(alpha_band, channel_signal)
    alpha_phase = np.angle(signal.hilbert(alpha_signal))
    
    # Extract amplitude from high frequencies
    gamma_band = signal.butter(4, [30, 100], btype='band', fs=fs, output='sos')
    gamma_signal = signal.sosfilt(gamma_band, channel_signal)
    gamma_amplitude = np.abs(signal.hilbert(gamma_signal))
    
    # PAC metrics
    # Theta-gamma PAC
    theta_gamma_pac = np.abs(np.mean(gamma_amplitude * np.exp(1j * theta_phase)))
    features.append(theta_gamma_pac)
    
    # Alpha-gamma PAC
    alpha_gamma_pac = np.abs(np.mean(gamma_amplitude * np.exp(1j * alpha_phase)))
    features.append(alpha_gamma_pac)
    
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
        
        # Modulation index
        pac_values = np.array(pac_values)
        if np.sum(pac_values) > 0:
            pac_norm = pac_values / np.sum(pac_values)
            mi = (np.log(n_phases) + np.sum(pac_norm * np.log(pac_norm + 1e-10))) / np.log(n_phases)
        else:
            mi = 0
        features.append(mi)
    
    # Cross-frequency ratios
    features.extend([
        np.mean(gamma_amplitude) / (np.mean(np.abs(theta_signal)) + 1e-10),
        np.mean(gamma_amplitude) / (np.mean(np.abs(alpha_signal)) + 1e-10),
        np.std(gamma_amplitude) / (np.std(theta_signal) + 1e-10),
        np.std(gamma_amplitude) / (np.std(alpha_signal) + 1e-10),
        np.max(gamma_amplitude) / (np.max(np.abs(theta_signal)) + 1e-10),
        np.max(gamma_amplitude) / (np.max(np.abs(alpha_signal)) + 1e-10),
    ])
    
    # Mean amplitude
    features.append(np.mean(gamma_amplitude))
    
    # Ensure we have exactly 150 features
    features = features[:150]
    while len(features) < 150:
        features.append(0)
    
    return np.array(features, dtype=np.float32)

def extract_hybrid_features(channel_signal, fs=500):
    """
    Combines extracted features with downsampled raw signal.
    """
    # Extract 100 key features
    features = extract_channel_features_enhanced(channel_signal, fs)
    
    # Add 50 downsampled time points
    downsampled = signal.resample(channel_signal, 100)
    
    # Combine
    combined = np.concatenate([features, downsampled])
    
    return combined

def load_matlab_string(matlab_str_array):
    """Convert MATLAB string array to Python string."""
    if isinstance(matlab_str_array, np.ndarray):
        if len(matlab_str_array.shape) == 2:
            if matlab_str_array.shape[0] > matlab_str_array.shape[1]:
                matlab_str_array = matlab_str_array.T
            return ''.join(chr(int(c)) for c in matlab_str_array.flatten() if c != 0)
        else:
            return ''.join(chr(int(c)) for c in matlab_str_array if c != 0)
    else:
        return str(matlab_str_array)


def process_single_trial(args):
    """Process a single trial - used for parallel processing."""
    trial_eeg, word_label = args
    n_channels, n_timepoints = trial_eeg.shape
    
    # Extract features for each channel
    channel_features = np.zeros((n_channels, 250), dtype=np.float32)
    
    for ch_idx in range(n_channels):
        features = extract_hybrid_features(trial_eeg[ch_idx])
        channel_features[ch_idx] = features
    
    # Replace NaN and inf values
    channel_features = np.nan_to_num(channel_features, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Compute correlation matrix (simplified)
    # Use only first 10 features for correlation to speed up
    corr_matrix = np.corrcoef(channel_features[:, :10])
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # Ensure correlation matrix is the right size
    if corr_matrix.shape != (n_channels, n_channels):
        corr_matrix = np.eye(n_channels, dtype=np.float32)
    
    return channel_features, word_label, corr_matrix

def extract_word_level_features_parallel(root_path, output_dir=None, max_samples=None, n_workers=None, use_reduced_channels=True):
    """
    Optimized parallel extraction of features from ZuCo dataset.
    
    Args:
        root_path: Path to directory containing ZuCo .mat files
        output_dir: Directory to save output files (default: same as root_path)
        max_samples: Maximum number of samples to process (for testing)
        n_workers: Number of parallel workers (default: CPU count - 1)
        use_reduced_channels: If True, use only 12 recommended channels (default: True)
    """
    if output_dir is None:
        output_dir = root_path
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    root_path = Path(root_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Define recommended channels for language processing
    if use_reduced_channels:
        recommended_channels = {
            # Left temporal (language comprehension)
            'E36': 28, 'E37': 29, 'E41': 33, 'E46': 38,
            # Left frontal (speech production)
            'E19': 14, 'E23': 17, 'E24': 18, 'E28': 21,
            # Central
            'Cz': 104, 'E6': 4, 'E13': 10,
            # Left parietal (integration)
            'E52': 42
        }
        # Get channel indices (0-based)
        channel_indices = sorted(recommended_channels.values())
        n_channels = len(channel_indices)
        print(f"Using {n_channels} recommended channels for language processing")
    else:
        channel_indices = None
        n_channels = 105
        print(f"Using all {n_channels} channels")
    
    # Lists to collect trial data for parallel processing
    all_trials = []
    
    # Word to label mapping
    word_to_label = {}
    label_counter = 0
    
    # Get all .mat files
    mat_files = list(root_path.glob("*.mat"))
    
    print(f"Found {len(mat_files)} .mat files")
    print(f"Using {n_workers} parallel workers")
    
    # First pass: collect all trials
    total_trials = 0
    sentences_processed = 0
    
    for file_path in tqdm(mat_files, desc="Collecting trials"):
        subject_id = file_path.stem
        
        with h5py.File(file_path, 'r') as f:
            if 'sentenceData' not in f:
                continue
            
            sentence_data = f['sentenceData']
            
            if 'content' not in sentence_data:
                continue
            
            content_data = sentence_data['content']
            n_sentences = content_data.shape[0]  # Use shape[0] for number of sentences
            
            print(f"\n  Processing {subject_id}: {n_sentences} sentences")
            
            # Process ALL sentences
            for sent_idx in range(n_sentences):
                try:
                    # Get sentence text
                    sentence_ref = content_data[sent_idx][0]
                    sentence_text = load_matlab_string(f[sentence_ref])
                    
                    # Access word data for this sentence
                    if 'word' not in sentence_data:
                        print(f"    No 'word' field in sentence_data")
                        continue
                    
                    word_field = sentence_data['word']
                    
                    # Check if this sentence has word data
                    if sent_idx >= word_field.shape[0]:
                        print(f"    Sentence {sent_idx} exceeds word field shape {word_field.shape}")
                        continue
                    
                    word_ref = word_field[sent_idx, 0]  # Changed indexing order!
                    if not word_ref:
                        print(f"    No word reference for sentence {sent_idx}")
                        continue
                    
                    word_group = f[word_ref]
                    
                    if 'content' not in word_group or 'rawEEG' not in word_group:
                        print(f"    Missing content or rawEEG in word group for sentence {sent_idx}")
                        continue
                    
                    content = word_group['content']
                    raw_eeg = word_group['rawEEG']
                    
                    n_words = content.shape[0]
                    
                    # Extract words
                    words = []
                    for word_idx in range(n_words):
                        word_ref = content[word_idx, 0]
                        word_data = f[word_ref][:]
                        word_text = word_data.tobytes().decode('utf-16-le').rstrip('\x00')
                        words.append(word_text)
                        
                        if word_text not in word_to_label:
                            word_to_label[word_text] = label_counter
                            label_counter += 1
                    
                    # Collect EEG trials
                    trials_in_sentence = 0
                    for word_idx in range(min(n_words, raw_eeg.shape[0])):
                        eeg_ref = raw_eeg[word_idx, 0]
                        if not eeg_ref:
                            continue
                        
                        eeg_word_data = f[eeg_ref]
                        
                        # Extract trials
                        trials = extract_trials(f, eeg_word_data, words[word_idx])
                        
                        for trial_eeg in trials:
                            # Select only recommended channels if specified
                            if use_reduced_channels and channel_indices:
                                # Ensure we don't exceed available channels
                                available_channels = trial_eeg.shape[0]
                                valid_indices = [idx for idx in channel_indices if idx < available_channels]
                                trial_eeg_filtered = trial_eeg[valid_indices, :]
                                
                                # Skip if we don't have enough valid channels
                                if trial_eeg_filtered.shape[0] < n_channels:
                                    continue
                            else:
                                trial_eeg_filtered = trial_eeg
                            
                            all_trials.append((trial_eeg_filtered, word_to_label[words[word_idx]]))
                            total_trials += 1
                            trials_in_sentence += 1
                            
                            if max_samples and total_trials >= max_samples:
                                break
                        
                        if max_samples and total_trials >= max_samples:
                            break
                    
                    if trials_in_sentence > 0:
                        sentences_processed += 1
                    
                    if max_samples and total_trials >= max_samples:
                        break
                
                except Exception as e:
                    print(f"    Error processing sentence {sent_idx}: {e}")
                    continue
            
            if max_samples and total_trials >= max_samples:
                break
            
            print(f"    Processed {sentences_processed} sentences with valid trials")
            print(f"    Collected {total_trials} trials so far...")
    
    print(f"\nCollected {len(all_trials)} trials")
    print(f"Unique words: {len(word_to_label)}")
    
    # Save channel information
    if use_reduced_channels:
        channel_info = {
            'recommended_channels': recommended_channels,
            'channel_indices': channel_indices,
            'n_channels': n_channels
        }
        import json
        with open(output_dir / 'channel_info.json', 'w') as f:
            json.dump(channel_info, f, indent=2)
    
    # Process trials in parallel
    print("\nProcessing trials in parallel...")
    
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_trial, all_trials, chunksize=100),
            total=len(all_trials),
            desc="Extracting features"
        ))
    
    # Unpack results
    all_features = [r[0] for r in results]
    all_labels = [r[1] for r in results]
    all_connections = [r[2] for r in results]
    
    # Convert to numpy arrays
    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int32)
    connections_array = np.array(all_connections, dtype=np.float32)
    
    print(f"\nExtraction complete!")
    print(f"Total samples: {len(features_array)}")
    print(f"Features shape: {features_array.shape}")
    print(f"Labels shape: {labels_array.shape}")
    print(f"Connections shape: {connections_array.shape}")
    
    # Save arrays
    np.save(output_dir / 'features.npy', features_array)
    np.save(output_dir / 'labels.npy', labels_array)
    np.save(output_dir / 'connections.npy', connections_array)
    
    # Save word-label mapping and channel info
    import json
    with open(output_dir / 'word_to_label.json', 'w') as f:
        json.dump(word_to_label, f, indent=2)
    
    print(f"\nSaved files to {output_dir}")
    print(f"Channel configuration: {n_channels} channels")
    if use_reduced_channels:
        print(f"Using recommended language-processing channels: {list(recommended_channels.keys())}")
    
    return features_array, labels_array, connections_array

def extract_trials(f, eeg_data, word):
    """Extract valid EEG trials from word-level EEG data."""
    trials = []
    
    # Handle case where eeg_data is invalid
    if hasattr(eeg_data, 'shape') and eeg_data.shape == (2,):
        return trials
    
    # Handle object array (multiple trials)
    if hasattr(eeg_data, 'dtype') and eeg_data.dtype == 'object':
        # Determine number of trials
        if len(eeg_data.shape) == 2:
            n_trials = eeg_data.shape[0]
        elif len(eeg_data.shape) == 1:
            n_trials = eeg_data.shape[0]
        else:
            n_trials = 1
        
        for trial_idx in range(n_trials):
            # Get trial reference
            if len(eeg_data.shape) == 2:
                trial_ref = eeg_data[trial_idx, 0]
            else:
                trial_ref = eeg_data[trial_idx]
            
            if trial_ref:
                actual_eeg = f[trial_ref][:]
                
                if is_valid_eeg(actual_eeg):
                    # Transpose to get [channels, timepoints]
                    trials.append(actual_eeg.T)
    else:
        # Single trial case
        actual_eeg = eeg_data[:]
        
        if is_valid_eeg(actual_eeg):
            # Transpose to get [channels, timepoints]
            trials.append(actual_eeg.T)
    
    return trials

def is_valid_eeg(eeg, min_samples=50):
    """Check if EEG data is valid."""
    if len(eeg.shape) != 2:
        return False
    
    # Check minimum samples
    if eeg.shape[0] < min_samples and eeg.shape[1] < min_samples:
        return False
    
    # Check for all NaN or all zero
    if np.all(np.isnan(eeg)) or np.all(eeg == 0):
        return False
    
    return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from ZuCo dataset (optimized)')
    parser.add_argument('--input', type=str, required=True, help='Path to ZuCo dataset directory')
    parser.add_argument('--output', type=str, default=None, help='Output directory (default: same as input)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples to process (for testing)')
    parser.add_argument('--n_workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--all_channels', action='store_true', help='Use all 105 channels instead of recommended 12')
    
    args = parser.parse_args()
    
    # Extract features with parallel processing
    features, labels, connections = extract_word_level_features_parallel(
        root_path=args.input,
        output_dir=args.output,
        max_samples=args.max_samples,
        n_workers=args.n_workers,
        use_reduced_channels=not args.all_channels
    )
    
    print("\nFeature extraction completed successfully!")