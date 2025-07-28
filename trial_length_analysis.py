import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')


class ZuCoTrialLengthAnalyzer:
    
    
    def __init__(self, root_path, n_workers=None):
        self.root_path = Path(root_path)
        self.n_workers = n_workers if n_workers else max(1, cpu_count() - 2)
        self.trial_lengths = None
        self.stats = None
        
    def analyze(self, max_files=None):
        mat_files = list(self.root_path.glob("*.mat"))
        
        if max_files:
            mat_files = mat_files[:max_files]
        
        print(f"Found {len(mat_files)} .mat files")
        print(f"Using {self.n_workers} parallel workers")
        
        # Parallel processing
        with Pool(self.n_workers) as pool:
            all_results = list(tqdm(
                pool.imap(self._process_single_file, mat_files),
                total=len(mat_files),
                desc="Processing files"
            ))
        
        # Combine results
        all_trial_lengths = []
        for file_lengths in all_results:
            all_trial_lengths.extend(file_lengths)
        
        if not all_trial_lengths:
            print("No valid trials found!")
            return None, None
        
        self.trial_lengths = np.array(all_trial_lengths)
        self.stats = self._calculate_statistics()
        
        self._print_summary()
        self._plot_analysis()
        
        return self.stats, self.trial_lengths
    
    @staticmethod
    def _process_single_file(file_path):
        """Process a single file and return trial lengths."""
        trial_lengths = []
        
        try:
            with h5py.File(file_path, 'r') as f:
                if 'sentenceData' not in f:
                    return trial_lengths
                
                sentence_data = f['sentenceData']
                
                if 'content' not in sentence_data or 'word' not in sentence_data:
                    return trial_lengths
                
                content_data = sentence_data['content']
                word_field = sentence_data['word']
                n_sentences = min(content_data.shape[0], word_field.shape[0])
                
                # Process sentences
                for sent_idx in range(n_sentences):
                    try:
                        word_ref = word_field[sent_idx, 0]
                        if not word_ref:
                            continue
                        
                        word_group = f[word_ref]
                        
                        if 'rawEEG' not in word_group:
                            continue
                        
                        raw_eeg = word_group['rawEEG']
                        
                        # Quick scan through EEG data
                        for word_idx in range(raw_eeg.shape[0]):
                            eeg_ref = raw_eeg[word_idx, 0]
                            if not eeg_ref:
                                continue
                            
                            eeg_word_data = f[eeg_ref]
                            
                            # Quick extraction of trial lengths
                            if hasattr(eeg_word_data, 'shape') and eeg_word_data.shape == (2,):
                                continue
                            
                            # Handle multiple trials
                            if hasattr(eeg_word_data, 'dtype') and eeg_word_data.dtype == 'object':
                                if len(eeg_word_data.shape) == 2:
                                    n_trials = eeg_word_data.shape[0]
                                else:
                                    n_trials = eeg_word_data.shape[0] if len(eeg_word_data.shape) == 1 else 1
                                
                                for trial_idx in range(n_trials):
                                    if len(eeg_word_data.shape) == 2:
                                        trial_ref = eeg_word_data[trial_idx, 0]
                                    else:
                                        trial_ref = eeg_word_data[trial_idx] if len(eeg_word_data.shape) == 1 else eeg_word_data
                                    
                                    if trial_ref:
                                        actual_eeg = f[trial_ref]
                                        if hasattr(actual_eeg, 'shape') and len(actual_eeg.shape) == 2:
                                            # Get timepoints (samples are in first dimension)
                                            n_timepoints = actual_eeg.shape[0]
                                            if n_timepoints >= 50:  # Min threshold
                                                trial_lengths.append(n_timepoints)
                            else:
                                # Single trial
                                actual_eeg = eeg_word_data
                                if hasattr(actual_eeg, 'shape') and len(actual_eeg.shape) == 2:
                                    n_timepoints = actual_eeg.shape[0]
                                    if n_timepoints >= 50:
                                        trial_lengths.append(n_timepoints)
                    
                    except:
                        continue
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        return trial_lengths
    
    def _calculate_statistics(self):
        """Calculate comprehensive statistics from trial lengths."""
        stats = {
            'total_trials': len(self.trial_lengths),
            'mean_length': np.mean(self.trial_lengths),
            'median_length': np.median(self.trial_lengths),
            'std_length': np.std(self.trial_lengths),
            'min_length': np.min(self.trial_lengths),
            'max_length': np.max(self.trial_lengths),
            'percentile_25': np.percentile(self.trial_lengths, 25),
            'percentile_75': np.percentile(self.trial_lengths, 75),
            'percentile_10': np.percentile(self.trial_lengths, 10),
            'percentile_90': np.percentile(self.trial_lengths, 90),
            'percentile_95': np.percentile(self.trial_lengths, 95),
        }
        
        # Mode calculation
        counts = np.bincount(self.trial_lengths.astype(int))
        stats['mode_length'] = np.argmax(counts)
        stats['unique_lengths'] = len(np.unique(self.trial_lengths))
        
        return stats
    
    def _print_summary(self):
        """Print analysis summary."""
        print("\n" + "="*60)
        print("TRIAL LENGTH ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total trials analyzed: {self.stats['total_trials']}")
        print(f"\nTrial Length Statistics:")
        print(f"  Mean: {self.stats['mean_length']:.2f} samples")
        print(f"  Median: {self.stats['median_length']:.2f} samples")
        print(f"  Mode: {self.stats['mode_length']} samples")
        print(f"  Std Dev: {self.stats['std_length']:.2f} samples")
        print(f"  Min: {self.stats['min_length']} samples")
        print(f"  Max: {self.stats['max_length']} samples")
        print(f"  25th percentile: {self.stats['percentile_25']:.2f} samples")
        print(f"  75th percentile: {self.stats['percentile_75']:.2f} samples")
        print(f"  90th percentile: {self.stats['percentile_90']:.2f} samples")
        print(f"  95th percentile: {self.stats['percentile_95']:.2f} samples")
        
        print("\n" + "-"*60)
        print("RECOMMENDED TARGET LENGTHS:")
        print("-"*60)
        print(f"Option 1 - Keep ALL data (pad shorter): {int(self.stats['max_length'])} samples")
        print(f"Option 2 - Keep 95% of data: {int(self.stats['percentile_95'])} samples")
        print(f"Option 3 - Keep 90% of data: {int(self.stats['percentile_90'])} samples")
        print(f"Option 4 - Balanced (median): {int(self.stats['median_length'])} samples")
        print(f"Option 5 - Most common (mode): {self.stats['mode_length']} samples")
        
        # Show impact of different choices
        print("\n" + "-"*60)
        print("IMPACT ANALYSIS FOR COMMON TARGET LENGTHS:")
        print("-"*60)
        for target in [100, 150, 200, 250, 300]:
            truncated = np.sum(self.trial_lengths > target)
            padded = np.sum(self.trial_lengths < target)
            exact = np.sum(self.trial_lengths == target)
            print(f"\nTarget {target} samples:")
            print(f"  - Would truncate: {truncated} trials ({100*truncated/len(self.trial_lengths):.1f}%)")
            print(f"  - Would pad: {padded} trials ({100*padded/len(self.trial_lengths):.1f}%)")
            print(f"  - Exact match: {exact} trials ({100*exact/len(self.trial_lengths):.1f}%)")
    
    def _plot_analysis(self):
        """Create visualization plots."""
        plt.figure(figsize=(12, 8))
        
        # Histogram
        plt.subplot(2, 2, 1)
        plt.hist(self.trial_lengths, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(self.stats['mean_length'], color='red', linestyle='--', 
                   label=f"Mean: {self.stats['mean_length']:.0f}")
        plt.axvline(self.stats['median_length'], color='green', linestyle='--', 
                   label=f"Median: {self.stats['median_length']:.0f}")
        plt.axvline(self.stats['percentile_90'], color='orange', linestyle='--', 
                   label=f"90%: {self.stats['percentile_90']:.0f}")
        plt.xlabel('Trial Length (samples)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Trial Lengths')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(self.trial_lengths, vert=True)
        plt.ylabel('Trial Length (samples)')
        plt.title('Box Plot of Trial Lengths')
        plt.grid(True, alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(2, 2, 3)
        sorted_lengths = np.sort(self.trial_lengths)
        cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
        plt.plot(sorted_lengths, cumulative)
        for p in [75, 90, 95]:
            plt.axhline(p, color='gray', linestyle='--', alpha=0.5)
            plt.text(sorted_lengths[0], p+1, f'{p}%', fontsize=8)
        plt.xlabel('Trial Length (samples)')
        plt.ylabel('Cumulative Percentage (%)')
        plt.title('Cumulative Distribution')
        plt.grid(True, alpha=0.3)
        
        # Length distribution ranges
        plt.subplot(2, 2, 4)
        ranges = ['<100', '100-150', '150-200', '200-250', '250-300', '>300']
        range_counts = [
            np.sum(self.trial_lengths < 100),
            np.sum((self.trial_lengths >= 100) & (self.trial_lengths < 150)),
            np.sum((self.trial_lengths >= 150) & (self.trial_lengths < 200)),
            np.sum((self.trial_lengths >= 200) & (self.trial_lengths < 250)),
            np.sum((self.trial_lengths >= 250) & (self.trial_lengths < 300)),
            np.sum(self.trial_lengths >= 300)
        ]
        plt.bar(ranges, range_counts)
        plt.xlabel('Length Range (samples)')
        plt.ylabel('Number of Trials')
        plt.title('Trial Length Ranges')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def get_recommendation(self, strategy='balanced'):
        """
        Get recommended target length based on strategy.
        
        Args:
            strategy: One of 'conservative' (95%), 'balanced' (median), 
                     'aggressive' (75%), 'mode', or 'all' (max)
                     
        Returns:
            Recommended target length in samples
        """
        if self.stats is None:
            raise ValueError("Must run analyze() first")
        
        strategies = {
            'all': int(self.stats['max_length']),
            'conservative': int(self.stats['percentile_95']),
            'balanced': int(self.stats['median_length']),
            'aggressive': int(self.stats['percentile_75']),
            'mode': int(self.stats['mode_length'])
        }
        
        if strategy not in strategies:
            raise ValueError(f"Strategy must be one of {list(strategies.keys())}")
        
        return strategies[strategy]
    
    def analyze_impact(self, target_length):
        """
        Analyze impact of choosing a specific target length.
        
        Args:
            target_length: Target length in samples
            
        Returns:
            Dictionary with impact statistics
        """
        if self.trial_lengths is None:
            raise ValueError("Must run analyze() first")
        
        return {
            'target_length': target_length,
            'trials_truncated': int(np.sum(self.trial_lengths > target_length)),
            'trials_padded': int(np.sum(self.trial_lengths < target_length)),
            'trials_exact': int(np.sum(self.trial_lengths == target_length)),
            'percent_truncated': float(100 * np.sum(self.trial_lengths > target_length) / len(self.trial_lengths)),
            'percent_padded': float(100 * np.sum(self.trial_lengths < target_length) / len(self.trial_lengths)),
            'percent_exact': float(100 * np.sum(self.trial_lengths == target_length) / len(self.trial_lengths)),
            'max_truncation': int(np.max(self.trial_lengths) - target_length) if np.any(self.trial_lengths > target_length) else 0,
            'max_padding': int(target_length - np.min(self.trial_lengths)) if np.any(self.trial_lengths < target_length) else 0
        }


if __name__ == '__main__':
    # Example usage
    analyzer = ZuCoTrialLengthAnalyzer(root_path='zuco')
    stats, lengths = analyzer.analyze(max_files=None)
    
    # Get recommendations
    print(f"\nRecommended length (balanced): {analyzer.get_recommendation('balanced')}")
    print(f"Recommended length (conservative): {analyzer.get_recommendation('conservative')}")
    
    # Analyze specific target
    impact = analyzer.analyze_impact(200)
    print(f"\nImpact of target length 200:")
    print(f"  Would truncate: {impact['percent_truncated']:.1f}% of trials")
    print(f"  Would pad: {impact['percent_padded']:.1f}% of trials")