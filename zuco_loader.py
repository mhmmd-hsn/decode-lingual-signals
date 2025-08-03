import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import json
import re
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

from feature_extraction import *
from utils import *

import pickle

def extract_word_level_features_optimized(
    root_path: str,
    output_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    n_workers: Optional[int] = None,
    use_reduced_channels: bool = True,
    batch_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if output_dir is None:
        output_dir = root_path
    
    config = FeatureConfig(n_workers=n_workers)
    processor = OptimizedDataProcessor(config)
    
    root_path = Path(root_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if use_reduced_channels:
        recommended_channels = {
            'E36': 28, 'E37': 29, 'E41': 33, 'E46': 38,
            'E19': 14, 'E23': 17, 'E24': 18, 'E28': 21,
            'Cz': 104, 'E6': 4, 'E13': 10,
            'E52': 42
        }
        channel_indices = sorted(recommended_channels.values())
        n_channels = len(channel_indices)
    else:
        channel_indices = None
        n_channels = 105
        with open('zuco/eeg_channel_mapping.json', 'r') as f:
            f = json.load(f)
            recommended_channels = f['channel_to_index']
    
    print(f"Using {n_channels} channels")
    print(f"Batch size: {batch_size}")
    
    all_trials = []
    word_to_label = {}
    label_counter = 0
    
    mat_files = list(root_path.glob("*.mat"))
    print(f"Found {len(mat_files)} .mat files")
    
    for file_path in tqdm(mat_files, desc="Collecting trials"):

        subject = str(file_path).split("results")[1].split("_")[0]

        trials, label_counter = load_trials_from_file(
            file_path, 
            word_to_label, 
            label_counter,
            channel_indices,
            n_channels,
            max_samples
        )

        all_trials.append(trials)

        print(f"\nCollected {len(trials)} trials from {subject}")
        print(f"Unique words with valid EEG: {len(word_to_label)}")
        
        if max_samples and len(trials) >= max_samples:
            trials = trials[:max_samples]


    with open(output_dir / 'all_trials.pkl', 'wb') as f:
        pickle.dump(trials, f)

        # n_batches = (len(trials) + batch_size - 1) // batch_size

        # all_trials = []
        # all_labels = []
        # all_connections = []

        # for batch_idx in range(n_batches):
        #     start_idx = batch_idx * batch_size
        #     end_idx = min((batch_idx + 1) * batch_size, len(trials))
        #     batch = trials[start_idx:end_idx]
            
        #     print(f"\nProcessing batch {batch_idx + 1}/{n_batches} ({len(batch)} samples)")
            
        #     features, labels, connections = processor.process_batch(batch, n_workers)
            
        #     all_features.append(features)
        #     all_labels.append(labels)
        #     all_connections.append(connections)

        # features_array = np.vstack(all_features)
        # labels_array = np.hstack(all_labels)
        # connections_array = np.vstack(all_connections)

        # print(f"\nProcessing complete!")
        # print(f"Features shape: {features_array.shape}")
        # print(f"Labels shape: {labels_array.shape}")
        # print(f"Connections shape: {connections_array.shape}")

        # save_final_results(output_dir, features_array, labels_array, connections_array, subject)


    save_metadata(output_dir, word_to_label, channel_indices, use_reduced_channels, recommended_channels, n_channels)
    # feature_names = get_feature_names()
    # with open(output_dir / 'feature_names.json', 'w') as f:
    #     json.dump(feature_names, f, indent=2)
    
def load_trials_from_file(
    file_path: Path,
    word_to_label: Dict[str, int],
    label_counter: int,
    channel_indices: Optional[List[int]],
    n_channels: int,
    max_samples: Optional[int]
) -> List[Tuple]:

    trials = []    
    with h5py.File(file_path, 'r') as f:

        sentence_data = f['sentenceData']
        content_data = sentence_data['content']
        n_sentences = content_data.shape[0]
        
        for sent_idx in range(n_sentences):
            try:
                sentence_trials, label_counter = process_sentence(
                    f, content_data, sentence_data,
                    sent_idx, word_to_label, label_counter,
                    channel_indices, n_channels
                )
                trials.extend(sentence_trials)
                
                if max_samples and len(trials) >= max_samples:
                    break
                    
            except Exception as e:
                print(f"Error processing sentence {sent_idx}: {e}")
                continue
    
    return trials, label_counter

def process_sentence(
    f, content_data, sentence_data,
    sent_idx: int, word_to_label: Dict[str, int],
    label_counter: int, channel_indices: Optional[List[int]],
    n_channels: int
) -> List[Tuple]:

    trials = []
    word_field = sentence_data['word']    
    word_ref = word_field[sent_idx, 0]
    word_group = f[word_ref]

    content = word_group['content']
    raw_eeg = word_group['rawEEG']
    n_words = content.shape[0]

    # First pass: check which words have valid EEG data
    valid_words = []
    for word_idx in range(n_words):
        # Get word text
        word_ref = content[word_idx, 0]
        word_data = f[word_ref][:]
        word_text = word_data.tobytes().decode('utf-16-le').rstrip('\x00')
        
        # Check if word contains alphanumeric characters
        if not re.search('[a-zA-Z0-9]', word_text):
            continue
            
        # Check if this word has valid EEG data
        if word_idx < raw_eeg.shape[0]:
            eeg_ref = raw_eeg[word_idx, 0]
            if eeg_ref:
                eeg_word_data = f[eeg_ref]
                # Check if this word has any valid EEG trials
                if has_valid_eeg_trials(f, eeg_word_data):
                    valid_words.append((word_idx, word_text))

    # Second pass: assign labels only to words with valid EEG and extract trials
    for word_idx, word_text in valid_words:
        # Assign label only if word has valid EEG
        if word_text not in word_to_label:
            word_to_label[word_text] = label_counter
            label_counter += 1
        
        # Extract EEG trials
        eeg_ref = raw_eeg[word_idx, 0]
        eeg_word_data = f[eeg_ref]
        eeg_trials = extract_trials(f, eeg_word_data, word_text)
        
        for trial_eeg in eeg_trials:
            if channel_indices:
                available_channels = trial_eeg.shape[0]
                valid_indices = [idx for idx in channel_indices if idx < available_channels]
                trial_eeg_filtered = trial_eeg[valid_indices, :]
                
                if trial_eeg_filtered.shape[0] < n_channels:
                    continue
            else:
                trial_eeg_filtered = trial_eeg
            
            trials.append((trial_eeg_filtered, word_to_label[word_text]))
    
    return trials, label_counter

def extract_trials(f, eeg_data, word):
    """Extract valid EEG trials from word-level EEG data"""
    trials = []
    
    if hasattr(eeg_data, 'shape') and eeg_data.shape == (2,):
        return trials
    
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
                actual_eeg = f[trial_ref][:]
                
                if is_valid_eeg(actual_eeg):
                    trials.append(actual_eeg.T)
    else:
        actual_eeg = eeg_data[:]
        
        if is_valid_eeg(actual_eeg):
            trials.append(actual_eeg.T)
    
    return trials


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from ZuCo dataset (optimized)')
    parser.add_argument('--input', type=str, required=True, help='Path to ZuCo dataset directory')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples to process')
    parser.add_argument('--n_workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing')
    
    args = parser.parse_args()
    
    extract_word_level_features_optimized(
        root_path=args.input,
        output_dir=args.output,
        max_samples=args.max_samples,
        n_workers=args.n_workers,
        use_reduced_channels=False,
        batch_size=args.batch_size
    )
    
    print("\nFeature extraction completed successfully!")