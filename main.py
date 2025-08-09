from Models.GAT import GATModel
from trainer import Trainer
import argparse
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from pathlib import Path
from feature_extraction import FeatureExtractor, FeatureConfig
from connections import GMatrixCalculator
from tqdm import tqdm


class SimpleEEGDataset(Dataset):
    def __init__(self, features=None, labels=None, connections=None, 
                 features_path=None, labels_path=None, connections_path=None):
        # If arrays are provided directly, use them
        if features is not None and labels is not None and connections is not None:
            self.features = features
            self.labels = labels
            self.connections = connections
        # Otherwise, load from files
        else:
            if features_path is None or labels_path is None or connections_path is None:
                raise ValueError("Either provide arrays directly or all file paths")
            self.features = np.load(features_path)
            self.labels = np.load(labels_path)
            self.connections = np.load(connections_path)
        
        print(f"Loaded dataset:")
        print(f"Features shape: {self.features.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Connections shape: {self.connections.shape}")
        
        assert self.features.shape[0] == self.labels.shape[0] == self.connections.shape[0], \
            "Number of samples must match across features, labels, and connections"
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        feature_matrix = torch.as_tensor(self.features[idx], dtype=torch.float32)
        adjacency_matrix = torch.as_tensor(self.connections[idx], dtype=torch.float32)
        label = torch.as_tensor(self.labels[idx], dtype=torch.long)
        return feature_matrix, adjacency_matrix, label


def prepare_data_from_pickle(pickle_path, target_length=250):
    with open(pickle_path, 'rb') as f:
        trials = pickle.load(f)
    
    print(f"Loaded {len(trials)} trials from pickle file")
    
    config = FeatureConfig(resample_length=target_length)
    feature_extractor = FeatureExtractor(config)
    g_matrix_calculator = GMatrixCalculator()
    
    features_list = []
    labels_list = []
    connections_list = []
    
    for trial_eeg, label in tqdm(trials, desc="Processing trials"):
        resampled_trial = feature_extractor.intelligent_resample_(trial_eeg, target_length)
        g_matrix = g_matrix_calculator._compute_G_matrix(resampled_trial)
        
        features_list.append(resampled_trial)
        labels_list.append(label)
        connections_list.append(g_matrix)
    
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    connections_array = np.array(connections_list)
    
    print(f"\nFinal data shapes:")
    print(f"Features: {features_array.shape}")
    print(f"Labels: {labels_array.shape}")
    print(f"Connections: {connections_array.shape}")
    
    return features_array, labels_array, connections_array


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--bs', type=int, default=16)    
    parser.add_argument('--training', type=str, default='2-cv', required=False, choices=['2-cv', '3-cv', 'k-fold'])
    parser.add_argument('--features_path', type=str, default='zuco/features.npy', help='Path to features numpy file')
    parser.add_argument('--labels_path', type=str, default='zuco/labels.npy', help='Path to labels numpy file')
    parser.add_argument('--connections_path', type=str, default='zuco/connections.npy', help='Path to connections numpy file')
    parser.add_argument('--pickle_path', type=str, default=None, help='Path to pickle file with trials')
    parser.add_argument('--target_length', type=int, default=150, help='Target length for resampling')
    args = parser.parse_args()

    if args.pickle_path:
        # Process pickle file and use data directly
        features, labels, connections = prepare_data_from_pickle(
            args.pickle_path, 
            args.target_length
        )
        # Create dataset with arrays directly
        dataset = SimpleEEGDataset(
            features=features,
            labels=labels,
            connections=connections
        )
    else:
        # Load from existing numpy files
        dataset = SimpleEEGDataset(
            features_path=args.features_path,
            labels_path=args.labels_path,
            connections_path=args.connections_path
        )

    print(dataset.features.shape[2])
    print(np.unique(dataset.labels).shape[0])

    model = GATModel(
        num_timepoints= dataset.features.shape[2], 
        num_classes= np.unique(dataset.labels).shape[0]
    )
    model.count_parameters()

    trainer = Trainer(
        model=model,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.bs,
        num_folds=args.folds,                    
        save_best=True,
        save_path="Outputs/best_model.pth",
        early_stopping=False,
        patience=10,
        top_k=10
    )
    
    if args.training == '2-cv':
        trainer.train_split(train_ratio=0.7, val_ratio=0.3)
    elif args.training == '3-cv':
        trainer.train_split(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    elif args.training == 'k-fold':
        trainer.train_kfold()