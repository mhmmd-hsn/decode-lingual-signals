from Models.GAT import GATModel
from trainer import Trainer
import argparse
import torch
from torch.utils.data import Dataset
import numpy as np


class SimpleEEGDataset(Dataset):
    """Simple dataset to load pre-processed numpy arrays"""
    def __init__(self, features_path='zuco/features.npy', labels_path='zuco/labels.npy', connections_path='zuco/connections.npy'):
        # Load the numpy arrays
        self.features = np.load(features_path)  # Shape: (50, 12, 250)
        self.labels = np.load(labels_path)      # Shape: (50,)
        self.connections = np.load(connections_path)  # Shape: (50, 12, 12)
        
        print(f"Loaded dataset:")
        print(f"Features shape: {self.features.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Connections shape: {self.connections.shape}")
        
        # Verify shapes match
        assert self.features.shape[0] == self.labels.shape[0] == self.connections.shape[0], \
            "Number of samples must match across features, labels, and connections"
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        # Return feature matrix, adjacency matrix, and label
        feature_matrix = torch.as_tensor(self.features[idx], dtype=torch.float32)
        adjacency_matrix = torch.as_tensor(self.connections[idx], dtype=torch.float32)
        label = torch.as_tensor(self.labels[idx], dtype=torch.long)
        return feature_matrix, adjacency_matrix, label


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
    args = parser.parse_args()

    dataset = SimpleEEGDataset(
        features_path=args.features_path,
        labels_path=args.labels_path,
        connections_path=args.connections_path
    )

    model = GATModel(
        num_timepoints= dataset.features.shape[2], 
        num_classes= np.unique(dataset.labels).shape[0]
    )
    model.count_parameters()

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.bs,
        num_folds=args.folds,                    
        save_best=True,
        save_path="Outputs/best_model.pth",
        early_stopping=False,
        patience=10
    )
    
    # Run training based on selected method
    if args.training == '2-cv':
        trainer.train_split(train_ratio=0.7, val_ratio=0.3)
    elif args.training == '3-cv':
        trainer.train_split(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    elif args.training == 'k-fold':
        trainer.train_kfold()