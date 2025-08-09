import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from visualization import Visualization
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, dataset, lr=0.0001, epochs=500, batch_size=80, 
                 num_folds=3, save_best=True, save_path="Outputs/best_model.pth",
                 early_stopping=False, patience=10, top_k=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_folds = num_folds
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        self.save_best = save_best
        self.save_path = save_path
        self.early_stopping = early_stopping
        self.patience = patience
        self.top_k = top_k  # Add top_k parameter
        
    
        self.best_acc = 0  
        self.best_top_k_acc = 0  # Track best top-k accuracy
        self.visualizer = Visualization()
        self.all_preds = []
        self.all_labels = []
        self.best_fold_metrics = None  
        
        
        self.train_losses = []
        self.train_accuracies = []
        self.train_top_k_accuracies = []  # Track top-k accuracies for training
        self.val_losses = []
        self.val_accuracies = []
        self.val_top_k_accuracies = []  # Track top-k accuracies for validation
        
       
        self.best_results = pd.DataFrame(columns=["Split", "Train Accuracy", "Train Top-10 Acc", 
                                                   "Valid Accuracy", "Valid Top-10 Acc", "Epoch Number"])
        
        if self.early_stopping and not self.save_best:
            print("Warning: Early stopping is enabled but save_best is False. The best model will not be saved.")

        assert hasattr(self.dataset, 'labels'), "Dataset must have a 'labels' attribute for stratification."

    def calculate_top_k_accuracy(self, outputs, labels, k=None):
        """
        Calculate top-k accuracy.
        
        Args:
            outputs: Model outputs (logits) of shape (batch_size, num_classes)
            labels: True labels of shape (batch_size,)
            k: Number of top predictions to consider (defaults to self.top_k)
        
        Returns:
            float: Top-k accuracy (proportion of samples where true label is in top-k predictions)
        """
        if k is None:
            k = self.top_k
            
        with torch.no_grad():
            # Get the top-k predictions
            _, top_k_preds = outputs.topk(k, dim=1, largest=True, sorted=True)
            
            # Check if true labels are in top-k predictions
            correct = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds))
            
            # Calculate accuracy
            top_k_acc = correct.any(dim=1).float().mean().item()
            
        return top_k_acc

    def train_kfold(self):
        """Train the model using k-fold cross-validation."""
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(self.dataset)), self.dataset.labels)):
            print(f"\nFold {fold + 1}/{self.num_folds}")
            
            
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)
            pin_memory = True if self.device.type == 'cuda' else False
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, pin_memory=pin_memory)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, pin_memory=pin_memory)

           
            self.model = self.model.__class__(num_timepoints=self.model.agacn1.weight.shape[0], 
                                              num_classes=self.model.fc.out_features).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0005)
            
           
            fold_train_losses, fold_train_accs, fold_train_top_k_accs = [], [], []
            fold_val_losses, fold_val_accs, fold_val_top_k_accs = [], [], []
            
            
            best_fold_acc = 0
            best_fold_top_k_acc = 0
            best_fold_epoch = 0
            epochs_no_improve = 0
            
            for epoch in range(self.epochs):
                
                avg_train_loss, train_acc, train_top_k_acc, _, _ = self._train_epoch(train_loader)
                val_loss, val_acc, val_top_k_acc, _, _ = self._validate_epoch(val_loader)

                fold_train_losses.append(avg_train_loss)
                fold_train_accs.append(train_acc)
                fold_train_top_k_accs.append(train_top_k_acc)
                fold_val_losses.append(val_loss)
                fold_val_accs.append(val_acc)
                fold_val_top_k_accs.append(val_top_k_acc)


                if epoch % 5 == 0:
                    print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, '
                          f'Train Acc: {train_acc:.4f}, Train Top-{self.top_k} Acc: {train_top_k_acc:.4f}, '
                          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                          f'Val Top-{self.top_k} Acc: {val_top_k_acc:.4f}')
                
                # You can choose to use either val_acc or val_top_k_acc for best model selection
                # Here using val_top_k_acc since it's more relevant for your use case
                if val_top_k_acc > best_fold_top_k_acc:
                    best_fold_acc = val_acc
                    best_fold_top_k_acc = val_top_k_acc
                    best_fold_epoch = epoch
                    epochs_no_improve = 0
                    
                    if self.save_best and val_top_k_acc > self.best_top_k_acc:
                        self.best_acc = val_acc
                        self.best_top_k_acc = val_top_k_acc
                        self.best_fold_metrics = (fold_train_losses, fold_train_accs, fold_val_losses, fold_val_accs,
                                                  fold_train_top_k_accs, fold_val_top_k_accs)
                        torch.save(self.model.state_dict(), self.save_path)
                else:
                    epochs_no_improve += 1
                
                
                if self.early_stopping and epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1} for fold {fold+1}")
                    break
            
            fold_results.append([f"Fold {fold+1}", 
                               fold_train_accs[best_fold_epoch],
                               fold_train_top_k_accs[best_fold_epoch],
                               best_fold_acc,
                               best_fold_top_k_acc,
                               best_fold_epoch+1])
        
        for result in fold_results:
            self.best_results = self.add_row(result)
        
        
        column_means = self.best_results.drop(columns=["Split"]).mean()
        average_row = pd.DataFrame([["Average"] + column_means.tolist()], columns=self.best_results.columns)
        self.best_results = pd.concat([self.best_results, average_row], ignore_index=True)
        
        print(self.best_results)

        if self.best_fold_metrics:
            self.visualizer.plot_loss(self.best_fold_metrics[0], self.best_fold_metrics[2], 
                                     save_path="Outputs/kfold_loss_curve.png")
            self.visualizer.plot_accuracy(self.best_fold_metrics[1], self.best_fold_metrics[3], 
                                         save_path="Outputs/kfold_accuracy_curve.png")
            # Optionally plot top-k accuracy curves
            self.visualizer.plot_accuracy(self.best_fold_metrics[4], self.best_fold_metrics[5], 
                                         save_path=f"Outputs/kfold_top{self.top_k}_accuracy_curve.png")
        
        return self.best_results

    def train_split(self, train_ratio=0.7, val_ratio=0.3, test_ratio=0.0, random_state=42):
        """
        Train using a simple split (train/valid or train/valid/test).
        
        Parameters:
        - train_ratio: Proportion of data for training
        - val_ratio: Proportion of data for validation
        - test_ratio: Proportion of data for testing (if 0, no test set is created)
        - random_state: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
        
        if test_ratio > 0:
            train_idx, val_idx, test_idx = self._custom_split(
                self.dataset, train_ratio, val_ratio, test_ratio, random_state
            )
            test_subset = Subset(self.dataset, test_idx)
            test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False, 
                                     pin_memory=(self.device.type == 'cuda'))
        else:
            train_idx, val_idx = self._custom_split(
                self.dataset, train_ratio, val_ratio, 0, random_state
            )
            test_loader = None
        
        train_subset = Subset(self.dataset, train_idx)
        val_subset = Subset(self.dataset, val_idx)
        
        pin_memory = True if self.device.type == 'cuda' else False
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, pin_memory=pin_memory)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, pin_memory=pin_memory)

        
        self.train_losses = []
        self.train_accuracies = []
        self.train_top_k_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_top_k_accuracies = []
        
        best_val_top_k_acc = 0.0
        best_val_acc = 0.0
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            
            avg_train_loss, train_acc, train_top_k_acc, _, _ = self._train_epoch(train_loader)
            val_loss, val_acc, val_top_k_acc, _, _ = self._validate_epoch(val_loader)

            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_acc)
            self.train_top_k_accuracies.append(train_top_k_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_top_k_accuracies.append(val_top_k_acc)


            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{self.epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Train Top-{self.top_k} Acc: {train_top_k_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                    f"Val Top-{self.top_k} Acc: {val_top_k_acc:.4f}")


            # Using top-k accuracy for best model selection
            if val_top_k_acc > best_val_top_k_acc:
                best_val_top_k_acc = val_top_k_acc
                best_val_acc = val_acc
                best_epoch = epoch
                
                if self.save_best:
                    self.best_results = self.add_row(["Train/Valid", train_acc, train_top_k_acc, 
                                                     val_acc, val_top_k_acc, epoch])
                    torch.save(self.model.state_dict(), self.save_path)
                
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if self.early_stopping and epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch} (no improvement in {self.patience} epochs).")
                    break

        
        if self.save_best and best_val_top_k_acc > 0:
            try:
                self.model.load_state_dict(torch.load(self.save_path))
            except Exception as e:
                print(f"Warning: Could not load the best model from '{self.save_path}' (error: {e}).")

        self.visualizer.plot_loss(self.train_losses, self.val_losses, save_path="split_loss_curve.png")
        self.visualizer.plot_accuracy(self.train_accuracies, self.val_accuracies, save_path="split_accuracy_curve.png")
        self.visualizer.plot_accuracy(self.train_top_k_accuracies, self.val_top_k_accuracies, 
                                     save_path=f"split_top{self.top_k}_accuracy_curve.png")

        if test_loader:
            test_loss, test_acc, test_top_k_acc, _, _ = self._validate_epoch(test_loader)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
                  f"Test Top-{self.top_k} Acc: {test_top_k_acc:.4f}")
            self.best_results = self.add_row(["Test", None, None, test_acc, test_top_k_acc, best_epoch])

        return self.best_results

    def _train_epoch(self, train_loader):
        """
        Trains the model for one epoch on the provided training data loader.

        Args:
            train_loader (DataLoader): DataLoader providing batches of training data.

        Returns:
            tuple: A tuple containing:
                - avg_train_loss (float): The average training loss for the epoch.
                - train_acc (float): The training accuracy for the epoch.
                - train_top_k_acc (float): The training top-k accuracy for the epoch.
                - all_train_preds (list): List of predicted labels for all training samples.
                - all_train_labels (list): List of true labels for all training samples.
        """

        self.model.train()
        total_train_loss = 0.0
        total_train_samples = 0
        total_top_k_correct = 0.0
        all_train_preds = []
        all_train_labels = []

        for feature_matrix, adjacency_matrix, labels in train_loader:
            feature_matrix = feature_matrix.to(self.device)
            adjacency_matrix = adjacency_matrix.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(feature_matrix, adjacency_matrix)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            batch_size_current = labels.size(0)
            total_train_loss += loss.item() * batch_size_current
            total_train_samples += batch_size_current
            
            # Calculate top-k accuracy for this batch
            batch_top_k_acc = self.calculate_top_k_accuracy(outputs, labels)
            total_top_k_correct += batch_top_k_acc * batch_size_current
            
            preds = outputs.argmax(dim=1)
            all_train_preds.extend(preds.detach().cpu().numpy())
            all_train_labels.extend(labels.detach().cpu().numpy())

        avg_train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0.0
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_top_k_acc = total_top_k_correct / total_train_samples if total_train_samples > 0 else 0.0
        
        return avg_train_loss, train_acc, train_top_k_acc, all_train_preds, all_train_labels

    def _validate_epoch(self, data_loader):
        """Validate the model on the given data loader."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        total_top_k_correct = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for feature_matrix, adjacency_matrix, labels in data_loader:
                feature_matrix = feature_matrix.to(self.device)
                adjacency_matrix = adjacency_matrix.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(feature_matrix, adjacency_matrix)
                loss = self.criterion(outputs, labels)
                
                batch_size_current = labels.size(0)
                total_loss += loss.item() * batch_size_current
                total_samples += batch_size_current
                
                # Calculate top-k accuracy for this batch
                batch_top_k_acc = self.calculate_top_k_accuracy(outputs, labels)
                total_top_k_correct += batch_top_k_acc * batch_size_current
                
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = accuracy_score(all_labels, all_preds) if total_samples > 0 else 0.0
        top_k_accuracy = total_top_k_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy, top_k_accuracy, all_preds, all_labels

    def add_row(self, data):
        """Add a new row to the results DataFrame."""
        new_row = pd.DataFrame([data], columns=self.best_results.columns)
        return pd.concat([self.best_results, new_row], ignore_index=True)

    def _custom_split(self, dataset, train_ratio=0.7, val_ratio=0.3, test_ratio=0.0, random_state=42):
        """
        Split dataset into stratified train, validation, and optionally test sets.
        
        Parameters:
        - dataset: Dataset to split
        - train_ratio: Proportion of data for training
        - val_ratio: Proportion of data for validation
        - test_ratio: Proportion of data for testing (if 0, no test set is created)
        - random_state: Random seed for reproducibility
        
        Returns:
        - List of indices for each split
        """
        labels = np.array(dataset.labels)
        unique_classes = np.unique(labels)
        
        train_idx, val_idx, test_idx = [], [], []
        np.random.seed(random_state)
        
        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            np.random.shuffle(cls_indices)
            
            if test_ratio > 0:
                # Three-way split (train/valid/test)
                train_size = int(len(cls_indices) * train_ratio)
                val_size = int(len(cls_indices) * val_ratio)
                
                train_idx.extend(cls_indices[:train_size])
                val_idx.extend(cls_indices[train_size:train_size + val_size])
                test_idx.extend(cls_indices[train_size + val_size:])
            else:
                # Two-way split (train/valid)
                train_size = int(len(cls_indices) * train_ratio)
                
                train_idx.extend(cls_indices[:train_size])
                val_idx.extend(cls_indices[train_size:])
        
        if test_ratio > 0:
            return train_idx, val_idx, test_idx
        else:
            return train_idx, val_idx