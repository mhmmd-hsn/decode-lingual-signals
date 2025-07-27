import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.pipeline import Pipeline
import joblib

class EEGClassifierEvaluator:
    def __init__(self, random_state=42, n_folds=5):
        """
        Initialize the classifier evaluator.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        n_folds : int
            Number of folds for cross-validation
        """
        self.random_state = random_state
        self.n_folds = n_folds
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = LabelEncoder()
        
        # Define the classifiers to evaluate
        self.classifiers = {
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Support Vector Machine': SVC(probability=True, random_state=random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'Naive Bayes': GaussianNB(),
            # 'LightGBM': LGBMClassifier(random_state=random_state)
        }
    
    def fit(self, X, y):
        """
        Fit and evaluate all classifiers using cross-validation.
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            The input features
        y : array-like
            Target variable (class labels)
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
    
        if not np.issubdtype(y.dtype, np.number):
            y = self.label_encoder.fit_transform(y)
            self.classes_ = self.label_encoder.classes_
        else:
            self.classes_ = np.unique(y)
            
        self.n_classes = len(self.classes_)
        
        # Create cross-validation folds
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Prepare results storage
        self.results = {
            'accuracy': {},
            'precision': {},
            'recall': {},
            'f1': {},
            'training_time': {},
            'prediction_time': {},
            'cv_predictions': {},
            'confusion_matrices': {},
            'feature_importance': {}
        }
        
        # Define preprocessing pipeline (scaling)
        scaler = StandardScaler()
        
        best_score = 0
        
        # Evaluate each classifier
        for name, clf in self.classifiers.items():
            print(f"Evaluating {name}...")
            start_time = time.time()
            
            # Create pipeline with preprocessing
            pipeline = Pipeline([
                ('scaler', scaler),
                ('classifier', clf)
            ])
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy')
            y_pred = cross_val_predict(pipeline, X, y, cv=skf)
            
            # Time measurements
            train_time = time.time() - start_time
            start_time = time.time()
            
            # Get confusion matrix
            cm = confusion_matrix(y, y_pred)
            
            # Store metrics
            self.results['accuracy'][name] = cv_scores.mean()
            self.results['precision'][name] = precision_score(y, y_pred, average='weighted')
            self.results['recall'][name] = recall_score(y, y_pred, average='weighted')
            self.results['f1'][name] = f1_score(y, y_pred, average='weighted')
            self.results['training_time'][name] = train_time
            self.results['prediction_time'][name] = time.time() - start_time
            self.results['cv_predictions'][name] = y_pred
            self.results['confusion_matrices'][name] = cm
            
            # Store feature importance if available
            if hasattr(clf, 'feature_importances_'):
                pipeline.fit(X, y)  # Fit on full dataset to get feature importances
                self.results['feature_importance'][name] = pipeline.named_steps['classifier'].feature_importances_
            
            # Check if this is the best model so far
            if self.results['accuracy'][name] > best_score:
                best_score = self.results['accuracy'][name]
                self.best_model_name = name
                
                # Fit the best model on the full dataset
                best_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', self.classifiers[name])
                ])
                best_pipeline.fit(X, y)
                self.best_model = best_pipeline
        
        # Convert results to DataFrame
        self.metrics_df = pd.DataFrame({
            'Classifier': list(self.classifiers.keys()),
            'Accuracy': [self.results['accuracy'][name] for name in self.classifiers],
            'Precision': [self.results['precision'][name] for name in self.classifiers],
            'Recall': [self.results['recall'][name] for name in self.classifiers],
            'F1 Score': [self.results['f1'][name] for name in self.classifiers],
            'Training Time (s)': [self.results['training_time'][name] for name in self.classifiers],
            'Prediction Time (s)': [self.results['prediction_time'][name] for name in self.classifiers]
        })
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the best model.
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            The input features
            
        Returns:
        --------
        y_pred : array
            Predicted class labels
        """
        if self.best_model is None:
            raise ValueError("Models have not been trained yet. Call fit() first.")
            
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        """
        Get class probabilities for prediction.
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            The input features
            
        Returns:
        --------
        y_proba : array
            Class probabilities
        """
        if self.best_model is None:
            raise ValueError("Models have not been trained yet. Call fit() first.")
            
        return self.best_model.predict_proba(X)

    def visualize_results(self, output_dir="classifier_evaluation_results"):
        """
        Visualize and save evaluation results.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
        """
        if not self.results:
            raise ValueError("No results to visualize. Call fit() first.")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics table
        self.metrics_df.to_csv(f"{output_dir}/classifier_metrics.csv", index=False)
        
        # Plot accuracy comparison
        plt.figure(figsize=(12, 6))
        accuracy_df = pd.DataFrame(self.results['accuracy'].items(), columns=['Classifier', 'Accuracy'])
        sns.barplot(x='Classifier', y='Accuracy', data=accuracy_df)
        plt.title('Classifier Accuracy Comparison')
        plt.ylim(accuracy_df['Accuracy'].min() * 0.9, min(1.0, accuracy_df['Accuracy'].max() * 1.1))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/accuracy_comparison.png")
        plt.close()
        
        # Plot metrics comparison
        metrics_df = pd.melt(self.metrics_df, 
                            id_vars=['Classifier'], 
                            value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                            var_name='Metric', value_name='Value')
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Classifier', y='Value', hue='Metric', data=metrics_df)
        plt.title('Classifier Performance Metrics')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_comparison.png")
        plt.close()
        
        # Plot training and prediction times
        times_df = pd.melt(self.metrics_df, 
                          id_vars=['Classifier'], 
                          value_vars=['Training Time (s)', 'Prediction Time (s)'],
                          var_name='Time Metric', value_name='Seconds')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Classifier', y='Seconds', hue='Time Metric', data=times_df)
        plt.title('Classifier Time Performance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_comparison.png")
        plt.close()
        
        # Plot confusion matrices for each classifier
        for name, cm in self.results['confusion_matrices'].items():
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.classes_, yticklabels=self.classes_)
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/confusion_matrix_{name.replace(' ', '_')}.png")
            plt.close()
        
        # Plot feature importance for models that support it
        if self.results['feature_importance']:
            for name, importance in self.results['feature_importance'].items():
                plt.figure(figsize=(12, len(importance) * 0.3 + 2))
                
                # Create a DataFrame for the feature importance
                feature_names = [f"Feature {i}" for i in range(len(importance))]
                importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                # Plot top 20 features (or all if less than 20)
                top_n = min(20, len(importance_df))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
                plt.title(f'Feature Importance - {name}')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/feature_importance_{name.replace(' ', '_')}.png")
                plt.close()
                
                # Save the full feature importance data
                importance_df.to_csv(f"{output_dir}/feature_importance_{name.replace(' ', '_')}.csv", index=False)
    
    def save_best_model(self, filename="best_eeg_classifier.joblib"):
        """
        Save the best model to file.
        
        Parameters:
        -----------
        filename : str
            File path to save the model
        """
        if self.best_model is None:
            raise ValueError("No best model to save. Call fit() first.")
            
        joblib.dump(self.best_model, filename)
        print(f"Best model ({self.best_model_name}) saved to {filename}")
        
        # Save label encoder if we used it
        if hasattr(self, 'label_encoder') and len(self.label_encoder.classes_) > 0:
            encoder_filename = filename.replace('.joblib', '_label_encoder.joblib')
            joblib.dump(self.label_encoder, encoder_filename)
            print(f"Label encoder saved to {encoder_filename}")
    
    def generate_report(self, output_dir="classifier_evaluation_results"):
        """
        Generate a comprehensive performance report.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the report
        """
        if not self.results:
            raise ValueError("No results to report. Call fit() first.")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate text report
        with open(f"{output_dir}/classification_report.txt", 'w') as f:
            f.write("EEG CLASSIFIER EVALUATION REPORT\n")
            f.write("===============================\n\n")
            
            f.write("SUMMARY\n")
            f.write("-------\n")
            f.write(f"Best classifier: {self.best_model_name}\n")
            f.write(f"Best accuracy: {self.results['accuracy'][self.best_model_name]:.4f}\n\n")
            
            f.write("DETAILED METRICS\n")
            f.write("---------------\n")
            f.write(self.metrics_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("CLASSIFICATION REPORTS\n")
            f.write("---------------------\n")
            for name in self.classifiers:
                f.write(f"\n{name}:\n")
                y_pred = self.results['cv_predictions'][name]
                report = classification_report(
                    self.label_encoder.inverse_transform(y_pred) if hasattr(self, 'label_encoder') and len(self.label_encoder.classes_) > 0 else y_pred,
                    self.label_encoder.inverse_transform(y_pred) if hasattr(self, 'label_encoder') and len(self.label_encoder.classes_) > 0 else y_pred
                )
                f.write(report)
                f.write("\n")
        
        print(f"Evaluation report saved to {output_dir}/classification_report.txt")