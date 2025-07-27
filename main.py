import numpy as np
import pandas as pd
from feature_extraction import EEGFeatureExtractor
from data_loader import EEGDataLoader
from feature_selection import EEGFeatureSelector
from trainer import EEGClassifierEvaluator
from test import EEGStatisticalAnalysis
# Example usage
if __name__ == "__main__":

    # dataset = EEGDataLoader(root_path = "D:\Work\MachineLearning\Projects\DRAFT\\Data",
    #                         class_type = "AK-SREP",
    #                         trial_type = "reading",
    #                         connection = "pearson",
    #                         track_quality = False,
    #                         visualize = False)
    
    # extractor = EEGFeatureExtractor(fs=500)
    # feature_df = extractor.extract_and_save_to_csv(dataset.data, "eeg_features.csv")
    # print(f"Extracted {feature_df.shape[1]} features for {feature_df.shape[0]} trials")

    # print(f"Original dataset shape: {feature_df.shape}")

    # selector = EEGFeatureSelector(quality_threshold=0.8)
    # X_selected = selector.fit_transform(feature_df, dataset.labels)
    # print(f"Selected dataset shape: {X_selected.shape}")
    # print(X_selected.head())

    df = pd.read_csv("selected_eeg_features_with_labels.csv").round(4)

    X = df.drop("labels", axis=1)
    y = df["labels"]

    evaluator = EEGClassifierEvaluator(n_folds=4)
    evaluator.fit(X, y)

    print("\nCLASSIFIER PERFORMANCE SUMMARY:")
    print(evaluator.metrics_df.sort_values('Accuracy', ascending=False))