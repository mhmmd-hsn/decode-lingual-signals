import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import mne
import networkx as nx


class Visualization:
    def __init__(self):
        sns.set(style="whitegrid")
    
    def plot_eeg_topomap(feature_matrix, adjacency_matrix, node_labels, threshold=0.1):
        """
        Plots EEG electrode positions and connectivity using MNE's built-in head model.
        """
        num_channels = len(node_labels)
        montage = mne.channels.make_standard_montage("standard_1020")
        pos_dict = montage.get_positions()['ch_pos']
        
        # Extract 2D positions for available channels
        xy_positions = np.array([pos_dict[ch][:2] for ch in node_labels])

        # Normalize node sizes based on mean EEG activity
        node_sizes = feature_matrix.mean(axis=1)
        node_sizes = (node_sizes - node_sizes.min()) / (node_sizes.max() - node_sizes.min() + 1e-6) * 10  # Normalize
        


        mne.viz.plot_topomap(node_sizes, xy_positions, cmap='coolwarm', contours=0, show=False)
        for i, label in enumerate(node_labels):
            plt.text(xy_positions[i, 0], xy_positions[i, 1], label, fontsize=9, ha='center', va='center', color='black')
        G = nx.Graph()
        
        for i, label in enumerate(node_labels):
            G.add_node(label, pos=xy_positions[i])

        
        for i in range(num_channels):
            for j in range(i + 1, num_channels):  
                weight = adjacency_matrix[i, j]
                if weight > threshold:  
                    G.add_edge(node_labels[i], node_labels[j], weight=weight)

        # Overlay connectivity graph
        pos = nx.get_node_attributes(G, 'pos')
        edges = G.edges(data=True)
        edge_widths = [d['weight'] * 2 for (u, v, d) in edges]  
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='red', alpha=0.8)

        plt.show()
    def plot_loss(self, train_losses, val_losses, save_path="loss_plot.png"):
        """
        Plots the training and validation loss over epochs.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss', marker='o')
        plt.plot(val_losses, label='Validation Loss', marker='s')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid()
        plt.savefig(save_path) 
        # plt.show()

    def plot_accuracy(self, train_accs, val_accs, save_path="accuracy_plot.png"):
        """
        Plots the training and validation accuarcy over epochs.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(train_accs, label='Train acc', marker='o')
        plt.plot(val_accs, label='Validation acc', marker='s')
        plt.xlabel('Epochs')
        plt.ylabel('accuracy')
        plt.title('Training and Validation accuracy Over Epochs')
        plt.legend()
        plt.grid()
        plt.savefig(save_path) 
        # plt.show()