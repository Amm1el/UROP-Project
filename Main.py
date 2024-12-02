#Computer won't allow me to test (conda / library issues)#

# Import required libraries
import pickle
import networkx as nx
import torch
from torch_geometric.data import Data
import pandas as pd
from graph_utils import display_graph

# Set file paths
path_to_adjacency_matrix = "path_to_data/electrode_graph/adj_mx_3d.pkl"
path_to_distance_csv = "path_to_data/electrode_graph/distances_3d.csv"
path_to_train_seizure_files = "path_to_data/file_markers_classification/trainSet_seizure_files.txt"
path_to_test_seizure_files = "path_to_data/file_markers_classification/testSet_seizure_files.txt"

# Load adjacency matrix
print("Loading adjacency matrix...")
with open(path_to_adjacency_matrix, "rb") as f:
    adj_matrix = pickle.load(f)

print(f"Adjacency Matrix Shape: {adj_matrix.shape}")

# Load distances (if needed)
print("\nLoading distances CSV...")
distances = pd.read_csv(path_to_distance_csv)
print("First 5 rows of distances CSV:")
print(distances.head())

# Create a NetworkX graph from the adjacency matrix
print("\nCreating NetworkX graph...")
G = nx.from_numpy_matrix(adj_matrix)

# Print basic graph info
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

display_graph(G, title="EEG Graph")

# Create a PyTorch Geometric graph
print("\nCreating PyTorch Geometric graph...")
edge_index = torch.tensor(list(G.edges)).t().contiguous()

# Generate dummy node features (replace with actual features if available)
node_features = torch.rand((adj_matrix.shape[0], 10))  # Example: 10-dimensional features

# Create PyTorch Geometric Data object
data = Data(x=node_features, edge_index=edge_index)

print("\nPyTorch Geometric Graph:")
print(data)

# Load train and test seizure file markers
print("\nLoading seizure file markers...")

# Train seizure files
with open(path_to_train_seizure_files, "r") as f:
    train_files = f.readlines()
print(f"Train Seizure Files ({len(train_files)}):")
print(train_files[:5])  # Display first 5 file paths

# Test seizure files
with open(path_to_test_seizure_files, "r") as f:
    test_files = f.readlines()
print(f"Test Seizure Files ({len(test_files)}):")
print(test_files[:5])  # Display first 5 file paths

# OPTIONAL: Process file markers (example)
def process_file_markers(file_list):
    return [file.strip() for file in file_list]

train_files_processed = process_file_markers(train_files)
test_files_processed = process_file_markers(test_files)

print("\nProcessed Train Files (First 5):")
print(train_files_processed[:5])

print("\nProcessed Test Files (First 5):")
print(test_files_processed[:5])

print("\nScript completed successfully.")
