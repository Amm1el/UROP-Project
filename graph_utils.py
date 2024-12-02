import matplotlib.pyplot as plt
import networkx as nx

def display_graph(graph, title="Graph Visualization"):
    """Display a graph using NetworkX and Matplotlib."""
    plt.figure(figsize=(10, 8))
    nx.draw(
        graph,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_weight="bold",
        edge_color="gray"
    )
    plt.title(title)
    plt.show()
    
