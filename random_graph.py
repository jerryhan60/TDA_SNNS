import matplotlib.pyplot as plt

import random
import rustworkx as rx
import rustworkx.generators
from rustworkx.visualization import mpl_draw

def generate_random_undirected_graph(n: int, max_weight: float) -> rx.PyGraph:
    """
    Generates a random undirected graph with `n` nodes and assigns random weights to the edges.

    Parameters:
        n (int): The number of nodes in the graph.
        max_weight (float): The maximum weight for the edges.

    Returns:
        rx.PyGraph: A random undirected graph with assigned random weights.

    Example:
        >>> generate_random_undirected_graph(5, 10.0)
        <rustworkx.PyGraph object at 0x...>
    """
    graph = rustworkx.generators.complete_graph(n)
    for i, j, _ in graph.weighted_edge_list():
        # Assigning random weights to the edges
        random_weight = random.uniform(0, max_weight)  # You can adjust the range of weights as needed
        graph.update_edge(source=i, target=j,edge=random_weight)
    return graph

def visualize_graph(graph: rx.PyGraph) -> None:
    """
    Visualizes the provided graph using Matplotlib.

    Args:
        graph (rx.PyGraph): The graph object to be visualized.

    Returns:
        None
    """
    mpl_draw(graph=graph)

    # uncomment to save to file
    # plt.savefig("random_graph.png")

def visualize_reduced_graph(graph: rx.PyGraph, filter_function: function) -> None:
    """
    Visualizes a reduced version of the provided graph using Matplotlib.

    This function filters the edges of the input graph based on a specified filter function,
    creates a reduced graph from the filtered edges, and then visualizes the reduced graph using Matplotlib.

    Args:
        graph (rx.PyGraph): The original graph object to be visualized.

    Returns:
        None

    Notes:
        - Uncomment the `plt.savefig` line at the end of the function to save the visualization
          as an image file named "reduced_graph.png".
    """
    filtered_edge_indices = graph.filter_edges(filter_function=filter_function)
    edge_index_map = graph.edge_index_map()
    filtered_edge_list = [(edge_index_map[i][0], edge_index_map[i][1]) for i in filtered_edge_indices]
    reduced_graph = graph.edge_subgraph(edge_list=filtered_edge_list)
    mpl_draw(graph=reduced_graph)

    # uncomment to save to file
    # plt.savefig("reduced_graph.png")


def filter_edges_by_threshold(edge: float) -> bool:
    """
    Determines whether an edge should be included based on a threshold (0.75).

    This function evaluates whether an edge weight exceeds a specified threshold value
    and returns True if it does, indicating that the edge should be included in the filtered set,
    otherwise, it returns False.

    Args:
        edge (float): The weight of the edge to be evaluated.

    Returns:
        bool: True if the edge weight is greater than 0.75, False otherwise.
    """
    if edge > 0.75:
        return True
    else:
        return False


if __name__ == "__main__":
    n = 10 # nodes in graph
    max_weight = 1 # maximum weight for edges
    graph = generate_random_undirected_graph(n=n, max_weight=max_weight)
    visualize_graph(graph)
    visualize_reduced_graph(graph, filter_edges_by_threshold)
