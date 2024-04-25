import matplotlib.pyplot as plt

import random
import rustworkx as rx
import rustworkx.generators
from rustworkx.visualization import mpl_draw

# create a random undirected graph with n nodes
def generate_random_undirected_graph(n, max_weight):
    graph = rustworkx.generators.complete_graph(n)
    for i, j, _ in graph.weighted_edge_list():
        # Assigning random weights to the edges
        random_weight = random.uniform(0, max_weight)  # You can adjust the range of weights as needed
        graph.update_edge(i, j,random_weight)
    return graph
def visualize_graph(graph):
    mpl_draw(graph)

    # uncomment to save to file
    # plt.savefig("random_graph.png")
def visualize_reduced_graph(graph):
    filtered_edge_indices = graph.filter_edges(filter_edges)
    edge_index_map = graph.edge_index_map()
    filtered_edge_list = [(edge_index_map[i][0], edge_index_map[i][1]) for i in filtered_edge_indices]
    reduced_graph = graph.edge_subgraph(filtered_edge_list)
    mpl_draw(reduced_graph)

    # uncomment to save to file
    # plt.savefig("reduced_graph.png")
def filter_edges(edge):
    if edge > 0.75:
        return True
    else:
        return False


if __name__ == "__main__":
    n = 10
    max_weight = 1
    graph = generate_random_undirected_graph(n, max_weight)
    visualize_graph(graph)
    visualize_reduced_graph(graph)
