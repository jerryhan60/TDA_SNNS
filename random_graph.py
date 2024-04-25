import random
import rustworkx.generators

# create a random undirected graph with n nodes
def generate_random_undirected_graph(n, max_weight):
    graph = rustworkx.generators.complete_graph(n)
    for i, j, _ in graph.weighted_edge_list():
        # Assigning random weights to the edges
        random_weight = random.uniform(0, max_weight)  # You can adjust the range of weights as needed
        graph.update_edge(i, j,random_weight)
    return graph
