from rich import print, inspect
import numpy as np
from pyvis.network import Network
import pandas as pd
import torch
from rich.progress import track

# DATA_PATH = "../data/train_10k_test_1.67k.pt"
DATA_PATH = "../data/train_10k_test_1.67k_layer_1.pt"

g = Network()
g.barnes_hut()

adj_mat = torch.load(DATA_PATH)

sources, targets = adj_mat.nonzero(as_tuple=True)
weights = [round(x, 4) for x in adj_mat[sources, targets].numpy()]
# print(max(weights))

def adj_mat_to_csv():
#     Matrix
# The sample below shows a graph with 5 nodes. An edge is created when the the cell is ‘1′.
# ;A;B;C;D;E
# A;0;1;0;1;0
# B;1;0;0;0;0
# C;0;0;1;0;0
# D;0;1;0;1;0
# E;0;0;0;0;0

# Edge weight
# Simply replace ‘1′ values by the edge weight value, formatted as a ‘double’.

    # convert the adj mat to a csv file
    sources, targets = adj_mat.nonzero(as_tuple=True)
    weights = [round(x, 4) for x in adj_mat[sources, targets].numpy()]

    outfile = "adj_mat_layer1.csv"
    with open(outfile, "w") as f:
        f.write(";")
        for i in range(adj_mat.shape[0]):
            f.write(f"{i};")
        f.write("\n")

        for i in range(adj_mat.shape[0]):
            f.write(f"{i};")
            for j in range(adj_mat.shape[1]):
                v = round(adj_mat[i, j].numpy().item(), 3)

                if v == 22.5: v = 0
                if v < 0.3: v = 0

                v *= 100

                # if v != 0:
                #     print(v, i, j)

                f.write(f"{v};")
            f.write("\n")

adj_mat_to_csv()

exit()
sources, targets = adj_mat.nonzero(as_tuple=True)
weights = [round(x, 4) for x in adj_mat[sources, targets].numpy()]

sources = sources[:1000]
targets = targets[:1000]
weights = weights[:1000]


# print([round(x, 3) for x in weights.tolist()])

edge_data = zip(sources.numpy(), targets.numpy(), weights)

for e in track(edge_data):
    src = e[0]
    dst = e[1]
    w = e[2]

    # print(src, dst, w)
    # exit()

    g.add_node(str(src), str(src))
    g.add_node(str(dst), str(dst))
    g.add_edge(str(src), str(dst), value=str(w))

neighbor_map = g.get_adj_list()
g.show_buttons(filter_=['physics'])
# g.show("extra/sbuttons.html")
g.show("out.html", notebook=False)
