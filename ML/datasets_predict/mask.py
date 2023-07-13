import networkx as nx

def mask_graph(G, nodes_to_mask):
    remained_nodes = sorted(set(G.nodes()) - set(nodes_to_mask))
    res_G = nx.subgraph(G, remained_nodes)
    id_to_index = {id: i for i, id in enumerate(remained_nodes)}
    res_G = nx.relabel_nodes(res_G, id_to_index)
    return res_G