import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Edge import Edge
from causallearn.utils.GraphUtils import GraphUtils

def plot_graph(graph,
               filename,
               figsize=(8,6),
               node_size=500,
               font_size=12,
               arrow_size=20,
               rad: float = 0.0):
    graphNx = nx.DiGraph()
    gUtils = GraphUtils()

    if isinstance(graph, CausalGraph):
        nodes = list(graph.G.get_nodes())
        edges = graph.G.get_graph_edges()
    else: # assume graph is an instance of GeneralGraph
        nodes = list(graph.get_nodes())
        edges = graph.get_graph_edges()

    # convert each Edge to (u,v) and split into different edge types
    directed = []
    undirected = []
    bidirected = []
    for e in edges:
        u, v = e.get_node1(), e.get_node2()
        if gUtils.undirected(e):
            undirected.append((u, v))
        elif gUtils.directed(e):
            directed.append((u, v))
        elif gUtils.bi_directed(e):
            bidirected.append((u, v))
        else:
            # unhandled edge types
            pass
    # edge_tuples = [(e.get_node1(), e.get_node2()) for e in edges]
    graphNx.add_nodes_from(nodes)
    # add all edges
    graphNx.add_edges_from(directed)
    graphNx.add_edges_from(undirected)
    graphNx.add_edges_from(bidirected)

    # pos = nx.spring_layout(graphNx, k=1.0, iterations=50)
    pos = nx.circular_layout(graphNx)

    # draw nodes
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(graphNx, pos,
                           node_size=node_size,
                           node_color='C0',
                           linewidths=1.0,
                           edgecolors='white')
    nx.draw_networkx_labels(graphNx, pos,
                            font_size=font_size,
                            # bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none'),
                            clip_on=False)

    # draw directed edges
    if directed:
        nx.draw_networkx_edges(graphNx, pos,
                               edgelist=directed,
                               arrows=True,
                               arrowstyle='-|>',
                               arrowsize=arrow_size,
                               width=1.5,
                               edge_color='k',
                               connectionstyle=f'arc3,rad={rad}')
    # draw undirected edges
    if undirected:
        nx.draw_networkx_edges(graphNx, pos,
                               edgelist=undirected,
                               arrows=False,
                               style='solid',
                               width=1.5,
                               edge_color='k',
                               connectionstyle=f'arc3,rad={rad}')

    # draw bidirected edges
    if bidirected:
        nx.draw_networkx_edges(graphNx, pos,
                               edgelist=bidirected,
                               arrows=True,
                               arrowstyle='<->',
                               arrowsize=arrow_size,
                               width=1.5,
                               edge_color='k',
                               connectionstyle=f'arc3,rad={rad}')

    plt.margins(0.2, 0.2)
    plt.tight_layout()
    plt.axis('off')
    # plt.show()
    filename = f"figs/{filename}.png"
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)

def ccpg_full_graph_connected_undirected(components, edges, node_names=None):
    all_nodes = set().union(*components)
    d = len(all_nodes)

    # build empty graph on d nodes
    cg = CausalGraph(d, node_names)
    for e in list(cg.G.get_graph_edges()):
        cg.G.remove_edge(e)

    # undirected edges within each component
    for comp in components:
        for u, v in combinations(sorted(comp), 2):
            edge = Edge(cg.G.nodes[u], cg.G.nodes[v],
                        Endpoint.TAIL, Endpoint.TAIL)
            # breakpoint()
            cg.G.add_edge(edge)

    # directed edges between components
    for i, j in edges:
        for u in components[i]:
            for v in components[j]:
                cg.G.add_directed_edge(cg.G.nodes[u], cg.G.nodes[v])

    return cg

def ccpg_full_graph_connected_bidirected(components, edges, node_names=None):
    all_nodes = set().union(*components)
    d = len(all_nodes)

    # build empty graph on d nodes
    cg = CausalGraph(d, node_names)
    for e in list(cg.G.get_graph_edges()):
        cg.G.remove_edge(e)

    # undirected edges within each component
    for comp in components:
        for u, v in combinations(sorted(comp), 2):
            edge = Edge(cg.G.nodes[u], cg.G.nodes[v],
                        Endpoint.ARROW, Endpoint.ARROW)
            cg.G.add_edge(edge)

    # directed edges between components
    for i, j in edges:
        for u in components[i]:
            for v in components[j]:
                cg.G.add_directed_edge(cg.G.nodes[u], cg.G.nodes[v])

    return cg