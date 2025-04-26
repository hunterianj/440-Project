import matplotlib.pyplot as plt
import networkx as nx

def plot_graph(graph,
               figsize=(8,6),
               node_size=500,
               font_size=12,
               arrow_size=20,
               rad: float = 0.0):
    graphNx = nx.DiGraph()
    nodes = list(graph.G.get_nodes())
    edges = graph.G.get_graph_edges()
    # convert each Edge → (u,v)
    edge_tuples = [(e.get_node1(), e.get_node2()) for e in edges]
    graphNx.add_nodes_from(nodes)
    graphNx.add_edges_from(edge_tuples)

    pos = nx.spring_layout(graphNx, k=1.0, iterations=50)

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
    nx.draw_networkx_edges(graphNx, pos,
                           arrows=True,
                           arrowstyle='-|>',
                           arrowsize=arrow_size,
                           connectionstyle=f'arc3,rad={rad}',
                           width=1.5,
                           edge_color='k')

    plt.margins(0.2, 0.2)
    plt.tight_layout()
    plt.axis('off')
    plt.show()