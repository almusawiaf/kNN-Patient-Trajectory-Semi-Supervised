import networkx as nx
import numpy as np
import random


def initialize_graph(num_nodes, edge_prob = 0.2):

    # Create a random graph
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)

    # Assign random labels to nodes
    for node in G.nodes:
        G.nodes[node]['weight'] = random.choice([i for i in range(5)])

    # Assign random similarity weights to edges
    for u, v in G.edges:
        G[u][v]['similarity'] = random.uniform(0, 1)

    return G


def label_propagation(G, max_iter = 1000):
    for _ in range(max_iter):
        label_changes = 0

        # Create a copy of the current labels
        new_labels = {}

        for node in G.nodes:
            if G.nodes[node]['weight'] == 0:  # Only update nodes with partial knowledge
                neighbor_labels = {}
                for neighbor in G.neighbors(node):
                    label = G.nodes[neighbor]['weight']
                    weight = G.edges[node, neighbor]['similarity']
                    if label in neighbor_labels:
                        neighbor_labels[label] += weight
                    else:
                        neighbor_labels[label] = weight

                # Get the label with the highest weighted frequency
                if neighbor_labels:
                    new_label = max(neighbor_labels, key=neighbor_labels.get)
                    if new_label != G.nodes[node]['weight']:
                        new_labels[node] = new_label
                        label_changes += 1

        # Update the labels
        for node, label in new_labels.items():
            G.nodes[node]['weight'] = label

        # Check for convergence
        if label_changes == 0:
            print(f'Converged after {_ + 1} iterations')
            break
    else:
        print(f'Reached max iterations: {max_iter}')

    return G


def main():
    num_nodes = 50
    G = initialize_graph(num_nodes)

    print("Initial labels:")
    print([G.nodes[node]['weight'] for node in G.nodes])

    G = label_propagation(G)

    print("Final labels:")
    print([G.nodes[node]['weight'] for node in G.nodes])


if __name__ == "__main__":
    main()
