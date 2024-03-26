import networkx as nx
import os
from sklearn.metrics.pairwise import cosine_similarity
import torch


class Node:
    def __init__(self, encoded, speaker):
        self.encoded = encoded
        self.speaker = speaker

    def get_speaker(self):
        return self.speaker

    def get_encoded(self):
        return self.encoded


class Graph:
    def __init__(self, inputs,
                 sim_threshold=0.5,
                 window_size=5, save_graph=False):
        self.inputs = inputs
        self.sim_threshold = sim_threshold
        self.window_size = window_size

        self.nodes = []
        for node in self.inputs:
            self.add_node(node["content"], node["speaker"])
            
        prebuilt_path = f"{sim_threshold}_{window_size}_graph"
        self.G = self.build_graph(prebuilt_path, save_graph)

    def add_node(self, utt, speaker):
        # encoded_tensor = torch.tensor(utt, dtype=float)
        node = Node(utt, speaker)
        self.nodes.append(node)

    def build_graph(self, path, save_graph):
        G = nx.Graph()
        for i, node in enumerate(self.nodes):
            G.add_node(i, vector=node.get_encoded())

        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):

                if self.nodes[i].get_speaker() != self.nodes[j].get_speaker() and abs(i - j) > self.window_size:
                    continue

                try:
                    sim = cosine_similarity([self.nodes[i].get_encoded()],
                                            [self.nodes[j].get_encoded()])[0, 0]
                except:
                    print([self.nodes[i].get_encoded()],
                          [self.nodes[j].get_encoded()])
                if sim >= self.sim_threshold:
                    G.add_edge(i, j, weight=sim)
                else:
                    G.add_edge(i, j, weight=self.sim_threshold)
                    
        if save_graph:
            nx.write_weighted_edgelist(G, path=path)
        return G

    def get_adjacency_matrix(self):
        adjacency_matrix = nx.adjacency_matrix(self.G)
        return torch.tensor(adjacency_matrix.todense(), dtype=torch.float) if adjacency_matrix.shape[0] > 0 else 'No edges found!'

    def get_num_nodes(self):
        return len(self.nodes)

    def get_nodes(self):
        return self.nodes
