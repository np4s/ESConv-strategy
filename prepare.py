from utils import build_model, convert_data_to_inputs, featurize
import json
from Graph import Graph
import torch

if __name__ == "__main__":
    data_name = "small_ESConv.json"
    with open('./dataset/'+data_name, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open('./dataset/strategy.json', "r", encoding="utf-8") as f:
        strategies = json.load(f)
    strategy2id = {strategy: i for i, strategy in enumerate(strategies)}
    tokenizer = build_model()

    processed_data = []
    for data in dataset:
        inputs, labels = convert_data_to_inputs(data, tokenizer, strategy2id)
        labels = torch.tensor(labels, dtype=torch.float)

        graph = Graph(inputs, sim_threshold=0.3)
        feature = featurize(graph.get_nodes())
        adj_matrix = graph.get_adjacency_matrix()

        processed_data.append((feature, adj_matrix, labels))
        
    torch.save(processed_data, './dataset/processed_data.pt')
