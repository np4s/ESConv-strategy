from utils import build_model, convert_data_to_inputs, featurize
import json
from Graph import Graph
import torch
import time
import torch.nn as nn

def train(model, inputs, labels, epochs):
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        # Calculate progress and estimated time
        progress = (epoch + 1) / epochs * 100
        estimated_time = (epochs - epoch - 1) * (time.time() - start_time) / (epoch + 1)
        # Print progress and estimated time using tqdm
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}, Progress: {progress}%, Estimated Time: {estimated_time} seconds")

if __name__ == "__main__":
    dataset = torch.load('./dataset/processed_data.pt')
    tokenizer, model = build_model(tokenizer_only=False)
    