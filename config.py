import torch

num_classes = 3
num_concepts = 60
class_mapping = {0: 'Pneumonia', 1: 'Covid', 2: 'Normal'}
W_F = torch.load('learned_weights.pt') 