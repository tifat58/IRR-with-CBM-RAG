import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pickle
from sklearn.metrics import accuracy_score
import numpy as np
from config import epochs as N

device = torch.device('mps')

class LungDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.labels = {'pneumonia': 0, 'covid': 1, 'normal': 2}  
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        e = torch.stack(item['e'])  
        e_min = e.min()
        e_max = e.max()
        e = (e - e_min) / (e_max - e_min)
        label = self.labels[item['label']]
        return e, label

data_dir = 'similarities/train'
data = []
for file in os.listdir(data_dir):
    f = open(os.path.join(data_dir, file), 'rb')
    data.append(pickle.load(f))
dataset = LungDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

test_data_dir = 'similarities/test'
test_data = []
for file in os.listdir(test_data_dir):
    f = open(os.path.join(test_data_dir, file), 'rb')
    test_data.append(pickle.load(f))
test_dataset = LungDataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class LungClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LungClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes, bias=False)

    def forward(self, x):
        out = self.fc(x)
        return F.log_softmax(out, dim=1)

input_size = 60
num_classes = 3
model = LungClassifier(input_size, num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)


num_epochs =  N

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

def train():
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        all_labels = []
        all_predictions = []
        
        for e, labels in dataloader:
            e, labels = e.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(e)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = accuracy_score(all_labels, all_predictions)

        scheduler.step(epoch_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

def save_model():
    torch.save(model.state_dict(), 'learned_weights.pth')

def test():

    model.eval()

    with torch.no_grad():
        test_loss = 0.0
        all_labels = []
        all_predictions = []

        for e, labels in test_dataloader:
            e, labels = e.to(device), labels.to(device)
            outputs = model(e)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        avg_test_loss = test_loss / len(test_dataloader)
        test_accuracy = accuracy_score(all_labels, all_predictions)

        print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

if __name__ == '__main__':
    train()
    save_model()
    test()