import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

csv_directory = "./dataFile"
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
csv_files_test_n = round(len(csv_files) * 0.2)

train_data = []
test_data = []

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        # x = x.float().double()
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

for index, csv_file in enumerate(csv_files):
    file_path = os.path.join(csv_directory, csv_file)
    df = pd.read_csv(file_path)  #  一篇csv的資料
    df['token'] = df['token'].apply(lambda x: [float(i) for i in eval(x)])  #  轉成list
    
    csv_data_collect = df['token'].tolist()
    
    for i in df.values.tolist():
        csv_data_collect.append(i[0])    

    if index > csv_files_test_n:
        train_data.append(csv_data_collect)
    else:
        test_data.append(csv_data_collect)

batch_size = 1
input_size = 185
hidden_size1 = 128
hidden_size2 = 64
output_size = 3

model = MLP(input_size, hidden_size1, hidden_size2, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
num_epochs = 10000

def get_top_k_probabilities(output, k=3):
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top_probs, top_indices = torch.topk(probabilities, k, dim=1)
    return top_probs, top_indices

def softmax(output):
    return torch.nn.functional.softmax(output, dim=1)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data in train_data:
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        X = data_tensor[:, :-1] 
        Y = data_tensor[:, -1].long()
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            optimizer.zero_grad()
            output = model(input_seq).to(device)
            loss = criterion(output, target_seq)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"[Training] : Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    def calculate_accuracy(y_pred, y_true):
        predicted = torch.argmax(y_pred, dim=1)
        correct = (predicted == y_true).float()
        accuracy = correct.sum() / len(correct)
        return accuracy


    model.eval()
    total_accuracy = 0
    total_test_loss = 0
    with torch.no_grad():
        for data in test_data:
            data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
            X = data_tensor[:, :-1] 
            Y = data_tensor[:, -1].long()
            dataset = TensorDataset(X, Y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

            for input_seq, target_seq in dataloader:
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                output = model(input_seq)
                
                probabilities = softmax(output)
                test_loss = criterion(output, target_seq)
                total_test_loss += test_loss.item()
        avg_test_loss = total_test_loss / len(dataloader)
        print(f"[Testing] : Epoch {epoch+1}/{num_epochs}, Average Test Loss: {avg_test_loss:.4f}")


torch.save(model.state_dict(), './Model/MLP_model.pth')

