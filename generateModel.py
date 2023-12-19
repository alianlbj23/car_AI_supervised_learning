import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import numpy as np

# 設置CSV文件所在的目錄

csv_directory = "./dataFile"

# 獲取目錄下所有CSV文件
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
csv_files_test_n = len(csv_files) * 0.2

# 初始化訓練集和測試集的空列表
train_data = []
test_data = []

# 迭代處理每個CSV文件
for index, csv_file in enumerate(csv_files):
    # 構建完整的文件路徑
    file_path = os.path.join(csv_directory, csv_file)
    
    # 讀取CSV文件
    df = pd.read_csv(file_path)
    
    # 使用train_test_split函數按照行號劃分數據，這裡我們根據行號切分
    # 根據行號順序切分，可以通過指定shuffle=False來實現
    train_set, test_set = train_test_split(df, test_size=0.2, shuffle=False)
    if index > csv_files_test_n:
    # 將劃分後的數據添加到訓練集和測試集列表中
        train_data.append(train_set)
    else:
        test_data.append(test_set)

tensor_train_data = []#53篇，每篇都有裡面紀錄
tensor_test_data = []
# 迭代处理训练集数据
for train_set in train_data:
    # 假设 'min_lidar' 是你的特定列名
    
    # min_lidar_values = [torch.tensor([float(val) for val in row['min_lidar'].strip('[]').split(',')], dtype=torch.float64) for _, row in train_set.iterrows()]
    min_lidar_values = [
        torch.tensor([float(val) for val in row['min_lidar'].strip('[]').split(',')], dtype=torch.float64).clone().detach()
        for _, row in train_set.iterrows()
    ]
    

    car_pos_x_values = [torch.tensor(float(val), dtype=torch.float64) for val in train_set['car_pos_x']]
    
    car_pos_y_values = [torch.tensor(float(val), dtype=torch.float64) for val in train_set['car_pos_y']]
    wheel_velocity_values = [torch.tensor([float(val) for val in row['wheelVelocity'].strip('[]').split(',')], dtype=torch.float64) for _, row in train_set.iterrows()]

    # 这里可以添加其他列的处理方式，例如 'wheel_velocity' 等

    # 这里可以添加其他列的处理方式，例如 'wheel_velocity' 等

    # 将处理后的数据添加到列表中
    tensor_train_data.append({
        # 'car_pos_x':car_pos_x,
        'min_lidar': min_lidar_values,
        'car_pos_x_values': car_pos_x_values,
        'car_pos_y_values': car_pos_y_values,
        'target': wheel_velocity_values,
        # 其他列的处理结果
    })
# print(tensor_train_data[2]['car_pos_x_values'])
for test_set in test_data:
    # 假设 'min_lidar' 是你的特定列名
    min_lidar_values = [torch.tensor([float(val) for val in row['min_lidar'].strip('[]').split(',')], dtype=torch.float64) for _, row in test_set.iterrows()]
    car_pos_x_values = [torch.tensor(float(val), dtype=torch.float64) for val in test_set['car_pos_x']]
    car_pos_y_values = [torch.tensor(float(val), dtype=torch.float64) for val in test_set['car_pos_y']]
    wheel_velocity_values = [torch.tensor([float(val) for val in row['wheelVelocity'].strip('[]').split(',')], dtype=torch.float64) for _, row in test_set.iterrows()]

    # 这里可以添加其他列的处理方式，例如 'wheel_velocity' 等

    # 这里可以添加其他列的处理方式，例如 'wheel_velocity' 等

    # 将处理后的数据添加到列表中
    tensor_test_data.append({
        # 'car_pos_x':car_pos_x,
        'min_lidar': min_lidar_values,
        'car_pos_x_values': car_pos_x_values,
        'car_pos_y_values': car_pos_y_values,
        'target': wheel_velocity_values,
        # 其他列的处理结果
    })


import torch
import torch.nn as nn
import torch.optim as optim

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1).double()
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2).double()
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size).double()

    def forward(self, x):
        x = x.float().double()
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

input_size = 182
hidden_size1 = 128  
hidden_size2 = 64
output_size = 2

model = MLP(input_size, hidden_size1, hidden_size2, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 20


for epoch in range(num_epochs):


    for data in tensor_train_data: #將每一篇每爭畫面抓出來
        car_pos_x = torch.tensor(data['car_pos_x_values'], dtype=torch.float64)
        car_pos_y = torch.tensor(data['car_pos_y_values'], dtype=torch.float64)
        
        # car_pos_x = torch.unsqueeze(car_pos_x, dim=0)
        # car_pos_y = torch.unsqueeze(car_pos_y, dim=0)
        
        for lidar_data, car_pos_x_data, car_pos_y_data,target_data in zip(data['min_lidar'],car_pos_x,car_pos_y, data['target']):
            model.train()
            lidar_data = torch.tensor(lidar_data, dtype=torch.float64)
            car_pos_x_data = car_pos_x_data.unsqueeze(0).double()
            car_pos_y_data = car_pos_y_data.unsqueeze(0).double()
            lidar_data = lidar_data.double()
            
            inputs = torch.cat([car_pos_x_data, car_pos_y_data, lidar_data], dim=0).double()
            inputs = inputs.double()
            output = model(inputs)
            loss = criterion(output, target_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(f"Epoch: {epoch} | Loss: {loss}")

    model.eval()
    with torch.inference_mode():
        for data in tensor_test_data:
            for lidar_data, car_pos_x_data, car_pos_y_data,target_data in zip(data['min_lidar'],car_pos_x,car_pos_y, data['target']):
                car_pos_x_data = car_pos_x_data.unsqueeze(0).double()
                car_pos_y_data = car_pos_y_data.unsqueeze(0).double()
                lidar_data = lidar_data.double()
                inputs = torch.cat([car_pos_x_data, car_pos_y_data, lidar_data], dim=0).double()
                output = model(inputs)
                test_loss = criterion(output, target_data)
        print(f"Epoch: {epoch} | Test_Loss: {test_loss}")




from pathlib import Path

#Create models directory
MODEL_PATH = Path("./models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

#create model save path
MODEL_NAME = "01_pytorch_workflow.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),f=MODEL_SAVE_PATH)