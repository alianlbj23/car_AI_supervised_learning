import torch
from datetime import datetime
from UnityAdaptor import transfer_obs
# from Entity import State


import threading
import sys
from rclpy.node import Node
import rclpy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

DEG2RAD = 0.01745329251

unityState = ""

class AiNode(Node):
    def __init__(self):
        super().__init__("aiNode")
        self.get_logger().info("Ai start")#ros2Ai #unity2Ros
        self.subsvriber_ = self.create_subscription(String, "unity2Ros", self.receive_data_from_ros, 10)
        self.dataList = list()
        self.publisher_Ai2ros = self.create_publisher(Float32MultiArray, 'ros2Unity', 10)#Ai2ros #ros2Unity

        input_size = 182
        hidden_size1 = 128  # 根据需要调整
        hidden_size2 = 64
        output_size = 2


        self.loaded_model_1 = MLP(input_size, hidden_size1, hidden_size2, output_size)


        self.loaded_model_1.load_state_dict(torch.load("./dataFile/ver2.pth", map_location=torch.device('cpu'))) #model loading

    
    def publish2Ros(self, data):
        
        self.data2Ros = Float32MultiArray()
        self.data2Ros.data = data
        self.publisher_Ai2ros.publish(self.data2Ros)

    def receive_data_from_ros(self, msg):
        global unityState        
        unityState = msg.data

        unityState = transfer_obs(unityState)
        if len(unityState.min_lidar) == 0:
            unityState.min_lidar = [100] * 180

        self.data = torch.tensor([
                    unityState.car_pos.x, 
                    unityState.car_pos.y, 
                    # unityState.car_vel.x, 
                    # unityState.car_vel.y,
                    # unityState.car_angular_vel,
                    # unityState.wheel_angular_vel.left_back,
                    # unityState.wheel_angular_vel.right_back,
                    
                    ]+unityState.min_lidar,dtype=torch.float32)


            
        

        self.loaded_model_1.eval()
        with torch.no_grad():
            # action = list()
            # action.append(0)
            # output = self.loaded_model_1(self.data2)
            # tensor = output.squeeze().tolist()

            # action.append(tensor[0]*1.2)
            # action.append(tensor[1]*1.2)
            # action = [float(value) for value in action]
            # self.publish2Ros(action)


            output = self.loaded_model_1(self.data)
            action = list()
            action.append(0)
            action.append(output[0])
            action.append(output[1])
            print(action)
            action = [float(value) for value in action]
            self.publish2Ros(action)

        


        
        

def spin_pros(node):
    exe = rclpy.executors.SingleThreadedExecutor()
    exe.add_node(node)
    exe.spin()
    rclpy.shutdown()
    sys.exit(0)

def returnUnityState():
    while len(unityState) == 0:
        pass
    return unityState

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

# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
        

#     def forward(self, x):
#         if x.ndim < 3:
#             x = x.unsqueeze(1)
#         lstm_out, _ = self.lstm(x)
#         # print(len(lstm_out))
#         output = self.fc(lstm_out[:, -1, :])  # 取序列的最后一个输出作为预测结果
#         return output
    
def main():
    

    rclpy.init()
    node = AiNode()
    pros = threading.Thread(target=spin_pros, args=(node,))
    pros.start()  


                  

if __name__ == '__main__':

    main()