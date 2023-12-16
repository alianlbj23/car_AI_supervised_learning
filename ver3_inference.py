#lstm inference
import numpy as np
from datetime import datetime
import os
from UnityAdaptor import transfer_obs
import threading
import sys
from rclpy.node import Node
import rclpy
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AiNode(Node):
    def __init__(self):
        super().__init__("aiNode")
        self.get_logger().info("LSTM inference")#ros2Ai #unity2Ros

        self.subscriber_fromUnity_thu_ROSbridge_ = self.create_subscription(
            String, 
            "Unity_2_AI", 
            self.callback_from_Unity, 
            10
        )

        self.publisher_AINode_2_unity_thu_ROSbridge = self.create_publisher(
            Float32MultiArray, 
            'AI_2_Unity', 
            10
        )

        self.input_size = 15 # input dimension 15
        self.hidden_size = 64
        self.num_layers = 2
        self.fc_hidden_size = 128
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True).to(device)
        self.fc1 = nn.Linear(self.hidden_size, self.fc_hidden_size).to(device)  # 额外的全连接层
        self.fc2 = nn.Linear(self.fc_hidden_size, 2).to(device)  # 输出层

        self.model_path = './Model/best_model.pth'
        
        self.model_weights = torch.load(self.model_path, map_location=device)
        self.lstm.load_state_dict(self.model_weights)

        self.lstm.eval()
        

    def publish_predict_data_2_unity(self, data):
        print(data)
        self.data = Float32MultiArray()
        self.data.data = data
        self.publisher_AINode_2_unity_thu_ROSbridge.publish(self.data)

    def unity_data_collect(self, unityState):
        token = list()
        self.state_detect, token = transfer_obs(unityState)
        if self.state_detect == 1:
            #  收到資料後將他丟入LSTM
            self.lstm.to(device).eval()
            with torch.inference_mode():
                test_input = [eval(token)]
                test_input_tensor = torch.tensor(test_input, dtype=torch.float32)
                test_input_tensor = test_input_tensor[:, :-2]
                test_input_tensor = test_input_tensor.unsqueeze(0).to(device)
                h0 = torch.zeros(self.num_layers, test_input_tensor.size(0), self.hidden_size).to(device)
                c0 = torch.zeros(self.num_layers, test_input_tensor.size(0), self.hidden_size).to(device)
                lstm_output, _ = self.lstm(test_input_tensor, (h0, c0))
                fc1_out = torch.relu(self.fc1(lstm_output[:, -1, :])) 
                output = self.fc2(fc1_out)
                predicted_output = output.detach().cpu().numpy().flatten().tolist()

                print(predicted_output)
                action = list()
                action.append(0)
                action.append(predicted_output[0])
                action.append(predicted_output[1])
                action = [float(value) for value in action]
                self.publish_predict_data_2_unity(action)
                
    def callback_from_Unity(self, msg):
        self.unity_data_collect(msg.data)



def spin_pros(node):
    exe = rclpy.executors.SingleThreadedExecutor()
    exe.add_node(node)
    exe.spin()
    rclpy.shutdown()
    sys.exit(0)

def main():
    rclpy.init()
    node = AiNode()
    pros = threading.Thread(target=spin_pros, args=(node,))
    pros.start()  

if __name__ == '__main__':
    main()
