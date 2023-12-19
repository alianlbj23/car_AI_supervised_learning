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

        input_size = 185
        hidden_size1 = 128
        hidden_size2 = 64
        output_size = 3

        self.model = MLP(input_size, hidden_size1, hidden_size2, output_size).to(device)

        self.model_path = './Model/MLP_model.pth'

        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))

    def publish_predict_data_2_unity(self, data):
        self.data = Float32MultiArray()
        self.data.data = data
        self.publisher_AINode_2_unity_thu_ROSbridge.publish(self.data)

    def unity_data_collect(self, unityState):
        token = list()
        self.state_detect, token = transfer_obs(unityState)
        if self.state_detect == 1:
            self.model.eval()
            new_frame = eval(token)
            new_frame = new_frame[:-1]
            new_frame_tensor = torch.tensor(new_frame, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension and move to the correct device
            print(new_frame_tensor.shape)
            with torch.no_grad():
                output = self.model(new_frame_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                max_prob, predicted_class = torch.max(probabilities, 1)
                action = max_prob.item(), predicted_class.item()
                print(f"Predicted class: {predicted_class.item()}, Probability: {max_prob.item()}")
                action = []
                action.append(0)
                print(predicted_class.item())
                action.append(float(predicted_class.item()))
                action = [float(value) for value in action]
                self.publish_predict_data_2_unity(action)
                
    def callback_from_Unity(self, msg):
        self.unity_data_collect(msg.data)

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
