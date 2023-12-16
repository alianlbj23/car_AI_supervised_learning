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
        self.frame_buffer = []
        self.action_buffer = []

        self.input_size = 185 # input dimension 15
        self.hidden_size = 64
        self.num_layers = 3
        self.output_size = 3
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True).to(device)
        self.linear = nn.Linear(self.hidden_size, 1).to(device)  


        self.model_path = './Model/best_model.pth'
        self.model_weights = torch.load(self.model_path)
        self.lstm.load_state_dict(self.model_weights)

    def publish_predict_data_2_unity(self, data):
        self.data = Float32MultiArray()
        self.data.data = data
        self.publisher_AINode_2_unity_thu_ROSbridge.publish(self.data)

    def unity_data_collect(self, unityState):
        token = list()
        self.state_detect, token = transfer_obs(unityState)
        if self.state_detect == 1:
            #  收到資料後將他丟入LSTM
            self.lstm.eval()
            with torch.inference_mode():
                test_input = [eval(token)]
                test_input_tensor = torch.tensor(test_input, dtype=torch.float32)

                X = test_input_tensor[:, :-1].unsqueeze(1).to(device)
                lstm_output = self.lstm(X)
                probabilities = F.softmax(lstm_output, dim=1)
                predicted_action = torch.argmax(probabilities, dim=1)
                action = [0]+[float(predicted_action.item())]
                print(action)
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
