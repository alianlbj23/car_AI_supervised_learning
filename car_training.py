import numpy as np
from datetime import datetime



from UnityAdaptor import transfer_obs

import threading
import sys
from rclpy.node import Node
import rclpy
from std_msgs.msg import String
import csv
from datetime import datetime

DEG2RAD = 0.01745329251


class AiNode(Node):
    def __init__(self):
        super().__init__("aiNode")
        self.get_logger().info("Ai start")#ros2Ai #unity2Ros
        self.subsvriber_ = self.create_subscription(String, "unity2Ros", self.receive_data_from_ros, 10)

        self.subsvriber_stopFlag = self.create_subscription(String, "unity2Ros_stop", self.storeDataToCsv, 10)

        self.allUnityData = list()
    
    def UnityDataCollect(self, unityState):
        self.unityState = transfer_obs(unityState)
        if len(self.unityState.min_lidar) != 0:
            self.allUnityData.append(
                [
                    self.unityState.car_pos.x, 
                    self.unityState.car_pos.y, 
                    self.unityState.car_vel.x, 
                    self.unityState.car_vel.y,
                    self.unityState.car_angular_vel,
                    self.unityState.wheel_angular_vel.left_back,
                    self.unityState.wheel_angular_vel.right_back,
                    self.unityState.min_lidar,
                    self.unityState.wheelVelocity
                ]
            )

    def receive_data_from_ros(self, msg):
        self.unityState = msg.data
        self.UnityDataCollect(self.unityState)

    def storeDataToCsv(self, msg):
    
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        csv_file_path = f'./dataFile/lstm_training_{timestamp}.csv'

        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            csv_writer.writerow(['car_pos_x', 
                                 'car_pos_y', 
                                 'car_vel_x', 
                                 'car_vel_y',
                                 'car_angular_vel',
                                 'wheel_angular_vel_left_back',
                                 'wheel_angular_vel_right_back',
                                 'min_lidar',
                                 'wheelVelocity'
                                 ])

            csv_writer.writerows(self.allUnityData)
        sys.exit()
        
        

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