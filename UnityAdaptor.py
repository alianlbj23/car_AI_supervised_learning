from Entity import State
from Entity import ROS2Point
from Entity import WheelAngularVel
from Utility import clamp
import json

def transfer_obs(obs):
    obs = jsonTransToDict(obs)
    state = State(
        car_pos = ROS2Point(x = obs['ROS2CarPosition'][0], 
                            y = obs['ROS2CarPosition'][1], 
                            z=0.0),#obs[2]
        car_vel = ROS2Point(x = obs['ROS2CarVelocity'][0], 
                            y = obs['ROS2CarVelocity'][1], 
                            z=0.0), #obs[20]
        car_angular_vel = obs['ROS2CarAugularVelocity'][2], #obs[21 22]
        wheel_angular_vel = WheelAngularVel(left_back = obs['ROS2WheelAngularVelocityLeftBack'][1], #obs[30]
                                            right_back = obs['ROS2WheelAngularVelocityRightBack'][1], #obs[34 36]
                                            ),
        min_lidar = obs['ROS2Range'], #57 58 59
        wheelVelocity = obs['Wheelvelocity'],
    )
    return state

def jsonTransToDict(obs):
    obs = json.loads(obs)
    
    for key, value in obs.items():
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            coordinate_str = value.strip('()')  
            coordinates = list(map(float, coordinate_str.split(',')))  
            obs[key] = coordinates
    return obs