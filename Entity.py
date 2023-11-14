from pydantic import BaseModel

class ROS2Point(BaseModel):
    x: float
    y: float
    z: float

# class lidar_direciton(BaseModel):
#     list(float, float, float)

# around ROS2 z axis, left +, right -, up 0, down 180
class WheelOrientation(BaseModel):
    left_front: float=0
    right_front: float=0

# around car wheel axis, front: +, back: -, r/s
class WheelAngularVel(BaseModel):
    left_back: float
    right_back: float

class State(BaseModel):
    car_pos: ROS2Point

    car_vel: ROS2Point # in ROS2 coordinate system
    car_angular_vel: float # r/s, in ROS2 around car z axis, yaw++: -, yaw--: +, counter-clockwise: +, clockwise: -, in Unity:  counter-clockwise: -, clockwise: +
    wheel_angular_vel: WheelAngularVel # around car wheel axis, front: +, back: -
    min_lidar: list # meter
    wheelVelocity: list

    # because orientation is transformed back to Unity coordinate system, here lidar direction alse needs to be transformed back from ROS2 to Unity
    # min_lidar_relative_angle: float # radian, base on car, right(x): 0, front(y): 90,  upper: 180 --->x 0, down: -180 --->x 0



class ControlSignal(BaseModel):
    wheel_vel: float # rad/s
    steering_angle: float # degree, left: -, right: +