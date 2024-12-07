#!/usr/bin/env python3
import rospy
from trajectory_msgs.msg import JointTrajectoryPoint
from actionlib import SimpleActionClient
import numpy as np
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryGoal
class ur10_move:
    def sayhello(self):
        print('Hello!!!')
    def __init__(self, action_name, default_duration):
        """
        初始化 Action 客户端和默认参数
        """
        self.client = SimpleActionClient(action_name, FollowJointTrajectoryAction)
        self.default_duration = default_duration
        
        # 设置默认关节名称
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        
        self.current_position = np.zeros(6)  # 初始化当前关节位置为零

    def init_client(self):
        """
        等待服务器连接
        """
        rospy.loginfo("Waiting for action server to start...")
        self.client.wait_for_server()
        rospy.loginfo("Action server connected.")

    def set_target_position(self, position):
        """
        设置目标位置
        :param position: 一个包含目标关节角度的 NumPy 数组
        """
        if len(position) != 6:
            raise ValueError("Target position must have exactly 6 values.")
        self.current_position = position

    def execute_movement(self):
        """
        执行移动
        """
        # 创建轨迹目标
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names  # 设置关节名称

        # 创建并填充一个轨迹点
        point = JointTrajectoryPoint()
        point.positions = self.current_position.tolist()  # 填充目标位置
        point.time_from_start = rospy.Duration(self.default_duration)  # 设置到达时间
        goal.trajectory.points.append(point)  # 添加轨迹点

        # 发送轨迹目标
        rospy.loginfo("Sending trajectory goal...")
        self.client.send_goal(goal)
        rospy.loginfo("Trajectory goal sent.")

    def set_default_duration(self, duration):
        """
        设置默认持续时间
        """
        self.default_duration = duration