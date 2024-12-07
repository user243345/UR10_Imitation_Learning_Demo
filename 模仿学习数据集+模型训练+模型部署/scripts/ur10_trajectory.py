#!/usr/bin/env python3
import rospy
import numpy as np
from ur_python_pkg import ur10_inverse
from ur_python_pkg import ur10_move
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Quaternion
#导入夹爪控制包
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output
from std_msgs.msg import Bool
#UR10采集到的轨迹复现 假设我以50Hz的频率做实验
#采集到目标点位
#将目标点位以50Hz的控制频率 0.1s的移动速率才能移动到目标位置
#因此python中的复现也应当与采集实验中的数据重合 并且采集频率一定要高才能够丝滑运行

global joint_position
global init_joint_flag
class GripperControl:
    def __init__(self, nh):
        # 初始化发布器
        self.gripper_pub = nh.advertise("/Robotiq2FGripperRobotOutput", 10)
        # 初始化夹爪命令
        self.gripper_command = Robotiq2FGripper_robot_output()
        self.gripper_command.rACT = 0  # 激活夹爪
        self.gripper_command.rGTO = 0  # 使能夹爪控制
        self.gripper_command.rSP = 0   # 设置最大速度
        self.gripper_command.rFR = 0   # 设置最大力量
        self.gripper_command.rPR = 0   # 初始位置，夹爪完全打开
        self.gripper_pub.publish(self.gripper_command)
        rospy.sleep(1)
        # 激活夹爪，设置速度、力量
        self.gripper_command.rACT = 1
        self.gripper_command.rGTO = 1  # 使能夹爪控制
        self.gripper_command.rSP = 255  # 设置最大速度
        self.gripper_command.rFR = 150  # 设置最大力量
        self.gripper_command.rPR = 0    # 初始位置，夹爪完全打开
        self.gripper_pub.publish(self.gripper_command)
        rospy.sleep(1)
        # 初始状态：夹爪未激活
    def close_gripper(self, target_position):
        # 确保目标位置在有效范围内（0到250）
        if target_position < 0:
            target_position = 0
        elif target_position > 250:
            target_position = 250

        # 设置夹爪位置
        self.gripper_command.rPR = target_position
        
        # 发布新的夹爪命令
        self.gripper_pub.publish(self.gripper_command)
        rospy.loginfo(f"Gripper position set to {target_position}")

    def update_gripper_state(self):
        # 更新夹爪状态
        if self.gripper_active:
            rospy.loginfo("Gripper activated.")
        else:
            rospy.loginfo("Gripper deactivated.")
# 假设你的逆运动学求解函数
def inverse_kinematics(position,current_pos, orientation):
    # 返回格式为 [q1, q2, q3, q4, q5, q6]
    joint_angles=ur10_inverse.ur10_inverse(position,current_pos,orientation)
    return joint_angles
def read_txt_file(file_path):
    """
    从 txt 文件中读取位置和四元数。
    假设文件以列存储，前 3 列为位置，后 4 列为四元数。
    返回：位置和四元数的列表
    """
    data = np.loadtxt(file_path)
    positions = data[:, 1:4]  # 前三列为位置
    orientations = data[:, 4:8]  # 后四列为四元数
    gripper_target_position=data[:,-1]
    return positions, orientations,gripper_target_position
def send_trajectory_command(joint_angles_list):
    traj_msg = JointTrajectory()
    traj_msg.header = Header()
    traj_msg.header.stamp = rospy.Time.now()
    traj_msg.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint","wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

    # 填充每个轨迹点
    for i, joint_angles in enumerate(joint_angles_list):
        point = JointTrajectoryPoint()
        point.positions = joint_angles
        point.time_from_start = rospy.Duration(1.0 + i * 1.0)  # 每个点间隔 1 秒
        traj_msg.points.append(point)
    rospy.loginfo("Sending trajectory command...")
# 回调函数，处理订阅到的关节角度数据
def joint_state_callback(msg):
    global joint_position
    global init_joint_flag
    # 获取关节角度
    joint_positions=msg.position
    joint_position[2] = joint_positions[0]  # msg.position是一个包含所有关节角度的列表
    joint_position[1] = joint_positions[1]
    joint_position[0] = joint_positions[2]
    joint_position[3] = joint_positions[3]
    joint_position[4] = joint_positions[4]
    joint_position[5] = joint_positions[5]
    if init_joint_flag==0:
        init_joint_flag=1

def main():
    global joint_position
    global init_joint_flag
    init_joint_flag=0
    joint_position=[0,0,0,0,0,0]
    rate=rospy.Rate(10)
    rospy.Subscriber('/joint_states', JointState, joint_state_callback)
    # 初始化轨迹控制器
    controller = ur10_move.ur10_move("/scaled_pos_joint_traj_controller/follow_joint_trajectory", 2.0)
    controller.init_client()
    controller.set_default_duration(5)
    # 初始化gripper_control类
    gripper_control = GripperControl(rospy)
    gripper_target_position = 0
    gripper_control.close_gripper(gripper_target_position)
    # target_position = np.array([1.0, 0.5, -1.0, 0.0, 0.5, -0.5])
    # controller.set_target_position(target_position)
    # 执行移动
    #controller.execute_movement()
    joint_angles_list=[]
    pos=[0.206161,-0.535404,0.954031]
    # 创建一个四元数对象
    pose=[-0.007825,0.019776,-0.379199,-0.925071]
    joint=inverse_kinematics(pos,joint_position,pose)
    print(joint)
    # 读取文件中所有目标位置 存储在变量joint_angles_list中 
    # txt 文件路径
    txt_file_path = "ur10_trajectory_copy.txt"  # 替换为实际路径
    # 读取位姿数据 
    positions, orientations,gripper_target_position = read_txt_file(txt_file_path)
    pose_index=0
    init_flag=0
    # 有while循环时不需要运行spin 后台自动回调 否则会卡住
    while not rospy.is_shutdown():
        #joint=inverse_kinematics(pos,joint_position,pose)
        #print(joint)
        # 循环中需要实现每0.1s发送一次运动指令 并且记录轨迹 
        if init_joint_flag==1:
            print(positions[pose_index])
            if init_flag==0:
                joint=inverse_kinematics(positions[pose_index],joint_position,orientations[pose_index])
                pose_index=pose_index+1
                controller.set_target_position(joint)
                controller.set_default_duration(5)
                controller.execute_movement()
                gripper_control.close_gripper(gripper_target_position[pose_index])
                rospy.sleep(5)
                init_flag=1
            joint=inverse_kinematics(positions[pose_index],joint_position,orientations[pose_index])
            pose_index=pose_index+1
            controller.set_target_position(joint)
            controller.set_default_duration(0.1)
            #控制夹爪/控制机械臂
            controller.execute_movement()
            gripper_control.close_gripper(gripper_target_position[pose_index])

            
        rate.sleep()
    

if __name__ == "__main__":
    rospy.init_node('trajectory_sender', anonymous=True)
    try:
        main()
    except rospy.ROSInterruptException:
        pass
