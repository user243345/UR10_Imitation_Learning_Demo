#!/home/robot/anaconda3/envs/mytorch/bin/python
#导入各种包
import torch
import torchvision
import rospy
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from ur_python_pkg import ur10_inverse as ur_ik
from ur_python_pkg import ur10_move as ur_move
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Quaternion
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output
from std_msgs.msg import Bool
from collections import deque
import time
from multiprocessing import Process, Manager
import threading

min=[np.array([ 0.306683, -0.400968,  0.331587]), np.array([ 0.306266, -0.401518,  0.33113 ])] #最小值：状态变量/动作变量
max=[np.array([ 0.683813, -0.193177,  0.620889]), np.array([ 0.684455, -0.192471,  0.621499])] #最大值：状态变量/动作变量
window_size=100
#写明回调函数和执行移动的函数逻辑 
#编写打包数据集的函数 将打包后的数据集输入到模型中 模型计算出结果发送给执行器执行
#编写观测状态用的程序

global arm_positions,arm_orientations,joint_positions,init_joint_flag,positions_shared,init_window_flag
def return_normalized(input_list, mode):
    # # 转换输入为 NumPy 数组，确保支持 NumPy 操作
    # input_array = np.array(input_list)
    # if input_array.shape[0] != 3:
    #     raise ValueError("Input must have exactly 3 rows.")
    # 定义不同模式下的归一化参数
    if mode == 'state':
        state_min_value = min[0]  # 假设 state 的最小值
        state_max_value = max[0]   # 假设 state 的最大值
        state_sample_normalized = (input_list - state_min_value) / (state_max_value - state_min_value)
        return state_sample_normalized
    elif mode == 'action':
        min_val = min[1]  # 假设 action 的最小值
        max_val = max[1]     # 假设 action 的最大值
        action_denormalized = input_list * (max_val - min_val) + min_val # 执行逆归一化：scale * normalized + min
        return action_denormalized
    else:
        raise ValueError("Mode must be 'state' or 'action'.")
    
#实时记录机械臂的末端位置，并打包成固定窗口大小的网络输入
def record_end_effector_positions():
    global arm_positions,positions_shared,init_window_flag,window_size
    print(window_size)
    arm_pose_update()  # 更新全局变量
    positions_shared.append(list(arm_positions))  # 记录当前值
    if init_window_flag==0:
        batch_positions = [list(arm_positions) for _ in range(window_size)] #如果数据量没有到100 就用100个原点数据填充
        positions_shared.extend(batch_positions)
        init_window_flag=1
    if len(positions_shared) > window_size:  # 只保留最近100个数据
        positions_shared.pop(0)

def arm_pose_update():
    global arm_positions,arm_orientations,joint_positions
    arm_positions,arm_orientations=ur_ik.ur10_forward(joint_positions)

def joint_state_callback(msg):
    global joint_positions
    global init_joint_flag
    # 获取关节角度
    joint_position=msg.position
    joint_positions[2] = joint_position[0]  # msg.position是一个包含所有关节角度的列表
    joint_positions[1] = joint_position[1]
    joint_positions[0] = joint_position[2]
    joint_positions[3] = joint_position[3]
    joint_positions[4] = joint_position[4]
    joint_positions[5] = joint_position[5]
    if init_joint_flag==0:
        init_joint_flag=1

# 定义基于 LSTM 的模仿学习网络
class RobotArmLSTMModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=2):
        super(RobotArmLSTMModel, self).__init__()
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        attn_out, _ = self.attn(last_time_step.unsqueeze(0), last_time_step.unsqueeze(0), last_time_step.unsqueeze(0))
        attn_out = attn_out.squeeze(0)
        fc1_out = torch.relu(self.fc1(attn_out))
        action = self.fc2(fc1_out)
        return action
    
def main():
    global joint_positions,init_joint_flag,arm_positions,arm_orientations,positions_shared,init_window_flag
    init_joint_flag=0;init_flag=0;init_window_flag=0
    joint_positions=[0,0,0,0,0,0];positions_shared=[]
    init_positions=[0.387899,-0.298762,0.444887]
    init_orientations=[0.655676,0.246287,0.315617,0.640170] #xyzw的赋值顺序
    target_positions=[]
    target_orientations=[]
    rate=rospy.Rate(50) #控制频率也是观测频率
    rospy.Subscriber('/joint_states', JointState, joint_state_callback)
    # 初始化轨迹控制器
    controller = ur_move.ur10_move("/scaled_pos_joint_traj_controller/follow_joint_trajectory", 2.0)
    controller.init_client()
    controller.set_default_duration(5)
    # 定义模型参数
    state_dim = 3  # 输入状态维度
    action_dim = 3  # 输出动作维度
    hidden_dim = 64  # 隐藏层维度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #加载设备
    model = RobotArmLSTMModel(state_dim, action_dim, hidden_dim).to(device) #加载模型
    model.load_state_dict(torch.load("robot_arm_lstm_model.pth", map_location=device)) #加载训练权重
    model.eval()  # 切换到评估模式
    while not rospy.is_shutdown():
        # 循环中需要实现每0.1s发送一次运动指令 并且记录轨迹 
        if init_joint_flag==1:
            record_end_effector_positions() #更新机械臂当前位姿并打包
            if init_flag==0:
                move_joint_command=ur_ik.ur10_inverse(init_positions,joint_positions,init_orientations)
                controller.set_target_position(move_joint_command)
                controller.set_default_duration(3)
                controller.execute_movement()
                rospy.sleep(5)
                init_flag=1
            
            #这里时刻运行打包数据集函数 并将函数输出传递到主函数中进行逆运动学求解
            data_input=return_normalized(positions_shared,'state')
            #注意将输入转化为张量形式
            test_input = torch.FloatTensor(data_input).unsqueeze(0).to(device)
            predicted_action = model(test_input)
            # 将张量转换为 NumPy 数组
            predicted_action_numpy = predicted_action.detach().cpu().numpy()
            target_positions = predicted_action_numpy[0]
            data_output=return_normalized(target_positions,'action')
            print(data_output)

            move_joint_command=ur_ik.ur10_inverse(data_output,joint_positions,init_orientations)
            controller.set_target_position(move_joint_command)
            controller.set_default_duration(0.1)
            controller.execute_movement()
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node('trajectory_sender', anonymous=True)
    try:
        main()
    except rospy.ROSInterruptException:
        pass


    # #开辟线程用于更新网络输入所用数据包
    # positions_shared = []
    # record_thread = threading.Thread(target=record_end_effector_positions, args=(positions_shared,))
    # record_thread.start()











