#!/home/robot/anaconda3/envs/mytorch/bin/python
import torch
import torchvision
import rospy
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

min=[np.array([ 0.306683, -0.400968,  0.331587]), np.array([ 0.306266, -0.401518,  0.33113 ])] #最小值：状态变量/动作变量
max=[np.array([ 0.683813, -0.193177,  0.620889]), np.array([ 0.684455, -0.192471,  0.621499])] #最大值：状态变量/动作变量

# 初始化参数
step = 1  # 无间隔取数据
window_size = 100  # 每组训练数据包含 100 个点
all_data = []  # 存储训练集的列表
#分割数据集 测试集和训练集的比例
states_data=[]
actions_data=[]

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
def load_trajectory_data(file_path):
    trajectories = []  # 用于存储每条轨迹的列表 这个
    current_trajectory = []  # 用于存储当前轨迹的数据
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line == "":  # 检测空行，表示新轨迹的结束
                if current_trajectory:  # 如果当前轨迹不为空，存储并重置
                    trajectories.append(np.array(current_trajectory, dtype=float))
                    current_trajectory = []
            else:
                # 将每行数据解析为浮点数并仅提取第1到第6列
                all_columns = list(map(float, line.split(' ')))  # 假设数据列以空格分隔
                current_trajectory.append(all_columns[:11])  # 提取第1到第6列
    # 如果文件最后一条轨迹没有空行结束，确保最后的轨迹被存储
    if current_trajectory:
        trajectories.append(np.array(current_trajectory, dtype=float))
    return trajectories
def load_model_input(trajectory):
    length=len(trajectory)
    #print(length)
    # for i in range(length):
    #     print(trajectory[i,:])
    # 遍历数据，按窗口和步长提取样本
    for start in range(0, len(trajectory) - step * (window_size - 1), step):
        # 每次取一个窗口大小的数据
        state_sample = trajectory[start:start + step * window_size:step,8:11]
        action_sample = trajectory[start:start + step * window_size:step,1:4]
        # 检查是否符合窗口要求
        if len(state_sample) == window_size:
            states_data.append(state_sample)
            actions_data.append(action_sample)

def return_normalized(input_list, mode):
    # 定义不同模式下的归一化参数
    if mode == 'state':
        state_min_value = min[0]  # 假设 state 的最小值
        state_max_value = max[0]   # 假设 state 的最大值
        state_sample_normalized = (input_list - state_min_value) / (state_max_value - state_min_value)
        return state_sample_normalized
    elif mode == 'action':
        min_val = min[1]  # 假设 action 的最小值
        max_val = max[1]     # 假设 action 的最大值
        action_sample_normalized = input_list * (max_val - min_val) + min_val # 执行逆归一化：scale * normalized + min
        return action_sample_normalized
    else:
        raise ValueError("Mode must be 'state' or 'action'.")
    
def main():
    file_path='/home/robot/test0_ws/ur10_trajectory_circle_data16.txt'
    trajectories = load_trajectory_data(file_path)
    for i, traj in enumerate(trajectories):
        print(f"Trajectory {i+1} shape: {traj.shape}")
        load_model_input(trajectories[i])
    print(states_data[0])
    print(actions_data[0])
    # 定义模型参数
    state_dim = 3  # 输入状态维度
    action_dim = 3  # 输出动作维度
    hidden_dim = 64  # 隐藏层维度
    # 加载设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型
    model = RobotArmLSTMModel(state_dim, action_dim, hidden_dim).to(device)
    # 加载训练好的权重
    model.load_state_dict(torch.load("robot_arm_lstm_model.pth", map_location=device))
    model.eval()  # 切换到评估模式
    # 测试数据（假设有一个输入序列）
    test_input = states_data[0]  # batch_size=1, sequence_length=25
    test_input = return_normalized(test_input,'state')
    #注意将输入转化为张量形式
    test_input = torch.FloatTensor(test_input).unsqueeze(0).to(device)
    predicted_action = model(test_input)
    # 将张量转换为 NumPy 数组
    predicted_action_numpy = predicted_action.detach().cpu().numpy()
    target_positions = predicted_action_numpy[0]
    data_output=return_normalized(target_positions,'action')
    print(data_output)



if __name__ == "__main__":
    main()
