#!/home/robot/anaconda3/envs/mytorch/bin/python
import torch
import torchvision
import rospy
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

print(torch.__version__)
flag=torch.cuda.is_available()
print(flag)
max=np.array([-3.70376967e-02,6.39087200e-01,1.05172992e+00,-3.66398245e-02,6.39733434e-01,1.05185831e+00])
min=np.array([-3.50500733e-01,4.02689129e-01,7.35534191e-01,-3.50699395e-01,4.02544379e-01,7.35628963e-01])

#数据前三列是状态，后三列是动作
train_data=[]
test_data=[]
# 初始化参数
step = 1  # 无间隔取数据
window_size = 100  # 每组训练数据包含 100 个点 相当于50Hz下100个点 2s的观测时长 实际上一个圆形的周期是6-7s
#分割数据集 测试集和训练集的比例
test_size = 0.1
train_size = 0.9
states_data=[]
actions_data=[]
# 自定义数据集
class RobotArmDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        #self.sequence_length = sequence_length #每条序列的时间步数
    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]
def normalize_trajectory(trajectories):
    # 拼接所有轨迹，计算全局最小值和最大值
    all_data = np.vstack(trajectories)  # 将所有轨迹垂直拼接成一个大矩阵
    # min_val = np.min(all_data, axis=0)  # 每列的全局最小值
    # max_val = np.max(all_data, axis=0)  # 每列的全局最大值
    max_val_8_to_11 = np.max(all_data[:, 8:11], axis=0)
    max_val_1_to_4 = np.max(all_data[:, 1:4], axis=0)
    min_val_8_to_11= np.min(all_data[:, 8:11], axis=0)
    min_val_1_to_4 = np.min(all_data[:, 1:4], axis=0)
    min_val=[min_val_8_to_11,min_val_1_to_4]
    max_val=[max_val_8_to_11,max_val_1_to_4]
    # 对每条轨迹归一化
    # normalized_trajectories = [(traj - min_val) / (max_val - min_val) for traj in trajectories]
    return min_val,max_val
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
#我的采样频率是50Hz 构造训练集时 一个样本是 25*6 的结构 滑动窗口对样本轨迹进行包装为数据集 每隔2个点采集一次数据
def load_model_input(trajectory,min_val,max_val):
    length=len(trajectory)
    state_min_value=min_val[0]
    state_max_value=max_val[0]
    action_min_value=min_val[1]
    action_max_value=max_val[1]
    # 遍历数据，按窗口和步长提取样本
    for start in range(0, len(trajectory) - step * (window_size - 1), step):
        # 每次取一个窗口大小的数据
        state_sample = trajectory[start:start + step * window_size:step,8:11]
        state_sample_normalized = (state_sample - state_min_value) / (state_max_value - state_min_value)

        action_sample = trajectory[start:start + step * window_size:step,1:4]
        action_sample_normalized = (action_sample - action_min_value) / (action_max_value - action_min_value)
        # 检查是否符合窗口要求
        if len(state_sample) == window_size:
            states_data.append(state_sample_normalized)
            actions_data.append(action_sample_normalized)
# 定义基于 LSTM 的模仿学习网络
class RobotArmLSTMModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=2):
        super(RobotArmLSTMModel, self).__init__()
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        # 多头自注意力机制
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        # 前馈神经网络部分（两个线性层）
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    def forward(self, x):
        # LSTM 输出序列特征
        lstm_out, _ = self.lstm(x)
        # 取 LSTM 输出的最后一个时间步的输出
        last_time_step = lstm_out[:, -1, :]
        # 通过多头自注意力机制
        # 为了符合输入要求，last_time_step 需要进行形状调整
        attn_out, _ = self.attn(last_time_step.unsqueeze(0), last_time_step.unsqueeze(0), last_time_step.unsqueeze(0))
        # 去掉多余的维度
        attn_out = attn_out.squeeze(0)
        # 通过第一个线性层
        fc1_out = torch.relu(self.fc1(attn_out))
        # 通过第二个线性层，得到最终的动作输出
        action = self.fc2(fc1_out)
        return action

# 训练函数
def train_model(model, train_loader, optimizer, criterion, device, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)
            # 前向传播
            predictions = model(states)
            # 只计算最后一个时间步的损失
            loss = criterion(predictions, actions[:, -1, :])
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #最后的loss是在10-5数量级
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.6f}")

def main():
    # 超参数设置
    state_dim = 3 # 状态维度
    action_dim = 3 # 动作维度
    hidden_dim = 64 # 隐藏层层数 隐藏层层数越多运动越稳定越保守 学习到更多隐藏的复杂特征
    batch_size = 64
    num_epochs = 200 # 迭代200次 迭代次数越多越像训练集 100次训练的效果不太稳定
    learning_rate = 0.0005

    file_path='/home/robot/test0_ws/ur10_trajectory_circle_data16.txt'
    trajectories = load_trajectory_data(file_path)
    # 对数据集进行归一化操作 提取出最大值和最小值并打印
    min_value,max_value=normalize_trajectory(trajectories)
    print(min_value)
    print(max_value)
    
    # tr=trajectories[0]
    # load_model_input(tr)
    # #打印封装后的第一个训练集数据
    # print(all_data[0])
    
    #返回结果是索引和元素 其中元素是第i个numpy数组 i从0开始
    for i, traj in enumerate(trajectories):
        print(f"Trajectory {i+1} shape: {traj.shape}")
        load_model_input(trajectories[i],min_value,max_value)
    # #len的长度是元素个数
    # print("一共有",len(all_data),"元素")
    # print(all_data[0])
    # print(all_data[len(all_data)-1])
    # train_data = all_data[:int(len(all_data)*train_size)]
    # test_data = all_data[int(len(all_data)*train_size):]
    # print("Data loaded")
    # print(f"Number of all data samples: {len(all_data)}")
    # print(f"Number of train data samples: {len(train_data)}")
    # print(f"Number of test data samples: {len(test_data)}")

    # 加载数据
    dataset = RobotArmDataset(torch.FloatTensor(states_data), torch.FloatTensor(actions_data))
    # print(states_data[0])
    # print(actions_data[0])

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 模型、损失函数、优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobotArmLSTMModel(state_dim, action_dim, hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 训练模型
    train_model(model, train_loader, optimizer, criterion, device, num_epochs)
    # 保存模型
    torch.save(model.state_dict(), "robot_arm_lstm_model.pth")
    print("Model saved to robot_arm_lstm_model.pth")
    
if __name__ == "__main__":
    main()








