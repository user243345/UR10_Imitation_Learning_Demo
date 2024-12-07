#!/home/robot/anaconda3/envs/mytorch/bin/python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 自定义数据集
class RobotArmDataset(Dataset):
    def __init__(self, states, actions, sequence_length):
        """
        :param states: 机械臂的状态 (N, T, state_dim) - 当前的位置
        :param actions: 对应的目标动作 (N, T, action_dim) - 期望的位置
        :param sequence_length: 每条序列的时间步数
        """
        self.states = states
        self.actions = actions
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


# 定义基于 LSTM 的模仿学习网络
class RobotArmLSTMModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=2):
        super(RobotArmLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        # LSTM 输出序列特征
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_time_step = lstm_out[:, -1, :]
        # 全连接网络输出动作
        return self.fc(last_time_step)


# 数据加载与预处理
def load_data(sequence_length=20):
    # 生成一些随机数据用于演示
    num_samples = 1000
    state_dim = 3
    action_dim = 3

    # 模拟状态和动作序列
    states = np.random.rand(num_samples, sequence_length, state_dim)
    actions = states + np.random.normal(0, 0.1, size=(num_samples, sequence_length, action_dim))

    return states, actions


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
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")


# 主函数
def main():
    # 超参数设置
    state_dim = 3
    action_dim = 3
    hidden_dim = 128
    sequence_length = 20
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001

    # 加载数据
    states, actions = load_data(sequence_length)
    dataset = RobotArmDataset(torch.FloatTensor(states), torch.FloatTensor(actions), sequence_length)
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
