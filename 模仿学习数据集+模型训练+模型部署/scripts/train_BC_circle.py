import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pickle
import numpy as np
import os

#训练机械臂画圆，使用手控器试教50次，每次初始位置尽量相同有轻微不同
#得到model1_final.pth是效果最好的，model1_12_4_20也还不错，final使用的以下参数进行训练

# max=np.array([-1.38204992e-01,5.88260591e-01,1.02165461e+00,-1.38017267e-01,5.91183126e-01,1.02208912e+00])
# min=np.array([-5.29938638e-01,1.67231530e-01,6.29852653e-01,-5.32091379e-01,1.65847614e-01,6.30220890e-01])
max=np.array([-3.70376967e-02,6.39087200e-01,1.05172992e+00,-3.66398245e-02,6.39733434e-01,1.05185831e+00])
min=np.array([-3.50500733e-01,4.02689129e-01,7.35534191e-01,-3.50699395e-01,4.02544379e-01,7.35628963e-01])

#数据前三列是状态，后三列是动作

all_data=[]
train_data=[]
test_data=[]
sample_rate = 5

test_size = 0.1
train_size = 0.9

def create_inout_sequences(input_data, window_len):
    size = len(input_data)

    padding_list=[[0]*6 for _in in range(window_len)]
    # for i in range(0, window_len-1, sample_rate):#WINDOW-1个需要padding的数据
        # padding_list.pop(0)
        # padding_list.append(((input_data[i,[2, 3, 4, 9, 10, 11]] - min) / (max-min)).tolist())
        # all_data.append(np.array(padding_list))
        # print(padding_list)
    train=[]
    for i in range(0, size - window_len, sample_rate): #同时以10HZ进行采样
        if (i + window_len ) > size:
            break
        train = input_data[i:i + window_len,[2, 3, 4, 9, 10, 11]] #需要234、91011列的数据
        normalized_data = (train - min) / (max-min)
        all_data.append(normalized_data)

class BC(nn.Module):
    def __init__(self, hidden_dim):
        super(BC, self).__init__()
        self.state_dim = 3         
        self.action_dim = 3

        self.lstm = nn.LSTM(self.state_dim, hidden_dim, num_layers=1,batch_first=True)

        # self.linear1 = nn.Linear(self.state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, self.action_dim)
        self.loss_func = nn.MSELoss()

    def encoder(self, input):
        output, (hn, cn) = self.lstm(input)
        h1 = torch.relu(self.linear2(output[:, -1, :]))
        return self.linear3(h1)

    def forward(self, x):
        input = x[:, :,:self.state_dim] #batch_size*window*3
        a_target = x[:, -1,self.action_dim:] #动作是时间序列上最后一个动作
        # if a_target.shape[0] != 64:
            # print("target",a_target.shape) #!!!!!!!!
        a_predicted = self.encoder(input)
        loss = self.loss(a_predicted, a_target)
        return loss

    def loss(self, a_predicted, a_target):
        return self.loss_func(a_predicted, a_target)

def main():
    
    # Define learning parameters
    EPOCH = 200
    LR = 0.001
    LR_STEP_SIZE = 1000
    LR_GAMMA = 0.1
    n_models = 1
    BATCH_SIZE_TRAIN = 64
    WINDOW = 20

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    dir = "/home/yan/Imitation_learning/"
    for i in range(1,50): #共有50个文件
        data = []
        # print(f"Loading data from file {i}")
        file = open(dir +  "数据/12.4/"+str(i) +".txt", "r")
        for line in file:
            data.append((line.strip('\n').split(',')))
        file.close()
        data = data[1:]  # Remove the first row of data
        data = np.array(data).astype(float)
        create_inout_sequences(data, WINDOW)
    
    train_data = all_data[:int(len(all_data)*train_size)]
    test_data = all_data[int(len(all_data)*train_size):]
    print("Data loaded")
    print(f"Number of all data samples: {len(all_data)}")
    print(f"Number of train data samples: {len(train_data)}")
    print(f"Number of test data samples: {len(test_data)}")

    train_set = DataLoader(dataset=torch.FloatTensor(train_data), batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    test_set = DataLoader(dataset=torch.FloatTensor(test_data),shuffle=True)
    
    for n in range(n_models):
        print(f'[*] Training model {n+1}')
        model = BC(32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

        for epoch in range(EPOCH):
            model.train()
            for batch, x in enumerate(train_set):
                x = x.to(device) #batch_size*window*6
                optimizer.zero_grad()
                loss = model(x)
                loss.backward()
                optimizer.step()
            scheduler.step()
            if epoch % 10 == 0: #评估模型
                model.eval()
                test_loss = 0
                num_batches = 0
                with torch.no_grad():
                    for x in test_set:
                        x = x.to(device)
                        loss = model(x)
                        test_loss += loss
                        num_batches += 1
                test_loss /= num_batches
                print(f"Epoch {epoch} Test Loss: {test_loss.item()}")

        torch.save(model.state_dict(), os.path.join(dir, f"model{n+1}.pth"))

if __name__ == "__main__":
    main()
