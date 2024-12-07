# UR10_Imitation_Learning_Demo
This document contains a simple UR10 robot imitation learning example program, which implements the UR10 robot training an LSTM model using teleoperation-collected end-effector trajectories to autonomously draw circles, allowing the robot to start from different initial points and draw circles with a certain level of generalization.

数据集文件夹中包含一个圆形轨迹 共16条轨迹 每条轨迹用空行隔开 收集的数据有时间+期望三维位置+期望四元数+当前末端三维位置+当前四元数+当前关节角+当前夹爪角度

核心是scripts文件夹 其中的说明如下：
imitation_test.py 我的模型训练代码
imitation_ceshi.py 模型简单验证 输入一组数据观察输出
test_torch.py 测试torch是否导入
train_BC_circle.py 我师姐的模型训练代码
ur10_imitation_test.py ur10机械臂模仿学习模型部署+数据实时采集
ur10_trajectory.py 一个简单的控制ur10机械臂的移动demo
本文件夹中的模型文件表现略有过拟合现象 表现得太像训练集数据

模型文件中是训练得比较好的两个模型
