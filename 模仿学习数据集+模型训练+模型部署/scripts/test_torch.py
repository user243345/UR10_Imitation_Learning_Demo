#!/home/robot/anaconda3/envs/mytorch/bin/python
import torch
import torchvision
import rospy
print(torch.__version__)
flag=torch.cuda.is_available()
print(flag)

