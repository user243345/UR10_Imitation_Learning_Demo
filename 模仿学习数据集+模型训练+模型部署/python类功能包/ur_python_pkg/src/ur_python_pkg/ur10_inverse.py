#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from scipy.spatial.transform import Rotation as R

d =[0, 0.1273,0,0,0.163941,0.1157,0.0922]
a =[0,-0.612,-0.5723,0,0,0]
alpha= [1.570796, 0, 0, 1.570796, -1.570796, 0]
def quaternion_to_rotation_matrix(quat):
    x, y, z, w = quat
    rotation_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)]
    ])
    return rotation_matrix



def say_hello_world():
    print('Hello world!')
def quaternion2rot(quaternion):
    r = R.from_quat(quaternion)
    rot = r.as_matrix()  # 返回一个3x3的旋转矩阵
    return rot
def rot2quaternion(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    quaternion = r.as_quat()
    return quaternion
def ur10_forward(theta_input):
    """
    正运动学计算
    :param theta_input: 关节角度列表 (单位：弧度)
    :param alpha: 链节的 alpha 参数列表 (单位：弧度)
    :param a: 链节的 a 参数列表
    :param d: 链节的 d 参数列表
    :return: 末端位置 pos_x_y_z 和旋转矩阵
    """
    T = []  # 存储每个关节的齐次变换矩阵
    for i in range(6):
        T_i = np.eye(4)  # 初始化 4x4 单位矩阵
        T_i[0, 0] = np.cos(theta_input[i])
        T_i[0, 1] = -np.sin(theta_input[i]) * np.cos(alpha[i])
        T_i[0, 2] = np.sin(theta_input[i]) * np.sin(alpha[i])
        T_i[0, 3] = a[i] * np.cos(theta_input[i])

        T_i[1, 0] = np.sin(theta_input[i])
        T_i[1, 1] = np.cos(theta_input[i]) * np.cos(alpha[i])
        T_i[1, 2] = -np.cos(theta_input[i]) * np.sin(alpha[i])
        T_i[1, 3] = a[i] * np.sin(theta_input[i])

        T_i[2, 0] = 0
        T_i[2, 1] = np.sin(alpha[i])
        T_i[2, 2] = np.cos(alpha[i])
        T_i[2, 3] = d[i + 1]  # 注意 d 的索引偏移

        T_i[3, 0:4] = [0, 0, 0, 1]
        T.append(T_i)

    # 累乘计算 T06
    T06 = T[0]
    for i in range(1, 6):
        T06 = np.dot(T06, T[i])
    # 提取末端位姿
    pos_x_y_z = T06[0:3, 3]  # 提取位置
    rotation_matrix = T06[0:3, 0:3]  # 提取旋转矩阵
    quaternion = rot2quaternion(rotation_matrix)  # 转为四元数
    # 输出调试信息
    # print(f"Position: {pos_x_y_z}")
    # print(f"Quaternion: {quaternion}")
    return pos_x_y_z, quaternion

def ur10_solution_filter_test(current, q_solutions):
    """
    过滤逆解，选择距离当前关节位置最近的解。
    :param current: 当前关节位置 (长度为6的列表或数组)
    :param q_solutions: 8x6 矩阵，表示八组逆解
    :return: 最优解 6x1 numpy数组
    """
    def calculate_distance(current, solution):
        """计算两组关节位置之间的欧几里得距离。"""
        return np.sqrt(np.sum((np.array(current) - np.array(solution)) ** 2))
    
    min_distance = 2 * np.pi * 6  # 初始化为一个很大的距离
    optimal_solution = np.zeros(6)  # 最优解的占位数组

    # 遍历8个逆解
    for i in range(8):
        is_solution_valid = True
        solution = np.zeros(6)
        
        for j in range(6):
            # 归一化到 [-2π, 2π]
            if q_solutions[i, j] < -2 * np.pi:
                q_solutions[i, j] += 2 * np.pi
            if q_solutions[i, j] > 2 * np.pi:
                q_solutions[i, j] -= 2 * np.pi

            solution[j] = q_solutions[i, j]
        
        # 如果解有效，计算距离
        if is_solution_valid:
            distance = calculate_distance(current, solution)
            # 如果当前解比之前的更优，则更新最优解
            if distance < min_distance:
                min_distance = distance
                optimal_solution = solution

    return optimal_solution

def ur10_inverse(target_pos, current, quat):
    """
    输入目标位置和四元数 返回8组逆解。
    """
    x, y, z = target_pos
    T06 = np.zeros((4, 4))  # 末端到基坐标系的变换矩阵
    q_solutions = np.zeros((8, 6))  # 用于存储8组逆解
    theta = np.zeros((9, 7))  # 用于存储计算中间值
    
    # 将四元数转换为旋转矩阵
    #rotation_matrix = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
    rotation_matrix=quaternion2rot(quat)
    
    T06[:3, :3] = rotation_matrix
    T06[0, 3] = x
    T06[1, 3] = y
    T06[2, 3] = z
    T06[3, 3] = 1
    #print(T06)

    # 定义DH参数中的常量
    d = [0, 0.1273,0,0,0.163941,0.1157,0.0922]  # 示例值
    a = [0,-0.612,-0.5723,0,0,0]  # 示例值
    #print(d[6])
    # 计算theta1的两个解
    A = d[6] * T06[1, 2] - T06[1, 3]
    B = d[6] * T06[0, 2] - T06[0, 3]
    C = d[4]
    R = A**2 + B**2
    #print(A,B,C)
    # theta1第一个解，赋值到1到4组
    theta[1][1] = np.arctan2(A, B) - np.arctan2(C, np.sqrt(A**2 + B**2 - C**2))
    #print(A**2 + B**2 - C**2)
    for i in range(1, 5):
        theta[i][1] = theta[1][1]
    # theta1第二个解，赋值到5到8组
    theta[5][1] = np.arctan2(A, B) - np.arctan2(C, -np.sqrt(A**2 + B**2 - C**2))
    for i in range(5, 9):
        theta[i][1] = theta[5][1]

    # theta5的四个解
    for i in range(1, 5):
        A = np.sin(theta[i][1]) * T06[0, 2] - np.cos(theta[i][1]) * T06[1, 2]
        theta[i][5] = np.arccos(A)
        theta[i + 4][5] = -np.arccos(A)

    # theta6的解
    for i in range(1, 9, 2):
        A = np.sin(theta[i][1]) * T06[0, 0] - np.cos(theta[i][1]) * T06[1, 0]
        B = np.sin(theta[i][1]) * T06[0, 1] - np.cos(theta[i][1]) * T06[1, 1]
        C = np.sin(theta[i][5])
        if np.abs(C) > 1e-5:
            theta[i][6] = np.arctan2(A, B) - np.arctan2(C, 0)
            theta[i + 1][6] = theta[i][6]
        else:
            theta[i][6] = theta[i + 1][6] = 0

    # theta3的解
    for i in range(1, 9, 2):
        C = np.cos(theta[i][1]) * T06[0, 0] + np.sin(theta[i][1]) * T06[1, 0]
        D = np.cos(theta[i][1]) * T06[0, 1] + np.sin(theta[i][1]) * T06[1, 1]
        E = np.cos(theta[i][1]) * T06[0, 2] + np.sin(theta[i][1]) * T06[1, 2]
        F = np.cos(theta[i][1]) * T06[0, 3] + np.sin(theta[i][1]) * T06[1, 3]
        G = np.cos(theta[i][6]) * T06[2, 1] + np.sin(theta[i][6]) * T06[2, 0]
        A = d[5] * (np.sin(theta[i][6]) * C + np.cos(theta[i][6]) * D) - d[6] * E + F
        B = T06[2, 3] - d[1] - T06[2, 2] * d[6] + d[5] * G
        if A**2 + B**2 <= (a[1] + a[2])**2:
            theta[i][3] = np.arccos((A**2 + B**2 - a[1]**2 - a[2]**2) / (2 * a[1] * a[2]))
            theta[i + 1][3] = -theta[i][3]
        else:
            theta[i][3] = theta[i + 1][3] = 0

    # theta2和theta4
    for i in range(1, 9):
        C = np.cos(theta[i][1]) * T06[0, 0] + np.sin(theta[i][1]) * T06[1, 0]
        D = np.cos(theta[i][1]) * T06[0, 1] + np.sin(theta[i][1]) * T06[1, 1]
        E = np.cos(theta[i][1]) * T06[0, 2] + np.sin(theta[i][1]) * T06[1, 2]
        F = np.cos(theta[i][1]) * T06[0, 3] + np.sin(theta[i][1]) * T06[1, 3]
        G = np.cos(theta[i][6]) * T06[2, 1] + np.sin(theta[i][6]) * T06[2, 0]
        A = d[5] * (np.sin(theta[i][6]) * C + np.cos(theta[i][6]) * D) - d[6] * E + F
        B = T06[2, 3] - d[1] - T06[2, 2] * d[6] + d[5] * G
        M = ((a[2] * np.cos(theta[i][3]) + a[1]) * B - a[2] * np.sin(theta[i][3]) * A) / (
            a[1]**2 + a[2]**2 + 2 * a[1] * a[2] * np.cos(theta[i][3])
        )
        N = (A + a[2] * np.sin(theta[i][3]) * M) / (a[2] * np.cos(theta[i][3]) + a[1])
        theta[i][2] = np.arctan2(M, N)
        theta[i][4] = np.arctan2((-np.sin(theta[i][6]) * C - np.cos(theta[i][6]) * D), G) - theta[i][2] - theta[i][3]
    #print(theta)
    # 将角度规范化到[-2π, 2π]并存储
    for i in range(1, 9):
        for j in range(1, 7):
            if theta[i][j] > 2*np.pi:
                theta[i][j]=theta[i][j] - 2*np.pi
            if theta[i][j] < -2*np.pi:
                theta[i][j]=theta[i][j] + 2*np.pi
            q_solutions[i - 1, j - 1] = theta[i][j]
    #print(q_solutions)
    # 过滤解并返回
    q_filter = ur10_solution_filter_test(current, q_solutions)
    return q_filter







