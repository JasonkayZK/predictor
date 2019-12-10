import os

import json
import numpy as np 

from LstmUtils import LstmUtils
from FftUtils import FftUtils

def process_line(line):
    res = [float(val) for val in line.strip().split(', ')]
    return res[0], res[1], res[2], res[3], res[4], res[5]

def process_data(ua_raw, ub_raw, uc_raw, ia_raw, ib_raw, ic_raw, 
    t, n_steps,
    normalize_voltage_high, normalize_voltage_low,
    normalize_current_high, normalize_current_low,
    normalize_y_high, normalize_y_low,
    find_basic_frame_length):
    
    # ############## 1. 窗口提取基波法 ############
    ua_basic_raw = FftUtils.mendBasicByFrame(ua_raw, t, find_basic_frame_length)
    ub_basic_raw = FftUtils.mendBasicByFrame(ub_raw, t, find_basic_frame_length)
    uc_basic_raw = FftUtils.mendBasicByFrame(uc_raw, t, find_basic_frame_length)

    ia_basic_raw = FftUtils.mendBasicByFrame(ia_raw, t, find_basic_frame_length)
    ib_basic_raw = FftUtils.mendBasicByFrame(ib_raw, t, find_basic_frame_length)
    ic_basic_raw = FftUtils.mendBasicByFrame(ic_raw, t, find_basic_frame_length)

    # ##############  2. 构建输入原始数据: 电压谐波, 电流基波; 输出量: 电流谐波 ############
    # 输入量
    # 电压量
    ua_harm_raw = ua_raw - ua_basic_raw
    ub_harm_raw = ub_raw - ub_basic_raw
    uc_harm_raw = uc_raw - uc_basic_raw
    # 电流量(已有)

    # 输出量
    y_raw = ia_raw - ia_basic_raw

    # ##############  3. 归一化数据 #############
    ua_harm_train = LstmUtils.normalizeByGiven(ua_harm_raw, normalize_voltage_high)
    ub_harm_train = LstmUtils.normalizeByGiven(ub_harm_raw, normalize_voltage_high)
    uc_harm_train = LstmUtils.normalizeByGiven(uc_harm_raw, normalize_voltage_high)

    ia_basic_train = LstmUtils.normalizeByGiven(ia_basic_raw, normalize_current_high)
    ib_basic_train = LstmUtils.normalizeByGiven(ib_basic_raw, normalize_current_high)
    ic_basic_train = LstmUtils.normalizeByGiven(ic_basic_raw, normalize_current_high)

    y_train = LstmUtils.normalizeByGiven(y_raw, normalize_y_high)

    # ########## 4. 原始数据池化 ##########
    train_ua = LstmUtils.split_sequence_x(ua_harm_train, n_steps)
    train_ub = LstmUtils.split_sequence_x(ub_harm_train, n_steps)
    train_uc = LstmUtils.split_sequence_x(uc_harm_train, n_steps)
    train_ia = LstmUtils.split_sequence_x(ia_basic_train, n_steps)
    train_ib = LstmUtils.split_sequence_x(ib_basic_train, n_steps)
    train_ic = LstmUtils.split_sequence_x(ic_basic_train, n_steps)

    # ########## 5. 构建LSTM神经网络输入 ##########
    data_x = np.hstack((train_ua, train_ub, train_uc, train_ia, train_ib, train_ic))
    # data_x = data_x.reshape((data_x.shape[0], n_steps, 6))  # 转换成（样本量，时间步，数据维度）

    data_y = LstmUtils.split_sequence_y(y_train, n_steps)

    return np.array(data_x), data_y

with open("config.json") as f:
    config = json.load(f)

    # 配置数据
    n_steps = config["data_process"]["n_steps"]
    normalize_voltage_high = config["data_process"]["normalize_voltage_high"]
    normalize_voltage_low = config["data_process"]["normalize_voltage_low"]
    normalize_current_high = config["data_process"]["normalize_current_high"]
    normalize_current_low = config["data_process"]["normalize_current_low"]
    normalize_y_high = config["data_process"]["normalize_y_high"]
    normalize_y_low = config["data_process"]["normalize_y_low"]

    find_basic_frame_length = config["data_process"]["find_basic_frame_length"]

    source_file = config["to_data"]["to_train"]["source_file"]
    save_file = config["to_data"]["to_train"]["save_file"]

    frame_size = config["to_data"]["to_train"]["frame_size"]

    sampling_rate = config["data_meta"]["sampling_rate"]
    t = np.arange(0, frame_size/sampling_rate, 1/sampling_rate)

source = None
save = None
try: 
    source = open(source_file, mode='r', encoding='utf-8')
    save = open(save_file, mode='a', encoding='utf-8')

    ua = []
    ub = []
    uc = []
    ia = []
    ib = []
    ic = []

    for line in source:
        # 文件未到一帧, 继续填充(若文件不足, 则填充未满时也会自动退出!)
        if len(ua) < frame_size:
            ua_line, ub_line, uc_line, ia_line, ib_line, ic_line = process_line(line)
            ua.append(ua_line)
            ub.append(ub_line)
            uc.append(uc_line)
            ia.append(ia_line)
            ib.append(ib_line)
            ic.append(ic_line)
            continue

        # 填充够了, 处理一个帧的数据
        else:
            # 生成一帧的结果, 写入文件
            x, y = process_data(ua, ub, uc, ia, ib, ic, t=t, n_steps=n_steps, 
                normalize_voltage_high=normalize_voltage_high, normalize_voltage_low=normalize_voltage_low,
                normalize_current_high=normalize_current_high, normalize_current_low=normalize_current_low,
                normalize_y_high=normalize_y_high, normalize_y_low=normalize_y_low,
                find_basic_frame_length=find_basic_frame_length)

            print("-------------------------- 一帧处理中 --------------------------")
            for i in range(len(x)):
                save.write(str(x[i].tolist()) + "; " + str(y[i]) + '\n')
                print("已完成: " + str(i * 100 / len(x)) + "%")
            
            print("-------------------------- 处理了一帧数据 --------------------------")

            # 处理结束后, 清空本帧数据
            ua.clear()
            ub.clear()
            uc.clear()
            ia.clear()
            ib.clear()
            ic.clear()

except Exception as e:
    print(e)

finally:
    if save is not None:
        save.close()
    if source is not None:
        source.close()


# import numpy as np

# def str_to_list(aim_str):
#     return np.array([float(val) for val in aim_str.strip().split(', ')])

# with open("/home/zk/workspace/NN_Python/paper_proj/process_data/motor_train_nSteps25_voltagehigh5_currenthigh3_yhigh0.05.txt") as f:
#     for line in f:
#         x, y = line.strip().split("; ")
#         x = str_to_list(x[1:-1])
#         print(np.shape(x))
#         print(y)





