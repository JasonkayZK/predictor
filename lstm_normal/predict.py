#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

import json
import numpy as np
import matplotlib.pyplot as plt
 
from LstmUtils import LstmUtils
from FftUtils import FftUtils

with open("config.json") as f:
    config = json.load(f)
    # 原始数据相关配置
    data_filepath = config["test"]["data_filepath"]
    model_name = config["test"]["model_name"]
    model_savepath = config["test"]["model_savepath"] + model_name

    # 源数据的元数据参数
    fft_size = config["data_meta"]["fft_size"]

    frame_size = config["to_data"]["to_train"]["frame_size"]

    sampling_rate = config["data_meta"]["sampling_rate"]
    time_length = config["data_meta"]["time_length"]
    t = np.arange(0, frame_size/sampling_rate, 1/sampling_rate)
    

    # 原始数据处理相关参数
    normalize_voltage_high = config["data_process"]["normalize_voltage_high"]
    normalize_voltage_low = config["data_process"]["normalize_voltage_low"]
    normalize_current_high = config["data_process"]["normalize_current_high"]
    normalize_current_low = config["data_process"]["normalize_current_low"]
    normalize_y_high = config["data_process"]["normalize_y_high"]
    normalize_y_low = config["data_process"]["normalize_y_low"]

    find_basic_frame_length = config["data_process"]["find_basic_frame_length"]
    n_steps = config["data_process"]["n_steps"]
    n_features = config["neuron_network"]["layers"][0]["input_dim"]

    test_savepath = config["test"]["test_savepath"]
    origin_savepath = test_savepath + config["test"]["origin_filename"]
    predict_savepath = test_savepath + config["test"]["predict_filename"]
    
# ############## 原始数据 ###################
ua_raw, ub_raw, uc_raw, ia_raw, ib_raw, ic_raw = LstmUtils.readRealData(data_filepath)

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
ua_harm_test = LstmUtils.normalizeByGiven(ua_harm_raw, normalize_voltage_high)
ub_harm_test = LstmUtils.normalizeByGiven(ub_harm_raw, normalize_voltage_high)
uc_harm_test = LstmUtils.normalizeByGiven(uc_harm_raw, normalize_voltage_high)

ia_basic_test = LstmUtils.normalizeByGiven(ia_basic_raw, normalize_current_high)
ib_basic_test = LstmUtils.normalizeByGiven(ib_basic_raw, normalize_current_high)
ic_basic_test = LstmUtils.normalizeByGiven(ic_basic_raw, normalize_current_high)

y_test = LstmUtils.normalizeByGiven(y_raw, normalize_y_high)

# ########## 4. 原始数据池化 ##########
test_ua = LstmUtils.split_sequence_x(ua_harm_test, n_steps)
test_ub = LstmUtils.split_sequence_x(ub_harm_test, n_steps)
test_uc = LstmUtils.split_sequence_x(uc_harm_test, n_steps)
test_ia = LstmUtils.split_sequence_x(ia_basic_test, n_steps)
test_ib = LstmUtils.split_sequence_x(ib_basic_test, n_steps)
test_ic = LstmUtils.split_sequence_x(ic_basic_test, n_steps)

# ########## 5. 构建LSTM神经网络输入 ##########
data_x = np.hstack((test_ua, test_ub, test_uc, test_ia, test_ib, test_ic))
data_x = data_x.reshape((data_x.shape[0], n_steps, n_features))  # 转换成（样本量，时间步，数据维度）

data_y = LstmUtils.split_sequence_y(y_test, n_steps)

print("-------------------------------构建LSTM神经网络输入完成-------------------------------")

############################################# Predict #############################################
import time
from keras.models import load_model

# ########## 1. 重装LSTM神经网络 ##########
model = load_model(model_savepath)

# ########## 2. 评估模型 ##########
predict_y = []
cur_time = time.time()
for i in range(len(data_x)):
    x_input = data_x[i].reshape(1, n_steps, n_features)
    predict_y.append(model.predict(x_input, verbose=0)[0][0])
    if i % 100 == 0:
        print("Percent: " + str(i / len(data_x) * 100) + "%, res: " + str((time.time() - cur_time) / (100 / len(data_x)) * (1 - i / len(data_x))) + "s")
        cur_time = time.time()

# 反归一化
data_y = LstmUtils.FNoramlize(data_y, high=normalize_y_high, low=normalize_y_low)
predict_y = LstmUtils.FNoramlize(predict_y, high=normalize_y_high, low=normalize_y_low)

# 保存预测数据
np.save(origin_savepath, data_y)
np.save(predict_savepath, predict_y)

# 时域波形
plt.plot(data_y, 'g', label='origin')
plt.plot(predict_y, 'r', label='predict')
plt.legend()
plt.show()

# 频域波形
predict_y = np.array(predict_y, dtype='float64')

FftUtils.plotFFT(ia_raw, sampling_rate=sampling_rate, fft_size=fft_size, labels='origin_normal', colors='y')
FftUtils.plotFFT(data_y, sampling_rate=sampling_rate, fft_size=fft_size, labels='origin_full', colors='r')
FftUtils.plotFFT(predict_y, sampling_rate=sampling_rate, fft_size=fft_size, labels='predict', colors='g')
plt.legend()
plt.show()
