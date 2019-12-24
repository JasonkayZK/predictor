#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

import json
import numpy as np
import matplotlib.pyplot as plt
 
from FftUtils import FftUtils

with open("plot_config.json") as f:
    config = json.load(f)

    sampling_rate = config["meta"]["sampling_rate"]
    fft_size = config["meta"]["fft_size"]

    origin_savepath = r"E:\\lstm_motor_diagnosis\\bp\data\\train_output_data\\y_vals_test.txt"
    predict_savepath = r"E:\\lstm_motor_diagnosis\\bp\data\\train_output_data\\result.txt"

data_y = np.loadtxt(origin_savepath)[:, 0:1]
predict_y = np.loadtxt(predict_savepath)[:, 0:1]
data_y = data_y.flatten()
predict_y = predict_y.flatten()

basic_data_y = FftUtils.mendBasicByFrame(data_y, time=np.arange(0, 20, 1/8000), windowWidth=1, sampling_rate=1.0/8000)
basic_predict_y = FftUtils.mendBasicByFrame(predict_y, time=np.arange(0, 20, 1/8000), windowWidth=1, sampling_rate=1.0/8000)

# 时域波形
# plt.plot(data_y, 'r*-', label='origin')
# plt.plot(predict_y, 'g.-.', label='predict')
# plt.legend(loc="best")
# plt.show()

# 频域波形
FftUtils.plotFFT(data_y - basic_data_y, sampling_rate=1/sampling_rate, fft_size=fft_size, labels='origin', colors='-')
FftUtils.plotFFT(predict_y - basic_predict_y, sampling_rate=1/sampling_rate, fft_size=fft_size, labels='predict', colors=':')
plt.legend(loc="best")
plt.show()
