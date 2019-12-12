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

    prefix = config["file"]["savepath_prefix"]
    origin_savepath = prefix + "/" + config["file"]["origin_filename"]
    predict_savepath = prefix + "/" + config["file"]["predict_filename"]

data_y = np.load(origin_savepath)
predict_y = np.load(predict_savepath)

# 时域波形
plt.plot(data_y, 'g', label='origin')
plt.plot(predict_y, 'r', label='predict')
plt.legend(loc="best")
plt.show()

# 频域波形
predict_y = np.array(predict_y, dtype='float64')

FftUtils.plotFFT(data_y, sampling_rate=sampling_rate, fft_size=fft_size, labels='origin', colors='r')
FftUtils.plotFFT(predict_y, sampling_rate=sampling_rate, fft_size=fft_size, labels='predict', colors='g')
plt.legend(loc="best")
plt.show()
