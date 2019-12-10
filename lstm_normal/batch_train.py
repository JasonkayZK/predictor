#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

import json
import numpy as np

from LstmUtils import LstmUtils
from FftUtils import FftUtils
from tensorflow.keras.callbacks import TensorBoard

print(" -------------- Config载入 -------------- ")

with open("config.json") as f:
    config = json.load(f)

    # 训练数据存放位置
    data_filepath = config["data"]["data_filepath"]

    checkpoint_name = config["data"]["checkpoint_name"]
    checkpoint_savepath = config["data"]["checkpoint_path"] + checkpoint_name

    model_name = config["data"]["model_name"]
    model_savepath = config["data"]["model_savepath"] + model_name

    n_steps = config["data_process"]["n_steps"]

    # 训练相关参数
    epoch_num = config["train_meta"]["epoch_num"]
    batch_size = config["train_meta"]["batch_size"]
    data_lines = config["train_meta"]["data_lines"]

    # 训练回调相关参数
    # 1. 提前结束轮数
    # monitor_patience = config["train_meta"]["callbacks"]["monitor_patience"]

    # 2. Tensorboard
    # tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

print(model_savepath)

print(" -------------- Config 载入完成 -------------- ")

# 根据model创建模型
model = LstmUtils.build_model(config)

print(" -------------- Model 构建完成 -------------- ")

############################################# Train #############################################
model.fit_generator(
    LstmUtils.generate_arrays_from_file(data_filepath, batch_size=batch_size, n_steps=n_steps),
    epochs = epoch_num,
    verbose=1,
    steps_per_epoch = int(np.floor(data_lines / batch_size)),
    validation_data=None,
    validation_steps=None,
    class_weight=None,
    max_queue_size=2,
    workers=1,
    use_multiprocessing=True,
    shuffle=True,
    initial_epoch=0,
    callbacks=None)

print(" -------------- Model 训练完成 -------------- ")

model.save(model_savepath)

print(" -------------- Model 已保存 -------------- ")

# import keras.utils.data_utils.Sequence as Sequence

# class My_Generator(Sequence):

#     def __init__(self, filename, batch_size):
#         self.filename = filename
#         self.batch_size = batch_size
     
#     def __len__(self):
#         return 

#     def __getitem__(self, index):
#         """
#         Args:
#             x_train: 输入文件路径列表
#             y_train: 标签文件路径列表
#             batch_size: 每一批的大小
#             capacity: 最大的容量
        
#         Returns:
#             x_train_batch:
#             y_train_batch:
#         """