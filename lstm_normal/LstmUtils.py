#!/usr/bin/python
# -*- coding: utf-8 -*-

import random

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

class LstmUtils(object):
    """
        封装一些用于Lstm常用的方法
    """

    @staticmethod
    def readFakeData(fileId):
        """Extract the data from the txt-type file

        Warnnng: the data in the file should be placed like this: 2..3..4 (two blank is needed)

        Args: fileId: the path of the file, Type: str

        return some list include the data"""
        with open(fileId) as file:
            var1 = []
            var2 = []
            var3 = []
            while(1):
                line = file.readline()
                if not line:
                    if var3 != []:
                        return var1, var2, var3
                    elif var2 != []:
                        return var1, var2
                    elif var1 != []:
                        return var1
                    else: 
                        return []        
                line = line.strip()
                str_list = line.split("  ")
                if len(str_list) == 3:
                    var1.append(float(str_list[0]))
                    var2.append(float(str_list[1]))
                    var3.append(float(str_list[2]))
                elif len(str_list) == 2:
                    var1.append(float(str_list[0]))
                    var2.append(float(str_list[1]))
                elif len(str_list) == 1:
                    var1.append(float(str_list[0]))
                else:
                    break
                if not line:
                    if len(str_list) == 3:
                        return var1, var2, var3
                    elif len(str_list) == 2:
                        return var1, var2
                    elif len(str_list) == 1:
                        return var1
                    else: 
                        return []

    @staticmethod
    def readRealData(fileId):
        """
            从txt文件中读取数据

            @param: fileId: 文件路径

            @return: [ua, ub, uc, ia, ib, ic]
        """
        f = None
        ua = []
        ub = []
        uc = []
        ia = []
        ib = []
        ic = []
        try:
            f = open(fileId, mode='r', encoding='utf-8')
            ua = f.readline().strip().split(', ')
            ua = [float(x) for x in ua]
            ub = f.readline().strip().split(', ')
            ub = [float(x) for x in ub]
            uc = f.readline().strip().split(', ')
            uc = [float(x) for x in uc]
            ia = f.readline().strip().split(', ')
            ia = [float(x) for x in ia]
            ib = f.readline().strip().split(', ')
            ib = [float(x) for x in ib]
            ic = f.readline().strip().split(', ')
            ic = [float(x) for x in ic]
        finally:
            if f is not None:
                f.close()        
        return ua[0:800000], ub[0:800000], uc[0:800000], ia[0:800000], ib[0:800000], ic[0:800000]

    @staticmethod
    def generate_arrays_from_file(filename, batch_size, n_steps, n_features = 6):   
        while 1:
            source = open(filename)

            cnt = 0
            x = []
            y = []
            for line in source:
                raw_x, raw_y = line.strip().split("; ")
                x.append(np.reshape(LstmUtils.str_to_list(raw_x[1:-1]), [n_steps, n_features]))
                y.append(float(raw_y))
                
                cnt += 1
                if cnt >= batch_size:
                    cnt = 0
                    yield (np.array(x), np.array(y))
                    x.clear()
                    y.clear()
                    
            source.close()

    @staticmethod
    def str_to_list(aim_str):
        return np.array([float(val) for val in aim_str.strip().split(', ')])

    @staticmethod
    def split_set(raw_data, percent):
        """
            将原始的一维数据按照percent拆分为trainset 和 testset

            return: raw_train_set, raw_test_set
        """
        index = int(len(raw_data) * percent)
        return raw_data[0:index], raw_data[index:]

    @staticmethod
    def split_sequence_x(sequence, n_steps):
        X = list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence)-1:
                break
            seq_x = sequence[i:end_ix]
            X.append(seq_x)
        return X

    @staticmethod
    def split_sequence_y(sequence, n_steps):
        y = list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence)-1:
                break
            seq_y = sequence[end_ix]
            y.append(seq_y)
        return np.array(y)

    @staticmethod
    def normalizeByGiven(raw_list, high, low = None):
        """
            根据给定的最大最小值对数据进行归一化处理

            Args:
                list: 所要归一化的数据
                high: 归一的最大值
                low:  归一的最小值 (默认low等于-high)
        """
        if low is None:
            low = -high
        delta = high - low
        if delta != 0:
            for i in range(len(raw_list)):
                raw_list[i] = (raw_list[i] - low) / delta   
        return raw_list
    
    @staticmethod
    def Normalize(raw_list):
        """
            单维最大最小归一化函数
            
            @param: list 需要归一化的数组数据

            @return
                list: 归一化之后的数据
                low: 归一化用到的最小值
                high: 归一化用到的最大值
        """
        raw_list = np.array(raw_list)
        low, high = np.percentile(raw_list, [0, 100])
        delta = high - low
        if delta != 0:
            for i in range(0, len(raw_list)):
                raw_list[i] = (raw_list[i]-low)/delta
        return  raw_list,low,high

    @staticmethod
    def FNoramlize(inputList,low,high):
        """
            单维最大最小反归一化函数

            @param:
                list: 归一化之后的数据
                low: 归一化用到的最小值
                high: 归一化用到的最大值
            
            @return: list 反归一化之后的值
        """
        res = []
        delta = high - low
        if delta != 0:
            for i in range(len(inputList)):
                res.append(inputList[i]*delta + low)
        return res

    @staticmethod
    def doReshape(raw_tuple):
        """
            将多维的list重新组合, 要求tuple中的列表长度严格相同

            @param: raw_tuple

                a = [1,2,3,4,5,6]
                b = [10,20,30,40,50, 60]
                c = [100,200,300,400,500,600]
                d = [1000,2000,3000,4000,5000,6000]

            @return: 
                a list

                [1, 10, 100, 1000, 2, 20, 200, 2000, 3, 30, 300, 3000, 4, 40, 400, 4000, 5, 50, 500, 5000, 6, 60, 600, 6000]
        """
        res = []
        for i in range(len(raw_tuple[0])):
            for j in range(len(raw_tuple)):
                res.append(raw_tuple[j][i])
        return res

    @staticmethod
    def build_model(configs):
        """
            根据config中的内容构建神经网络

            @Args:
                configs: 读取的神经网络的配置

            @Return:
                一个Sequential()实例

        """
        model = Sequential()

        for layer in configs["neuron_network"]["layers"]:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                model.add(LSTM(neurons, input_shape=(
                    input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                model.add(Dropout(dropout_rate))            

        model.compile(loss=configs['neuron_network']['loss_method'], optimizer=configs['neuron_network']['optimize_method'])

        return model


if __name__ == '__main__':
    fileID = "E:\\Spyder_python\\paper_proj\\data\\voltage_harm_Train.txt"
    ua, ub, uc = LstmUtils.readFakeData(fileID)
    fileID = "E:\\Spyder_python\\paper_proj\\data\\normal_current_Train.txt"
    ia_normal, ib_normal, ic_normal = LstmUtils.readFakeData(fileID)
    fileID = "E:\\Spyder_python\\paper_proj\\data\\current_harm_Train.txt"
    ia_harm, ib_harm, ic_harm = LstmUtils.readFakeData(fileID)

    from FftUtils import FftUtils
    FftUtils.plotFFT(ua, labels='ua', sampling_rate = 1000, fft_size = 2 ** 16)

    plt.show()


