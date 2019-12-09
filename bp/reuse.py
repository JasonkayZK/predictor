# BP network reuse
# Version: 1.0
# Auther: ZK
# Last Updata: 2019.03.05

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Prepare the original data
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

# Normalize, set the data to 0~1
def Normalize_col(m):
    col_max = m.max(axis = 0)
    col_min = m.min(axis = 0)
    return (m - col_max) / (col_max - col_min)

sampling_rate = 1 / 8000
fft_size = 2 ** 16
t = np.arange(0, 100, sampling_rate)
find_basic_frame_length = 1

# 负载变化测试数据
# 数据读入内存
fileID = "/home/zk/workspace/paper_proj/bp/data/test_data/motor_normal_solved_24.txt"
ua, ub, uc, ia_harm, ib_harm, ic_harm = readRealData(fileID)

from FftUtils import FftUtils

ia_normal = FftUtils.mendBasicByFrame(ia_harm, t, find_basic_frame_length)
ib_normal = FftUtils.mendBasicByFrame(ib_harm, t, find_basic_frame_length)
ic_normal = FftUtils.mendBasicByFrame(ic_harm, t, find_basic_frame_length)

x_new_vals = []
y_new_vals = []

for i in range(len(ua)):
    x_new_vals.append([ua[i], ub[i], uc[i], ia_normal[i], ib_normal[i], ic_normal[i]])
    y_new_vals.append([ia_normal[i], ib_normal[i], ic_normal[i]])

# Prepare the original data
x_new_vals = np.array(x_new_vals)
y_new_vals = np.array(y_new_vals)

x_new_vals = np.nan_to_num(Normalize_col(x_new_vals))

#提取神经网络参数
saver = tf.train.import_meta_graph("/home/zk/workspace/paper_proj/bp/save/myNN.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint('/home/zk/workspace/paper_proj/bp/save/'))
    graph = tf.get_default_graph()
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    xs = graph.get_tensor_by_name("xs:0")
    prediction = graph.get_tensor_by_name("prediction/mul:0")
    result = sess.run(prediction, feed_dict={xs:x_new_vals,keep_prob:1})

plt.plot(result, label='predict')
plt.plot(y_new_vals, label='origin')
plt.legend()
plt.show()

result = np.array(result)
y_vals_test = np.array(y_new_vals)

result_a = result[:, 0]
y_vals_test_a = y_vals_test[:, 0]

FftUtils.plotFFT(np.array(result_a), labels='predict', colors='y')
FftUtils.plotFFT(np.array(y_vals_test_a), labels='origin', colors='r')
plt.legend()
plt.show()