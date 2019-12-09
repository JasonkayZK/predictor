# BP network for simul the motor data
# Version: 1.0
# Auther: ZK
# Last Updata: 2019.03.05

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

from FftUtils import FftUtils

sampling_rate = 1 / 8000
fft_size = 2 ** 16
t = np.arange(0, 100, sampling_rate)
find_basic_frame_length = 1

# 数据读入内存
fileID = "/home/zk/workspace/paper_proj/bp/data/train_data/motor_normal_solved_31.txt"
ua, ub, uc, ia_harm, ib_harm, ic_harm = readRealData(fileID)

ia_normal = FftUtils.mendBasicByFrame(ia_harm, t, find_basic_frame_length)
ib_normal = FftUtils.mendBasicByFrame(ib_harm, t, find_basic_frame_length)
ic_normal = FftUtils.mendBasicByFrame(ic_harm, t, find_basic_frame_length)


x_vals = []
y_vals = []
for i in range(len(ua)):
    x_vals.append([ua[i], ub[i], uc[i], ia_normal[i], ib_normal[i], ic_normal[i]])
    y_vals.append([ia_harm[i], ib_harm[i], ic_harm[i]])

# Prepare the original data
x_vals = np.array(x_vals)
y_vals = np.array(y_vals)

# 1. 随机分配训练集方法
# 将数据随机分割为Train Set: 80%,  Test Set: 20%
# train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
# test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

# 2. 不随机分配训练集方法
train_indices = np.array(list(range(640000)))
test_indices = np.array(list(range(640000, 800000)))
# print(train_indices)
# print(test_indices)

# print(len(train_indices), len(test_indices))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Normalize, set the data to 0~1
def Normalize_col(m):
    col_max = m.max(axis = 0)
    col_min = m.min(axis = 0)
    return (m - col_max) / (col_max - col_min)

x_vals_train = np.nan_to_num(Normalize_col(x_vals_train))
x_vals_test = np.nan_to_num(Normalize_col(x_vals_test))

# 2.定义节点准备接收数据 
xs = tf.placeholder(tf.float32, [None,6], name='xs') 
ys = tf.placeholder(tf.float32, [None,3], name='ys')
keep_prob = tf.placeholder(tf.float32,name="keep_prob")

# 3.定义神经层：隐藏层和预测层 
# add hidden layer1 输入值是 xs，隐藏层有 25 个神经元 
Weights1 = tf.Variable(tf.random_normal([6, 25]), name='W1')
biases1 = tf.Variable(tf.zeros([25]) + 0.1, name='b1') 
Wx_plus_b1 = tf.add(tf.matmul(xs, Weights1), biases1)
Wx_plus_b11 = tf.nn.dropout(Wx_plus_b1, keep_prob)
l1 = tf.nn.tanh(Wx_plus_b11)

# add hidden layer2 输入值是隐藏层 l1，输出 10个结果
Weights2 = tf.Variable(tf.random_normal([25, 10]), name='W2')
biases2 = tf.Variable(tf.zeros([10]) + 0.1, name='b2') 
Wx_plus_b2 = tf.add(tf.matmul(l1, Weights2), biases2)
Wx_plus_b22 = tf.nn.dropout(Wx_plus_b2, keep_prob)
l2 = tf.nn.tanh(Wx_plus_b22)

# add hidden layer3 输入层是隐藏层 l2，输出 5个结果
Weights3 = tf.Variable(tf.random_normal([10, 5]), name='W3')
biases3 = tf.Variable(tf.zeros([5]) + 0.05, name='b3')
Wx_plus_b3 = tf.add(tf.matmul(l2, Weights3), biases3)
Wx_plus_b33 = tf.nn.dropout(Wx_plus_b3, keep_prob)
l3 = tf.nn.tanh(Wx_plus_b33)

# add output layer 输入值是隐藏层 l3，在预测层输出 3 个结果 
Weights4 = tf.Variable(tf.random_normal([5, 3]), name='W4')
biases4 = tf.Variable(tf.zeros([3]) + 0.05, name='b4')
Wx_plus_b4 = tf.add(tf.matmul(l3, Weights4), biases4)
prediction = tf.nn.dropout(Wx_plus_b4, keep_prob,name="prediction")

# 4.定义 loss 表达式 
with tf.name_scope('loss'):
    loss =  tf.reduce_mean(tf.square(y_vals_train - prediction))

# 5.选择 optimizer 使 loss 达到最小 
# 这一行定义了用什么方式去减少 loss，学习率是 0.005
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.005).minimize(loss) 

saver = tf.train.Saver()
with tf.Session() as sess: 
    init = tf.global_variables_initializer()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
    sess.run(init) 
    # 迭代 5000 次学习，sess.run optimizer 
    for i in range(5000): 
    # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数 
        sess.run(train_step, feed_dict={xs: x_vals_train, ys: y_vals_train,keep_prob:0.95})
        if i % 10 == 0:
            print("Generation: " + str(i) + '  loss: ' + str(sess.run(loss, feed_dict={xs: x_vals_train, ys: y_vals_train, keep_prob:1})))
    result1= sess.run(prediction, feed_dict={xs: x_vals_train, ys:y_vals_train,keep_prob:1})
    print(sess.run(loss, feed_dict={xs: x_vals_train, ys: y_vals_train,keep_prob:0.95}))
    saver.save(sess, "/home/zk/workspace/paper_proj/bp/save/myNN.ckpt")


#提取神经网络参数
saver = tf.train.import_meta_graph("/home/zk/workspace/paper_proj/bp/save/myNN.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint('/home/zk/workspace/paper_proj/bp/save'))
    graph = tf.get_default_graph()
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    xs = graph.get_tensor_by_name("xs:0")
    prediction = graph.get_tensor_by_name("prediction/mul:0")
    result = sess.run(prediction, feed_dict={xs:x_vals_test,keep_prob:1})

plt.plot(result)
plt.plot(y_vals_test)
np.savetxt('/home/zk/workspace/paper_proj/bp/data/train_output_data/result.txt', result)
np.savetxt('/home/zk/workspace/paper_proj/bp/data/train_output_data/y_vals_test.txt', y_vals_test)
plt.show()


result = np.array(result)
y_vals_test = np.array(y_vals_test)

result_a = result[:, 0]
y_vals_test_a = y_vals_test[:, 0]

FftUtils.plotFFT(np.array(result_a), labels='result', colors='y')
FftUtils.plotFFT(np.array(y_vals_test_a), labels='y_vals_test', colors='r')
plt.legend()
plt.show()


