## 一个基于BP网络的自回归时序预测(Deprecated)

### 神经网络说明

项目仅仅基于tensorflow实现, 没有使用keras.

BP网络结构由多个隐藏层组成:

-   输入层: 6个输入
-   layer1: 25个神经元
-   layer2: 10个神经元
-   layer3: 5个神经元
-   output: 3个结果输出

整个训练过程迭代5000次, 最终保存训练结果, 并输出训练集结果.

<br/>

### 目录结构说明

项目中目录结构如下:

```bash
$ tree
.
├── BP.py
├── data
│   ├── fault_data
│   ├── fault_output
│   ├── test_data
│   ├── test_output_data
│   ├── train_data
│   └── train_output_data
├── FftUtils.py
├── reuse.py
└── save
    ├── checkpoint
    └── ...

```

项目中主要文件说明如下:

|  文件名称   |           文件说明           |
| :---------: | :--------------------------: |
|    BP.py    | 提供具体BP网络的实现, 训练等 |
| FftUtils.py |  与傅里叶变换相关的工具函数  |
|  reuse.py   | BP神经网络模型的重用, 预测等 |
|    data     | 存放train, test, fault等数据 |
|    save     |         模型保存目录         |

<br/>

### 使用方法

**训练:**

① 将待训练数据放入data/train_data/目录下;

② 修改BP.py中神经网络结构, 优化器类型, loss, 以及迭代次数等;

③ 运行BP.py

<br/>

**预测:**

① 将待预测数据放入data/test_data/目录下;

② 运行reuse.py, 可查看输出结果

>   <br/>
>
>   **注:**
>
>   **代码中, 包括readRealData()等方法, 用于解析并处理源数据, 使用时根据需求修改**

<br/>