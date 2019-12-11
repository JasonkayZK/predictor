A repository for motor diagnosis via LSTM or some other ANN.

## 基于LSTM神经网络的自回归预测

### 零. 项目依赖

项目基于tensorflow与keras开发, 使用到的相关依赖以及版本如下

|         包名         |  版本   |
| :------------------: | :-----: |
|      tensorflow      | 2.0.0b1 |
|     tensorboard      |  0.4.0  |
|        numpy         | 1.17.1  |
|        scipy         |  1.3.3  |
|        keras         |  2.3.1  |
| keras-rectified-adam | 0.17.0  |

<br/>

### 一. 项目说明

项目目录结构:

```bash
zk@jasonkay:~/workspace/lstm_motor_diagnosis/lstm_normal$ tree
.
├── batch_train.py
├── config.json
├── data
│   ├── fault_data
│   │   └── motor_pianxin_solved_5.txt
│   ├── fault_output_data
│   ├── test_data
│   │   └── motor_normal_solved_24.txt
│   ├── test_output_data
│   ├── train_data
│   │   ├── motor_normal_flow_train.txt
│   │   ├── motor_normal_solved_31.txt
│   │   ├── motor_normal_solved_32.txt
│   │   └── motor_train_nSteps20_voltagehigh8_currenthigh5_yhigh0.1.txt
│   └── train_output_data
├── FftUtils.py
├── LstmUtils.py
├── predict.py
├── __pycache__
│   ├── FftUtils.cpython-36.pyc
│   └── LstmUtils.cpython-36.pyc
├── README.md
├── save
│   └── motor_train_nSteps20_voltagehigh8_currenthigh5_yhigh0.1_2file_lstm148_dropout0.3_lstm246_dropout0.2_lstm_216_dropout_0.2_lstm_164_lstm74_tanh_epoch6_batchsize2909_sgd_2file.h5
├── toFlowData.py
└── toTrainData.py
```

项目文件说明:

|    文件名称    |                         文件内容                         |
| :------------: | :------------------------------------------------------: |
| toFlowData.py  | 将原始数据进行处理, 转化为每行六个特征元素(转为流式数据) |
| toTrainData.py |    将流式数据转化为训练时的实际输入数据结构(每个一行)    |
|  config.json   |                    整个项目的配置文件                    |
|  FftUtils.py   |                 傅里叶变换相关的工具函数                 |
|  LstmUtils.py  |                LSTM神经网络相关的工具函数                |
| batch_train.py |                        批训练文件                        |
|   predict.py   |                        批预测文件                        |
|     save/      |                      模型保存文件夹                      |
|     data/      |                      数据保存文件夹                      |

><br/>
>
>**说明:**
>
>data目录的结构如下:
>
>```bash
>$ tree
>.
>├── fault_data
>│   └── ...
>├── fault_output_data
>│   └── ...
>├── test_data
>│   └── ...
>├── test_output_data
>│   └── ...
>├── train_data
>│   └── ...
>└── train_output_data
>```
>
>**对于fault, test, train分别采用了不同的目录存放数据, 防止数据过多而混乱**

<br/>

**① config.json文件说明**

一个项目经典的config.json文件如下所示:

```json
{
    "to_data": {
        "to_flow": {
            "file_list": [
                "data/train_data/motor_normal_solved_31.txt", 
                "data/train_data/motor_normal_solved_32.txt"
            ],
            "save_file": "data/train_data/motor_normal_flow_train.txt"
        },
        "to_train": {
            "source_file": "data/train_data/motor_normal_flow_train.txt",
            "save_file": "data/train_data/motor_train_nSteps20_voltagehigh8_currenthigh5_yhigh0.1.txt",
            
            "frame_size": 800000
        }
    },
    "data": {
        "data_filepath": "data/train_data/motor_train_nSteps20_voltagehigh8_currenthigh5_yhigh0.1.txt",
        "checkpoint_name": "",
        "checkpoint_path": "",
        "model_name": "motor_train_nSteps20_voltagehigh8_currenthigh5_yhigh0.1_2file_lstm148_dropout0.3_lstm246_dropout0.2_lstm_216_dropout_0.2_lstm_164_lstm74_tanh_epoch6_batchsize2909_sgd_2file.h5",
        "model_savepath": "save/"
    },
    "data_meta": {
        "sampling_rate": 8000,
        "fft_size": 65536,
        "time_length": 100
    },
    "data_process": {
        "normalize_voltage_high": 8,
        "normalize_voltage_low": -8,
        "normalize_current_high": 5,
        "normalize_current_low": -5,
        "normalize_y_high": 0.1,
        "normalize_y_low": -0.1,

        "find_basic_frame_length": 1,
        "data_use_rate": 0.8,

        "n_steps": 20
    },
    "neuron_network": {
        "layers": [
			{
				"type": "lstm",
				"neurons": 148,
				"input_timesteps": 20,
				"input_dim": 6,
				"return_seq": true
			},
			{
				"type": "dropout",
				"rate": 0.3
			},
			{
				"type": "lstm",
				"neurons": 246,
				"return_seq": true
			},
			{
		        "type": "dropout",
		        "rate": 0.2
			},
			{
			    "type": "lstm",
			    "neurons": 216,
			    "return_seq": true
			},
			{
			    "type": "dropout",
			    "rate": 0.2
			},
			{
				"type": "lstm",
				"neurons": 164,
				"return_seq": true
			},
			{
			    "type": "lstm",
			    "neurons": 74,
			    "return_seq": false
			},
			{
				"type": "dense",
				"neurons": 1,
                "activation": "tanh"
			}
        ],

        "loss_method": "mse",

        "optimize_method": "sgd"
    },
    "train_meta": {
        "epoch_num": 3,
        "batch_size": 2909,
        "data_lines": 799975,

        "callbacks": {
            "monitor_patience": 2
        }
    }, 

    "test": {
        "data_filepath": "data/test_data/motor_normal_solved_24.txt",
        "model_name": "motor_train_nSteps20_voltagehigh8_currenthigh5_yhigh0.1_2file_lstm148_dropout0.3_lstm246_dropout0.2_lstm_216_dropout_0.2_lstm_164_lstm74_tanh_epoch6_batchsize2909_sgd_2file.h5",
        "model_savepath": "/home/zk/workspace/lstm_motor_diagnosis/lstm_normal/save/",

        "test_savepath": "/home/zk/workspace/lstm_motor_diagnosis/lstm_normal/data/test_output_data/",
        "origin_filename": "origin_nStep25_voltageHigh8_currentHigh5_yHigh_0.05_3file_lstm148_dropout0.3_lstm246_dropout0.2_lstm_216_dropout_0.2_lstm_164_lstm74_tanh_epoch6_batchsize2909_4file_motor_normal_solved_57.npy",
        "predict_filename": "predict_nStep25_voltageHigh8_currentHigh5_yHigh_0.05_3file_lstm148_dropout0.3_lstm246_dropout0.2_lstm_216_dropout_0.2_lstm_164_lstm74_tanh_epoch6_batchsize2909_4file_motor_normal_solved_57.npy"
    },

    "selection": {
        "activation": ["relu", "softmax", "tanh"],
        "loss_method": ["mse", "mae", "mse", "mape"],
        "optimize_method": ["adam", "sgd", "Nadam", "rmsprop"]
    }
}

```

><br/>
>
>**说明:**
>
>**① `to_data`**: 主要是与toFlowData和toTrainData有关的配置;
>
>**② `data`**: 配置训练数据存放位置, 训练模型保存位置等;
>
>**③ `data_meta`**: 与数据处理相关的元数据, 如单个文件数据时长, 采样率等;
>
>**④ `data_process`**: 与数据处理相关的配置, 如数据归一标准, 基波提取滑窗长度, 池化长度等;
>
>**⑤ `neuron_network`**: 神经网络模型结构**(在此配置即可, 代码内部自动生成相应的神经网络)**
>
>**⑥ `train_meta`**: 神经网络训练相关的元数据, 如: 训练迭代次数, 每一批大小, callbacks等;
>
>**⑦ `test`**: 和测试相关的配置, 如: 模型的存放位置, 测试数据的位置, 测试预测结果的输出位置等;

<br/>

### 二. 如何使用

① 修改config.json, 配置需要格式化的数据

② 通过toFlowData和toTrainData等工具将源数据转为符合LSTM神经网络格式的数据(归一, 池化等);

③ 配置config.json中train_meta等配置, 并配置神经网络模型, 之后运行batch_train.py;

④ batch_train.py的训练模型被保存在config.json中声明的位置, 通过predict.py可以重载模型, 并进行预测;

