## 基于LSTM等神经网络的自回归时序预测

目前已发布版本:

|                           更新日志                           |    日期    |         更新内容          |
| :----------------------------------------------------------: | :--------: | :-----------------------: |
|    [bp](https://github.com/JasonkayZK/predictor/tree/bp)     | 2019-12-09 |     发布bp网络的实现      |
| [lstm_normal](https://github.com/JasonkayZK/predictor/tree/lstm_normal) | 2019-12-10 |  发布基本lstm网络的实现   |
| [lstm_radam_non_warmup](https://github.com/JasonkayZK/predictor/tree/lstm_radam_non_warmup) | 2019-12-11 |    发布lstm+RAdam版本     |
| [lstm_radam_warmup](https://github.com/JasonkayZK/predictor/tree/lstm_radam_warmup) | 2019-12-11 | 发布lstm+RAdam+warmup版本 |

### 项目特点

① 项目开发了FftUtils工具类, 提供大量与傅里叶变换相关的实用方法(滤波, 提取基波等);

② 项目提供了LstmUtils工具类, 提供构建LSTM等神经网络相关的数据处理, 神经网络构建相关的方法;

③ 整个项目的神经网络结构, 输入等均可通过config.json配置, 便于针对神经网络进行调参等操作;

