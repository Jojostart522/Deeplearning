# 深度学习ANC实现说明

## 概述

本文件(`dl_anc_model.py`)实现了使用深度学习方法替代维纳滤波器的主动噪声控制(ANC)系统。目标是达到或超过维纳滤波器的降噪效果（7.65 dBA）。

## 安装依赖

```bash
pip install -r requirements_dl.txt
```

或者手动安装：
```bash
pip install numpy scipy matplotlib torch torchvision
```

## 快速开始

```bash
python dl_anc_model.py
```

## 模型架构

### 1. CNN-LSTM混合模型（默认）

**特点**：
- 使用1D卷积提取局部特征
- 使用LSTM捕捉时序依赖关系
- 适合处理时序信号

**架构**：
```
输入 (18通道 × 300样本点)
  ↓
1D Conv (64 filters) + BatchNorm + ReLU + MaxPool
  ↓
1D Conv (128 filters) + BatchNorm + ReLU + MaxPool
  ↓
1D Conv (64 filters) + BatchNorm + ReLU + MaxPool
  ↓
LSTM (2层, hidden_size=128)
  ↓
全连接层 (64 → 1)
  ↓
输出 (单个预测值)
```

### 2. TCN模型（可选）

**特点**：
- 时间卷积网络，使用因果卷积和膨胀卷积
- 感受野大，能捕捉长距离依赖
- 训练速度快

要使用TCN模型，在代码中取消注释相应部分。

## 关键参数

### 数据参数
- `window_size = 300`: 滑动窗口大小（与维纳滤波器长度一致）
- `stride = 10`: 滑动步长（越小数据量越大，训练越慢）
- `train_split = 0.8`: 训练集比例

### 训练参数
- `batch_size = 64`: 批次大小
- `num_epochs = 100`: 训练轮数
- `learning_rate = 0.001`: 学习率
- `dropout = 0.3`: Dropout比例

## 工作流程

1. **数据加载**：读取32通道传感器数据
2. **数据预处理**：
   - 提取18个加速度传感器作为输入
   - 提取1个麦克风信号作为目标
   - 数据归一化（零均值，单位方差）
3. **数据集创建**：使用滑动窗口创建训练样本
4. **模型训练**：
   - 使用MSE损失函数
   - Adam优化器
   - 学习率自适应调整
5. **预测**：在完整信号上进行预测
6. **评估**：
   - 计算误差信号
   - 应用A计权
   - 计算dBA降噪量
   - 与维纳滤波对比

## 评估指标

与维纳滤波器使用相同的评估方法：

1. **BeforeANC**: 原始麦克风信号的dBA值
2. **AfterANC**: 误差信号（原始 - 估计）的dBA值
3. **ANC降噪量**: `10 * log10(RMS_error^2 / RMS_target^2)`

**目标**：ANC降噪量 ≤ -7.65 dBA（负值表示降噪）

## 输出文件

运行后会在`results/`目录生成：

1. **dl_training_curve.png**: 训练和验证损失曲线
2. **dl_anc_result.png**: ANC结果对比图
   - 时域波形对比
   - 误差信号
   - PSD频域对比
3. **dl_anc_model.pth**: 训练好的模型及归一化参数

## 性能优化建议

### 如果性能不达标：

1. **增加训练数据**：
   - 减小`stride`（例如从10改为5或1）
   - 使用数据增强（添加噪声、时间拉伸等）

2. **调整模型架构**：
   - 增加网络深度或宽度
   - 尝试不同的模型（TCN、Transformer等）
   - 调整LSTM的hidden_size

3. **优化训练过程**：
   - 增加训练轮数
   - 调整学习率
   - 使用不同的优化器（AdamW、SGD等）
   - 调整正则化（dropout、weight_decay）

4. **特征工程**：
   - 添加频域特征
   - 使用小波变换
   - 添加相位信息

5. **损失函数**：
   - 尝试频域损失
   - 使用感知损失
   - 多任务学习

## 与维纳滤波的对比

| 特性 | 维纳滤波 | 深度学习 |
|------|---------|---------|
| 理论基础 | 最优线性滤波器 | 非线性函数逼近 |
| 训练时间 | 快（直接求解） | 慢（迭代优化） |
| 推理时间 | 快 | 中等 |
| 数据需求 | 少 | 多 |
| 非线性建模 | 不支持 | 支持 |
| 可解释性 | 高 | 低 |
| 泛化能力 | 依赖数据统计特性 | 可能更好 |

## 使用训练好的模型

```python
import torch
import numpy as np

# 加载模型
checkpoint = torch.load('results/dl_anc_model.pth')
model = ANCModel_CNN_LSTM(num_channels=18, window_size=300)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 加载归一化参数
input_mean = checkpoint['input_mean']
input_std = checkpoint['input_std']
target_mean = checkpoint['target_mean']
target_std = checkpoint['target_std']

# 预测新数据
# new_input: (18, num_samples)
new_input_norm = (new_input - input_mean) / input_std
predictions_norm = predict_full_signal(model, new_input_norm, window_size=300)
predictions = predictions_norm * target_std + target_mean
```

## 故障排除

### GPU内存不足
- 减小`batch_size`
- 减小模型大小

### 训练不收敛
- 降低学习率
- 检查数据归一化
- 增加训练轮数

### 过拟合
- 增加dropout
- 增加weight_decay
- 使用更多训练数据
- 使用数据增强

### 性能不如维纳滤波
- 这是正常的，因为数据量较少
- 尝试上述"性能优化建议"
- 考虑使用更多数据或数据增强

## 进阶：多文件训练

如果有多个数据文件，可以修改代码以支持：

```python
# 加载多个文件
data_files = ['data/file1.dat', 'data/file2.dat', 'data/file3.dat']
all_inputs = []
all_targets = []

for file in data_files:
    data = load_data(file)
    all_inputs.append(data[acc_indices, :])
    all_targets.append(data[mic_index, :])

# 拼接所有数据
input_signals = np.concatenate(all_inputs, axis=1)
target_signal = np.concatenate(all_targets)
```

## 参考

- 原始维纳滤波实现：`main_analysis.py`
- MATLAB实现：`Wiener/miso_firwiener2.m`
- 项目文档：`README.md`
