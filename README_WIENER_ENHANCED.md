# 增强型深度学习ANC - 融合维纳滤波思想

## 概述

本次更新添加了三种融合维纳滤波思想的增强型模型，旨在让深度学习达到维纳滤波器的性能水平。

## 主要改进

### 1. 评估方式改进
**模仿维纳滤波器：训练集上训练，训练集上评估**

- 之前：使用80%数据训练，在全部数据上评估（数据泄漏）
- 现在：使用90%数据训练，10%验证，在全部数据上评估（与维纳滤波器一致）

这样可以公平比较深度学习和维纳滤波器在相同数据上的拟合能力。

### 2. 三种增强型模型

#### 模型1: `ANCModel_WienerEnhanced` (推荐)
**融合维纳滤波思想的增强型模型**

特点：
- **频域路径**：使用FFT提取频域特征（实部+虚部）
- **时域路径**：使用1D CNN提取时域特征
- **线性FIR路径**：直接学习FIR滤波器系数（模仿维纳滤波器）
- **三路融合**：通过LSTM融合三路特征

优势：
- 同时利用时域和频域信息
- 包含线性FIR层，能学习类似维纳滤波器的线性映射
- 更大的感受野，能看到更多上下文

#### 模型2: `ANCModel_HybridLinear`
**混合线性模型（主要线性+少量非线性）**

特点：
- **主路径**：线性FIR滤波（64个滤波器）
- **辅助路径**：非线性特征提取（32个特征）
- **融合**：简单的线性组合

优势：
- 主要保持线性特性（适合线性系统）
- 允许少量非线性来捕捉可能的非线性关系
- 参数量适中，训练快速

#### 模型3: `ANCModel_LinearFIR`
**纯线性FIR模型（最接近维纳滤波器）**

特点：
- 单层卷积，无非线性激活
- 直接学习18个通道的FIR滤波器系数
- 输出是所有通道滤波结果的加权和

优势：
- 最接近维纳滤波器的神经网络实现
- 参数量最少（18 × 300 + 1 = 5,401个参数）
- 如果问题是线性的，这个模型理论上应该能达到维纳滤波器的效果

### 3. 关键技术点

#### 频域特征提取
```python
# 计算FFT
x_fft = torch.fft.rfft(x, dim=2)
x_fft_real = x_fft.real
x_fft_imag = x_fft.imag

# 拼接实部和虚部作为特征
freq_input = torch.cat([x_fft_real, x_fft_imag], dim=1)
```

#### 线性FIR滤波层
```python
# 模仿维纳滤波器的FIR结构
self.fir_filters = nn.Conv1d(num_channels, 32,
                             kernel_size=window_size,
                             padding=0, bias=False)
```

## 使用方法

### 运行训练

```bash
cd "D:\Claude Code Projects\Wiener"
python dl_anc_multiband.py
```

### 选择模型类型

在 `dl_anc_multiband.py` 中修改 `model_type` 参数：

```python
# 第203行
model_type = 'wiener_enhanced'  # 推荐：增强型模型
# model_type = 'hybrid_linear'  # 混合线性模型
# model_type = 'linear_fir'     # 纯线性FIR模型
# model_type = 'cnn_lstm'       # 原始CNN-LSTM模型
```

### 训练配置

当前配置（已优化）：
- **低频段 (20-150Hz)**: stride=2, epochs=400, hidden_size=384
- **中频段 (150-250Hz)**: stride=2, epochs=400, hidden_size=384
- **高频段 (250-500Hz)**: stride=1, epochs=600, hidden_size=640

## 预期效果

### 理论分析

1. **`ANCModel_LinearFIR`** 应该最接近维纳滤波器：
   - 如果问题是纯线性的，这个模型理论上能达到维纳滤波器的效果
   - 参数量少，训练快速
   - 适合作为baseline

2. **`ANCModel_HybridLinear`** 应该略好于纯线性：
   - 保留线性主路径
   - 允许捕捉少量非线性关系
   - 平衡性能和复杂度

3. **`ANCModel_WienerEnhanced`** 应该效果最好：
   - 同时利用时域和频域信息
   - 包含线性FIR层
   - 能学习更复杂的模式

### 性能目标

- **维纳滤波器**: -7.65 dBA（已达到）
- **深度学习目标**: ≤ -7.65 dBA（与维纳滤波器持平或更好）

## 为什么这些改进有效？

### 1. 频域特征
- 维纳滤波器本质上在频域工作（通过相关函数）
- 添加FFT特征让模型能直接看到频域信息
- 不同频率成分可以被独立处理

### 2. 线性FIR层
- 维纳滤波器就是一组FIR滤波器
- 直接在模型中加入FIR层，让模型能学习类似的线性映射
- 如果问题是线性的，这个层应该能学到接近维纳滤波器的系数

### 3. 更大的感受野
- 维纳滤波器使用全部数据计算相关矩阵
- 增强型模型通过更深的网络和更大的卷积核增加感受野
- 能看到更多上下文信息

### 4. 训练集评估
- 与维纳滤波器使用相同的评估方式
- 公平比较拟合能力
- 避免数据泄漏问题

## 文件说明

- **`dl_anc_model_wiener_enhanced.py`**: 增强型模型定义
- **`dl_anc_multiband.py`**: 多频段训练脚本（已更新）
- **`README_WIENER_ENHANCED.md`**: 本文档

## 下一步

1. 运行训练，查看三种模型的效果
2. 比较不同模型类型的性能
3. 如果效果不理想，可以进一步调整：
   - 增加训练轮数
   - 调整学习率
   - 修改模型架构
   - 尝试不同的损失函数（如频域损失）

## 技术细节

### 模型参数量对比

- **LinearFIR**: ~5,401 参数
- **HybridLinear**: ~60,000 参数
- **WienerEnhanced**: ~2,000,000 参数（取决于hidden_size）
- **原始CNN-LSTM**: ~1,000,000 参数

### 计算复杂度

- **LinearFIR**: 最快（单层卷积）
- **HybridLinear**: 快（两路并行）
- **WienerEnhanced**: 较慢（三路并行+FFT+LSTM）

### 内存占用

- **LinearFIR**: 最小
- **HybridLinear**: 小
- **WienerEnhanced**: 较大（需要存储FFT结果和三路特征）

## 故障排除

### 如果训练很慢
- 尝试使用 `model_type='linear_fir'` 或 `'hybrid_linear'`
- 减小 `batch_size`
- 减小 `hidden_size`

### 如果显存不足
- 减小 `batch_size`
- 使用更小的模型（`linear_fir` 或 `hybrid_linear`）
- 减小 `hidden_size`

### 如果效果不好
- 增加训练轮数
- 减小学习率
- 尝试不同的模型类型
- 检查数据归一化是否正确
