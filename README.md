# 维纳滤波器分析项目

## 项目简介

本项目实现了多输入单输出(MISO) FIR维纳滤波器，用于从加速度传感器信号估计麦克风信号，实现主动噪声控制(ANC)分析。

## 项目结构

```
Wiener/
├── main_analysis.py                    # Python主程序（推荐使用）
├── MATLAB_Python_对比修正记录.md      # 修正记录文档
├── README.md                           # 本文件
│
├── data/                               # 数据文件目录
│   ├── 260107_ddr_32x5_05.dat         # 主数据文件（32通道，5秒）
│   └── ...
│
├── results/                            # 结果输出目录
│   ├── psd_accelerometers.png         # 加速度传感器PSD图
│   ├── psd_microphones.png            # 麦克风PSD图
│   └── wiener_filter_result.png        # 维纳滤波结果对比图
│
└── Wiener/                             # MATLAB程序目录
    ├── z_importBASE.m                  # MATLAB主程序
    ├── run_matlab_simple.m             # MATLAB运行脚本
    ├── miso_firwiener2.m               # 维纳滤波器核心函数
    ├── test_wiener.m                   # 滤波器测试函数
    ├── aweighting.m                    # A计权函数
    ├── psdcomp.m                       # PSD比较函数
    ├── freq_rms.m                      # 频域RMS计算函数
    ├── save_wiener_filters.m           # 保存滤波器系数函数
    └── ...
```

## 快速开始

### Python版本（推荐）

```bash
python main_analysis.py
```

**输出结果：**
- BeforeANC: 28.81 dBA
- AfterANC: 21.16 dBA
- ANC降噪量: -7.65 dBA

**生成图表：**
- `results/psd_accelerometers.png` - 加速度传感器PSD图
- `results/psd_microphones.png` - 麦克风PSD图
- `results/wiener_filter_result.png` - 维纳滤波结果对比图

### MATLAB版本

在MATLAB中运行：
```matlab
cd('Wiener')
run_matlab_simple
```

或直接运行：
```matlab
cd('Wiener')
z_importBASE
```

## 程序说明

### Python主程序 (main_analysis.py)

**主要功能：**
1. 读取32通道数据文件
2. 应用A计权滤波
3. 计算所有通道的PSD和dB值
4. 绘制PSD结果图
5. 计算MISO维纳滤波器（使用18个加速度传感器估计麦克风信号）
6. 应用滤波器并计算误差信号
7. 对结果进行A计权并计算降噪量

**关键参数：**
- 采样率: 3000 Hz
- 数据时长: 5秒
- 滤波器长度: 300
- 输入通道: 加速度传感器（通道7-24，共18个）
- 目标通道: 麦克风（通道25）

### MATLAB主程序 (z_importBASE.m)

功能与Python版本相同，使用Block Levinson算法求解维纳滤波器。

## 技术要点

### 关键修正（详见 `MATLAB_Python_对比修正记录.md`）

1. **滤波器长度**: 300（与MATLAB一致）
2. **A计权应用顺序**: 先计算滤波器，再对结果A计权
3. **reshape操作**: 所有reshape必须使用`order='F'`（列优先）
4. **Block Levinson算法**: 完全复刻MATLAB实现

### reshape操作的关键规则

**重要：** MATLAB的`reshape`默认使用列优先（Fortran顺序），Python的`reshape`默认使用行优先（C顺序）。

**修正方法：**
```python
# 错误（行优先）
R = R.reshape((N+1, M, M))

# 正确（列优先，与MATLAB一致）
R = R.reshape((N+1, M, M), order='F')
```

所有涉及reshape的地方都必须使用`order='F'`参数。

## 依赖库

### Python
- numpy
- scipy
- matplotlib

### MATLAB
- Signal Processing Toolbox

## 文件说明

### 核心文件

- **main_analysis.py**: Python主程序，包含完整的数据分析和维纳滤波流程
- **z_importBASE.m**: MATLAB主程序
- **miso_firwiener2.m**: MISO维纳滤波器核心算法（Block Levinson）
- **test_wiener.m**: 应用维纳滤波器并计算误差信号

### 辅助函数

- **aweighting.m**: A计权滤波器实现
- **psdcomp.m**: PSD比较和绘图函数
- **freq_rms.m**: 频域RMS计算函数
- **save_wiener_filters.m**: 保存滤波器系数到.mat文件

## 结果验证

Python和MATLAB程序的结果已完全一致：

| 指标 | MATLAB | Python | 状态 |
|------|--------|--------|------|
| BeforeANC | 28.81 dBA | 28.81 dBA | ✅ |
| AfterANC | 21.16 dBA | 21.16 dBA | ✅ |
| ANC降噪量 | -7.65 dBA | -7.65 dBA | ✅ |

## 注意事项

1. 数据文件路径：确保`data/260107_ddr_32x5_05.dat`文件存在
2. 结果目录：程序会自动创建`results/`目录保存图表
3. MATLAB版本：建议使用MATLAB R2018a或更高版本
4. Python版本：建议使用Python 3.8或更高版本

## 参考文档

- `MATLAB_Python_对比修正记录.md`: 详细的修正过程和技术要点

## 作者

修正完成日期：2024年
