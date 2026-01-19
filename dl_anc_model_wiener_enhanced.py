"""
增强型深度学习ANC模型 - 融合维纳滤波思想
包含频域特征、线性FIR层等仿维纳操作
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ANCModel_WienerEnhanced(nn.Module):
    """
    融合维纳滤波思想的增强型ANC模型

    特点：
    1. 频域特征提取（FFT）
    2. 时域和频域双路径处理
    3. 线性FIR滤波层（模仿维纳滤波器）
    4. 更大的感受野
    """
    def __init__(self, num_channels=18, window_size=300, hidden_size=256):
        super(ANCModel_WienerEnhanced, self).__init__()

        self.num_channels = num_channels
        self.window_size = window_size

        # ========== 时域路径 ==========
        # 1D卷积提取时域特征
        self.time_conv1 = nn.Conv1d(num_channels, 64, kernel_size=7, padding=3)
        self.time_bn1 = nn.BatchNorm1d(64)
        self.time_conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.time_bn2 = nn.BatchNorm1d(128)
        self.time_conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.time_bn3 = nn.BatchNorm1d(64)

        # ========== 频域路径 ==========
        # 处理FFT特征（实部和虚部）
        fft_size = window_size // 2 + 1  # FFT输出大小
        self.freq_conv1 = nn.Conv1d(num_channels * 2, 64, kernel_size=5, padding=2)  # *2因为有实部和虚部
        self.freq_bn1 = nn.BatchNorm1d(64)
        self.freq_conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.freq_bn2 = nn.BatchNorm1d(128)
        self.freq_conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.freq_bn3 = nn.BatchNorm1d(64)

        # ========== 线性FIR滤波层（模仿维纳滤波器）==========
        # 直接学习滤波器系数
        self.fir_filters = nn.Conv1d(num_channels, 32, kernel_size=window_size, padding=0, bias=False)

        # ========== 融合层 ==========
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        # LSTM处理时序依赖
        # 时域路径: 64 channels, 频域路径: 64 channels, FIR: 32 channels
        lstm_input_size = 64 + 64 + 32  # 三路融合
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers=2,
                           batch_first=True, dropout=0.2)

        # 输出层
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        x: (batch, num_channels, window_size)
        """
        batch_size = x.size(0)

        # ========== 时域路径 ==========
        time_feat = self.relu(self.time_bn1(self.time_conv1(x)))
        time_feat = self.pool(time_feat)
        time_feat = self.dropout(time_feat)

        time_feat = self.relu(self.time_bn2(self.time_conv2(time_feat)))
        time_feat = self.pool(time_feat)
        time_feat = self.dropout(time_feat)

        time_feat = self.relu(self.time_bn3(self.time_conv3(time_feat)))
        time_feat = self.pool(time_feat)
        # time_feat: (batch, 64, window_size//8)

        # ========== 频域路径 ==========
        # 计算FFT（实部和虚部）
        x_fft = torch.fft.rfft(x, dim=2)  # (batch, num_channels, fft_size)
        x_fft_real = x_fft.real
        x_fft_imag = x_fft.imag

        # 拼接实部和虚部
        freq_input = torch.cat([x_fft_real, x_fft_imag], dim=1)  # (batch, num_channels*2, fft_size)

        freq_feat = self.relu(self.freq_bn1(self.freq_conv1(freq_input)))
        freq_feat = self.pool(freq_feat)
        freq_feat = self.dropout(freq_feat)

        freq_feat = self.relu(self.freq_bn2(self.freq_conv2(freq_feat)))
        freq_feat = self.pool(freq_feat)
        freq_feat = self.dropout(freq_feat)

        freq_feat = self.relu(self.freq_bn3(self.freq_conv3(freq_feat)))
        # 调整大小以匹配时域特征
        if freq_feat.size(2) > time_feat.size(2):
            freq_feat = freq_feat[:, :, :time_feat.size(2)]
        elif freq_feat.size(2) < time_feat.size(2):
            padding = time_feat.size(2) - freq_feat.size(2)
            freq_feat = torch.nn.functional.pad(freq_feat, (0, padding))
        # freq_feat: (batch, 64, window_size//8)

        # ========== 线性FIR滤波路径（模仿维纳滤波器）==========
        fir_feat = self.fir_filters(x)  # (batch, 32, 1)
        # 扩展到与其他特征相同的长度
        fir_feat = fir_feat.expand(-1, -1, time_feat.size(2))  # (batch, 32, window_size//8)

        # ========== 融合三路特征 ==========
        combined = torch.cat([time_feat, freq_feat, fir_feat], dim=1)  # (batch, 160, window_size//8)

        # 转换为LSTM输入格式
        combined = combined.permute(0, 2, 1)  # (batch, seq_len, features)

        # LSTM处理
        lstm_out, (h_n, c_n) = self.lstm(combined)

        # 取最后一个时间步
        x_out = lstm_out[:, -1, :]

        # 全连接层
        x_out = self.relu(self.fc1(x_out))
        x_out = self.dropout(x_out)
        x_out = self.fc2(x_out)

        return x_out


class ANCModel_LinearFIR(nn.Module):
    """
    纯线性FIR模型 - 最接近维纳滤波器的神经网络实现

    这个模型直接学习FIR滤波器系数，类似于维纳滤波器
    """
    def __init__(self, num_channels=18, filter_length=300):
        super(ANCModel_LinearFIR, self).__init__()

        # 为每个通道学习一个FIR滤波器
        # 输出是所有通道滤波结果的加权和
        self.fir_filters = nn.Conv1d(num_channels, 1, kernel_size=filter_length,
                                     padding=0, bias=True)

        # 不使用非线性激活，保持线性特性

    def forward(self, x):
        """
        x: (batch, num_channels, window_size)
        输出: (batch, 1)
        """
        # 应用FIR滤波
        out = self.fir_filters(x)  # (batch, 1, 1)
        out = out.squeeze(-1)  # (batch, 1)
        return out


class ANCModel_HybridLinear(nn.Module):
    """
    混合线性模型 - 结合线性FIR和少量非线性

    主要是线性的，但允许少量非线性来捕捉可能的非线性关系
    """
    def __init__(self, num_channels=18, filter_length=300, hidden_size=128):
        super(ANCModel_HybridLinear, self).__init__()

        # 主路径：线性FIR滤波
        self.fir_filters = nn.Conv1d(num_channels, 64, kernel_size=filter_length,
                                     padding=0, bias=True)

        # 辅助路径：提取非线性特征（如果存在）
        self.nonlinear_conv = nn.Conv1d(num_channels, 32, kernel_size=filter_length,
                                       padding=0, bias=True)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()

        # 融合层
        self.fc = nn.Linear(64 + 32, 1)

    def forward(self, x):
        """
        x: (batch, num_channels, window_size)
        """
        # 线性路径
        linear_out = self.fir_filters(x).squeeze(-1)  # (batch, 64)

        # 非线性路径
        nonlinear_out = self.nonlinear_conv(x)  # (batch, 32, 1)
        nonlinear_out = self.relu(self.bn(nonlinear_out))
        nonlinear_out = nonlinear_out.squeeze(-1)  # (batch, 32)

        # 融合
        combined = torch.cat([linear_out, nonlinear_out], dim=1)  # (batch, 96)
        out = self.fc(combined)  # (batch, 1)

        return out


class ANCModel_TCN(nn.Module):
    """
    时间卷积网络(TCN)模型用于ANC
    使用因果卷积和膨胀卷积，适合处理有因果性的时序信号

    特点：
    1. 因果卷积（不会看到未来信息）
    2. 膨胀卷积（指数级增长的感受野）
    3. 残差连接（更深的网络）
    4. 参数效率高
    """
    def __init__(self, num_channels=18, window_size=300, num_filters=64, num_layers=6):
        super(ANCModel_TCN, self).__init__()

        layers = []
        in_channels = num_channels

        for i in range(num_layers):
            dilation = 2 ** i
            padding = dilation * (3 - 1)  # kernel_size=3

            layers.append(nn.Conv1d(in_channels, num_filters, kernel_size=3,
                                   padding=padding, dilation=dilation))
            layers.append(nn.BatchNorm1d(num_filters))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

            in_channels = num_filters

        self.tcn = nn.Sequential(*layers)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(num_filters, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        x: (batch, num_channels, window_size)
        """
        x = self.tcn(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x


def create_model(model_type='wiener_enhanced', num_channels=18, window_size=300, hidden_size=256):
    """
    创建模型的工厂函数

    Parameters:
    -----------
    model_type : str
        'wiener_enhanced': 增强型模型（频域+时域+FIR）
        'linear_fir': 纯线性FIR模型
        'hybrid_linear': 混合线性模型
        'tcn': 时间卷积网络（适合因果时序信号）
        'cnn_lstm': 原始CNN-LSTM模型
    """
    if model_type == 'wiener_enhanced':
        return ANCModel_WienerEnhanced(num_channels, window_size, hidden_size)
    elif model_type == 'linear_fir':
        return ANCModel_LinearFIR(num_channels, window_size)
    elif model_type == 'hybrid_linear':
        return ANCModel_HybridLinear(num_channels, window_size, hidden_size)
    elif model_type == 'tcn':
        # TCN模型，num_filters根据hidden_size调整
        num_filters = max(64, hidden_size // 4)
        return ANCModel_TCN(num_channels, window_size, num_filters=num_filters, num_layers=6)
    elif model_type == 'cnn_lstm':
        # 导入原始模型
        from dl_anc_model import ANCModel_CNN_LSTM
        return ANCModel_CNN_LSTM(num_channels, window_size, hidden_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
