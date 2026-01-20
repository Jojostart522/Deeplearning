import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy import signal
import warnings
import os

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 开启交互模式，让图形窗口可编辑
plt.ion()

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# A计权滤波器系数（与main_analysis.py保持一致）
# ============================================================================

MIC_WEIGHT_B = np.array([
    0.525940985213594,
    -1.05188197042719,
    -0.525940985213595,
    2.10376394085438,
    -0.525940985213593,
    -1.05188197042719,
    0.525940985213595,
])

MIC_WEIGHT_A = np.array([
    1,
    -1.13163468428868,
    -1.33375196803962,
    1.65328506906755,
    0.347752998752847,
    -0.603093973377823,
    0.0685263316042962,
])

# ============================================================================
# 辅助函数
# ============================================================================

def load_data(filename, num_channels=32, num_samples=15000):
    """读取二进制数据文件"""
    print(f"正在读取数据文件: {filename}")
    raw_data = np.fromfile(filename, dtype=np.float32)
    data = raw_data.reshape((num_channels, num_samples))
    print(f"数据形状: {data.shape}")
    return data

def apply_a_weighting(data):
    """应用A计权滤波"""
    b = MIC_WEIGHT_B
    a = MIC_WEIGHT_A

    if len(data.shape) == 1:
        filtered_data = signal.lfilter(b, a, data)
    elif len(data.shape) == 2:
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i, :] = signal.lfilter(b, a, data[i, :])
    else:
        raise ValueError("数据必须是1D或2D数组")

    return filtered_data

def apply_highpass_filter(data, fs=3000, cutoff=100, order=5):
    """
    应用高通滤波器，去除低频成分

    Parameters:
    -----------
    data : ndarray
        输入信号，1D或2D数组
    fs : int
        采样率 (Hz)
    cutoff : float
        截止频率 (Hz)
    order : int
        滤波器阶数

    Returns:
    --------
    filtered_data : ndarray
        滤波后的信号
    """
    # 设计Butterworth高通滤波器
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)

    if len(data.shape) == 1:
        # 1D信号
        filtered_data = signal.filtfilt(b, a, data)
    elif len(data.shape) == 2:
        # 2D信号，逐通道滤波
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i, :] = signal.filtfilt(b, a, data[i, :])
    else:
        raise ValueError("数据必须是1D或2D数组")

    return filtered_data

def apply_bandpass_filter(data, fs=3000, lowcut=50, highcut=150, order=5):
    """
    应用带通滤波器，提取特定频段

    Parameters:
    -----------
    data : ndarray
        输入信号，1D或2D数组
    fs : int
        采样率 (Hz)
    lowcut : float
        低截止频率 (Hz)
    highcut : float
        高截止频率 (Hz)
    order : int
        滤波器阶数

    Returns:
    --------
    filtered_data : ndarray
        滤波后的信号
    """
    # 设计Butterworth带通滤波器
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band', analog=False)

    if len(data.shape) == 1:
        # 1D信号
        filtered_data = signal.filtfilt(b, a, data)
    elif len(data.shape) == 2:
        # 2D信号，逐通道滤波
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i, :] = signal.filtfilt(b, a, data[i, :])
    else:
        raise ValueError("数据必须是1D或2D数组")

    return filtered_data

def compute_freq_rms(signal_data, fs, f_min, f_max):
    """计算指定频率范围内的RMS值"""
    n = len(signal_data)
    fft_vals = np.fft.fft(signal_data)
    fft_mag = np.abs(fft_vals) / fs

    df = fs / n
    freqs = np.arange(n) * df

    half_n = n // 2
    fft_mag_half = fft_mag[:half_n]
    freqs_half = freqs[:half_n]

    idx = (freqs_half >= f_min) & (freqs_half <= f_max)

    if np.any(idx):
        mag_selected = fft_mag_half[idx]
        rms_sq = np.sum(mag_selected**2 * df**2 * 2)
        rms = np.sqrt(rms_sq)
    else:
        rms = 0.0

    return rms

def compute_db_value(signal_data, fs, f_min=50, f_max=500, reference=4e-10):
    """计算dB值"""
    rms = compute_freq_rms(signal_data, fs, f_min, f_max)
    db = 10 * np.log10((rms**2) / reference)
    return db

# ============================================================================
# 数据集类
# ============================================================================

class ANCDataset(Dataset):
    """
    主动噪声控制数据集
    使用滑动窗口创建训练样本
    """
    def __init__(self, input_signals, target_signal, window_size=300, stride=1):
        """
        Parameters:
        -----------
        input_signals : ndarray
            输入信号，形状 (num_channels, num_samples)
        target_signal : ndarray
            目标信号，形状 (num_samples,)
        window_size : int
            窗口大小（与维纳滤波器长度一致）
        stride : int
            滑动步长
        """
        self.input_signals = input_signals
        self.target_signal = target_signal
        self.window_size = window_size
        self.stride = stride

        # 计算可以创建的样本数量
        self.num_samples = (target_signal.shape[0] - window_size) // stride + 1

        print(f"数据集创建: {self.num_samples} 个样本 (窗口大小={window_size}, 步长={stride})")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        返回一个样本
        输入: (num_channels, window_size) 的窗口
        输出: 窗口末尾的目标值（单个值）
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size

        # 输入窗口
        x = self.input_signals[:, start_idx:end_idx]  # (num_channels, window_size)

        # 目标值（窗口末尾的值）
        y = self.target_signal[end_idx - 1]  # 单个值

        return torch.FloatTensor(x), torch.FloatTensor([y])

# ============================================================================
# 深度学习模型
# ============================================================================

class ANCModel_CNN_LSTM(nn.Module):
    """
    混合CNN-LSTM模型用于ANC
    - 1D CNN提取局部特征
    - LSTM捕捉时序依赖
    """
    def __init__(self, num_channels=18, window_size=300, hidden_size=128):
        super(ANCModel_CNN_LSTM, self).__init__()

        # 1D卷积层
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        # LSTM层
        # 经过3次池化后，序列长度变为 window_size // 8
        lstm_input_size = 64
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers=2,
                           batch_first=True, dropout=0.3)

        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, num_channels, window_size)

        # CNN特征提取
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # 转换为LSTM输入格式: (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最后一个时间步的输出
        x = lstm_out[:, -1, :]

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ANCModel_TCN(nn.Module):
    """
    时间卷积网络(TCN)模型用于ANC
    使用因果卷积和膨胀卷积
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
        # x: (batch, num_channels, window_size)
        x = self.tcn(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# ============================================================================
# 训练和评估函数
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cpu'):
    """训练模型（GPU优化版 - 混合精度训练 + 更好的学习率策略和早停）"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.6, patience=15, verbose=True, min_lr=1e-7)

    # GPU优化：启用混合精度训练（AMP）
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # GPU优化：启用cudnn benchmark（自动寻找最优卷积算法）
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("GPU优化已启用：混合精度训练(AMP) + cudnn.benchmark")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 50

    print(f"\n开始训练 (设备: {device})...")
    print(f"训练样本数: {len(train_loader.dataset)}, 验证样本数: {len(val_loader.dataset)}")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            # GPU优化：异步数据传输
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # GPU优化：更快的梯度清零

            # GPU优化：混合精度训练
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 学习率调整
        scheduler.step(val_loss)

        # 保存最佳模型和早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"Patience: {patience_counter}/{early_stop_patience}")

        # 早停机制
        if patience_counter >= early_stop_patience:
            print(f"\n早停触发！验证损失连续{early_stop_patience}轮未改善")
            print(f"在第 {epoch+1} 轮停止训练")
            break

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.6f}")

    return train_losses, val_losses

def predict_full_signal(model, input_signals, window_size=300, device='cpu', batch_size=512):
    """
    使用训练好的模型预测完整信号（GPU优化版 - 批量处理）

    Parameters:
    -----------
    model : nn.Module
        训练好的模型
    input_signals : ndarray
        输入信号 (num_channels, num_samples)
    window_size : int
        窗口大小
    device : torch.device
        计算设备
    batch_size : int
        批处理大小（GPU优化：一次处理多个窗口）

    Returns:
    --------
    predictions : ndarray
        预测结果 (num_samples,)
    """
    model.eval()
    num_samples = input_signals.shape[1]
    predictions = np.zeros(num_samples)

    # GPU优化：批量处理窗口
    with torch.no_grad():
        # 准备所有窗口
        windows = []
        indices = []
        for i in range(window_size - 1, num_samples):
            start_idx = i - window_size + 1
            end_idx = i + 1
            x_window = input_signals[:, start_idx:end_idx]
            windows.append(x_window)
            indices.append(i)

        # 转换为tensor
        windows_tensor = torch.FloatTensor(np.array(windows)).to(device)

        # 批量预测
        all_preds = []
        for i in range(0, len(windows_tensor), batch_size):
            batch = windows_tensor[i:i+batch_size]

            # GPU优化：混合精度推理
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    preds = model(batch)
            else:
                preds = model(batch)

            all_preds.append(preds.cpu().numpy())

        # 合并结果
        all_preds = np.concatenate(all_preds, axis=0)
        for idx, pred in zip(indices, all_preds):
            predictions[idx] = pred[0]

    # 前window_size-1个点使用零填充
    predictions[:window_size-1] = 0

    return predictions

def evaluate_anc_performance(target_signal, estimated_signal, fs=3000):
    """
    评估ANC性能（与维纳滤波相同的评估方法）
    """
    # 计算误差信号
    error_signal = target_signal - estimated_signal

    # 应用A计权
    target_weighted = apply_a_weighting(target_signal)
    error_weighted = apply_a_weighting(error_signal)

    # 计算dB值
    db_before = compute_db_value(target_weighted, fs)
    db_after = compute_db_value(error_weighted, fs)

    # 计算ANC降噪量
    anc_db = 10 * np.log10((compute_freq_rms(error_weighted, fs, 50, 500)**2) /
                           (compute_freq_rms(target_weighted, fs, 50, 500)**2))

    return {
        'db_before': db_before,
        'db_after': db_after,
        'anc_reduction': anc_db,
        'reduction_diff': db_before - db_after,
        'error_signal': error_signal,
        'target_weighted': target_weighted,
        'error_weighted': error_weighted
    }

# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""

    # ========== 配置参数 ==========
    
    # ===== 模型选择 (修改这里来选择不同的模型) =====
    # 选项: 'CNN-LSTM' 或 'TCN'
    MODEL_TYPE = 'CNN-LSTM'  # <--- 在这里修改模型类型
    # =============================================
    
    filename = 'data/260107_ddr_32x5_05.dat'
    num_channels = 32
    num_samples = 15000
    fs = 3000

    window_size = 300  # 与维纳滤波器长度一致
    stride = 3  # 滑动步长（进一步减小以最大化训练样本）

    batch_size = 32  # 减小batch size以获得更稳定的梯度
    num_epochs = 300  # 增加到300轮
    learning_rate = 0.0005  # 降低初始学习率以获得更稳定的训练

    # 检测GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ========== 步骤1: 加载数据 ==========
    print("\n" + "="*60)
    print("步骤1: 加载数据")
    print("="*60)
    data = load_data(filename, num_channels, num_samples)

    # 定义输入和目标
    acc_indices = list(range(6, 24))  # 加速度传感器 (Ch 7-24)
    mic_index = 24  # 麦克风 (Ch 25)

    input_signals = data[acc_indices, :]  # (18, 15000)
    target_signal = data[mic_index, :]    # (15000,)

    print(f"输入信号形状: {input_signals.shape}")
    print(f"目标信号形状: {target_signal.shape}")

    # ========== 步骤1.5: 应用100Hz高通滤波器 ==========
    print("\n" + "="*60)
    print("步骤1.5: 应用100Hz高通滤波器（去除低频成分）")
    print("="*60)

    # 对输入信号和目标信号都应用高通滤波
    input_signals_filtered = apply_highpass_filter(input_signals, fs=fs, cutoff=100, order=5)
    target_signal_filtered = apply_highpass_filter(target_signal, fs=fs, cutoff=100, order=5)

    print("高通滤波完成 - 已去除100Hz以下的频率成分")
    print("模型将专注于100-400Hz频段的拟合")

    # 使用滤波后的信号替换原始信号
    input_signals = input_signals_filtered
    target_signal = target_signal_filtered

    # ========== 步骤2: 数据归一化 ==========
    print("\n" + "="*60)
    print("步骤2: 数据归一化")
    print("="*60)

    # 计算归一化参数（使用训练集统计）
    input_mean = np.mean(input_signals, axis=1, keepdims=True)
    input_std = np.std(input_signals, axis=1, keepdims=True) + 1e-8
    target_mean = np.mean(target_signal)
    target_std = np.std(target_signal) + 1e-8

    input_signals_norm = (input_signals - input_mean) / input_std
    target_signal_norm = (target_signal - target_mean) / target_std

    print("归一化完成")

    # ========== 步骤3: 划分训练集和测试集 ==========
    print("\n" + "="*60)
    print("步骤3: 划分数据集")
    print("="*60)

    # 使用前80%作为训练集，后20%作为测试集
    split_idx = int(0.8 * num_samples)

    train_input = input_signals_norm[:, :split_idx]
    train_target = target_signal_norm[:split_idx]

    test_input = input_signals_norm[:, split_idx:]
    test_target = target_signal_norm[split_idx:]

    print(f"训练集: {train_input.shape[1]} 个样本")
    print(f"测试集: {test_input.shape[1]} 个样本")

    # 创建数据集
    train_dataset = ANCDataset(train_input, train_target, window_size, stride)
    # 验证集使用训练集的最后10%
    val_split = int(0.9 * train_input.shape[1])
    val_input = train_input[:, val_split:]
    val_target = train_target[val_split:]
    val_dataset = ANCDataset(val_input, val_target, window_size, stride)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ========== 步骤4: 创建模型 ==========
    print("\n" + "="*60)
    print("步骤4: 创建深度学习模型")
    print("="*60)

    # 根据配置选择模型架构
    if MODEL_TYPE == 'CNN-LSTM':
        print("使用模型: CNN-LSTM混合架构（增强版）")
        model = ANCModel_CNN_LSTM(num_channels=18, window_size=window_size, hidden_size=256)  # 增加隐藏层大小
        model_name = 'CNN_LSTM'
    elif MODEL_TYPE == 'TCN':
        print("使用模型: 时间卷积网络(TCN)（增强版）")
        model = ANCModel_TCN(num_channels=18, window_size=window_size, num_filters=128)  # 增加滤波器数量
        model_name = 'TCN'
    else:
        raise ValueError(f"未知的模型类型: {MODEL_TYPE}，请选择 'CNN-LSTM' 或 'TCN'")

    model = model.to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # ========== 步骤5: 训练模型 ==========
    print("\n" + "="*60)
    print("步骤5: 训练模型")
    print("="*60)

    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        num_epochs=num_epochs, lr=learning_rate, device=device
    )

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'训练过程 - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    training_curve_file = f'results/dl_training_curve_{model_name}.png'
    plt.savefig(training_curve_file, dpi=150)
    print(f"训练曲线已保存: {training_curve_file}")
    plt.show()  # 显示图片

    # ========== 步骤6: 在完整数据上预测 ==========
    print("\n" + "="*60)
    print("步骤6: 预测完整信号")
    print("="*60)

    # 使用归一化的数据进行预测
    estimated_signal_norm = predict_full_signal(model, input_signals_norm, window_size, device)

    # 反归一化
    estimated_signal = estimated_signal_norm * target_std + target_mean

    print("预测完成")

    # ========== 步骤7: 评估性能 ==========
    print("\n" + "="*60)
    print("步骤7: 评估ANC性能")
    print("="*60)

    results = evaluate_anc_performance(target_signal, estimated_signal, fs)

    print(f"\n=== {model_name} 模型ANC结果 ===")
    print(f"BeforeANC (原始麦克风): {results['db_before']:.2f} dBA")
    print(f"AfterANC (误差信号): {results['db_after']:.2f} dBA")
    print(f"ANC降噪量: {results['anc_reduction']:.2f} dBA")
    print(f"降噪量 (原始 - 误差): {results['reduction_diff']:.2f} dBA")

    print(f"\n=== 与维纳滤波对比 ===")
    wiener_reduction = -7.65
    print(f"维纳滤波降噪量: {wiener_reduction:.2f} dBA")
    print(f"深度学习降噪量: {results['anc_reduction']:.2f} dBA")
    print(f"差异: {results['anc_reduction'] - wiener_reduction:.2f} dBA")

    if results['anc_reduction'] <= wiener_reduction:
        print("✅ 深度学习模型达到或超过维纳滤波性能！")
    else:
        print("⚠️ 深度学习模型性能略低于维纳滤波")

    # ========== 步骤8: 可视化结果 ==========
    print("\n" + "="*60)
    print("步骤8: 可视化结果")
    print("="*60)

    # 创建两个图：1) 拟合精度对比  2) 降噪效果对比
    from scipy.signal import welch

    # ===== 图1: 拟合精度对比（原始 vs 估计） =====
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 12))

    time_axis = np.arange(num_samples) / fs
    show_samples = 1000

    # 1.1 时域对比
    axes1[0].plot(time_axis[:show_samples], target_signal[:show_samples],
                label='原始麦克风', alpha=0.8, color='blue', linewidth=1.5)
    axes1[0].plot(time_axis[:show_samples], estimated_signal[:show_samples],
                label=f'{model_name}估计信号', alpha=0.8, color='red', linewidth=1.5)
    axes1[0].set_title(f'时域波形对比 (前1000点) - {model_name}', fontsize=13, fontweight='bold')
    axes1[0].set_xlabel('时间 (s)')
    axes1[0].set_ylabel('幅值')
    axes1[0].legend(fontsize=11)
    axes1[0].grid(True, alpha=0.3)

    # 1.2 拟合误差
    fit_error = target_signal - estimated_signal
    axes1[1].plot(time_axis[:show_samples], fit_error[:show_samples],
                color='purple', alpha=0.7, linewidth=1.5)
    axes1[1].set_title('拟合误差 (原始 - 估计)', fontsize=13, fontweight='bold')
    axes1[1].set_xlabel('时间 (s)')
    axes1[1].set_ylabel('幅值')
    axes1[1].grid(True, alpha=0.3)

    # 1.3 频域拟合对比（PSD）- 使用A计权信号
    # 对估计信号也应用A计权
    estimated_signal_weighted = apply_a_weighting(estimated_signal)

    f_target_weighted, Pxx_target_weighted = welch(results['target_weighted'], fs, nperseg=750, noverlap=375)
    f_est_weighted, Pxx_est_weighted = welch(estimated_signal_weighted, fs, nperseg=750, noverlap=375)

    Pxx_target_weighted_db = 10 * np.log10(Pxx_target_weighted / 4e-10)
    Pxx_est_weighted_db = 10 * np.log10(Pxx_est_weighted / 4e-10)

    axes1[2].plot(f_target_weighted, Pxx_target_weighted_db,
                label='原始麦克风(A计权)', linewidth=2.5, color='blue', alpha=0.8)
    axes1[2].plot(f_est_weighted, Pxx_est_weighted_db,
                label=f'{model_name}估计(A计权)', linewidth=2.5, color='red', alpha=0.8, linestyle='--')

    # 标注关键频段
    axes1[2].axvspan(50, 100, alpha=0.1, color='yellow', label='50-100Hz')
    axes1[2].axvspan(100, 300, alpha=0.1, color='green', label='100-300Hz')
    axes1[2].axvspan(300, 500, alpha=0.1, color='orange', label='300-500Hz')

    axes1[2].set_title(f'PSD拟合对比（A计权信号） - {model_name}', fontsize=13, fontweight='bold')
    axes1[2].set_xlabel('频率 (Hz)', fontsize=11)
    axes1[2].set_ylabel('PSD (dB/Hz)', fontsize=11)
    axes1[2].set_xlim([20, 500])
    axes1[2].legend(fontsize=10, loc='upper right')
    axes1[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fit_plot_file = f'results/dl_fitting_comparison_{model_name}.png'
    plt.savefig(fit_plot_file, dpi=150)
    print(f"拟合对比图已保存: {fit_plot_file}")
    plt.show()

    # ===== 图2: 降噪效果对比（原始 vs 误差） =====
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 12))

    # 2.1 时域对比（A计权后）
    axes2[0].plot(time_axis[:show_samples], results['target_weighted'][:show_samples],
                label='原始麦克风(A计权)', alpha=0.8, color='blue', linewidth=1.5)
    axes2[0].plot(time_axis[:show_samples], results['error_weighted'][:show_samples],
                label='误差信号(A计权)', alpha=0.8, color='green', linewidth=1.5)
    axes2[0].set_title(f'ANC降噪效果 - 时域对比 (A计权) - {model_name}', fontsize=13, fontweight='bold')
    axes2[0].set_xlabel('时间 (s)')
    axes2[0].set_ylabel('幅值')
    axes2[0].legend(fontsize=11)
    axes2[0].grid(True, alpha=0.3)

    # 2.2 误差信号（A计权）
    axes2[1].plot(time_axis[:show_samples], results['error_weighted'][:show_samples],
                color='green', alpha=0.7, linewidth=1.5)
    axes2[1].set_title('ANC误差信号 (A计权)', fontsize=13, fontweight='bold')
    axes2[1].set_xlabel('时间 (s)')
    axes2[1].set_ylabel('幅值')
    axes2[1].grid(True, alpha=0.3)

    # 2.3 频域降噪对比（PSD，A计权）
    f_target, Pxx_target = welch(results['target_weighted'], fs, nperseg=750, noverlap=375)
    f_error, Pxx_error = welch(results['error_weighted'], fs, nperseg=750, noverlap=375)

    Pxx_target_db = 10 * np.log10(Pxx_target / 4e-10)
    Pxx_error_db = 10 * np.log10(Pxx_error / 4e-10)

    axes2[2].plot(f_target, Pxx_target_db,
                label=f'原始麦克风 ({results["db_before"]:.1f}dBA)',
                linewidth=2.5, color='blue', alpha=0.8)
    axes2[2].plot(f_error, Pxx_error_db,
                label=f'误差信号 ({results["db_after"]:.1f}dBA)',
                linewidth=2.5, color='green', alpha=0.8)

    # 标注降噪量
    axes2[2].text(0.98, 0.95, f'ANC降噪量: {results["anc_reduction"]:.2f} dBA\n目标: -7.65 dBA',
                transform=axes2[2].transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes2[2].set_title(f'PSD降噪对比 (A计权) - {model_name}', fontsize=13, fontweight='bold')
    axes2[2].set_xlabel('频率 (Hz)', fontsize=11)
    axes2[2].set_ylabel('PSD (dB/Hz)', fontsize=11)
    axes2[2].set_xlim([20, 500])
    axes2[2].legend(fontsize=10, loc='upper right')
    axes2[2].grid(True, alpha=0.3)

    plt.tight_layout()
    anc_plot_file = f'results/dl_anc_result_{model_name}.png'
    plt.savefig(anc_plot_file, dpi=150)
    print(f"降噪效果图已保存: {anc_plot_file}")
    plt.show()

    # ========== 步骤9: 保存模型 ==========
    print("\n" + "="*60)
    print("步骤9: 保存模型")
    print("="*60)

    model_file = f'results/dl_anc_model_{model_name}.pth'
    torch.save({
        'model_type': MODEL_TYPE,
        'model_state_dict': model.state_dict(),
        'input_mean': input_mean,
        'input_std': input_std,
        'target_mean': target_mean,
        'target_std': target_std,
        'window_size': window_size,
        'results': results
    }, model_file)

    print(f"模型已保存: {model_file}")

    print("\n" + "="*60)
    print("完成！")
    print("="*60)

    return model, results

if __name__ == '__main__':
    model, results = main()
