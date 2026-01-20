"""
多频段深度学习ANC实现
将频谱分为三个频段分别训练，然后融合结果：
- 低频：50-150Hz
- 中频：150-350Hz
- 高频：350-500Hz
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import welch
import warnings
import os

# 导入基础函数和模型
from dl_anc_model import (
    load_data, apply_a_weighting, compute_db_value, compute_freq_rms,
    apply_bandpass_filter, ANCModel_CNN_LSTM, ANCDataset,
    train_model, predict_full_signal, evaluate_anc_performance
)

# 导入增强型模型
from dl_anc_model_wiener_enhanced import (
    ANCModel_WienerEnhanced, ANCModel_LinearFIR, ANCModel_HybridLinear,
    create_model
)

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 开启交互模式，让图形窗口可编辑
plt.ion()

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 配置开关：分频 vs 不分频
# ============================================================================
USE_MULTIBAND = True  # True=分频训练（3个频段）, False=不分频（全频段）

# ============================================================================
# 多频段融合函数
# ============================================================================

def fuse_multiband_predictions(predictions_dict, fs=3000):
    """
    融合多个频段的预测结果

    Parameters:
    -----------
    predictions_dict : dict
        字典，键为频段名称，值为预测信号
        例如: {'low': signal1, 'mid': signal2, 'high': signal3}
    fs : int
        采样率

    Returns:
    --------
    fused_signal : ndarray
        融合后的信号
    """
    # 方法1: 简单相加（因为各频段是正交的）
    fused_signal = np.zeros_like(list(predictions_dict.values())[0])

    for band_name, prediction in predictions_dict.items():
        fused_signal += prediction

    return fused_signal


def train_single_band(input_signals, target_signal, band_name, lowcut, highcut,
                     window_size=300, stride=3, batch_size=32, num_epochs=300,
                     learning_rate=0.0005, hidden_size=256, model_type='wiener_enhanced',
                     device='cpu', fs=3000):
    """
    训练单个频段的模型

    Parameters:
    -----------
    input_signals : ndarray
        输入信号 (num_channels, num_samples)
    target_signal : ndarray
        目标信号 (num_samples,)
    band_name : str
        频段名称（用于保存模型）
    lowcut, highcut : float
        频段范围
    其他参数同main函数

    Returns:
    --------
    model : nn.Module
        训练好的模型
    normalization_params : dict
        归一化参数
    """
    print("\n" + "="*70)
    print(f"训练频段: {band_name} ({lowcut}-{highcut} Hz)")
    print(f"模型类型: {model_type}")
    print(f"训练配置: stride={stride}, epochs={num_epochs}, hidden_size={hidden_size}, lr={learning_rate}")
    print("="*70)

    # 步骤1: 应用带通滤波
    print(f"\n应用带通滤波器: {lowcut}-{highcut} Hz")
    input_filtered = apply_bandpass_filter(input_signals, fs=fs, lowcut=lowcut, highcut=highcut, order=5)
    target_filtered = apply_bandpass_filter(target_signal, fs=fs, lowcut=lowcut, highcut=highcut, order=5)

    # 步骤2: 数据归一化
    print("数据归一化...")
    input_mean = np.mean(input_filtered, axis=1, keepdims=True)
    input_std = np.std(input_filtered, axis=1, keepdims=True) + 1e-8
    target_mean = np.mean(target_filtered)
    target_std = np.std(target_filtered) + 1e-8

    input_norm = (input_filtered - input_mean) / input_std
    target_norm = (target_filtered - target_mean) / target_std

    # 步骤3: 划分训练集和验证集（模仿维纳滤波：使用全部数据训练）
    # 使用全部数据作为训练集，取最后10%作为验证集监控训练过程
    val_split = int(0.9 * len(target_norm))
    train_input = input_norm[:, :val_split]
    train_target = target_norm[:val_split]
    val_input = input_norm[:, val_split:]
    val_target = target_norm[val_split:]

    print(f"使用全部数据训练（模仿维纳滤波）")

    # 步骤4: 创建数据集
    train_dataset = ANCDataset(train_input, train_target, window_size, stride)
    val_dataset = ANCDataset(val_input, val_target, window_size, stride)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, persistent_workers=True,
                             prefetch_factor=4)  # GPU优化：增加workers和预取
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=8, pin_memory=True, persistent_workers=True,
                           prefetch_factor=4)

    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    # 步骤5: 创建模型
    print(f"创建模型 (type={model_type}, hidden_size={hidden_size})...")
    model = create_model(model_type=model_type, num_channels=18,
                        window_size=window_size, hidden_size=hidden_size)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 步骤6: 训练模型
    print(f"\n开始训练 {band_name} 频段模型...")
    train_losses, val_losses = train_model(model, train_loader, val_loader,
                                          num_epochs=num_epochs, lr=learning_rate, device=device)

    # 步骤7: 保存模型
    model_file = f'results/multiband_model_{band_name}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_mean': input_mean,
        'input_std': input_std,
        'target_mean': target_mean,
        'target_std': target_std,
        'band_name': band_name,
        'lowcut': lowcut,
        'highcut': highcut,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, model_file)
    print(f"模型已保存: {model_file}")

    # 返回归一化参数
    normalization_params = {
        'input_mean': input_mean,
        'input_std': input_std,
        'target_mean': target_mean,
        'target_std': target_std
    }

    return model, normalization_params


def main_multiband():
    """多频段训练主程序"""

    print("="*70)
    print("多频段深度学习ANC系统")
    print("="*70)

    # ========== 配置参数 ==========
    filename = 'data/260107_ddr_32x60_01.dat'  # 使用60秒数据文件
    num_channels = 32
    num_samples = 180000  # 60秒 * 3000Hz = 180000个点
    fs = 3000

    window_size = 300
    stride = 3
    batch_size = 512  # GPU优化：进一步增大batch size（256→512）
    num_epochs = 300
    learning_rate = 0.0005

    # 模型类型选择
    # 'wiener_enhanced': 增强型模型（频域+时域+FIR）- 推荐
    # 'hybrid_linear': 混合线性模型（主要线性+少量非线性）
    # 'linear_fir': 纯线性FIR模型（最接近维纳滤波器）
    # 'tcn': 时间卷积网络（适合因果时序信号）
    # 'cnn_lstm': 原始CNN-LSTM模型
    model_type = 'wiener_enhanced'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"模型类型: {model_type}")

    # 创建结果目录
    os.makedirs('results', exist_ok=True)

    # ========== 加载原始数据 ==========
    print("\n" + "="*70)
    print("加载原始数据")
    print("="*70)
    data = load_data(filename, num_channels, num_samples)

    acc_indices = list(range(6, 24))
    mic_index = 24

    input_signals = data[acc_indices, :]
    target_signal = data[mic_index, :]

    print(f"输入信号形状: {input_signals.shape}")
    print(f"目标信号形状: {target_signal.shape}")

    # ========== 定义频段（根据USE_MULTIBAND开关）==========
    if USE_MULTIBAND:
        print("\n使用多频段训练模式")
        frequency_bands = {
            'low': (20, 150),      # 低频：20-150Hz
            'mid': (150, 250),     # 中频：150-250Hz（拟合效果好）
            'high': (250, 500)     # 高频：250-500Hz
        }
    else:
        print("\n使用全频段训练模式（不分频）")
        frequency_bands = {
            'full': (20, 500)      # 全频段：20-500Hz
        }

    print("\n频段划分:")
    for band_name, (low, high) in frequency_bands.items():
        print(f"  {band_name}: {low}-{high} Hz")

    # ========== 训练各频段模型（使用频段特定的优化配置）==========
    models = {}
    norm_params = {}

    # 为不同频段定义不同的训练配置（根据USE_MULTIBAND开关）
    if USE_MULTIBAND:
        # 多频段模式：每个频段使用不同的训练配置
        band_configs = {
            'low': {
                'stride': 2,           # 减小步长，增加训练样本
                'num_epochs': 400,     # 增加训练轮数
                'hidden_size': 384,    # 增大模型容量
                'learning_rate': 0.0004,
                'model_type': model_type
            },
            'mid': {
                'stride': 2,           # 减小步长，增加训练样本
                'num_epochs': 400,     # 增加训练轮数
                'hidden_size': 384,    # 增大模型容量
                'learning_rate': 0.0004,
                'model_type': model_type
            },
            'high': {
                'stride': 1,           # 最小步长，最多训练样本
                'num_epochs': 600,     # 最多训练轮数
                'hidden_size': 640,    # 最大模型容量
                'learning_rate': 0.0002,  # 更小的学习率，更稳定的收敛
                'model_type': model_type
            }
        }
    else:
        # 全频段模式：使用统一的训练配置
        band_configs = {
            'full': {
                'stride': 1,           # 使用最小步长
                'num_epochs': 500,     # 充分训练
                'hidden_size': 512,    # 中等模型容量
                'learning_rate': 0.0003,
                'model_type': model_type
            }
        }

    print("\n频段特定训练配置:")
    for band_name, config in band_configs.items():
        print(f"  {band_name}: stride={config['stride']}, epochs={config['num_epochs']}, "
              f"hidden_size={config['hidden_size']}, lr={config['learning_rate']}")

    for band_name, (lowcut, highcut) in frequency_bands.items():
        config = band_configs[band_name]
        model, params = train_single_band(
            input_signals, target_signal, band_name, lowcut, highcut,
            window_size=window_size,
            stride=config['stride'],
            batch_size=batch_size,
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            hidden_size=config['hidden_size'],
            model_type=config['model_type'],
            device=device,
            fs=fs
        )
        models[band_name] = model
        norm_params[band_name] = params

    # ========== 生成各频段预测 ==========
    print("\n" + "="*70)
    print("生成各频段预测")
    print("="*70)

    predictions = {}

    for band_name, (lowcut, highcut) in frequency_bands.items():
        print(f"\n预测 {band_name} 频段...")

        # 对输入信号应用带通滤波
        input_filtered = apply_bandpass_filter(input_signals, fs=fs, lowcut=lowcut, highcut=highcut, order=5)

        # 归一化
        params = norm_params[band_name]
        input_norm = (input_filtered - params['input_mean']) / params['input_std']

        # 预测（GPU优化：批量预测）
        model = models[band_name]
        prediction_norm = predict_full_signal(model, input_norm, window_size=window_size,
                                             device=device, batch_size=1024)

        # 反归一化
        prediction = prediction_norm * params['target_std'] + params['target_mean']

        predictions[band_name] = prediction
        print(f"{band_name} 频段预测完成，信号长度: {len(prediction)}")

    # ========== 融合预测结果 ==========
    print("\n" + "="*70)
    print("融合多频段预测")
    print("="*70)

    fused_prediction = fuse_multiband_predictions(predictions, fs=fs)
    print(f"融合完成，信号长度: {len(fused_prediction)}")

    # ========== 评估性能 ==========
    print("\n" + "="*70)
    print("评估ANC性能")
    print("="*70)

    # 计算误差信号
    error_signal = target_signal - fused_prediction

    # 应用A计权
    target_weighted = apply_a_weighting(target_signal)
    error_weighted = apply_a_weighting(error_signal)

    # 计算dB值
    db_before = compute_db_value(target_weighted, fs)
    db_after = compute_db_value(error_weighted, fs)
    anc_reduction = 10 * np.log10((compute_freq_rms(error_weighted, fs, 50, 500)**2) /
                                   (compute_freq_rms(target_weighted, fs, 50, 500)**2))

    print(f"\n多频段融合结果:")
    print(f"  BeforeANC: {db_before:.2f} dBA")
    print(f"  AfterANC: {db_after:.2f} dBA")
    print(f"  ANC降噪量: {anc_reduction:.2f} dBA")
    print(f"  目标: -7.65 dBA")

    if anc_reduction <= -7.65:
        print(f"  达到目标！超出 {abs(anc_reduction + 7.65):.2f} dBA")
    else:
        print(f"  未达到目标，差距 {anc_reduction + 7.65:.2f} dBA")

    # ========== 可视化结果 ==========
    print("\n" + "="*70)
    print("生成可视化结果")
    print("="*70)

    visualize_multiband_results(target_signal, predictions, fused_prediction,
                               error_signal, frequency_bands, fs,
                               db_before, db_after, anc_reduction)

    # ========== 保存结果 ==========
    results = {
        'target_signal': target_signal,
        'fused_prediction': fused_prediction,
        'error_signal': error_signal,
        'predictions': predictions,
        'db_before': db_before,
        'db_after': db_after,
        'anc_reduction': anc_reduction,
        'frequency_bands': frequency_bands
    }

    np.save('results/multiband_results.npy', results)
    print("结果已保存: results/multiband_results.npy")

    print("\n" + "="*70)
    print("多频段训练完成！")
    print("="*70)

    return models, results


def visualize_multiband_results(target_signal, predictions, fused_prediction,
                                error_signal, frequency_bands, fs,
                                db_before, db_after, anc_reduction):
    """可视化多频段结果"""

    # 应用A计权
    target_weighted = apply_a_weighting(target_signal)
    error_weighted = apply_a_weighting(error_signal)
    fused_weighted = apply_a_weighting(fused_prediction)

    # 创建大图 - 调整为3列布局
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    time_axis = np.arange(len(target_signal)) / fs
    show_samples = 1000

    # ===== 第一行：各频段时域对比 =====
    band_names = ['low', 'mid', 'high']
    colors = ['blue', 'green', 'orange']
    band_labels = {'low': '低频', 'mid': '中频', 'high': '高频'}

    for idx, (band_name, color) in enumerate(zip(band_names, colors)):
        ax = fig.add_subplot(gs[0, idx])

        lowcut, highcut = frequency_bands[band_name]
        target_band = apply_bandpass_filter(target_signal, fs=fs, lowcut=lowcut, highcut=highcut)
        pred_band = predictions[band_name]

        ax.plot(time_axis[:show_samples], target_band[:show_samples],
               label='原始', alpha=0.7, linewidth=1.5, color='gray')
        ax.plot(time_axis[:show_samples], pred_band[:show_samples],
               label='预测', alpha=0.8, linewidth=1.5, color=color)

        ax.set_title(f'{band_labels[band_name]} ({lowcut}-{highcut}Hz)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('时间 (s)', fontsize=10)
        ax.set_ylabel('幅值', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # ===== 第二行：各频段PSD对比 =====
    for idx, (band_name, color) in enumerate(zip(band_names, colors)):
        ax = fig.add_subplot(gs[1, idx])

        lowcut, highcut = frequency_bands[band_name]
        target_band = apply_bandpass_filter(target_signal, fs=fs, lowcut=lowcut, highcut=highcut)
        target_band_weighted = apply_a_weighting(target_band)
        pred_band_weighted = apply_a_weighting(predictions[band_name])

        f_target, Pxx_target = welch(target_band_weighted, fs, nperseg=750, noverlap=375)
        f_pred, Pxx_pred = welch(pred_band_weighted, fs, nperseg=750, noverlap=375)

        Pxx_target_db = 10 * np.log10(Pxx_target / 4e-10)
        Pxx_pred_db = 10 * np.log10(Pxx_pred / 4e-10)

        ax.plot(f_target, Pxx_target_db, label='原始', linewidth=2.5, color='gray', alpha=0.7)
        ax.plot(f_pred, Pxx_pred_db, label='预测', linewidth=2.5, color=color, alpha=0.8)

        ax.axvspan(lowcut, highcut, alpha=0.1, color=color)
        ax.set_title(f'{band_labels[band_name]} PSD对比', fontsize=12, fontweight='bold')
        ax.set_xlabel('频率 (Hz)', fontsize=10)
        ax.set_ylabel('PSD (dB/Hz)', fontsize=10)
        ax.set_xlim([20, 500])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # ===== 第三行：融合结果时域对比 =====
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(time_axis[:show_samples], target_signal[:show_samples],
            label='原始麦克风', alpha=0.7, linewidth=1.5, color='blue')
    ax3.plot(time_axis[:show_samples], fused_prediction[:show_samples],
            label='融合预测', alpha=0.8, linewidth=1.5, color='red')
    ax3.set_title('多频段融合 - 时域对比 (前1000点)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('时间 (s)', fontsize=10)
    ax3.set_ylabel('幅值', fontsize=10)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ===== 第四行：融合结果PSD对比 =====
    ax4 = fig.add_subplot(gs[3, :])

    f_target, Pxx_target = welch(target_weighted, fs, nperseg=750, noverlap=375)
    f_fused, Pxx_fused = welch(fused_weighted, fs, nperseg=750, noverlap=375)
    f_error, Pxx_error = welch(error_weighted, fs, nperseg=750, noverlap=375)

    Pxx_target_db = 10 * np.log10(Pxx_target / 4e-10)
    Pxx_fused_db = 10 * np.log10(Pxx_fused / 4e-10)
    Pxx_error_db = 10 * np.log10(Pxx_error / 4e-10)

    ax4.plot(f_target, Pxx_target_db, label=f'原始麦克风 ({db_before:.1f}dBA)',
            linewidth=2.5, color='blue', alpha=0.8)
    ax4.plot(f_fused, Pxx_fused_db, label='融合预测',
            linewidth=2, color='red', alpha=0.7, linestyle='--')
    ax4.plot(f_error, Pxx_error_db, label=f'误差信号 ({db_after:.1f}dBA)',
            linewidth=2.5, color='green', alpha=0.8)

    # 标注频段
    band_colors_map = {'low': 'blue', 'mid': 'green', 'high': 'orange'}
    for band_name, (low, high) in frequency_bands.items():
        ax4.axvspan(low, high, alpha=0.05, color=band_colors_map[band_name])

    # 标注降噪量
    ax4.text(0.98, 0.95, f'ANC降噪量: {anc_reduction:.2f} dBA\n目标: -7.65 dBA',
            transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax4.set_title('多频段融合 - PSD对比 (A计权)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('频率 (Hz)', fontsize=10)
    ax4.set_ylabel('PSD (dB/Hz)', fontsize=10)
    ax4.set_xlim([20, 500])
    ax4.legend(fontsize=10, loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.savefig('results/multiband_anc_result.png', dpi=150, bbox_inches='tight')
    print("多频段结果图已保存: results/multiband_anc_result.png")
    plt.show(block=False)  # 非阻塞模式显示图形


if __name__ == '__main__':
    models, results = main_multiband()

    # ========== 单独显示PSD对比图（用户自定义坐标范围）==========
    print("\n" + "="*70)
    print("生成单独的PSD对比图")
    print("="*70)
    
    # 提取结果数据
    target_signal = results['target_signal']
    fused_prediction = results['fused_prediction']
    error_signal = results['error_signal']
    db_before = results['db_before']
    db_after = results['db_after']
    anc_reduction = results['anc_reduction']
    frequency_bands = results['frequency_bands']
    fs = 3000
    
    # 应用A计权
    from dl_anc_model import apply_a_weighting
    target_weighted = apply_a_weighting(target_signal)
    error_weighted = apply_a_weighting(error_signal)
    fused_weighted = apply_a_weighting(fused_prediction)
    
    # 计算PSD
    f_target, Pxx_target = welch(target_weighted, fs, nperseg=750, noverlap=375)
    f_fused, Pxx_fused = welch(fused_weighted, fs, nperseg=750, noverlap=375)
    f_error, Pxx_error = welch(error_weighted, fs, nperseg=750, noverlap=375)
    
    Pxx_target_db = 10 * np.log10(Pxx_target / 4e-10)
    Pxx_fused_db = 10 * np.log10(Pxx_fused / 4e-10)
    Pxx_error_db = 10 * np.log10(Pxx_error / 4e-10)
    
    # 创建单独的PSD图
    fig_psd = plt.figure(figsize=(14, 8))
    ax_psd = fig_psd.add_subplot(111)
    
    # 绘制PSD曲线
    ax_psd.plot(f_target, Pxx_target_db, label=f'原始麦克风 ({db_before:.1f}dBA)',
                linewidth=3, color='blue', alpha=0.8)
    ax_psd.plot(f_fused, Pxx_fused_db, label='融合预测',
                linewidth=2.5, color='red', alpha=0.7, linestyle='--')
    ax_psd.plot(f_error, Pxx_error_db, label=f'误差信号 ({db_after:.1f}dBA)',
                linewidth=3, color='green', alpha=0.8)
    
    # 标注频段（淡色背景）
    for band_name, (low, high) in frequency_bands.items():
        if 'low' in band_name:
            color = 'blue'
        else:
            color = 'green'
        ax_psd.axvspan(low, high, alpha=0.08, color=color)
    
    # 标注降噪量
    ax_psd.text(0.98, 0.95, f'ANC降噪量: {anc_reduction:.2f} dBA\n目标: -7.65 dBA',
                transform=ax_psd.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, 
                         edgecolor='black', linewidth=1.5))
    
    # 设置坐标范围（用户要求）
    ax_psd.set_xlim([20, 500])
    ax_psd.set_ylim([-30, 20])
    
    # 设置标签和标题
    ax_psd.set_title('多频段深度学习ANC - PSD对比 (A计权)', 
                     fontsize=14, fontweight='bold', pad=15)
    ax_psd.set_xlabel('频率 (Hz)', fontsize=12, fontweight='bold')
    ax_psd.set_ylabel('PSD (dB/Hz)', fontsize=12, fontweight='bold')
    
    # 图例和网格
    ax_psd.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax_psd.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    
    # 保存单独的PSD图
    plt.savefig('results/multiband_psd_only.png', dpi=150, bbox_inches='tight')
    print("单独PSD图已保存: results/multiband_psd_only.png")
    plt.show(block=False)

    # 保持图形窗口打开，等待用户交互
    print("\n" + "="*70)
    print("图形窗口已打开，您可以：")
    print("  - 使用鼠标缩放、平移查看细节")
    print("  - 点击工具栏按钮进行编辑")
    print("  - 关闭所有图形窗口或按Ctrl+C退出程序")
    print("="*70)
    plt.show(block=True)  # 阻塞模式，保持窗口打开直到用户关闭
