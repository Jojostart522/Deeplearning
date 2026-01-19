"""
对比不同模型类型的ANC性能
测试: linear_fir, hybrid_linear, wiener_enhanced, cnn_lstm
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import signal
import os
import time

from dl_anc_model import (
    load_data, apply_a_weighting, compute_db_value, compute_freq_rms,
    apply_bandpass_filter, ANCDataset, train_model, predict_full_signal
)

from dl_anc_model_wiener_enhanced import create_model

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def train_and_evaluate_single_model(model_type, input_signals, target_signal,
                                   window_size=300, stride=2, batch_size=32,
                                   num_epochs=200, learning_rate=0.0005,
                                   hidden_size=256, device='cpu', fs=3000):
    """
    训练并评估单个模型
    """
    print("\n" + "="*70)
    print(f"模型类型: {model_type}")
    print("="*70)

    # 数据归一化
    input_mean = np.mean(input_signals, axis=1, keepdims=True)
    input_std = np.std(input_signals, axis=1, keepdims=True) + 1e-8
    target_mean = np.mean(target_signal)
    target_std = np.std(target_signal) + 1e-8

    input_norm = (input_signals - input_mean) / input_std
    target_norm = (target_signal - target_mean) / target_std

    # 划分训练集和验证集（使用全部数据，模仿维纳滤波）
    val_split = int(0.9 * len(target_norm))
    train_input = input_norm[:, :val_split]
    train_target = target_norm[:val_split]
    val_input = input_norm[:, val_split:]
    val_target = target_norm[val_split:]

    # 创建数据集
    train_dataset = ANCDataset(train_input, train_target, window_size, stride)
    val_dataset = ANCDataset(val_input, val_target, window_size, stride)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    # 创建模型
    model = create_model(model_type=model_type, num_channels=18,
                        window_size=window_size, hidden_size=hidden_size)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 训练模型
    start_time = time.time()
    train_losses, val_losses = train_model(model, train_loader, val_loader,
                                          num_epochs=num_epochs, lr=learning_rate,
                                          device=device)
    training_time = time.time() - start_time
    print(f"训练时间: {training_time:.1f} 秒")

    # 预测（在全部数据上）
    prediction_norm = predict_full_signal(model, input_norm, window_size=window_size,
                                         device=device)
    prediction = prediction_norm * target_std + target_mean

    # 计算误差
    error_signal = target_signal - prediction

    # A计权
    target_weighted = apply_a_weighting(target_signal)
    error_weighted = apply_a_weighting(error_signal)

    # 计算性能指标
    db_before = compute_db_value(target_weighted, fs)
    db_after = compute_db_value(error_weighted, fs)
    anc_reduction = 10 * np.log10((compute_freq_rms(error_weighted, fs, 50, 500)**2) /
                                   (compute_freq_rms(target_weighted, fs, 50, 500)**2))

    print(f"\n性能结果:")
    print(f"  BeforeANC: {db_before:.2f} dBA")
    print(f"  AfterANC: {db_after:.2f} dBA")
    print(f"  ANC降噪量: {anc_reduction:.2f} dBA")

    results = {
        'model_type': model_type,
        'prediction': prediction,
        'error_signal': error_signal,
        'db_before': db_before,
        'db_after': db_after,
        'anc_reduction': anc_reduction,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_time': training_time,
        'num_params': total_params
    }

    return results


def compare_all_models():
    """对比所有模型类型"""

    print("="*70)
    print("对比不同模型类型的ANC性能")
    print("="*70)

    # 加载数据
    print("\n加载数据...")
    filename = 'data/260107_ddr_32x5_05.dat'
    data = load_data(filename, num_channels=32, num_samples=15000)

    acc_indices = list(range(6, 24))
    mic_index = 24

    input_signals = data[acc_indices, :]
    target_signal = data[mic_index, :]
    fs = 3000

    print(f"输入信号形状: {input_signals.shape}")
    print(f"目标信号形状: {target_signal.shape}")

    # 配置参数
    window_size = 300
    stride = 2
    batch_size = 32
    num_epochs = 200  # 减少轮数以加快对比
    learning_rate = 0.0005
    hidden_size = 256

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 测试的模型类型
    model_types = [
        'linear_fir',        # 纯线性FIR（最简单）
        'hybrid_linear',     # 混合线性
        'tcn',               # 时间卷积网络（因果时序）
        'cnn_lstm',          # 原始CNN-LSTM
        'wiener_enhanced'    # 增强型（最复杂）
    ]

    # 训练和评估每个模型
    all_results = {}

    for model_type in model_types:
        try:
            results = train_and_evaluate_single_model(
                model_type, input_signals, target_signal,
                window_size=window_size, stride=stride, batch_size=batch_size,
                num_epochs=num_epochs, learning_rate=learning_rate,
                hidden_size=hidden_size, device=device, fs=fs
            )
            all_results[model_type] = results
        except Exception as e:
            print(f"\n错误: 模型 {model_type} 训练失败: {e}")
            continue

    # 可视化对比结果
    print("\n" + "="*70)
    print("生成对比图...")
    print("="*70)

    visualize_comparison(target_signal, all_results, fs)

    # 打印总结
    print("\n" + "="*70)
    print("性能总结")
    print("="*70)

    print(f"\n{'模型类型':<20} {'参数量':<15} {'训练时间(s)':<15} {'ANC降噪(dBA)':<15}")
    print("-"*70)

    for model_type, results in all_results.items():
        print(f"{model_type:<20} {results['num_params']:<15,} "
              f"{results['training_time']:<15.1f} {results['anc_reduction']:<15.2f}")

    # 找出最佳模型
    best_model = min(all_results.items(), key=lambda x: x[1]['anc_reduction'])
    print(f"\n最佳模型: {best_model[0]} (ANC降噪: {best_model[1]['anc_reduction']:.2f} dBA)")

    # 与维纳滤波器对比
    wiener_target = -7.65
    print(f"\n维纳滤波器: {wiener_target:.2f} dBA")
    print(f"最佳深度学习: {best_model[1]['anc_reduction']:.2f} dBA")
    print(f"差距: {best_model[1]['anc_reduction'] - wiener_target:.2f} dBA")

    # 保存结果
    os.makedirs('results', exist_ok=True)
    np.save('results/model_comparison_results.npy', all_results)
    print("\n结果已保存: results/model_comparison_results.npy")

    return all_results


def visualize_comparison(target_signal, all_results, fs):
    """可视化对比结果"""

    num_models = len(all_results)
    fig = plt.figure(figsize=(18, 4 * num_models))

    time_axis = np.arange(len(target_signal)) / fs
    show_samples = 1000

    # 为每个模型创建一行图
    for idx, (model_type, results) in enumerate(all_results.items()):
        # 时域对比
        ax1 = plt.subplot(num_models, 3, idx * 3 + 1)
        ax1.plot(time_axis[:show_samples], target_signal[:show_samples],
                label='原始', alpha=0.7, linewidth=1.5, color='blue')
        ax1.plot(time_axis[:show_samples], results['prediction'][:show_samples],
                label='预测', alpha=0.7, linewidth=1.5, color='red')
        ax1.set_title(f'{model_type} - 时域对比', fontsize=11, fontweight='bold')
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('幅值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 误差信号
        ax2 = plt.subplot(num_models, 3, idx * 3 + 2)
        ax2.plot(time_axis[:show_samples], results['error_signal'][:show_samples],
                alpha=0.7, linewidth=1.5, color='green')
        ax2.set_title(f'{model_type} - 误差信号', fontsize=11, fontweight='bold')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('幅值')
        ax2.grid(True, alpha=0.3)

        # PSD对比
        ax3 = plt.subplot(num_models, 3, idx * 3 + 3)

        target_weighted = apply_a_weighting(target_signal)
        error_weighted = apply_a_weighting(results['error_signal'])

        f_target, Pxx_target = signal.welch(target_weighted, fs, nperseg=750, noverlap=375)
        f_error, Pxx_error = signal.welch(error_weighted, fs, nperseg=750, noverlap=375)

        Pxx_target_db = 10 * np.log10(Pxx_target / 4e-10)
        Pxx_error_db = 10 * np.log10(Pxx_error / 4e-10)

        ax3.plot(f_target, Pxx_target_db,
                label=f'原始 ({results["db_before"]:.1f}dBA)',
                linewidth=2, color='blue')
        ax3.plot(f_error, Pxx_error_db,
                label=f'误差 ({results["db_after"]:.1f}dBA)',
                linewidth=2, color='green')

        ax3.set_title(f'{model_type} - PSD (降噪{results["anc_reduction"]:.2f}dBA)',
                     fontsize=11, fontweight='bold')
        ax3.set_xlabel('频率 (Hz)')
        ax3.set_ylabel('PSD (dB/Hz)')
        ax3.set_xlim([20, 500])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/model_types_comparison.png', dpi=150, bbox_inches='tight')
    print("对比图已保存: results/model_types_comparison.png")
    plt.close()

    # 创建性能对比柱状图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    model_names = list(all_results.keys())
    anc_reductions = [all_results[m]['anc_reduction'] for m in model_names]
    training_times = [all_results[m]['training_time'] for m in model_names]
    num_params = [all_results[m]['num_params'] for m in model_names]

    # ANC降噪量对比
    ax1 = axes[0]
    bars = ax1.bar(model_names, anc_reductions, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    ax1.axhline(y=-7.65, color='black', linestyle='--', linewidth=2, label='维纳滤波器')
    ax1.set_ylabel('ANC降噪量 (dBA)')
    ax1.set_title('ANC降噪性能对比', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom' if height < 0 else 'top')

    # 训练时间对比
    ax2 = axes[1]
    ax2.bar(model_names, training_times, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    ax2.set_ylabel('训练时间 (秒)')
    ax2.set_title('训练时间对比', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 参数量对比
    ax3 = axes[2]
    ax3.bar(model_names, [p/1000 for p in num_params],
           color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    ax3.set_ylabel('参数量 (千)')
    ax3.set_title('模型参数量对比', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/model_performance_bars.png', dpi=150, bbox_inches='tight')
    print("性能柱状图已保存: results/model_performance_bars.png")
    plt.close()


if __name__ == '__main__':
    results = compare_all_models()
