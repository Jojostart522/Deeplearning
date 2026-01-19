"""
对比维纳滤波和深度学习模型的ANC性能
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
import sys

# 导入维纳滤波实现
from main_analysis import (
    load_data, apply_a_weighting, compute_db_value,
    compute_freq_rms, compute_wiener_filter, apply_miso_filter
)

# 导入深度学习模型
from dl_anc_model import (
    ANCModel_CNN_LSTM, predict_full_signal, evaluate_anc_performance
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def compare_methods():
    """对比维纳滤波和深度学习方法"""

    print("="*70)
    print("维纳滤波 vs 深度学习 - ANC性能对比")
    print("="*70)

    # ========== 加载数据 ==========
    print("\n步骤1: 加载数据...")
    filename = 'data/260107_ddr_32x5_05.dat'
    data = load_data(filename, num_channels=32, num_samples=15000)

    acc_indices = list(range(6, 24))  # 加速度传感器
    mic_index = 24  # 麦克风

    input_signals = data[acc_indices, :]
    target_signal = data[mic_index, :]

    fs = 3000

    # ========== 方法1: 维纳滤波 ==========
    print("\n" + "="*70)
    print("方法1: 维纳滤波")
    print("="*70)

    print("计算维纳滤波器...")
    wiener_filters = compute_wiener_filter(input_signals, target_signal, filter_length=300)

    print("应用维纳滤波器...")
    wiener_estimated = apply_miso_filter(input_signals, wiener_filters)
    wiener_error = target_signal - wiener_estimated

    # A计权
    target_weighted = apply_a_weighting(target_signal)
    wiener_error_weighted = apply_a_weighting(wiener_error)

    # 计算性能
    wiener_db_before = compute_db_value(target_weighted, fs)
    wiener_db_after = compute_db_value(wiener_error_weighted, fs)
    wiener_anc = 10 * np.log10((compute_freq_rms(wiener_error_weighted, fs, 50, 500)**2) /
                                (compute_freq_rms(target_weighted, fs, 50, 500)**2))

    print(f"\n维纳滤波结果:")
    print(f"  BeforeANC: {wiener_db_before:.2f} dBA")
    print(f"  AfterANC: {wiener_db_after:.2f} dBA")
    print(f"  ANC降噪量: {wiener_anc:.2f} dBA")

    # ========== 方法2: 深度学习 ==========
    print("\n" + "="*70)
    print("方法2: 深度学习")
    print("="*70)

    # 检查是否有训练好的模型
    import os
    model_path = 'results/dl_anc_model.pth'

    if not os.path.exists(model_path):
        print("错误: 未找到训练好的深度学习模型！")
        print("请先运行 'python dl_anc_model.py' 训练模型。")
        return

    print(f"加载训练好的模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ANCModel_CNN_LSTM(num_channels=18, window_size=300, hidden_size=128)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 加载归一化参数
    input_mean = checkpoint['input_mean']
    input_std = checkpoint['input_std']
    target_mean = checkpoint['target_mean']
    target_std = checkpoint['target_std']

    # 归一化输入
    input_signals_norm = (input_signals - input_mean) / input_std

    print("使用深度学习模型预测...")
    dl_estimated_norm = predict_full_signal(model, input_signals_norm, window_size=300, device=device)

    # 反归一化
    dl_estimated = dl_estimated_norm * target_std + target_mean

    # 评估性能
    dl_results = evaluate_anc_performance(target_signal, dl_estimated, fs)

    print(f"\n深度学习结果:")
    print(f"  BeforeANC: {dl_results['db_before']:.2f} dBA")
    print(f"  AfterANC: {dl_results['db_after']:.2f} dBA")
    print(f"  ANC降噪量: {dl_results['anc_reduction']:.2f} dBA")

    # ========== 对比分析 ==========
    print("\n" + "="*70)
    print("性能对比")
    print("="*70)

    print(f"\n{'指标':<20} {'维纳滤波':<15} {'深度学习':<15} {'差异':<15}")
    print("-"*70)
    print(f"{'BeforeANC (dBA)':<20} {wiener_db_before:<15.2f} {dl_results['db_before']:<15.2f} "
          f"{dl_results['db_before'] - wiener_db_before:<15.2f}")
    print(f"{'AfterANC (dBA)':<20} {wiener_db_after:<15.2f} {dl_results['db_after']:<15.2f} "
          f"{dl_results['db_after'] - wiener_db_after:<15.2f}")
    print(f"{'ANC降噪量 (dBA)':<20} {wiener_anc:<15.2f} {dl_results['anc_reduction']:<15.2f} "
          f"{dl_results['anc_reduction'] - wiener_anc:<15.2f}")

    # 判断哪个方法更好
    print("\n结论:")
    if dl_results['anc_reduction'] < wiener_anc:
        improvement = wiener_anc - dl_results['anc_reduction']
        print(f"✅ 深度学习模型性能更好，额外降噪 {improvement:.2f} dBA")
    elif dl_results['anc_reduction'] > wiener_anc:
        degradation = dl_results['anc_reduction'] - wiener_anc
        print(f"⚠️ 维纳滤波性能更好，深度学习差 {degradation:.2f} dBA")
    else:
        print("两种方法性能相当")

    # ========== 可视化对比 ==========
    print("\n" + "="*70)
    print("生成对比图...")
    print("="*70)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    time_axis = np.arange(len(target_signal)) / fs
    show_samples = 1000

    # 1. 时域对比 - 维纳滤波
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_axis[:show_samples], target_signal[:show_samples],
             label='原始麦克风', alpha=0.7, color='blue', linewidth=1.5)
    ax1.plot(time_axis[:show_samples], wiener_estimated[:show_samples],
             label='维纳估计', alpha=0.7, color='red', linewidth=1.5)
    ax1.set_title('维纳滤波 - 时域对比 (前1000点)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 时域对比 - 深度学习
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_axis[:show_samples], target_signal[:show_samples],
             label='原始麦克风', alpha=0.7, color='blue', linewidth=1.5)
    ax2.plot(time_axis[:show_samples], dl_estimated[:show_samples],
             label='DL估计', alpha=0.7, color='orange', linewidth=1.5)
    ax2.set_title('深度学习 - 时域对比 (前1000点)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('幅值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 误差信号对比
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time_axis[:show_samples], wiener_error[:show_samples],
             label='维纳误差', alpha=0.7, color='green', linewidth=1.5)
    ax3.set_title('维纳滤波 - 误差信号', fontsize=12, fontweight='bold')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('幅值')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_axis[:show_samples], dl_results['error_signal'][:show_samples],
             label='DL误差', alpha=0.7, color='purple', linewidth=1.5)
    ax4.set_title('深度学习 - 误差信号', fontsize=12, fontweight='bold')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('幅值')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 4. PSD对比 - 维纳滤波
    ax5 = fig.add_subplot(gs[2, 0])
    f_w, Pxx_w_target = signal.welch(target_weighted, fs, nperseg=750, noverlap=375)
    f_w, Pxx_w_error = signal.welch(wiener_error_weighted, fs, nperseg=750, noverlap=375)

    Pxx_w_target_db = 10 * np.log10(Pxx_w_target / 4e-10)
    Pxx_w_error_db = 10 * np.log10(Pxx_w_error / 4e-10)

    ax5.plot(f_w, Pxx_w_target_db, label=f'原始 ({wiener_db_before:.1f}dBA)',
             linewidth=2, color='blue')
    ax5.plot(f_w, Pxx_w_error_db, label=f'误差 ({wiener_db_after:.1f}dBA)',
             linewidth=2, color='green')
    ax5.set_title(f'维纳滤波 - PSD对比 (降噪{wiener_anc:.2f}dBA)',
                  fontsize=12, fontweight='bold')
    ax5.set_xlabel('频率 (Hz)')
    ax5.set_ylabel('PSD (dB/Hz)')
    ax5.set_xlim([20, 500])
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 5. PSD对比 - 深度学习
    ax6 = fig.add_subplot(gs[2, 1])
    f_dl, Pxx_dl_target = signal.welch(dl_results['target_weighted'], fs, nperseg=750, noverlap=375)
    f_dl, Pxx_dl_error = signal.welch(dl_results['error_weighted'], fs, nperseg=750, noverlap=375)

    Pxx_dl_target_db = 10 * np.log10(Pxx_dl_target / 4e-10)
    Pxx_dl_error_db = 10 * np.log10(Pxx_dl_error / 4e-10)

    ax6.plot(f_dl, Pxx_dl_target_db, label=f'原始 ({dl_results["db_before"]:.1f}dBA)',
             linewidth=2, color='blue')
    ax6.plot(f_dl, Pxx_dl_error_db, label=f'误差 ({dl_results["db_after"]:.1f}dBA)',
             linewidth=2, color='purple')
    ax6.set_title(f'深度学习 - PSD对比 (降噪{dl_results["anc_reduction"]:.2f}dBA)',
                  fontsize=12, fontweight='bold')
    ax6.set_xlabel('频率 (Hz)')
    ax6.set_ylabel('PSD (dB/Hz)')
    ax6.set_xlim([20, 500])
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 6. 性能对比柱状图
    ax7 = fig.add_subplot(gs[3, :])

    methods = ['维纳滤波', '深度学习']
    before_values = [wiener_db_before, dl_results['db_before']]
    after_values = [wiener_db_after, dl_results['db_after']]
    reduction_values = [wiener_anc, dl_results['anc_reduction']]

    x = np.arange(len(methods))
    width = 0.25

    bars1 = ax7.bar(x - width, before_values, width, label='BeforeANC', color='blue', alpha=0.7)
    bars2 = ax7.bar(x, after_values, width, label='AfterANC', color='green', alpha=0.7)
    bars3 = ax7.bar(x + width, reduction_values, width, label='ANC降噪量', color='red', alpha=0.7)

    ax7.set_ylabel('dBA')
    ax7.set_title('性能对比总结', fontsize=14, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(methods)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 在柱状图上添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9)

    plt.savefig('results/comparison_wiener_vs_dl.png', dpi=150, bbox_inches='tight')
    print("对比图已保存: results/comparison_wiener_vs_dl.png")
    plt.close()

    # ========== 保存对比结果 ==========
    comparison_results = {
        'wiener': {
            'db_before': wiener_db_before,
            'db_after': wiener_db_after,
            'anc_reduction': wiener_anc
        },
        'deep_learning': {
            'db_before': dl_results['db_before'],
            'db_after': dl_results['db_after'],
            'anc_reduction': dl_results['anc_reduction']
        }
    }

    np.save('results/comparison_results.npy', comparison_results)
    print("对比结果已保存: results/comparison_results.npy")

    print("\n" + "="*70)
    print("对比完成！")
    print("="*70)

    return comparison_results

if __name__ == '__main__':
    results = compare_methods()
