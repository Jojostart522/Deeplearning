"""
单独显示多频段ANC的PSD对比图
横坐标: 20-500 Hz
纵坐标: -30 ~ 20 dB
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from dl_anc_model import apply_a_weighting

# 开启交互模式
plt.ion()

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("加载并显示PSD对比图")
print("="*70)

# 加载结果数据
results = np.load('results/multiband_results.npy', allow_pickle=True).item()

target_signal = results['target_signal']
fused_prediction = results['fused_prediction']
error_signal = results['error_signal']
db_before = results['db_before']
db_after = results['db_after']
anc_reduction = results['anc_reduction']
frequency_bands = results['frequency_bands']

fs = 3000

# 应用A计权
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

# 创建图形
fig, ax = plt.subplots(figsize=(14, 8))

# 绘制PSD曲线
ax.plot(f_target, Pxx_target_db, label=f'原始麦克风 ({db_before:.1f}dBA)',
        linewidth=3, color='blue', alpha=0.8)
ax.plot(f_fused, Pxx_fused_db, label='融合预测',
        linewidth=2.5, color='red', alpha=0.7, linestyle='--')
ax.plot(f_error, Pxx_error_db, label=f'误差信号 ({db_after:.1f}dBA)',
        linewidth=3, color='green', alpha=0.8)

# 标注频段（淡色背景）
for band_name, (low, high) in frequency_bands.items():
    if 'low' in band_name:
        color = 'blue'
    else:
        color = 'green'
    ax.axvspan(low, high, alpha=0.08, color=color)

# 标注降噪量
ax.text(0.98, 0.95, f'ANC降噪量: {anc_reduction:.2f} dBA\n目标: -7.65 dBA',
        transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, 
                 edgecolor='black', linewidth=1.5))

# 设置坐标范围
ax.set_xlim([20, 500])
ax.set_ylim([-30, 20])

# 设置标签和标题
ax.set_title('多频段深度学习ANC - PSD对比 (A计权)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('频率 (Hz)', fontsize=12, fontweight='bold')
ax.set_ylabel('PSD (dB/Hz)', fontsize=12, fontweight='bold')

# 图例和网格
ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)

plt.tight_layout()

# 保存图片
plt.savefig('results/multiband_psd_only.png', dpi=150, bbox_inches='tight')
print("\n✅ PSD图已保存: results/multiband_psd_only.png")

# 显示图形
plt.show(block=False)

print("\n" + "="*70)
print("图形窗口已打开！")
print("  - 横坐标: 20-500 Hz")
print("  - 纵坐标: -30 ~ 20 dB")
print("  - 可以使用鼠标缩放、平移")
print("  - 关闭窗口或按Ctrl+C退出")
print("="*70)

# 保持窗口打开
plt.show(block=True)
