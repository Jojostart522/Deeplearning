"""
使用测试集数据评估多频段ANC模型的真实降噪效果
不修改原程序，独立评估脚本
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import welch
from dl_anc_model import (
    load_data, apply_a_weighting, compute_db_value, compute_freq_rms,
    apply_bandpass_filter, predict_full_signal, ANCModel_CNN_LSTM
)
from dl_anc_model_wiener_enhanced import create_model

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.ion()

print("="*70)
print("多频段ANC - 测试集真实降噪效果评估")
print("="*70)

# ========== 配置参数 ==========
filename = 'data/260107_ddr_32x5_05.dat'
num_channels = 32
num_samples = 15000
fs = 3000
window_size = 300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ========== 加载原始数据并划分 ==========
print("\n" + "="*70)
print("加载数据并划分训练集/测试集")
print("="*70)

data = load_data(filename, num_channels, num_samples)
acc_indices = list(range(6, 24))
mic_index = 24

input_signals_full = data[acc_indices, :]
target_signal_full = data[mic_index, :]

# 划分：80%训练，20%测试
split_idx = int(0.8 * num_samples)

# 只使用测试集数据（模型从未见过的数据）
input_signals_test = input_signals_full[:, split_idx:]
target_signal_test = target_signal_full[split_idx:]

print(f"原始数据: {num_samples} 个点")
print(f"训练集: {split_idx} 个点 (索引 0-{split_idx-1})")
print(f"测试集: {num_samples-split_idx} 个点 (索引 {split_idx}-{num_samples-1})")
print(f"\n✅ 使用测试集数据评估: {len(target_signal_test)} 个点")

# ========== 加载已训练的模型 ==========
print("\n" + "="*70)
print("加载已训练的多频段模型")
print("="*70)

# 根据实际保存的模型定义频段
frequency_bands = {
    'low': (20, 150),
    'mid_high': (150, 500)
}

models = {}
norm_params = {}
models_loaded = True

for band_name in frequency_bands.keys():
    model_file = f'results/multiband_model_{band_name}.pth'
    try:
        checkpoint = torch.load(model_file, map_location=device)
        print(f"✅ 加载模型: {band_name} - {model_file}")
        
        # 提取归一化参数
        norm_params[band_name] = {
            'input_mean': checkpoint['input_mean'],
            'input_std': checkpoint['input_std'],
            'target_mean': checkpoint['target_mean'],
            'target_std': checkpoint['target_std']
        }
        
        # 重建模型（先尝试加载wiener_enhanced，失败则尝试CNN-LSTM）
        model_type = checkpoint.get('model_type', 'wiener_enhanced')
        hidden_size = 384
        
        print(f"  尝试加载模型类型: {model_type}, hidden_size: {hidden_size}")
        
        # 先尝试 wiener_enhanced
        try:
            model = create_model(model_type='wiener_enhanced', num_channels=18, 
                               window_size=window_size, hidden_size=hidden_size)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  ✅ 成功加载 wiener_enhanced 模型")
        except (RuntimeError, KeyError) as e:
            # 如果失败，尝试 CNN-LSTM
            print(f"  ⚠️ wiener_enhanced 加载失败，尝试 CNN-LSTM...")
            try:
                # 尝试不同的 hidden_size
                for hs in [256, 384, 512]:
                    try:
                        model = ANCModel_CNN_LSTM(num_channels=18, window_size=window_size, hidden_size=hs)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        hidden_size = hs
                        print(f"  ✅ 成功加载 CNN-LSTM 模型 (hidden_size={hs})")
                        break
                    except RuntimeError:
                        continue
            except Exception as e2:
                print(f"  ❌ 加载失败: {str(e2)[:100]}")
                models_loaded = False
                continue
        
        model = model.to(device)
        model.eval()
        models[band_name] = model
        
    except FileNotFoundError:
        print(f"❌ 未找到模型文件: {model_file}")
        models_loaded = False

if not models_loaded:
    print("\n⚠️ 请先运行 dl_anc_multiband.py 训练模型")
    exit(1)

# ========== 在测试集上生成预测 ==========
print("\n" + "="*70)
print("在测试集上生成预测（模型从未见过的数据）")
print("="*70)

predictions = {}

for band_name, (lowcut, highcut) in frequency_bands.items():
    print(f"\n预测 {band_name} 频段 ({lowcut}-{highcut} Hz)...")
    
    # 1. 对测试集应用带通滤波
    input_filtered = apply_bandpass_filter(input_signals_test, fs=fs, 
                                          lowcut=lowcut, highcut=highcut, order=5)
    
    # 2. 归一化（使用训练时的参数）
    params = norm_params[band_name]
    input_norm = (input_filtered - params['input_mean']) / params['input_std']
    
    # 3. 预测
    model = models[band_name]
    with torch.no_grad():
        prediction_norm = predict_full_signal(model, input_norm, 
                                             window_size=window_size, device=device)
    
    # 4. 反归一化
    prediction = prediction_norm * params['target_std'] + params['target_mean']
    
    predictions[band_name] = prediction
    print(f"  ✅ {band_name} 频段预测完成，信号长度: {len(prediction)}")

# ========== 融合多频段预测 ==========
print("\n" + "="*70)
print("融合多频段预测")
print("="*70)

fused_prediction = np.zeros_like(target_signal_test)
for band_name, prediction in predictions.items():
    fused_prediction += prediction

print(f"✅ 融合完成，信号长度: {len(fused_prediction)}")

# ========== 评估测试集性能 ==========
print("\n" + "="*70)
print("评估测试集ANC性能（真实泛化能力）")
print("="*70)

# 计算误差信号
error_signal = target_signal_test - fused_prediction

# 应用A计权
target_weighted = apply_a_weighting(target_signal_test)
error_weighted = apply_a_weighting(error_signal)

# 计算dB值
db_before = compute_db_value(target_weighted, fs)
db_after = compute_db_value(error_weighted, fs)
anc_reduction = 10 * np.log10((compute_freq_rms(error_weighted, fs, 50, 500)**2) /
                               (compute_freq_rms(target_weighted, fs, 50, 500)**2))

print(f"\n【测试集ANC性能】")
print(f"  BeforeANC: {db_before:.2f} dBA")
print(f"  AfterANC: {db_after:.2f} dBA")
print(f"  ANC降噪量: {anc_reduction:.2f} dBA")
print(f"  目标: -7.65 dBA")

if anc_reduction <= -7.65:
    print(f"  ✅ 达到目标！超出 {abs(anc_reduction + 7.65):.2f} dBA")
else:
    print(f"  ⚠️ 未达到目标，差距 {anc_reduction + 7.65:.2f} dBA")

# ========== 生成PSD对比图 ==========
print("\n" + "="*70)
print("生成测试集PSD对比图")
print("="*70)

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

# 标注频段
for band_name, (low, high) in frequency_bands.items():
    if 'low' in band_name:
        color = 'blue'
    else:
        color = 'green'
    ax.axvspan(low, high, alpha=0.08, color=color)

# 标注降噪量
ax.text(0.98, 0.95, f'测试集ANC降噪量: {anc_reduction:.2f} dBA\n目标: -7.65 dBA',
        transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, 
                 edgecolor='black', linewidth=1.5))

# 设置坐标范围
ax.set_xlim([20, 500])
ax.set_ylim([-30, 20])

# 设置标签和标题
ax.set_title('多频段深度学习ANC - 测试集PSD对比 (A计权)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('频率 (Hz)', fontsize=12, fontweight='bold')
ax.set_ylabel('PSD (dB/Hz)', fontsize=12, fontweight='bold')

# 图例和网格
ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)

plt.tight_layout()

# 保存图片
plt.savefig('results/testset_psd_result.png', dpi=150, bbox_inches='tight')
print("\n✅ 测试集PSD图已保存: results/testset_psd_result.png")

# 显示图形
plt.show(block=False)

print("\n" + "="*70)
print("测试集评估完成！")
print("="*70)
print("\n图形窗口已打开:")
print("  - 横坐标: 20-500 Hz")
print("  - 纵坐标: -30 ~ 20 dB")
print("  - 可以使用鼠标缩放、平移")
print("  - 关闭窗口或按回车键退出")
print("="*70)

input("\n按回车键退出...")
