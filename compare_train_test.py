"""
对比训练集和测试集的降噪效果
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
print("对比训练集和测试集的ANC降噪效果")
print("="*70)

# ========== 配置参数 ==========
filename = 'data/260107_ddr_32x5_05.dat'
num_channels = 32
num_samples = 15000
fs = 3000
window_size = 300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}\n")

# ========== 加载数据并划分 ==========
print("="*70)
print("加载数据并划分")
print("="*70)

data = load_data(filename, num_channels, num_samples)
acc_indices = list(range(6, 24))
mic_index = 24

input_signals_full = data[acc_indices, :]
target_signal_full = data[mic_index, :]

# 划分：80%训练，20%测试
split_idx = int(0.8 * num_samples)

input_signals_train = input_signals_full[:, :split_idx]
target_signal_train = target_signal_full[:split_idx]

input_signals_test = input_signals_full[:, split_idx:]
target_signal_test = target_signal_full[split_idx:]

print(f"全部数据: {num_samples} 个点")
print(f"训练集: {split_idx} 个点 (0-{split_idx-1})")
print(f"测试集: {num_samples-split_idx} 个点 ({split_idx}-{num_samples-1})\n")

# ========== 加载模型 ==========
print("="*70)
print("加载已训练的模型")
print("="*70)

frequency_bands = {
    'low': (20, 150),
    'mid_high': (150, 500)
}

models = {}
norm_params = {}

for band_name in frequency_bands.keys():
    model_file = f'results/multiband_model_{band_name}.pth'
    try:
        checkpoint = torch.load(model_file, map_location=device)
        print(f"✅ 加载: {band_name}")
        
        norm_params[band_name] = {
            'input_mean': checkpoint['input_mean'],
            'input_std': checkpoint['input_std'],
            'target_mean': checkpoint['target_mean'],
            'target_std': checkpoint['target_std']
        }
        
        # 尝试加载模型
        try:
            model = create_model(model_type='wiener_enhanced', num_channels=18, 
                               window_size=window_size, hidden_size=384)
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            model = ANCModel_CNN_LSTM(num_channels=18, window_size=window_size, hidden_size=256)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(device)
        model.eval()
        models[band_name] = model
        
    except FileNotFoundError:
        print(f"❌ 未找到: {model_file}")
        exit(1)

print()

# ========== 函数：评估ANC性能 ==========
def evaluate_anc(input_signals, target_signal, models, norm_params, frequency_bands, dataset_name):
    """评估ANC性能"""
    print("="*70)
    print(f"评估{dataset_name}ANC性能")
    print("="*70)
    
    predictions = {}
    
    for band_name, (lowcut, highcut) in frequency_bands.items():
        # 带通滤波
        input_filtered = apply_bandpass_filter(input_signals, fs=fs, 
                                              lowcut=lowcut, highcut=highcut, order=5)
        
        # 归一化
        params = norm_params[band_name]
        input_norm = (input_filtered - params['input_mean']) / params['input_std']
        
        # 预测
        model = models[band_name]
        with torch.no_grad():
            prediction_norm = predict_full_signal(model, input_norm, 
                                                 window_size=window_size, device=device)
        
        # 反归一化
        prediction = prediction_norm * params['target_std'] + params['target_mean']
        predictions[band_name] = prediction
    
    # 融合
    fused_prediction = np.zeros_like(target_signal)
    for prediction in predictions.values():
        fused_prediction += prediction
    
    # 计算ANC性能
    error_signal = target_signal - fused_prediction
    
    target_weighted = apply_a_weighting(target_signal)
    error_weighted = apply_a_weighting(error_signal)
    
    db_before = compute_db_value(target_weighted, fs)
    db_after = compute_db_value(error_weighted, fs)
    anc_reduction = 10 * np.log10((compute_freq_rms(error_weighted, fs, 50, 500)**2) /
                                   (compute_freq_rms(target_weighted, fs, 50, 500)**2))
    
    print(f"BeforeANC: {db_before:.2f} dBA")
    print(f"AfterANC:  {db_after:.2f} dBA")
    print(f"ANC降噪量: {anc_reduction:.2f} dBA")
    print(f"与目标差距: {anc_reduction + 7.65:.2f} dBA\n")
    
    return {
        'predictions': predictions,
        'fused_prediction': fused_prediction,
        'error_signal': error_signal,
        'db_before': db_before,
        'db_after': db_after,
        'anc_reduction': anc_reduction
    }

# ========== 评估训练集 ==========
train_results = evaluate_anc(input_signals_train, target_signal_train, 
                            models, norm_params, frequency_bands, "训练集")

# ========== 评估测试集 ==========
test_results = evaluate_anc(input_signals_test, target_signal_test, 
                           models, norm_params, frequency_bands, "测试集")

# ========== 对比总结 ==========
print("="*70)
print("训练集 vs 测试集 对比总结")
print("="*70)

print(f"\n{'指标':<20} {'训练集':<15} {'测试集':<15} {'差异':<15}")
print("-"*70)
print(f"{'BeforeANC (dBA)':<20} {train_results['db_before']:<15.2f} {test_results['db_before']:<15.2f} {test_results['db_before']-train_results['db_before']:<15.2f}")
print(f"{'AfterANC (dBA)':<20} {train_results['db_after']:<15.2f} {test_results['db_after']:<15.2f} {test_results['db_after']-train_results['db_after']:<15.2f}")
print(f"{'ANC降噪量 (dBA)':<20} {train_results['anc_reduction']:<15.2f} {test_results['anc_reduction']:<15.2f} {test_results['anc_reduction']-train_results['anc_reduction']:<15.2f}")
print(f"{'与目标差距 (dBA)':<20} {train_results['anc_reduction']+7.65:<15.2f} {test_results['anc_reduction']+7.65:<15.2f} -")

print("\n" + "="*70)
print("结论")
print("="*70)

performance_gap = train_results['anc_reduction'] - test_results['anc_reduction']
print(f"训练集比测试集性能好 {abs(performance_gap):.2f} dBA")

if abs(performance_gap) > 2:
    print("⚠️  存在明显过拟合，模型泛化能力不足")
elif abs(performance_gap) > 1:
    print("⚠️  存在一定程度过拟合")
else:
    print("✅ 模型泛化能力良好")

print(f"\n训练集降噪效果: {train_results['anc_reduction']:.2f} dBA")
print(f"测试集降噪效果: {test_results['anc_reduction']:.2f} dBA")
print(f"目标: -7.65 dBA")

print("\n程序执行完成！")
