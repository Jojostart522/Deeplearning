import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, linalg
import warnings

# 忽略字体警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*Font.*not found.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing.*')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ============================================================================
# 常量定义
# ============================================================================

# MIC加权滤波器系数（A计权）
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
# 基础函数：数据读取
# ============================================================================

def load_data(filename, num_channels=32, num_samples=15000):
    """
    读取二进制数据文件
    
    Parameters:
    -----------
    filename : str
        数据文件路径
    num_channels : int
        通道数，默认32
    num_samples : int
        每通道采样点数，默认15000
    
    Returns:
    --------
    data : ndarray
        形状为 (num_channels, num_samples) 的数据数组
    """
    print(f"正在读取数据文件: {filename}")
    raw_data = np.fromfile(filename, dtype=np.float32)
    print(f"读取的数据总长度: {len(raw_data)}")
    
    data = raw_data.reshape((num_channels, num_samples))
    print(f"数据形状: {data.shape}")
    
    return data


# ============================================================================
# 基础函数：A计权（MIC加权）
# ============================================================================

def apply_a_weighting(data):
    """
    对数据应用A计权（MIC加权）
    
    Parameters:
    -----------
    data : ndarray
        输入数据，可以是1D或2D数组
        - 1D: 形状为 (num_samples,)
        - 2D: 形状为 (num_channels, num_samples)
    
    Returns:
    --------
    filtered_data : ndarray
        A计权后的数据，形状与输入相同
    """
    b = MIC_WEIGHT_B
    a = MIC_WEIGHT_A
    
    if len(data.shape) == 1:
        # 一维数据
        filtered_data = signal.lfilter(b, a, data)
    elif len(data.shape) == 2:
        # 二维数据，对每个通道分别滤波
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i, :] = signal.lfilter(b, a, data[i, :])
    else:
        raise ValueError("数据必须是1D或2D数组")
    
    return filtered_data


# ============================================================================
# 基础函数：频域RMS计算
# ============================================================================

def compute_freq_rms(signal_data, fs, f_min, f_max):
    """
    计算指定频率范围内的RMS值（使用FFT）
    
    Parameters:
    -----------
    signal_data : ndarray
        输入信号，1D数组
    fs : float
        采样频率 (Hz)
    f_min : float
        最小频率 (Hz)
    f_max : float
        最大频率 (Hz)
    
    Returns:
    --------
    rms : float
        指定频率范围内的RMS值
    """
    n = len(signal_data)
    
    # FFT
    fft_vals = np.fft.fft(signal_data)
    fft_mag = np.abs(fft_vals) / fs
    
    # 频率轴
    df = fs / n
    freqs = np.arange(n) * df
    
    # 单边谱
    half_n = n // 2
    fft_mag_half = fft_mag[:half_n]
    freqs_half = freqs[:half_n]
    
    # 选择频率范围
    idx = (freqs_half >= f_min) & (freqs_half <= f_max)
    
    if np.any(idx):
        mag_selected = fft_mag_half[idx]
        # 计算RMS（单边谱需乘以2）
        rms_sq = np.sum(mag_selected**2 * df**2 * 2)
        rms = np.sqrt(rms_sq)
    else:
        rms = 0.0
    
    return rms


# ============================================================================
# 基础函数：dB值计算
# ============================================================================

def compute_db_value(signal_data, fs, f_min=50, f_max=500, reference=4e-10):
    """
    计算信号在指定频率范围内的dB值
    
    Parameters:
    -----------
    signal_data : ndarray
        输入信号，1D数组
    fs : float
        采样频率 (Hz)
    f_min : float
        最小频率 (Hz)，默认50
    f_max : float
        最大频率 (Hz)，默认500
    reference : float
        参考值，默认4e-10
    
    Returns:
    --------
    db : float
        dB值
    """
    rms = compute_freq_rms(signal_data, fs, f_min, f_max)
    db = 10 * np.log10((rms**2) / reference)
    return db


# ============================================================================
# 基础函数：PSD计算
# ============================================================================

def compute_psd(signal_data, fs, win_length, noverlap):
    """
    计算单通道的功率谱密度（PSD）
    
    Parameters:
    -----------
    signal_data : ndarray
        输入信号，1D数组
    fs : float
        采样频率 (Hz)
    win_length : int
        窗口长度
    noverlap : int
        重叠样本数
    
    Returns:
    --------
    f : ndarray
        频率数组
    Pxx : ndarray
        功率谱密度数组
    """
    # 确保是1D数组
    if len(signal_data.shape) > 1:
        signal_data = signal_data.flatten()
    
    # 使用Welch方法计算PSD
    f, Pxx = signal.welch(
        signal_data, fs, 
        window='hann',
        nperseg=win_length, 
        noverlap=noverlap,
        nfft=win_length
    )
    
    return f, Pxx


# ============================================================================
# 分析函数：计算所有通道的PSD和dB值
# ============================================================================

def analyze_all_channels(data_weighted, fs, win_length, noverlap, f_min=50, f_max=500):
    """
    计算所有通道的PSD和dB值
    
    Parameters:
    -----------
    data_weighted : ndarray
        A计权后的数据，形状为 (num_channels, num_samples)
    fs : float
        采样频率 (Hz)
    win_length : int
        窗口长度
    noverlap : int
        重叠样本数
    f_min : float
        dB计算的最小频率，默认50 Hz
    f_max : float
        dB计算的最大频率，默认500 Hz
    
    Returns:
    --------
    results : list of dict
        每个通道的结果字典，包含:
        - 'channel': 通道编号（从1开始）
        - 'f': 频率数组
        - 'Pxx': PSD数组
        - 'db': dB值
    """
    num_channels = data_weighted.shape[0]
    results = []
    
    print(f"\n=== 计算PSD（基于A计权后的数据） ===")
    print(f"窗口长度: {win_length}")
    print(f"重叠样本数: {noverlap}")
    print(f"频率范围: {f_min}-{f_max} Hz")
    
    for ch in range(num_channels):
        signal_data = data_weighted[ch, :]
        
        # 计算PSD
        f, Pxx = compute_psd(signal_data, fs, win_length, noverlap)
        
        # 计算dB值
        db = compute_db_value(signal_data, fs, f_min, f_max)
        
        results.append({
            'channel': ch + 1,
            'f': f,
            'Pxx': Pxx,
            'db': db
        })
        
        # 显示有效通道信息（通道7-26）
        if 6 <= ch <= 25:
            channel_type = ""
            if 6 <= ch <= 23:
                channel_type = " (加速度)"
            elif ch == 24 or ch == 25:
                channel_type = " (麦克风)"
            print(f"  通道 {ch+1}{channel_type}: dB = {db:.2f}")
    
    return results


# ============================================================================
# 绘图函数：按传感器类型绘制PSD
# ============================================================================

def plot_psd_by_type(results, save_dir='results'):
    """
    按传感器类型分组绘制PSD（两张独立图片同时弹出）
    
    Parameters:
    -----------
    results : list of dict
        analyze_all_channels返回的结果列表
    save_dir : str
        保存目录，默认'results'
    """
    print("\n绘制PSD结果图...")
    
    # 1. 绘制加速度传感器 (通道7-24)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    for ch_idx in range(6, 24):  # 索引6-23，对应通道7-24
        result = results[ch_idx]
        f = result['f']
        Pxx = result['Pxx']
        db = result['db']
        
        # 转换为dB/Hz
        Pxx_db = 10 * np.log10(Pxx / 4e-10)
        
        ax1.plot(f, Pxx_db, linewidth=1, alpha=0.7, 
                label=f"Ch{result['channel']} ({db:.2f}dB)")
    
    ax1.set_xlabel('频率 (Hz)', fontsize=12)
    ax1.set_ylabel('PSD (dB/Hz)', fontsize=12)
    ax1.set_xlim([20, 500])
    ax1.set_ylim([-10, 40])
    ax1.set_title('加速度传感器 (通道7-24) PSD（A计权）', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8, ncol=3)
    plt.tight_layout()
    
    save_path1 = f'{save_dir}/psd_accelerometers.png'
    plt.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"已保存: {save_path1}")
    
    # 2. 绘制麦克风 (通道25-26)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    for ch_idx in [24, 25]:  # 索引24-25，对应通道25-26
        result = results[ch_idx]
        f = result['f']
        Pxx = result['Pxx']
        db = result['db']
        
        # 转换为dB/Hz
        Pxx_db = 10 * np.log10(Pxx / 4e-10)
        
        ax2.plot(f, Pxx_db, linewidth=2, 
                label=f"Ch{result['channel']} 麦克风 ({db:.2f}dB)")
    
    ax2.set_xlabel('频率 (Hz)', fontsize=12)
    ax2.set_ylabel('PSD (dB/Hz)', fontsize=12)
    ax2.set_xlim([20, 500])
    ax2.set_ylim([-10, 40])
    ax2.set_title('麦克风信号 (通道25-26) PSD（A计权）', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    save_path2 = f'{save_dir}/psd_microphones.png'
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"已保存: {save_path2}")
    
    # 关闭所有图表
    plt.close('all')


# ============================================================================
# 维纳滤波相关函数
# ============================================================================

def compute_wiener_filter(inputs, target, filter_length=128):
    """
    计算多输入单输出(MISO) FIR维纳滤波器系数
    
    Parameters:
    -----------
    inputs : ndarray
        输入信号矩阵，形状 (num_channels, num_samples)
    target : ndarray
        目标信号向量，形状 (num_samples,)
    filter_length : int
        滤波器长度 (FIR阶数+1)
        
    Returns:
    --------
    filters : ndarray
        滤波器系数矩阵，形状 (num_channels, filter_length)
    """
    num_channels, num_samples = inputs.shape
    
    # 确保target是1D
    if len(target.shape) > 1:
        target = target.flatten()
        
    # 截断以匹配长度
    min_len = min(inputs.shape[1], len(target))
    inputs = inputs[:, :min_len]
    target = target[:min_len]
    
    # 去均值 (非常重要，维纳滤波假设零均值或平稳随机过程)
    # inputs = inputs - np.mean(inputs, axis=1, keepdims=True)
    # target = target - np.mean(target)
    
    print(f"开始计算维纳滤波器 (输入通道数={num_channels}, 滤波器长度={filter_length})...")
    
    # 完全复刻MATLAB的miso_firwiener2实现
    # MATLAB: N = filtlength - 1
    N = filter_length - 1
    M = num_channels
    
    # 1. 构建R矩阵 - 完全复刻MATLAB逻辑
    # MATLAB: R = zeros(M*(N+1), M);
    R = np.zeros((M * (N+1), M))
    
    for m in range(M):  # MATLAB: for m = 1:M
        for i in range(M):  # MATLAB: for i = 1:M
            # MATLAB: rmi = xcorr(X(:,m), X(:,i), N);
            # xcorr(x, y, N) 返回长度为 2*N+1 的互相关序列
            # 对应 lag = -N, ..., 0, ..., N
            xc_rmi = signal.correlate(inputs[m], inputs[i], mode='full')
            lags_rmi = signal.correlation_lags(len(inputs[m]), len(inputs[i]), mode='full')
            zero_idx = np.where(lags_rmi == 0)[0][0]
            
            # MATLAB: rmi(1:N+1) 对应 lag -N 到 0
            # 注意：MATLAB索引从1开始，所以1:N+1是前N+1个元素
            rmi_first = xc_rmi[zero_idx - N : zero_idx + 1]  # lag -N 到 0
            
            # MATLAB: Rmi = flipud(rmi(1:N+1));
            # flipud 将列向量上下翻转，所以 lag 0 到 -N (即 0, -1, -2, ..., -N)
            Rmi = np.flipud(rmi_first)
            
            # MATLAB: top = (m-1) * (N+1) + 1; bottom = m * (N+1);
            # MATLAB: R(top:bottom,i) = Rmi;
            top = m * (N+1)  # Python索引从0开始
            bottom = (m+1) * (N+1)
            R[top:bottom, i] = Rmi
    
    # MATLAB: R = reshape(R, [N+1,M,M]);
    # MATLAB reshape是列优先（Fortran顺序），必须使用order='F'
    R = R.reshape((N+1, M, M), order='F')
    # MATLAB: R = permute(R, [2,1,3]);
    R = np.transpose(R, (1, 0, 2))  # 维度从 (N+1,M,M) 变为 (M,N+1,M)
    # MATLAB: R = reshape(R, [M*(N+1),M]);
    R = R.reshape((M*(N+1), M), order='F')  # 最终reshape也必须使用列优先
    
    # 2. 构建P向量 - 完全复刻MATLAB逻辑
    # MATLAB: P = zeros(1, M*(N+1));
    P = np.zeros(M * (N+1))
    
    for i in range(M):  # MATLAB: for i = 1:M
        # MATLAB: top = (i-1)*(N+1)+1; bottom = i * (N+1);
        # MATLAB: p = xcorr(y, X(:,i), N);
        xc_p = signal.correlate(target, inputs[i], mode='full')
        lags_p = signal.correlation_lags(len(target), len(inputs[i]), mode='full')
        zero_idx = np.where(lags_p == 0)[0][0]
        
        # MATLAB: p(N+1:2*N+1) 对应 lag 0 到 N
        # xcorr输出长度为2*N+1，索引N+1对应lag 0
        p_vec = xc_p[zero_idx : zero_idx + N + 1]  # lag 0 到 N
        
        # MATLAB: P(top:bottom) = p(N+1:2*N+1)';
        top = i * (N+1)  # Python索引从0开始
        bottom = (i+1) * (N+1)
        P[top:bottom] = p_vec
    
    # MATLAB: P = reshape(P, [N+1,M]);
    P = P.reshape((N+1, M), order='F')  # 使用列优先
    # MATLAB: P = reshape(P', [M*(N+1),1]);
    P = P.T.reshape((M*(N+1),), order='F')  # 转置后reshape也要用列优先

    # 3. 求解线性方程组
    # 先尝试直接求解验证R矩阵构建是否正确
    print("正在求解线性方程组...")
    
    # 尝试直接求解 R * w = P
    try:
        # 使用Cholesky分解（如果R是正定的）
        w_vec = linalg.solve(R, P, assume_a='pos')
        print("使用直接求解方法成功")
    except (linalg.LinAlgError, ValueError):
        # 如果失败，尝试使用Block Levinson算法（完全复刻MATLAB）
        print("直接求解失败，使用Block Levinson算法...")
        y = P.copy()  # MATLAB: y=P;
        L = R.copy()  # MATLAB: L=R;
        
        s = L.shape
        d = s[1]  # MATLAB: d = s(2); Block dimension
        N2 = s[0] // d  # MATLAB: N2 = s(1) / d; Number of blocks
        
        # MATLAB: B = reshape(L, [d,N2,d]);
        B = L.reshape((d, N2, d), order='F')  # 使用列优先
        # MATLAB: B = permute(B, [1,3,2]);
        B = np.transpose(B, (0, 2, 1))  # 从 (d,N2,d) 变为 (d,d,N2)
        # MATLAB: B = flipdim(B, 3);
        B = np.flip(B, axis=2)  # 翻转第3维（索引2）
        # MATLAB: B = reshape(B, [d,N2*d]);
        B = B.reshape((d, N2*d), order='F')  # 使用列优先
        
        # MATLAB: f = L(1:d,:)^-1;
        f = linalg.inv(L[0:d, :])
        # MATLAB: b = f;
        b = f.copy()
        # MATLAB: x = f * y(1:d);
        # 确保y[0:d]是列向量
        y_first = y[0:d].reshape((d, 1)) if len(y[0:d].shape) == 1 else y[0:d]
        x = f @ y_first
        
        # MATLAB: for n = 2:N2
        for n in range(2, N2+1):
            # MATLAB: ef = B(:,(N2-n)*d+1:N2*d) * [f;zeros(d)];
            ef = B[:, (N2-n)*d:N2*d] @ np.vstack([f, np.zeros((d, d))])
            # MATLAB: eb = L(1:n*d,:)' * [zeros(d);b];
            eb = L[0:n*d, :].T @ np.vstack([np.zeros((d, d)), b])
            # MATLAB: ex = B(:,(N2-n)*d+1:N2*d) * [x;zeros(d,1)];
            # 确保x是列向量
            if len(x.shape) == 1:
                x = x.reshape((len(x), 1))
            ex = B[:, (N2-n)*d:N2*d] @ np.vstack([x, np.zeros((d, 1))])
            # ex应该是列向量
            if len(ex.shape) == 1:
                ex = ex.reshape((len(ex), 1))
            
            # MATLAB: A = [eye(d),eb;ef,eye(d)]^-1;
            A_top = np.hstack([np.eye(d), eb])
            A_bottom = np.hstack([ef, np.eye(d)])
            A = linalg.inv(np.vstack([A_top, A_bottom]))
            
            # MATLAB: fn = [[f;zeros(d)],[zeros(d);b]] * A(:,1:d);
            f_zeros = np.vstack([f, np.zeros((d, d))])
            zeros_b = np.vstack([np.zeros((d, d)), b])
            fn = np.hstack([f_zeros, zeros_b]) @ A[:, 0:d]
            
            # MATLAB: bn = [[f;zeros(d)],[zeros(d);b]] * A(:,d+1:end);
            bn = np.hstack([f_zeros, zeros_b]) @ A[:, d:]
            
            # MATLAB: f = fn; b = bn;
            f = fn
            b = bn
            
            # MATLAB: x = [x;zeros(d,1)] + b * (y((n-1)*d+1:n*d) - ex);
            # 确保y[(n-1)*d:n*d]是列向量
            y_segment = y[(n-1)*d:n*d]
            if len(y_segment.shape) == 1:
                y_segment = y_segment.reshape((d, 1))
            x = np.vstack([x, np.zeros((d, 1))]) + b @ (y_segment - ex)
        
        # MATLAB: W=x;
        W = x
        # MATLAB: W = reshape(W, [M,N+1]);
        # MATLAB reshape是列优先，所以这里要用order='F'
        W = W.reshape((M, N+1), order='F')
        # MATLAB: W = reshape(W', [1, M*(N+1)]);
        # 转置后reshape，MATLAB也是列优先
        W = W.T.reshape((M*(N+1),), order='F')
        # MATLAB: W=W';
        W = W.reshape((M*(N+1), 1))  # 转置为列向量
        
        # MATLAB: for i=1:M
        #     temp=(i-1)*(N+1);
        #     Warr(:,i)=W(temp+1:temp+N+1);
        # end
        # 注意：MATLAB索引从1开始，所以temp+1:temp+N+1对应Python的temp:temp+N+1
        # 但这里W已经是列向量，直接提取即可
        w_vec = W.flatten()
    
    # 将w_vec重塑为滤波器矩阵
    # 根据MATLAB代码，Warr(:,i) = W(temp+1:temp+N+1)
    # 这意味着每个通道的滤波器系数是W中连续的一段
    # 由于MATLAB reshape是列优先，我们需要按列优先reshape
    filters = np.zeros((N+1, M))
    for i in range(M):
        temp = i * (N+1)
        filters[:, i] = w_vec[temp:temp+N+1]
    filters = filters.T  # 转置为(M, N+1)以匹配Python代码的期望格式
    print("维纳滤波器计算完成。")
    
    return filters

def apply_miso_filter(inputs, filters):
    """
    应用MISO滤波器到输入信号
    MATLAB: 
    for i=1:s1
        temp=filter(wf(:,i),1,ref(:,i));
        tempsum=tempsum+temp;
    end
    error=err-tempsum; (注意 MATLAB test_wiener 返回的是 error)
    
    filter(b, a, x) 是 IIR/FIR 滤波。这里 a=1, b=wf(:,i)
    这对应 Python 的 signal.lfilter(filters[i], 1, inputs[i])
    或者 signal.convolve(..., mode='same') 但要注意相位
    
    lfilter 是因果滤波，与 filter 对应
    """
    num_channels, num_samples = inputs.shape
    output = np.zeros(num_samples)
    
    # 移除临时的去均值，因为 MATLAB 没有
    # inputs_mean = np.mean(inputs, axis=1, keepdims=True)
    # inputs_demean = inputs - inputs_mean
    
    for i in range(num_channels):
        # 使用 lfilter 对应 MATLAB 的 filter
        # filters[i] 是系数 b
        filt_out = signal.lfilter(filters[i], 1, inputs[i])
        output += filt_out
        
    return output


# ============================================================================
# 主程序
# ============================================================================


def main():
    """
    主程序 - 数据分析流程
    """
    # ========== 配置参数 ==========
    filename = 'data/260107_ddr_32x5_05.dat'
    num_channels = 32
    num_samples = 15000
    fs = 3000  # 采样率 (Hz)
    
    win_length = fs // 4  # 750个采样点
    noverlap = fs // 8    # 375个采样点重叠
    
    wiener_filter_len = 300 # 维纳滤波器长度 (与MATLAB一致: FL=300)
    
    # ========== 步骤1: 读取数据 ==========
    print("="*60)
    print("步骤1: 读取数据")
    print("="*60)
    data = load_data(filename, num_channels, num_samples)
    
    print(f"\n=== 数据统计信息 ===")
    print(f"采样率: {fs} Hz")
    print(f"时长: {num_samples/fs} 秒")
    print(f"每通道采样点数: {num_samples}")
    
    # ========== 步骤2: 应用A计权 ==========
    print("\n" + "="*60)
    print("步骤2: 应用A计权（MIC加权）")
    print("="*60)
    data_weighted = apply_a_weighting(data)
    print("A计权完成")
    
    # ========== 步骤3: 计算PSD和dB值 ==========
    print("\n" + "="*60)
    print("步骤3: 计算PSD和dB值")
    print("="*60)
    results = analyze_all_channels(data_weighted, fs, win_length, noverlap)
    
    # ========== 步骤4: 绘制PSD图 ==========
    print("\n" + "="*60)
    print("步骤4: 绘制PSD结果")
    print("="*60)
    plot_psd_by_type(results)
    
    # ========== 步骤5: 维纳滤波 (MISO) ==========
    print("\n" + "="*60)
    print("步骤5: 维纳滤波分析 (多输入单输出)")
    print("="*60)
    
    # 定义输入和输出
    # 输入: 加速度传感器 (通道7-24, 索引6-23)
    acc_indices = list(range(6, 24))
    # 目标: 麦克风 (通道25, 索引24)
    mic_index = 24
    
    print(f"输入通道: 加速度传感器 ({len(acc_indices)}个, Ch{acc_indices[0]+1}-Ch{acc_indices[-1]+1})")
    print(f"目标通道: 麦克风 (Ch{mic_index+1})")
    
    # 关键修正：MATLAB使用原始数据计算滤波器，然后对结果A计权
    # 而不是先A计权再计算滤波器
    input_signals = data[acc_indices, :]  # 使用原始数据，不是A计权后的
    target_signal = data[mic_index, :]    # 使用原始数据
    
    print("使用原始数据计算维纳滤波器 (与MATLAB一致)...")
    
    # 5.2 计算滤波器 - 使用原始数据
    filters = compute_wiener_filter(input_signals, target_signal, filter_length=wiener_filter_len)
    
    # 5.3 应用滤波器 - 使用原始数据
    estimated_mic = apply_miso_filter(input_signals, filters)
    
    # 5.4 计算误差信号 - 使用原始数据
    error_signal = target_signal - estimated_mic
    
    # 5.5 对结果应用A计权（与MATLAB一致：先计算滤波器，再A计权）
    print("对结果应用A计权...")
    target_signal_weighted = apply_a_weighting(target_signal)
    error_signal_weighted = apply_a_weighting(error_signal)
    
    # 5.6 结果分析 - 使用A计权后的信号
    # 计算A计权后信号的PSD和dB值
    f_mic, Pxx_mic = compute_psd(target_signal_weighted, fs, win_length, noverlap)
    f_err, Pxx_err = compute_psd(error_signal_weighted, fs, win_length, noverlap)
    
    db_mic = compute_db_value(target_signal_weighted, fs)
    db_err = compute_db_value(error_signal_weighted, fs)
    
    # 计算ANC降噪量（与MATLAB psdcomp函数一致）
    anc_db = 10 * np.log10((compute_freq_rms(error_signal_weighted, fs, 50, 500)**2) / 
                           (compute_freq_rms(target_signal_weighted, fs, 50, 500)**2))
    
    print(f"\n=== 维纳滤波结果 (与MATLAB一致) ===")
    print(f"BeforeANC (原始麦克风): {db_mic:.2f} dBA")
    print(f"AfterANC (误差信号): {db_err:.2f} dBA")
    print(f"ANC降噪量: {anc_db:.2f} dBA")
    print(f"降噪量 (原始 - 误差): {db_mic - db_err:.2f} dBA")
    
    # 5.5 绘制对比图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 时域对比 (只显示前0.1秒或一部分)
    time_axis = np.arange(num_samples) / fs
    show_samples = min(1000, num_samples)
    
    ax1.plot(time_axis[:show_samples], target_signal_weighted[:show_samples], label='原始麦克风 (A计权)', alpha=0.7, color='red')
    ax1.plot(time_axis[:show_samples], error_signal_weighted[:show_samples], label='误差信号 (A计权)', alpha=0.7, color='green')
    ax1.set_title('时域波形对比 (A计权后, 前1000点)')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 频域对比 - 使用A计权后的信号
    Pxx_mic_db = 10 * np.log10(Pxx_mic / 4e-10)
    Pxx_err_db = 10 * np.log10(Pxx_err / 4e-10)
    
    ax2.plot(f_mic, Pxx_mic_db, label=f'原始麦克风 ({db_mic:.1f}dBA)', linewidth=2, color='red')
    ax2.plot(f_err, Pxx_err_db, label=f'误差信号 ({db_err:.1f}dBA)', color='green', linewidth=2)
    
    ax2.set_title('PSD 对比')
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('PSD (dB/Hz)')
    ax2.set_xlim([20, 500])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = 'results/wiener_filter_result.png'
    plt.savefig(save_path, dpi=150)
    print(f"已保存维纳滤波分析图: {save_path}")
    plt.close()

    # ========== 总结 ==========
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
    print(f"\n已生成的图表:")
    print(f"  - results/psd_accelerometers.png")
    print(f"  - results/psd_microphones.png")
    print(f"  - results/wiener_filter_result.png")
    
    # 返回数据和结果
    return {
        'data': data,
        'data_weighted': data_weighted,
        'results': results,
        'filters': filters,
        'estimated_mic': estimated_mic
    }


if __name__ == '__main__':
    main()
