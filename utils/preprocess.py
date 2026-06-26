import pywt
from PyEMD import EMD
import numpy as np


def shoulder_feature_extraction(I, th1=1.5, th2=0.6):
    """
    Algorithm 1 : 从信号序列中提取肩部特征。

    参数:
    I (list): 输入的信号序列
    th1 (float): 判断肩部点的阈值1
    th2 (float): 判断肩部点的阈值2

    返回:
    NShoulder (int): 肩部点的数量
    LShoulder_max (int): 最长肩部持续时间
    """
    # Step 1: Process Current Sequence
    shoulder = []
    for n in range(len(I) - 3):
        if abs(I[n]) < th1 and abs(I[n + 1]) < th1 and abs(I[n + 2]) < th1 and abs(I[n + 3]) < th1 and abs(I[n + 3] - I[n]) < th2:
            shoulder.append(n)

    # Step 2: Calculate shoulder lasting duration
    duration = []
    if shoulder:
        len_shoulder = 1
        for p in range(len(shoulder) - 1):
            if shoulder[p + 1] - shoulder[p] == 1:
                len_shoulder += 1
            else:
                duration.append(len_shoulder)
                len_shoulder = 1
        duration.append(len_shoulder)  # 添加最后一个肩部持续时间

    # Step 3: 提取特征
    NShoulder = len(duration) if duration else 0
    LShoulder_max = max(duration) if duration else 0

    return NShoulder, LShoulder_max


def frequency_domain_feature_Extraction(I, fs, fundamental_freq=50):
    """
    Algorithm 2 : 频域特征提取算法

    参数:
    I (list or np.array): 输入的电流信号序列
    fs (int): 采样频率，默认为6400/10000Hz
    fundamental_freq (int): 基波频率，默认为50Hz

    返回:
    dict: 包含 sumPodd, sumPeven, stdPodd, stdPeven, sumP450, stdP450 的特征字典
    """
    # Step 1: Fourier Transform of Current Sequence
    N = len(I)
    Y = np.fft.fft(I, N)
    psd = np.abs(Y) ** 2 / N  # Power spectral density

    # Step 2: Map harmonic frequencies
    m0 = int(fundamental_freq * N / fs)  # 基波频率对应的索引
    m450 = int(450 * N / fs)  # 450Hz 对应的索引
    harmonics = [2, 3, 4, 5, 6, 7, 8, 9]  # 谐波序号
    harmonic_indices = [int(h * fundamental_freq * N / fs) for h in harmonics]

    # Step 3: Extract frequency features
    psd0 = psd[m0] if m0 < len(psd) else 1e-10  # 避免除以零
    psd_harmonics = [psd[idx] if idx < len(psd) else 0 for idx in harmonic_indices]

    # Individual harmonic power spectral densities
    P2 = psd_harmonics[0] / psd0  # P^(2)
    P3 = psd_harmonics[1] / psd0  # P^(3)
    P4 = psd_harmonics[2] / psd0  # P^(4)
    P5 = psd_harmonics[3] / psd0  # P^(5)

    # Step 3.1: sumPodd = sum(psd3, psd5, psd7, psd9) / psd0
    sumPodd = sum(psd_harmonics[i] for i in [1, 3, 5, 7]) / psd0

    # Step 3.2: sumPeven = sum(psd2, psd4, psd6, psd8) / psd0
    sumPeven = sum(psd_harmonics[i] for i in [0, 2, 4, 6]) / psd0

    # Step 3.3: stdPodd = std(psd3, psd5, psd7, psd9) / psd0
    stdPodd = np.std([psd_harmonics[i] / psd0 for i in [1, 3, 5, 7]])

    # Step 3.4: stdPeven = std(psd2, psd4, psd6, psd8) / psd0
    stdPeven = np.std([psd_harmonics[i] / psd0 for i in [0, 2, 4, 6]])

    # Step 3.5: sumP450 = sum(psd[m0], psd[m2], ..., psd[m9]) where m <= m450
    sumP450 = sum(psd[idx] for idx in [m0] + harmonic_indices if idx <= m450)

    # Step 3.6: stdP450 = std(psd[m0], psd[m2], ..., psd[m9]) where m <= m450
    stdP450 = np.std([psd[idx] for idx in [m0] + harmonic_indices if idx <= m450])

    return sumPodd, sumPeven, stdPodd, stdPeven, sumP450, stdP450, P2, P3, P4, P5


def time_domain_current_based_morphological_features(I, fs):
    """
    Algorithm 3 : 时间域电流形态特征提取算法

    参数:
    I (list or np.array): 输入的电流信号序列
    fs (int): 采样频率，默认为10000Hz
    resolution (str): 数据分辨率，"HR"表示高分辨率，"LR"表示低分辨率

    返回:
    dict: 包含 peakstd, peakrange, peakmedian, valleystd, valleyrange, valleymedian 的特征字典
    """
    # Step 1: Select suitable number of periods
    N = 5 if fs == 10000 else 8  # HR数据用5个周期，LR数据用8个周期
    period_samples = int(0.02 * fs)  # 每个周期的采样点数 (0.02秒窗口)
    total_samples = N * period_samples  # 总采样点数

    # 如果输入信号长度不足，截断或补零
    if len(I) < total_samples:
        I = np.pad(I, (0, total_samples - len(I)), mode="constant")
    else:
        I = I[:total_samples]

    # Step 1.2: Apply N periodic windows
    Iw = np.reshape(I, (N, period_samples))  # 将信号分成N个窗口

    # Step 2: Calculate peak and valley
    Ipeak = np.max(Iw, axis=1)  # 每个窗口的峰值
    Ivalley = np.min(Iw, axis=1)  # 每个窗口的谷值

    # Step 3: Extract morphological features
    peakstd = np.std(Ipeak)  # 峰值的标准差
    peakrange = np.ptp(Ipeak)  # 峰值的范围 (最大值 - 最小值)
    peakmedian = np.median(Ipeak)  # 峰值的中位数

    valleystd = np.std(Ivalley)  # 谷值的标准差
    valleyrange = np.ptp(Ivalley)  # 谷值的范围 (最大值 - 最小值)
    valleymedian = np.median(Ivalley)  # 谷值的中位数

    return peakstd, peakrange, peakmedian, valleystd, valleyrange, valleymedian


def feature_extraction_based_on_phase_space(I, time_delays=[3, 8, 11], flag_plot=False):
    """
    Algorithm 4 : 基于相空间的非线性动态特征提取算法

    参数:
    I (list or np.array): 输入的电流信号序列
    time_delays (list): 时间延迟的列表，默认为 [3, 8, 11]

    返回:
    dict: 包含 Acc3_small, Acc3_large, Acc8_small, Acc8_large, Acc11_small, Acc11_large 的特征字典
    """
    # Step 1: Normalize current signals
    features = {}
    if flag_plot:
        # 每隔多少个t绘制一张图，初始化
        count_plot = 0
    # 将 I 归一化到 [0, 20]
    I_normalized = (I - np.min(I)) / (np.max(I) - np.min(I)) * 20

    # Step 2: Construct phase space
    for t in time_delays:
        # 初始化 20x20 的相空间矩阵
        space = np.zeros((20, 20))
        for n in range(t, len(I_normalized)):
            x = int(I_normalized[n])  # 当前值
            y = int(I_normalized[n - t])  # 延迟值
            if 0 <= x < 20 and 0 <= y < 20:  # 确保索引在范围内
                space[x, y] += 1
        if flag_plot:
            from utils.utils_plot import plot_space_matrix

            count_plot += 1
            if count_plot % 10 == 1:
                # 每隔10个t绘制一张图
                plot_space_matrix(space, t)

        # Step 3: Calculate Energy at Space Origin
        # Acct_small: 累计能量在 [9, 10] 范围
        small_indices = [(i, j) for i in range(9, 11) for j in range(9, 11)]
        features[f"Acc{t}_small"] = sum(space[i, j] for i, j in small_indices)

        # Acct_large: 累计能量在 [8, 12] 范围
        large_indices = [(i, j) for i in range(8, 13) for j in range(8, 13)]
        features[f"Acc{t}_large"] = sum(space[i, j] for i, j in large_indices)

    return (features["Acc3_small"], features["Acc3_large"], features["Acc8_small"], features["Acc8_large"], features["Acc11_small"], features["Acc11_large"])


def wavelet_transform(I):
    """
    Algorithm 5 : 进行小波变换并提取高频系数的标准差

    参数:
    I (list or np.array): 输入的信号序列

    返回:
    tuple: 包含第二层、第三层和第四层高频系数标准差的元组 (std_cD2, std_cD3, std_cD4)
    """
    # Step 1: 选择小波类型和分解层数
    wavelet = "db1"  # Daubechies小波
    level = 4  # 分解层数

    # Step 2: 进行小波分解
    coeffs = pywt.wavedec(I, wavelet, level=level)

    # Step 3: 提取高频系数
    cD2 = coeffs[-2]  # 第二层高频系数
    cD3 = coeffs[-3]  # 第三层高频系数
    cD4 = coeffs[-4]  # 第四层高频系数

    # Step 4: 计算标准差
    std_cD2 = np.std(cD2)
    std_cD3 = np.std(cD3)
    std_cD4 = np.std(cD4)

    return std_cD2, std_cD3, std_cD4


def calculate_EMD_std_and_dEMD_std(I):
    """
    计算经验模态分解（EMD）的标准差（EMD_std）和 EMD 差值的标准差（dEMD_std）

    参数:
    signal (np.array): 输入信号

    返回:
    EMD_std (float): EMD 分解后的 IMFs 的标准差
    dEMD_std (float): EMD 差值的标准差
    """
    # Step 1: 进行经验模态分解（EMD）
    emd = EMD()
    IMFs = emd(I)  # 分解为 IMFs

    # Step 2: 计算 EMD 的标准差（EMD_std）
    EMD_std = np.std(IMFs, axis=1)  # 对每个 IMF 计算标准差
    EMD_std = np.mean(EMD_std)  # 取所有 IMF 标准差的平均值

    # Step 3: 计算 EMD 差值的标准差（dEMD_std）
    dIMFs = np.diff(IMFs, axis=1)  # 计算 IMFs 的差值
    dEMD_std = np.std(dIMFs, axis=1)  # 对每个差值 IMF 计算标准差
    dEMD_std = np.mean(dEMD_std)  # 取所有差值 IMF 标准差的平均值

    return EMD_std, dEMD_std


def preprocess(input, sample_rate):
    features_input = np.zeros((len(input), 29), dtype=float)

    for i in range(len(input)):
        # Algorithm 1
        NShoulder, LShoulder_max = shoulder_feature_extraction(input[i])

        # Algorithm 2
        sumPodd, sumPeven, stdPodd, stdPeven, sumP450, stdP450, P2, P3, P4, P5 = frequency_domain_feature_Extraction(input[i], fs=sample_rate, fundamental_freq=50)

        # Algorithm 3
        peakstd, peakrange, peakmedian, valleystd, valleyrange, valleymedian = time_domain_current_based_morphological_features(input[i], fs=sample_rate)

        # Algorithm 4
        (Acc3_small, Acc3_large, Acc8_small, Acc8_large, Acc11_small, Acc11_large) = feature_extraction_based_on_phase_space(input[i])

        cD2, cD3, cD4 = wavelet_transform(input[i])

        EMD_std, dEMD_std = calculate_EMD_std_and_dEMD_std(input[i])

        features_input[i] = [
            Acc3_small,
            Acc3_large,
            Acc8_small,
            Acc8_large,
            Acc11_small,
            Acc11_large,
            NShoulder,
            LShoulder_max,
            peakstd,
            peakrange,
            peakmedian,
            valleystd,
            valleyrange,
            valleymedian,
            cD2,
            cD3,
            cD4,
            EMD_std,
            dEMD_std,
            P2,
            P3,
            P4,
            P5,
            sumPodd,
            sumPeven,
            stdPodd,
            stdPeven,
            sumP450,
            stdP450,
        ]

    return features_input


if __name__ == "__main__":
    from GetData import DataProcess, load_data

    # data_path = "yaml/16-6400_multi_electricoven-refrigerator_m3.yaml"
    # data_path = "yaml/11-10000_single_electricoven_m1.yaml"
    data_path = "yaml/05-10000_multi_airpurifier-airconditioner03_m2.yaml"
    # 从配置文件中加载数据
    U, I, label, window_size, sample_rate = load_data(data_path)
    features_input = preprocess(I.values, sample_rate)
    print(features_input)
