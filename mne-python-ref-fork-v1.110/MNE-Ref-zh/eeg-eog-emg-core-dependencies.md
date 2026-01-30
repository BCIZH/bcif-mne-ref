# EEG/EOG/EMG 信号处理核心依赖清单

> **目标用户**: 专注于脑电(EEG)、眼电(EOG)、肌电(EMG)原始数据与预处理  
> **创建日期**: 2026-01-30  
> **核心库**: NumPy, SciPy, scikit-learn

---

## 目录

1. [核心工作流程](#核心工作流程)
2. [NumPy 核心依赖](#numpy-核心依赖)
3. [SciPy 核心依赖](#scipy-核心依赖)
4. [scikit-learn 核心依赖](#scikit-learn-核心依赖)
5. [最重要的依赖总结](#最重要的依赖总结)
6. [官方文档链接](#官方文档链接)

---

## 核心工作流程

### EEG/EOG/EMG 典型处理管道

```
原始数据读取
    ↓
┌─────────────────────────┐
│ 1. 数据加载             │ ← NumPy arrays
│  raw.load_data()        │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ 2. 滤波                 │ ← scipy.signal, scipy.fft
│  raw.filter(l_freq, h)  │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ 3. 伪迹检测             │ ← NumPy statistics
│  - EOG: find_eog_events │ ← scipy.signal (peaks)
│  - ECG: find_ecg_events │
│  - EMG: muscle artifacts│
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ 4. ICA 去伪迹           │ ← sklearn.FastICA (可选)
│  ica.fit(raw)           │ ← scipy.stats (评分)
│  ica.apply(raw)         │ ← NumPy (矩阵运算)
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ 5. Epochs 分段          │ ← NumPy indexing
│  epochs = Epochs(...)   │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ 6. Evoked 平均          │ ← NumPy mean/std
│  evoked = epochs.avg()  │
└─────────────────────────┘
```

---

## NumPy 核心依赖

### 1. 数据存储与访问 ⭐⭐⭐⭐⭐

**必需度**: **100% (无可替代)**

#### 1.1 ndarray - 核心数据结构

```python
import numpy as np

# 所有 EEG/EOG/EMG 数据都是 NumPy 数组
raw.get_data()  # 返回 (n_channels, n_times) ndarray
epochs.get_data()  # 返回 (n_epochs, n_channels, n_times) ndarray
```

**文档**: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html

**为什么重要**:
- MNE 所有数据存储都是 `numpy.ndarray`
- 内存高效 (C 连续数组)
- 支持快速索引和切片

---

#### 1.2 数组索引与切片 ⭐⭐⭐⭐⭐

```python
# 选择通道
data = raw.get_data(picks=['Fz', 'Cz', 'Pz'])  # NumPy fancy indexing

# 时间切片
data_segment = data[:, 1000:2000]  # 第 1000-2000 采样点

# 条件索引
bad_epochs = np.where(ptp > threshold)[0]  # 超阈值的 epoch 索引
```

**文档**: https://numpy.org/doc/stable/user/basics.indexing.html

---

#### 1.3 统计计算 ⭐⭐⭐⭐⭐

```python
# 基本统计
mean = data.mean(axis=1)  # 每个通道的平均值
std = data.std(axis=1)    # 标准差
ptp = data.ptp(axis=1)    # 峰峰值 (Peak-to-Peak)

# 百分位数 (异常检测)
threshold = np.percentile(data, 95, axis=1)

# Z-score 标准化
z_data = (data - mean[:, None]) / std[:, None]
```

**文档**: 
- https://numpy.org/doc/stable/reference/generated/numpy.mean.html
- https://numpy.org/doc/stable/reference/generated/numpy.std.html
- https://numpy.org/doc/stable/reference/generated/numpy.ptp.html
- https://numpy.org/doc/stable/reference/generated/numpy.percentile.html

---

#### 1.4 线性代数 ⭐⭐⭐⭐

```python
# ICA/PCA 中的矩阵运算
import numpy.linalg as la

# 伪逆 (用于投影矩阵)
proj_matrix = la.pinv(mixing_matrix)

# SVD 分解 (PCA)
U, s, Vt = la.svd(data, full_matrices=False)

# 协方差矩阵
cov = np.cov(data)  # (n_channels, n_channels)

# 矩阵乘法
reconstructed = mixing_matrix @ sources
```

**文档**:
- https://numpy.org/doc/stable/reference/routines.linalg.html
- https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
- https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
- https://numpy.org/doc/stable/reference/generated/numpy.cov.html

---

#### 1.5 FFT (频域分析) ⭐⭐⭐⭐

```python
# 功率谱密度 (PSD)
from numpy.fft import rfft, rfftfreq

# 实数 FFT
fft_data = np.fft.rfft(data, axis=-1)
freqs = np.fft.rfftfreq(n_samples, 1/sfreq)

# 功率谱
psd = np.abs(fft_data) ** 2

# 逆 FFT
data_reconstructed = np.fft.irfft(fft_data)
```

**文档**: https://numpy.org/doc/stable/reference/routines.fft.html

**注意**: MNE 实际使用 `scipy.fft` (更快)，但 API 与 NumPy 兼容。

---

### NumPy 核心模块总结

| 模块 | 用途 | 必需度 | 文档链接 |
|------|------|--------|----------|
| **numpy.ndarray** | 数据存储 | ⭐⭐⭐⭐⭐ | https://numpy.org/doc/stable/reference/arrays.ndarray.html |
| **numpy indexing** | 数据访问 | ⭐⭐⭐⭐⭐ | https://numpy.org/doc/stable/user/basics.indexing.html |
| **numpy.mean/std** | 统计计算 | ⭐⭐⭐⭐⭐ | https://numpy.org/doc/stable/reference/routines.statistics.html |
| **numpy.linalg** | 线性代数 | ⭐⭐⭐⭐ | https://numpy.org/doc/stable/reference/routines.linalg.html |
| **numpy.fft** | 频域分析 | ⭐⭐⭐⭐ | https://numpy.org/doc/stable/reference/routines.fft.html |

---

## SciPy 核心依赖

### 1. scipy.signal - 信号处理 ⭐⭐⭐⭐⭐

**必需度**: **100% (滤波核心)**

#### 1.1 滤波器设计

```python
from scipy import signal

# Butterworth 滤波器
b, a = signal.butter(
    N=5,           # 滤波器阶数
    Wn=30,         # 截止频率 (Hz)
    btype='low',   # 'low', 'high', 'bandpass', 'bandstop'
    fs=sfreq       # 采样率
)

# FIR 滤波器 (窗函数法)
fir_coefs = signal.firwin(
    numtaps=101,    # 滤波器长度
    cutoff=30,      # 截止频率
    window='hamming',
    fs=sfreq
)
```

**文档**:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html

---

#### 1.2 滤波应用

```python
# IIR 滤波 (Butterworth, Chebyshev)
filtered = signal.filtfilt(b, a, data, axis=-1)  # 零相位滤波

# FIR 滤波 (卷积)
filtered = signal.convolve(data, fir_coefs, mode='same')

# 重采样
resampled = signal.resample(data, num=new_n_samples, axis=-1)
```

**文档**:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html

---

#### 1.3 峰值检测 (EOG/ECG)

```python
# 峰值检测
peaks, properties = signal.find_peaks(
    ecg_signal,
    height=threshold,    # 最小高度
    distance=200,        # 最小间隔 (采样点)
    prominence=0.5       # 显著性
)

# Hilbert 变换 (包络提取)
analytic = signal.hilbert(emg_signal)
envelope = np.abs(analytic)
```

**文档**:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html

---

#### 1.4 窗函数

```python
# 常用窗函数
hann_window = signal.windows.hann(window_length)
hamming_window = signal.windows.hamming(window_length)
blackman_window = signal.windows.blackman(window_length)

# 加窗 (减少频谱泄漏)
windowed_data = data * hann_window
```

**文档**: https://docs.scipy.org/doc/scipy/reference/signal.windows.html

---

### 2. scipy.fft - 快速傅里叶变换 ⭐⭐⭐⭐⭐

```python
from scipy import fft

# 实数 FFT (比 numpy.fft 更快)
fft_data = fft.rfft(data, axis=-1)
freqs = fft.rfftfreq(n_samples, 1/sfreq)

# 逆 FFT
data_reconstructed = fft.irfft(fft_data, axis=-1)

# FFT 卷积 (快速滤波)
filtered = fft.fftconvolve(data, kernel, mode='same', axes=-1)
```

**文档**: https://docs.scipy.org/doc/scipy/reference/fft.html

**为什么用 SciPy FFT**:
- 比 NumPy FFT 快 2-5 倍
- 支持更多 FFT 算法 (FFTPACK, FFTW)

---

### 3. scipy.stats - 统计检验 ⭐⭐⭐⭐

#### 3.1 ICA 成分评分

```python
from scipy import stats

# Pearson 相关系数 (ICA-EOG 相关性)
r, p_value = stats.pearsonr(
    ica_component,  # ICA 成分时间序列
    eog_reference   # EOG 通道
)

# Spearman 秩相关
rho, p = stats.spearmanr(ica_component, eog_reference)
```

**文档**:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

---

#### 3.2 Z-score 标准化

```python
from scipy.stats import zscore

# Z-score (每个通道独立标准化)
z_data = zscore(data, axis=1)  # axis=1: 沿时间轴
```

**文档**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html

---

#### 3.3 峰度 (Kurtosis) - ICA 收敛指标

```python
from scipy.stats import kurtosis

# 峰度 (衡量分布的非高斯性)
kurt = kurtosis(ica_sources, axis=1)  # 每个成分

# 用于 Infomax ICA 的 Extended ICA
```

**文档**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html

---

### 4. scipy.special - 特殊函数 ⭐⭐⭐

```python
from scipy.special import expit

# Sigmoid 函数 (Infomax ICA)
# expit(x) = 1 / (1 + exp(-x))
activation = expit(sources)
```

**文档**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html

---

### 5. scipy.linalg - 优化的线性代数 ⭐⭐⭐⭐

```python
from scipy import linalg

# SVD (比 numpy.linalg.svd 更快)
U, s, Vt = linalg.svd(data, full_matrices=False)

# Cholesky 分解 (协方差矩阵)
L = linalg.cholesky(cov_matrix, lower=True)

# 伪逆 (数值稳定性更好)
pinv = linalg.pinv(matrix)
```

**文档**: https://docs.scipy.org/doc/scipy/reference/linalg.html

**优势**:
- BLAS/LAPACK 优化
- 比 NumPy 快 2-10 倍
- 更好的数值稳定性

---

### SciPy 核心模块总结

| 模块 | 用途 | 必需度 | 文档链接 |
|------|------|--------|----------|
| **scipy.signal** | 滤波、重采样、峰值检测 | ⭐⭐⭐⭐⭐ | https://docs.scipy.org/doc/scipy/reference/signal.html |
| **scipy.fft** | 快速傅里叶变换 | ⭐⭐⭐⭐⭐ | https://docs.scipy.org/doc/scipy/reference/fft.html |
| **scipy.stats** | 统计检验、相关性 | ⭐⭐⭐⭐ | https://docs.scipy.org/doc/scipy/reference/stats.html |
| **scipy.linalg** | 优化线性代数 | ⭐⭐⭐⭐ | https://docs.scipy.org/doc/scipy/reference/linalg.html |
| **scipy.special** | 特殊函数 (sigmoid) | ⭐⭐⭐ | https://docs.scipy.org/doc/scipy/reference/special.html |

---

## scikit-learn 核心依赖

**必需度**: ⭐⭐⭐⭐ **(可选但强烈推荐)**

### 1. FastICA - ICA 算法 ⭐⭐⭐⭐

```python
from sklearn.decomposition import FastICA

# 创建 ICA 对象
ica = FastICA(
    n_components=20,       # 提取成分数
    algorithm='parallel',  # 'parallel' (快) or 'deflation'
    fun='logcosh',         # 非高斯性度量: 'logcosh', 'exp', 'cube'
    max_iter=200,
    random_state=42
)

# 拟合并转换
sources = ica.fit_transform(data.T)  # (n_samples, n_components)

# 提取矩阵
mixing_matrix = ica.mixing_        # (n_features, n_components)
unmixing_matrix = ica.components_  # (n_components, n_features)
```

**文档**: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

**为什么用 scikit-learn**:
- 比 MNE 内置 Infomax 更快
- 更稳定的收敛性
- 业界标准实现

---

### 2. PCA - 降维预处理 ⭐⭐⭐

```python
from sklearn.decomposition import PCA

# PCA 降维 (ICA 预处理)
pca = PCA(
    n_components=0.95,  # 保留 95% 方差
    whiten=True         # 白化 (归一化主成分)
)

# 拟合并转换
data_pca = pca.fit_transform(data.T)

# 提取属性
pca.explained_variance_ratio_  # 每个成分解释的方差比例
pca.components_                # 主成分向量
```

**文档**: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

---

### 3. LocalOutlierFactor - 异常检测 ⭐⭐⭐

```python
from sklearn.neighbors import LocalOutlierFactor

# LOF 异常检测 (肌电伪迹)
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1  # 预期异常比例
)

# 预测: -1 异常, 1 正常
labels = lof.fit_predict(features)
muscle_artifacts = np.where(labels == -1)[0]
```

**文档**: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html

---

### 4. StandardScaler - 数据标准化 ⭐⭐⭐

```python
from sklearn.preprocessing import StandardScaler

# Z-score 标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.T).T

# 等价于
# data_scaled = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
```

**文档**: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

---

### scikit-learn 核心模块总结

| 模块 | 用途 | 必需度 | 文档链接 |
|------|------|--------|----------|
| **FastICA** | ICA 伪迹去除 | ⭐⭐⭐⭐ | https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html |
| **PCA** | 降维预处理 | ⭐⭐⭐ | https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html |
| **LocalOutlierFactor** | 异常检测 | ⭐⭐⭐ | https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html |
| **StandardScaler** | 标准化 | ⭐⭐⭐ | https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html |

---

## 最重要的依赖总结

### ⭐⭐⭐⭐⭐ 必需核心 (无可替代)

#### NumPy
1. **`numpy.ndarray`** - 所有数据的存储格式
2. **`numpy.mean/std/ptp`** - 基本统计计算
3. **`numpy array indexing`** - 数据访问与切片
4. **`numpy.linalg.svd/pinv`** - ICA/PCA 矩阵分解

#### SciPy
1. **`scipy.signal.butter`** - 滤波器设计
2. **`scipy.signal.filtfilt`** - 零相位滤波
3. **`scipy.fft.rfft/irfft`** - 快速傅里叶变换
4. **`scipy.signal.find_peaks`** - EOG/ECG 峰值检测

---

### ⭐⭐⭐⭐ 强烈推荐 (可选但重要)

#### SciPy
1. **`scipy.stats.pearsonr`** - ICA 成分评分
2. **`scipy.linalg.svd`** - 优化的 SVD (比 NumPy 快)
3. **`scipy.signal.hilbert`** - EMG 包络提取

#### scikit-learn
1. **`sklearn.decomposition.FastICA`** - 最佳 ICA 实现
2. **`sklearn.decomposition.PCA`** - 降维预处理

---

### ⭐⭐⭐ 辅助功能

#### SciPy
1. **`scipy.stats.zscore`** - Z-score 标准化
2. **`scipy.stats.kurtosis`** - 峰度计算
3. **`scipy.special.expit`** - Infomax ICA

#### scikit-learn
1. **`sklearn.neighbors.LocalOutlierFactor`** - 肌电异常检测

---

## 官方文档链接

### NumPy
- **官网**: https://numpy.org/
- **用户指南**: https://numpy.org/doc/stable/user/index.html
- **API 参考**: https://numpy.org/doc/stable/reference/index.html
- **快速入门**: https://numpy.org/doc/stable/user/quickstart.html

**重点阅读**:
- Array creation: https://numpy.org/doc/stable/user/basics.creation.html
- Indexing: https://numpy.org/doc/stable/user/basics.indexing.html
- Linear algebra: https://numpy.org/doc/stable/reference/routines.linalg.html
- Statistics: https://numpy.org/doc/stable/reference/routines.statistics.html

---

### SciPy
- **官网**: https://scipy.org/
- **用户指南**: https://docs.scipy.org/doc/scipy/tutorial/index.html
- **API 参考**: https://docs.scipy.org/doc/scipy/reference/index.html

**重点阅读**:
- Signal processing: https://docs.scipy.org/doc/scipy/reference/signal.html
- FFT: https://docs.scipy.org/doc/scipy/reference/fft.html
- Statistics: https://docs.scipy.org/doc/scipy/reference/stats.html
- Linear algebra: https://docs.scipy.org/doc/scipy/reference/linalg.html

---

### scikit-learn
- **官网**: https://scikit-learn.org/
- **用户指南**: https://scikit-learn.org/stable/user_guide.html
- **API 参考**: https://scikit-learn.org/stable/modules/classes.html

**重点阅读**:
- Decomposition (ICA/PCA): https://scikit-learn.org/stable/modules/decomposition.html
- Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
- Outlier detection: https://scikit-learn.org/stable/modules/outlier_detection.html

---

## 实战代码示例

### 完整 EEG/EOG 预处理流程

```python
import numpy as np
from scipy import signal, stats, fft
from sklearn.decomposition import FastICA, PCA
import mne

# 1. 加载数据 (NumPy)
raw = mne.io.read_raw_fif('eeg_data.fif', preload=True)
data = raw.get_data()  # NumPy ndarray (n_channels, n_times)

# 2. 滤波 (SciPy)
sfreq = raw.info['sfreq']
b, a = signal.butter(5, [1, 40], btype='bandpass', fs=sfreq)
data_filt = signal.filtfilt(b, a, data, axis=-1)

# 3. EOG 峰值检测 (SciPy)
eog_channel = data_filt[eog_idx]
peaks, _ = signal.find_peaks(np.abs(eog_channel), distance=int(0.5*sfreq))

# 4. ICA 去伪迹 (scikit-learn + SciPy)
# 4.1 PCA 降维
pca = PCA(n_components=0.95, whiten=True)
data_pca = pca.fit_transform(data_filt.T)

# 4.2 FastICA
ica = FastICA(n_components=20, max_iter=200, random_state=42)
sources = ica.fit_transform(data_pca)  # ICA 成分
mixing = ica.mixing_

# 4.3 ICA 成分评分 (SciPy)
eog_ref = data_filt[eog_idx]
scores = [stats.pearsonr(sources[:, i], eog_ref)[0] for i in range(20)]

# 4.4 排除 EOG 成分
exclude_idx = np.where(np.abs(scores) > 0.3)[0]
sources[:, exclude_idx] = 0  # 置零

# 4.5 重建数据 (NumPy)
data_clean = (sources @ mixing.T @ pca.components_) + pca.mean_
data_clean = data_clean.T

# 5. 分段 (NumPy)
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5)
epochs_data = epochs.get_data()  # (n_epochs, n_channels, n_times)

# 6. 统计 (NumPy)
evoked = epochs_data.mean(axis=0)  # 平均
evoked_std = epochs_data.std(axis=0)  # 标准差

# 7. 频谱分析 (SciPy)
fft_data = fft.rfft(evoked, axis=-1)
freqs = fft.rfftfreq(evoked.shape[-1], 1/sfreq)
psd = np.abs(fft_data) ** 2
```

---

## 最终建议

### 对于 EEG/EOG/EMG 处理，你需要掌握:

1. **NumPy** (⭐⭐⭐⭐⭐ 必需)
   - `ndarray` 操作
   - 索引与切片
   - 基本统计函数
   - 线性代数 (`linalg.svd`, `linalg.pinv`)

2. **SciPy** (⭐⭐⭐⭐⭐ 必需)
   - `scipy.signal`: 滤波器设计与应用
   - `scipy.fft`: 快速傅里叶变换
   - `scipy.stats`: 相关性分析
   - `scipy.signal.find_peaks`: 峰值检测

3. **scikit-learn** (⭐⭐⭐⭐ 强烈推荐)
   - `FastICA`: ICA 伪迹去除
   - `PCA`: 降维预处理

**学习顺序**:
1. 先掌握 NumPy 基础 (数组操作)
2. 再学 SciPy signal 模块 (滤波)
3. 最后学 scikit-learn ICA (伪迹去除)

**时间投入**:
- NumPy: 2-3 天 (基础) + 持续实践
- SciPy signal: 1-2 天 (滤波器) + 实践
- scikit-learn ICA: 1 天 (FastICA) + 实践

---

**返回**: [依赖分析总览](dependency-analysis-overview.md)
