# MNE-Python NumPy 依赖详细分析

> **核心依赖**: `numpy >= 1.26, < 3`  
> **使用频率**: 🔥🔥🔥🔥🔥 (100%)  
> **角色**: 数据存储、数学运算、线性代数、FFT

---

## 目录

1. [NumPy 在 MNE 中的角色](#numpy-在-mne-中的角色)
2. [核心模块使用](#核心模块使用)
3. [数据结构设计](#数据结构设计)
4. [线性代数应用](#线性代数应用)
5. [FFT 应用场景](#fft-应用场景)
6. [数学函数使用](#数学函数使用)
7. [性能优化技巧](#性能优化技巧)
8. [代码示例](#代码示例)

---

## NumPy 在 MNE 中的角色

### 1. 数据存储基础

**所有 MNE 数据对象的底层都是 NumPy 数组**:

```python
# MNE 核心对象内部结构
class Raw(BaseRaw):
    def __init__(self, ...):
        self._data = np.ndarray  # shape: (n_channels, n_times)
        
class Epochs(BaseEpochs):
    def __init__(self, ...):
        self._data = np.ndarray  # shape: (n_epochs, n_channels, n_times)
        
class Evoked(Evoked):
    def __init__(self, ...):
        self.data = np.ndarray  # shape: (n_channels, n_times)
        
class SourceEstimate:
    def __init__(self, ...):
        self.data = np.ndarray  # shape: (n_vertices, n_times)
```

**数据流**:
```
文件读取 → NumPy 数组 → 信号处理 → NumPy 数组 → 可视化/保存
```

---

### 2. NumPy 模块使用统计

| NumPy 模块 | 使用文件数 | 主要用途 | 关键函数 |
|-----------|-----------|---------|---------|
| **核心数组操作** | ~500 | 数据处理 | `np.array`, `np.zeros`, `np.ones`, `np.concatenate` |
| **numpy.linalg** | ~150 | 线性代数 | `np.linalg.norm`, `np.linalg.svd`, `np.linalg.eig` |
| **numpy.fft** | ~80 | 频域分析 | `np.fft.rfft`, `np.fft.irfft`, `np.fft.fftfreq` |
| **numpy.random** | ~120 | 随机数生成 | `np.random.randn`, `np.random.permutation` |
| **numpy.testing** | ~200 | 单元测试 | `assert_allclose`, `assert_array_equal` |
| **numpy.polynomial** | ~5 | 多项式计算 | `legendre.legval` (Legendre 多项式) |

---

## 核心模块使用

### 1. 数组创建与操作

**位置**: 几乎所有模块

**常用函数**:
```python
# 创建数组
np.zeros((n_channels, n_times))      # 初始化全零数组
np.ones(shape)                        # 全一数组
np.empty(shape, dtype=np.float64)    # 未初始化数组(性能优化)
np.arange(start, stop, step)          # 等差数列
np.linspace(start, stop, num)         # 线性空间

# 数组操作
np.concatenate([arr1, arr2], axis=0)  # 拼接
np.stack([arr1, arr2], axis=0)        # 堆叠
np.split(arr, indices_or_sections)    # 分割
np.transpose(arr, axes)               # 转置
np.reshape(arr, new_shape)            # 重塑
arr.ravel()                           # 展平

# 索引与切片
arr[start:stop:step]                  # 基础切片
arr[indices]                          # 索引数组
arr[mask]                             # 布尔掩码
np.where(condition, x, y)             # 条件选择
```

**示例** - `mne/epochs.py`:
```python
def _get_data(self, item=None, ...):
    # 使用 NumPy 索引提取 epoch 数据
    data = self._data[item]  # shape: (n_selected, n_channels, n_times)
    
    # 使用 NumPy 拼接
    if self.preload:
        data = np.concatenate([self._data[i] for i in indices], axis=0)
```

---

### 2. numpy.linalg - 线性代数

**位置**: `mne/rank.py`, `mne/cov.py`, `mne/minimum_norm/`, `mne/beamformer/`

**核心函数使用**:

#### 2.1 矩阵范数

```python
# 位置: mne/rank.py
import numpy as np

def compute_rank(data, tol='auto'):
    """计算数据矩阵的秩"""
    # 计算 Frobenius 范数
    norm = np.linalg.norm(data, 'fro')  
    
    # 奇异值分解
    s = np.linalg.svd(data, compute_uv=False)
    
    # 确定秩
    rank = np.sum(s > tol * s[0])
    return rank
```

**常用范数函数**:
- `np.linalg.norm(x, ord=2)` - 向量 2-范数 (欧几里得距离)
- `np.linalg.norm(x, ord=1)` - 1-范数 (曼哈顿距离)
- `np.linalg.norm(A, 'fro')` - Frobenius 范数 (矩阵)

---

#### 2.2 奇异值分解 (SVD)

```python
# 位置: mne/utils/linalg.py
def _safe_svd(A, full_matrices=True, **kwargs):
    """安全的 SVD 计算，处理 NaN 和无穷值"""
    # NumPy SVD
    U, s, Vh = np.linalg.svd(A, full_matrices=full_matrices)
    
    # 对于复数矩阵，确保特征向量相位一致
    if np.iscomplexobj(A):
        # 调整相位
        U = U * np.sign(U[0, :])
    
    return U, s, Vh
```

**SVD 应用场景**:
- **PCA 降维**: `mne/preprocessing/xdawn.py`
- **伪逆计算**: `mne/channels/interpolation.py`
- **秩估计**: `mne/rank.py`
- **去噪**: `mne/preprocessing/maxwell.py`

---

#### 2.3 特征值分解

```python
# 位置: mne/decoding/_ged.py
import numpy as np

def _compute_ged(S, R):
    """广义特征值分解 (Generalized Eigenvalue Decomposition)"""
    # 标准特征值分解
    evals, evecs = np.linalg.eigh(S)  # 对称矩阵
    
    # 或者广义形式
    from scipy.linalg import eigh
    evals, evecs = eigh(S, R)  # S evecs = R evecs * evals
    
    return evals, evecs
```

**特征值分解应用**:
- **空间滤波器**: CSP, GED
- **协方差对角化**: `mne/cov.py`
- **源定位**: eLORETA

---

#### 2.4 矩阵求逆与伪逆

```python
# 位置: mne/utils/numerics.py
def _reg_pinv(x, reg=0, rank='full', rcond=1e-15):
    """正则化伪逆"""
    U, s, Vh = np.linalg.svd(x, full_matrices=False)
    
    # 正则化奇异值
    s_inv = 1.0 / (s + reg * s[0])
    
    # 计算伪逆
    pinv = (Vh.T * s_inv) @ U.T
    
    return pinv
```

**应用场景**:
- **通道插值**: `mne/channels/interpolation.py`
- **正向模型求逆**: `mne/forward/`
- **最小二乘**: `mne/preprocessing/xdawn.py`

---

### 3. numpy.fft - 快速傅里叶变换

**位置**: `mne/filter.py`, `mne/time_frequency/`, `mne/cuda.py`

**核心函数**:

#### 3.1 实数 FFT (rfft/irfft)

```python
# 位置: mne/filter.py
import numpy as np

def filter_data_fft(data, sfreq, l_freq, h_freq):
    """使用 FFT 实现频域滤波"""
    n_times = data.shape[-1]
    
    # 前向 FFT (实数输入)
    data_fft = np.fft.rfft(data, n=n_times, axis=-1)
    
    # 频率向量
    freqs = np.fft.rfftfreq(n_times, 1.0 / sfreq)
    
    # 构建频域滤波器
    mask = (freqs >= l_freq) & (freqs <= h_freq)
    data_fft[..., ~mask] = 0
    
    # 逆 FFT
    data_filtered = np.fft.irfft(data_fft, n=n_times, axis=-1)
    
    return data_filtered
```

**rfft vs fft**:
- `rfft`: 实数输入 → 减半的复数输出 (利用对称性)
- `fft`: 复数输入 → 完整复数输出

**性能**: `rfft` 速度约为 `fft` 的 2 倍

---

#### 3.2 频率向量生成

```python
# 位置: mne/time_frequency/_stft.py
def stftfreq(wsize, sfreq=None):
    """STFT 频率向量"""
    from scipy.fft import rfftfreq
    
    # 使用 SciPy 版本 (与 NumPy 兼容)
    freqs = rfftfreq(wsize, 1.0 / sfreq)
    
    return freqs
```

**相关函数**:
- `np.fft.fftfreq(n, d)` - 完整频率向量
- `np.fft.rfftfreq(n, d)` - 实数 FFT 频率向量
- `np.fft.fftshift(x)` - 将零频率移到中心

---

#### 3.3 FFT 应用场景

| 应用 | 模块 | 函数 | 用途 |
|------|------|------|------|
| **频域滤波** | `filter.py` | `rfft/irfft` | FIR 滤波器卷积 |
| **功率谱密度** | `time_frequency/multitaper.py` | `rfft` | PSD 计算 |
| **时频分析** | `time_frequency/_stft.py` | `rfft` | STFT, 频谱图 |
| **重采样** | `filter.py` | `rfft` | FFT-based 重采样 |
| **CUDA 加速** | `cuda.py` | `rfft` | GPU FFT 卷积 |

---

### 4. numpy.random - 随机数生成

**位置**: `mne/stats/`, `mne/simulation/`, `mne/utils/`

**随机数生成器 (RNG) 管理**:

```python
# 位置: mne/utils/check.py
def check_random_state(seed):
    """将种子转换为 NumPy RandomState"""
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(f"Invalid seed: {seed}")
```

**常用随机函数**:

```python
# 正态分布
rng = np.random.RandomState(42)
noise = rng.randn(n_channels, n_times)  # 标准正态

# 排列组合
perm = rng.permutation(n_samples)  # 随机排列索引

# 整数随机
idx = rng.randint(0, n_samples, size=n_bootstraps)  # 有放回抽样

# 均匀分布
weights = rng.rand(n_features)  # [0, 1) 均匀分布
```

**应用**:
- **Permutation Test**: `mne/stats/permutations.py`
- **Bootstrap**: `mne/stats/cluster_level.py`
- **数据模拟**: `mne/simulation/`

---

### 5. numpy.testing - 单元测试

**位置**: 所有 `tests/` 目录

**核心断言函数**:

```python
from numpy.testing import (
    assert_allclose,         # 浮点数近似相等
    assert_array_equal,      # 数组完全相等
    assert_array_almost_equal,  # 数组近似相等 (旧式)
    assert_array_less,       # 数组元素逐个比较 <
)

# 示例用法
def test_filter():
    data = np.random.randn(10, 1000)
    filtered = filter_data(data, sfreq=100, l_freq=1, h_freq=40)
    
    # 检查形状
    assert_array_equal(filtered.shape, data.shape)
    
    # 检查数值精度
    assert_allclose(filtered.mean(), 0, atol=0.1)
```

**参数说明**:
- `rtol`: 相对容差 (relative tolerance)
- `atol`: 绝对容差 (absolute tolerance)
- `equal_nan`: 将 NaN 视为相等

---

## 数据结构设计

### 1. MNE 数据对象与 NumPy 数组映射

```python
# Raw 对象
raw._data: np.ndarray
    shape: (n_channels, n_times)
    dtype: np.float64 or np.float32
    layout: C-contiguous (行优先)

# Epochs 对象  
epochs._data: np.ndarray
    shape: (n_epochs, n_channels, n_times)
    dtype: np.float64
    layout: C-contiguous

# Evoked 对象
evoked.data: np.ndarray
    shape: (n_channels, n_times)
    dtype: np.float64
    layout: C-contiguous

# SourceEstimate 对象
stc.data: np.ndarray
    shape: (n_vertices, n_times)
    dtype: np.float64
    layout: C-contiguous
```

---

### 2. NumPy dtype 选择

**MNE 使用策略**:

| 数据类型 | NumPy dtype | 使用场景 |
|---------|-------------|---------|
| **EEG/MEG 数据** | `np.float64` | 高精度需求 (默认) |
| **存储优化** | `np.float32` | 减少内存 (可选) |
| **整数索引** | `np.int64` | 事件编码、索引数组 |
| **布尔掩码** | `np.bool_` | 数据选择、坏通道标记 |
| **复数** | `np.complex128` | 频域表示、Fourier 系数 |

**示例** - `mne/io/base.py`:
```python
def _read_data(self, dtype=np.float64):
    """读取数据到指定精度"""
    data = self._read_raw_data()
    
    # 转换精度以节省内存
    if dtype == np.float32:
        data = data.astype(np.float32, copy=False)
    
    return data
```

---

### 3. 内存布局优化

**C-contiguous vs Fortran-contiguous**:

```python
# C-contiguous (行优先) - MNE 默认
arr_c = np.array([[1, 2, 3],
                   [4, 5, 6]], order='C')
# 内存布局: [1, 2, 3, 4, 5, 6]

# Fortran-contiguous (列优先) - LAPACK 偏好
arr_f = np.array([[1, 2, 3],
                   [4, 5, 6]], order='F')
# 内存布局: [1, 4, 2, 5, 3, 6]
```

**MNE 策略**:
- **数据存储**: C-contiguous (沿时间轴操作高效)
- **LAPACK 调用**: 自动转换为 Fortran-contiguous (避免复制)

**示例** - `mne/utils/linalg.py`:
```python
def _safe_svd(A, **kwargs):
    """确保输入为 Fortran-contiguous 以提高 LAPACK 性能"""
    if not np.isfortran(A):
        A = np.asfortranarray(A)  # 转换但不一定复制
    
    U, s, Vh = np.linalg.svd(A, **kwargs)
    return U, s, Vh
```

---

## 线性代数应用

### 1. 协方差矩阵计算

**位置**: `mne/cov.py`

```python
def _compute_covariance_from_epochs(epochs):
    """计算 epochs 的协方差矩阵"""
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape
    
    # 方法 1: 使用 NumPy 广播
    data_centered = data - data.mean(axis=2, keepdims=True)
    cov = np.einsum('ijk,ilk->jl', data_centered, data_centered)
    cov /= (n_epochs * n_times - 1)
    
    # 方法 2: 使用矩阵乘法 (更快)
    data_flat = data.reshape(n_epochs * n_times, n_channels)
    data_flat -= data_flat.mean(axis=0)
    cov = (data_flat.T @ data_flat) / (data_flat.shape[0] - 1)
    
    return cov
```

**性能**: `einsum` vs `@` 取决于数据大小和形状

---

### 2. 白化 (Whitening)

**位置**: `mne/cov.py`, `mne/preprocessing/ica.py`

```python
def compute_whitener(cov, reg=0.1):
    """计算白化矩阵"""
    # 特征值分解
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # 正则化 (避免数值不稳定)
    eigvals_reg = eigvals + reg * eigvals.max()
    
    # 白化矩阵: W = V * diag(1/sqrt(lambda)) * V^T
    whitener = eigvecs @ np.diag(1.0 / np.sqrt(eigvals_reg)) @ eigvecs.T
    
    return whitener

def apply_whitening(data, whitener):
    """应用白化"""
    # data shape: (n_channels, n_times)
    data_white = whitener @ data
    return data_white
```

**应用**:
- **ICA 预处理**: `mne/preprocessing/ica.py`
- **CSP 空间滤波**: `mne/decoding/csp.py`
- **噪声归一化**: `mne/minimum_norm/inverse.py`

---

### 3. 矩阵分解技巧

#### 3.1 Cholesky 分解

```python
# 位置: mne/cov.py
def _regularized_covariance_cholesky(data, reg=0.1):
    """使用 Cholesky 分解的协方差正则化"""
    cov = np.cov(data)
    
    # 添加正则化项
    cov[np.diag_indices_from(cov)] += reg
    
    # Cholesky 分解 (要求正定)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # 如果失败，使用 SVD 替代
        U, s, Vh = np.linalg.svd(cov)
        L = U @ np.diag(np.sqrt(s))
    
    return L
```

---

#### 3.2 QR 分解

```python
# 位置: mne/preprocessing/_csd.py
def _orthogonalize_vectors(X):
    """使用 QR 分解正交化列向量"""
    Q, R = np.linalg.qr(X)
    
    # Q 的列向量是正交的
    # X ≈ Q @ R
    
    return Q
```

**应用**: CSD (Current Source Density) 计算

---

## FFT 应用场景

### 1. FIR 滤波器实现

**位置**: `mne/filter.py`

**原理**: 时域卷积 = 频域乘法

```python
def fir_filter_fft(data, h, n_fft):
    """使用 FFT 实现 FIR 滤波"""
    # h: 滤波器系数 (impulse response)
    # n_fft: FFT 长度
    
    # 1. 零填充滤波器系数
    h_padded = np.zeros(n_fft)
    h_padded[:len(h)] = h
    
    # 2. FFT
    H = np.fft.rfft(h_padded)        # 滤波器频域表示
    X = np.fft.rfft(data, n=n_fft)   # 数据频域表示
    
    # 3. 频域乘法
    Y = X * H
    
    # 4. 逆 FFT
    y = np.fft.irfft(Y, n=n_fft)
    
    # 5. 提取有效部分 (去除 padding)
    y = y[:len(data)]
    
    return y
```

**优势**:
- 时域卷积: O(N * M)
- FFT 卷积: O(N log N)

---

### 2. 时频分析 (STFT)

**位置**: `mne/time_frequency/_stft.py`

```python
def stft(x, wsize, tstep):
    """短时傅里叶变换"""
    # x: 信号, shape (n_times,)
    # wsize: 窗口大小
    # tstep: 时间步长
    
    n_times = len(x)
    n_freqs = wsize // 2 + 1
    n_windows = (n_times - wsize) // tstep + 1
    
    # 初始化输出
    X = np.zeros((n_freqs, n_windows), dtype=np.complex128)
    
    # 汉宁窗
    window = np.hanning(wsize)
    
    # 滑动窗口 FFT
    for i in range(n_windows):
        start = i * tstep
        end = start + wsize
        
        # 加窗 + FFT
        x_windowed = x[start:end] * window
        X[:, i] = np.fft.rfft(x_windowed)
    
    return X
```

**应用**:
- **频谱图**: `mne.viz.plot_epochs_psd_topomap()`
- **时频分解**: `mne.time_frequency.tfr_morlet()`

---

### 3. Hilbert 变换

**位置**: `mne/preprocessing/ctps_.py`

```python
def hilbert_transform(data):
    """使用 FFT 实现 Hilbert 变换"""
    from scipy.signal import hilbert
    
    # SciPy 内部实现也使用 FFT
    analytic_signal = hilbert(data, axis=-1)
    
    # 提取相位
    phase = np.angle(analytic_signal)
    
    # 提取幅度包络
    amplitude = np.abs(analytic_signal)
    
    return phase, amplitude
```

**应用**: CTPS (Cross-Trial Phase Statistics) - 相位一致性分析

---

## 数学函数使用

### 1. 三角函数

```python
# 位置: mne/transforms.py
def rotation3d(x=0, y=0, z=0):
    """3D 旋转矩阵"""
    cos_x, sin_x = np.cos(x), np.sin(x)
    cos_y, sin_y = np.cos(y), np.sin(y)
    cos_z, sin_z = np.cos(z), np.sin(z)
    
    R = np.array([
        [cos_y*cos_z, -cos_x*sin_z + sin_x*sin_y*cos_z,  ...],
        [cos_y*sin_z,  cos_x*cos_z + sin_x*sin_y*sin_z,  ...],
        [-sin_y,       sin_x*cos_y,                       ...]
    ])
    
    return R
```

---

### 2. 统计函数

```python
# 位置: mne/utils/numerics.py
def compute_corr(x, y):
    """计算 Pearson 相关系数"""
    # x: shape (n_features,)
    # y: shape (n_samples, n_features)
    
    # 中心化
    x_centered = x - x.mean()
    y_centered = y - y.mean(axis=-1, keepdims=True)
    
    # 相关系数
    corr = (y_centered @ x_centered) / (
        np.sqrt((y_centered ** 2).sum(axis=-1)) * 
        np.sqrt((x_centered ** 2).sum())
    )
    
    return corr
```

---

### 3. 数值稳定性技巧

```python
# 位置: mne/utils/numerics.py
def _log_sum_exp(x, axis=None):
    """数值稳定的 log(sum(exp(x)))"""
    x_max = x.max(axis=axis, keepdims=True)
    
    # log(sum(exp(x))) = log(sum(exp(x - x_max) * exp(x_max)))
    #                  = log(sum(exp(x - x_max))) + x_max
    out = np.log(np.sum(np.exp(x - x_max), axis=axis)) + x_max.squeeze()
    
    return out
```

**应用**: 避免上溢/下溢

---

## 性能优化技巧

### 1. 广播 (Broadcasting)

```python
# 低效: 使用循环
result = np.zeros((n_epochs, n_channels, n_times))
for i in range(n_epochs):
    for j in range(n_channels):
        result[i, j, :] = data[i, j, :] - baseline[j]

# 高效: 使用广播
baseline = data.mean(axis=2, keepdims=True)  # shape: (n_epochs, n_channels, 1)
result = data - baseline  # 自动广播
```

---

### 2. 预分配数组

```python
# 低效: 动态增长
result = []
for i in range(n_samples):
    result.append(process(data[i]))
result = np.array(result)

# 高效: 预分配
result = np.empty((n_samples, output_size))
for i in range(n_samples):
    result[i] = process(data[i])
```

---

### 3. 原地操作 (In-place)

```python
# 创建新数组
data_norm = data / np.linalg.norm(data, axis=-1, keepdims=True)

# 原地修改 (节省内存)
data /= np.linalg.norm(data, axis=-1, keepdims=True)
```

---

### 4. einsum 优化

```python
# 低效: 多步矩阵乘法
result = (A @ B @ C.T)

# 高效: 一次 einsum
result = np.einsum('ij,jk,lk->il', A, B, C)
```

**何时使用 einsum**:
- ✅ 复杂的张量收缩
- ✅ 自定义求和路径
- ❌ 简单的矩阵乘法 (`@` 更快)

---

## 代码示例

### 示例 1: Epoch 平均 (Evoked)

```python
# 位置: mne/epochs.py
def average(epochs):
    """计算 epochs 平均"""
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    
    # NumPy 平均
    evoked_data = data.mean(axis=0)  # (n_channels, n_times)
    
    return evoked_data
```

---

### 示例 2: 基线校正

```python
# 位置: mne/baseline.py
def rescale(data, times, baseline, mode='mean'):
    """基线校正"""
    # 找到基线时间索引
    bmin, bmax = baseline
    baseline_mask = (times >= bmin) & (times <= bmax)
    
    if mode == 'mean':
        # 减去基线均值
        baseline_mean = data[..., baseline_mask].mean(axis=-1, keepdims=True)
        data -= baseline_mean
        
    elif mode == 'ratio':
        # 除以基线均值
        baseline_mean = data[..., baseline_mask].mean(axis=-1, keepdims=True)
        data /= baseline_mean
        
    elif mode == 'zscore':
        # Z-score 标准化
        baseline_mean = data[..., baseline_mask].mean(axis=-1, keepdims=True)
        baseline_std = data[..., baseline_mask].std(axis=-1, keepdims=True)
        data = (data - baseline_mean) / baseline_std
    
    return data
```

---

### 示例 3: PCA 降维

```python
# 位置: mne/utils/numerics.py
class _PCA:
    """简化的 PCA 实现"""
    
    def fit(self, X):
        """拟合 PCA"""
        # X: (n_samples, n_features)
        
        # 中心化
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        
        # 协方差矩阵
        cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
        
        # 特征值分解
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # 降序排序
        idx = eigvals.argsort()[::-1]
        self.explained_variance_ = eigvals[idx]
        self.components_ = eigvecs[:, idx].T  # (n_components, n_features)
        
    def transform(self, X, n_components=None):
        """转换数据"""
        X_centered = X - self.mean_
        X_pca = X_centered @ self.components_[:n_components].T
        return X_pca
```

---

## 总结

### NumPy 在 MNE 中的重要性

| 维度 | 评分 | 说明 |
|------|------|------|
| **数据存储** | ⭐⭐⭐⭐⭐ | 所有数据对象的底层 |
| **数学运算** | ⭐⭐⭐⭐⭐ | 核心计算引擎 |
| **线性代数** | ⭐⭐⭐⭐⭐ | SVD, 特征值, 矩阵运算 |
| **FFT** | ⭐⭐⭐⭐⭐ | 频域分析、滤波 |
| **测试** | ⭐⭐⭐⭐⭐ | numpy.testing 覆盖所有测试 |
| **性能影响** | ⭐⭐⭐⭐⭐ | BLAS/LAPACK 优化至关重要 |

---

### 关键要点

1. **NumPy 是 MNE 的基石** - 无法替代
2. **数组操作模式** - 广播、向量化、避免循环
3. **线性代数** - SVD 和特征值分解是核心
4. **FFT 应用广泛** - 滤波、时频分析、重采样
5. **性能关键** - 内存布局、预分配、原地操作

---

**返回**: [依赖分析总览](dependency-analysis-overview.md)  
**下一步**: [SciPy 依赖分析](dependency-scipy.md)
