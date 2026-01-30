# 模块 2: Preprocessing - 预处理与伪迹去除

> **在数据流中的位置**: 第二步 - 原始数据清理  
> **核心职责**: 滤波、伪迹检测与去除、信号增强  
> **模块路径**: `mne/preprocessing/`, `mne/filter.py`

---

## 目录

1. [模块概述](#模块概述)
2. [滤波算法](#滤波算法)
3. [伪迹去除算法](#伪迹去除算法)
4. [信号增强算法](#信号增强算法)
5. [模态专用算法](#模态专用算法)
6. [应用场景](#应用场景)

---

## 模块概述

### 预处理管道

```
Raw Data
   ↓
[滤波] filter(), notch_filter()
   ↓
[伪迹检测] annotate_muscle_zscore(), find_ecg_events()
   ↓
[伪迹去除] ICA, SSP, Maxwell Filter, Regression
   ↓
[信号增强] CSD, Xdawn
   ↓
Clean Data
```

### 模块结构

```
mne/
├── filter.py                    # 滤波核心 (FilterMixin)
├── preprocessing/
│   ├── ica.py                  # ICA 实现 (3558行)
│   ├── ssp.py                  # 信号空间投影 (606行)
│   ├── maxwell.py              # Maxwell滤波 (3003行)
│   ├── _regress.py             # 回归方法 (391行)
│   ├── _csd.py                 # 电流源密度 (324行)
│   ├── xdawn.py                # Xdawn空间滤波 (534行)
│   │
│   ├── 伪迹检测
│   ├── ecg.py                  # 心电检测 (540行)
│   ├── eog.py                  # 眼电检测
│   ├── artifact_detection.py  # 自动化检测 (656行)
│   ├── _annotate_amplitude.py # 幅度异常
│   ├── _annotate_nan.py        # NaN检测
│   │
│   ├── 模态专用
│   ├── nirs/                   # fNIRS预处理
│   ├── eyetracking/            # 眼动预处理
│   └── ieeg/                   # 颅内脑电
```

---

## 滤波算法

### 1. FIR 滤波器设计

**算法位置**: `mne/filter.py:filter_data()` (行 400-800)

**核心算法**:
```python
def filter_data(data, sfreq, l_freq, h_freq, 
                filter_length='auto', method='fir'):
    """
    FIR 滤波器实现
    
    参数:
        l_freq: 高通截止频率 (None = 不应用高通)
        h_freq: 低通截止频率 (None = 不应用低通)
        filter_length: 滤波器长度 (影响频率分辨率和计算量)
        method: 'fir' | 'iir'
    
    FIR 设计方法:
        - firwin: 窗函数法 (默认)
        - firwin2: 任意频率响应
    """
    
    # 1. 计算滤波器参数
    if filter_length == 'auto':
        # 根据过渡带宽自动计算
        transition_bandwidth = min(
            0.25 * min([freq for freq in [l_freq, h_freq] if freq]),
            2.0  # 最大 2 Hz
        )
        filter_length = int(np.ceil(
            3.3 / (transition_bandwidth / sfreq) 
        ))
        # 确保奇数长度（对称性）
        filter_length += (filter_length % 2 == 0)
    
    # 2. 设计滤波器
    if l_freq is None and h_freq is not None:
        # 低通滤波器
        h = firwin(filter_length, h_freq, window='hamming', 
                   fs=sfreq, pass_zero=True)
    elif l_freq is not None and h_freq is None:
        # 高通滤波器
        h = firwin(filter_length, l_freq, window='hamming',
                   fs=sfreq, pass_zero=False)
    else:
        # 带通滤波器
        h = firwin(filter_length, [l_freq, h_freq], 
                   window='hamming', fs=sfreq, pass_zero=False)
    
    # 3. 应用滤波器（FFT 卷积）
    if method == 'fft':
        # 分段处理避免内存溢出
        n_fft = _get_optim_nfft(data.shape[-1], filter_length)
        filtered = fftconvolve(data, h[np.newaxis, :], mode='same')
    else:
        # 直接卷积（小数据）
        filtered = np.apply_along_axis(
            lambda m: np.convolve(m, h, mode='same'), 
            axis=-1, arr=data
        )
    
    return filtered
```

**特点**:
- **线性相位**: 无相位失真
- **稳定性**: FIR 总是稳定的
- **可控过渡带**: 精确控制频率响应

**计算复杂度**: O(N × M)，其中 N=数据长度，M=滤波器长度  
**内存需求**: O(N + M)

---

### 2. IIR 滤波器

**算法位置**: `mne/filter.py:filter_data(method='iir')`

**实现**:
```python
from scipy.signal import iirfilter, filtfilt

def iir_filter(data, sfreq, l_freq, h_freq, order=5):
    """
    IIR (Butterworth) 滤波器
    
    优势:
        - 更陡峭的滚降（相同阶数）
        - 更短的滤波器长度
        - 适用于实时处理
    
    劣势:
        - 非线性相位（需双向滤波）
        - 可能不稳定
    """
    # 设计 Butterworth 滤波器
    if l_freq and h_freq:
        btype = 'bandpass'
        Wn = [l_freq, h_freq]
    elif l_freq:
        btype = 'highpass'
        Wn = l_freq
    else:
        btype = 'lowpass'
        Wn = h_freq
    
    b, a = iirfilter(
        order, Wn, btype=btype, 
        ftype='butter', fs=sfreq
    )
    
    # 双向滤波（零相位）
    filtered = filtfilt(b, a, data, axis=-1)
    
    return filtered
```

**应用**:
- 实时处理系统
- 计算资源受限环境

---

### 3. 陷波滤波器

**算法位置**: `mne/filter.py:notch_filter()` (行 1200-1400)

**用途**: 去除工频噪声（50/60 Hz）

```python
def notch_filter(data, sfreq, freqs, notch_widths=None):
    """
    陷波滤波器实现
    
    参数:
        freqs: 陷波频率列表 [50, 100, 150] (基频+谐波)
        notch_widths: 陷波宽度 (默认为频率的 ±1 Hz)
    
    方法:
        1. 设计多个带阻滤波器
        2. 级联应用
    """
    if notch_widths is None:
        notch_widths = freqs / 200  # 默认 0.5% 带宽
    
    filtered = data.copy()
    for freq, width in zip(freqs, notch_widths):
        # 设计陷波器
        Q = freq / width  # 品质因子
        b, a = iirnotch(freq, Q, sfreq)
        
        # 应用
        filtered = filtfilt(b, a, filtered, axis=-1)
    
    return filtered
```

**常见配置**:
```python
# 欧洲: 50 Hz + 谐波
raw.notch_filter(freqs=[50, 100, 150, 200])

# 北美: 60 Hz + 谐波
raw.notch_filter(freqs=[60, 120, 180, 240])

# 自适应带宽
raw.notch_filter(freqs=np.arange(60, 241, 60), 
                 notch_widths=np.arange(60, 241, 60) / 200)
```

---

## 伪迹去除算法

### 1. ICA (独立成分分析)

**算法位置**: `mne/preprocessing/ica.py` (3558 行)

#### 1.1 FastICA 算法

**实现位置**: `ica.py:_fit_fastica()` (行 1800-2000)

**数学原理**:
$$
\text{最大化非高斯性: } J(w) = E\{G(w^T x)\}^2
$$

其中 $G$ 是非二次函数（如 $G(u) = \log \cosh(u)$）

**算法流程**:
```python
def _fit_fastica(X, n_components, max_iter=200):
    """
    FastICA 实现
    
    步骤:
        1. 中心化: X' = X - mean(X)
        2. 白化: Z = whitening_matrix @ X'
        3. 迭代优化权重矩阵 W
        4. 提取独立成分: S = W @ Z
    """
    # 1. 中心化
    X_mean = X.mean(axis=1, keepdims=True)
    X_centered = X - X_mean
    
    # 2. 白化（SVD）
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    K = U @ np.diag(1.0 / S)  # 白化矩阵
    Z = K.T @ X_centered
    
    # 3. FastICA 迭代
    W = np.random.randn(n_components, n_components)
    
    for iteration in range(max_iter):
        # 计算非线性函数
        gwtx = np.tanh(W @ Z)
        g_wtx = 1 - gwtx ** 2  # 导数
        
        # 更新权重
        W_new = (gwtx @ Z.T) / Z.shape[1] - \
                np.diag(g_wtx.mean(axis=1)) @ W
        
        # 正交化（对称）
        W_new = _sym_decorrelation(W_new)
        
        # 检查收敛
        if _converged(W, W_new, tolerance=1e-4):
            break
        
        W = W_new
    
    # 4. 提取独立成分
    unmixing_matrix = W @ K.T  # 去混合矩阵
    sources = unmixing_matrix @ X
    
    return sources, unmixing_matrix
```

**计算复杂度**: O(n²p + n³)，其中 n=通道数，p=时间点

---

#### 1.2 Infomax 算法

**实现位置**: `preprocessing/infomax_.py`

**原理**: 最大化输出熵
$$
H(Y) = H(g(W^T X)) \rightarrow \max
$$

**自然梯度更新**:
```python
def _infomax_update(W, X, learning_rate):
    """
    Infomax 自然梯度更新
    
    Extended Infomax 支持超高斯和亚高斯源
    """
    Y = W @ X
    
    # 非线性函数
    if extended:
        # 自适应选择
        u = np.tanh(Y)
        signs = np.sign(np.mean(Y * u, axis=1))
    else:
        u = np.tanh(Y)
        signs = np.ones(len(W))
    
    # 自然梯度
    dW = (np.eye(len(W)) - signs[:, None] * (u @ Y.T) / Y.shape[1]) @ W
    
    # 更新
    W += learning_rate * dW
    
    return W
```

---

#### 1.3 ICA 成分自动识别

**算法位置**: `ica.py:find_bads_ecg()`, `find_bads_eog()`

**ECG 成分检测**:
```python
def find_bads_ecg(ica, inst, method='correlation'):
    """
    基于 ECG 通道的心电成分自动检测
    
    方法:
        1. correlation: 与 ECG 通道的相关性
        2. ctps: 跨试次相位统计 (CTPS)
    """
    # 获取 ECG 通道或创建虚拟 ECG
    ecg_ch = _get_ecg_channel_index(inst.info)
    if ecg_ch is None:
        ecg_events = create_ecg_epochs(inst)
        ecg_signal = ecg_events.average().data.mean(axis=0)
    else:
        ecg_signal = inst.get_data(picks=[ecg_ch])[0]
    
    # 计算ICA成分与ECG的相关性
    ica_sources = ica.get_sources(inst).get_data()
    correlations = []
    
    for ic in range(ica.n_components_):
        r = np.corrcoef(ica_sources[ic], ecg_signal)[0, 1]
        correlations.append(abs(r))
    
    # 阈值选择（默认 r > 0.25）
    threshold = 0.25
    ecg_components = np.where(np.array(correlations) > threshold)[0]
    
    return ecg_components, np.array(correlations)
```

**EOG 成分检测**:
- 类似方法，使用 EOG 通道
- 或基于前额电极的差分

---

### 2. SSP (信号空间投影)

**算法位置**: `mne/preprocessing/ssp.py` (606 行)

**数学原理**:

投影算子：
$$
P = I - UU^T
$$

其中 $U$ 是伪迹子空间的正交基

**实现**:
```python
def compute_proj_ecg(raw, n_grad=2, n_mag=2, n_eeg=2):
    """
    计算 ECG 伪迹的 SSP 投影向量
    
    算法:
        1. 检测 ECG 事件
        2. 创建心跳锁定的 epochs
        3. 对 epochs 做 PCA
        4. 选取前 k 个主成分作为伪迹子空间
    """
    # 1. 检测心跳
    ecg_events = find_ecg_events(raw, ch_name=None)
    
    # 2. 创建 epochs
    ecg_epochs = Epochs(
        raw, ecg_events, 
        tmin=-0.2, tmax=0.4,
        baseline=None,
        preload=True
    )
    
    # 3. 分通道类型计算投影
    projs = []
    
    for ch_type, n_proj in [('grad', n_grad), ('mag', n_mag), ('eeg', n_eeg)]:
        if n_proj == 0:
            continue
        
        # 选择通道
        picks = pick_types(raw.info, meg=ch_type, eeg=(ch_type=='eeg'))
        data = ecg_epochs.get_data(picks=picks)
        
        # 计算协方差矩阵
        data_avg = data.mean(axis=0)  # 平均心跳模式
        cov = np.dot(data_avg, data_avg.T)
        
        # 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # 选取最大的 n_proj 个
        idx = np.argsort(eigenvalues)[::-1][:n_proj]
        vectors = eigenvectors[:, idx]
        
        # 创建投影对象
        proj = _make_proj(vectors, ch_type, 'ECG')
        projs.append(proj)
    
    return projs
```

**应用投影**:
```python
# 添加投影
raw.add_proj(projs)

# 应用投影（实际修改数据）
raw.apply_proj()

# 数据变换: X_clean = (I - UU^T) @ X
```

---

### 3. Maxwell 滤波 (仅 MEG)

**算法位置**: `mne/preprocessing/maxwell.py` (3003 行)

**理论基础**: 信号空间分离 (SSS)

**球面谐波展开**:
$$
\mathbf{B}(\mathbf{r}) = \mathbf{B}_{int}(\mathbf{r}) + \mathbf{B}_{ext}(\mathbf{r})
$$

内源（大脑）:
$$
\mathbf{B}_{int} = \sum_{l=1}^{L_{in}} \sum_{m=-l}^{l} a_l^m \mathbf{Y}_l^m(\theta, \phi)
$$

外源（环境）:
$$
\mathbf{B}_{ext} = \sum_{l=1}^{L_{out}} \sum_{m=-l}^{l} b_l^m \mathbf{Z}_l^m(\theta, \phi)
$$

**算法流程**:
```python
def maxwell_filter(raw, origin='auto', int_order=8, ext_order=3):
    """
    Maxwell 滤波实现
    
    参数:
        origin: 球心位置（头中心）
        int_order: 内源球谐阶数 (通常 8)
        ext_order: 外源球谐阶数 (通常 3)
    
    步骤:
        1. 构建球谐基函数矩阵 S [n_channels × n_moments]
        2. 最小二乘求解系数: α = (S^T S)^{-1} S^T B
        3. 仅保留内源重建: B_clean = S_int @ α_int
    """
    # 1. 获取传感器位置
    meg_info = pick_info(raw.info, pick_types(raw.info, meg=True))
    coil_positions = np.array([ch['loc'][:3] for ch in meg_info['chs']])
    coil_orientations = np.array([ch['loc'][3:6] for ch in meg_info['chs']])
    
    # 2. 构建球谐基
    n_int = _get_n_moments(int_order)  # (int_order + 1)^2 - 1
    n_ext = _get_n_moments(ext_order)
    
    S_int = _compute_sph_harm(coil_positions, origin, int_order, 'in')
    S_ext = _compute_sph_harm(coil_positions, origin, ext_order, 'out')
    S = np.hstack([S_int, S_ext])  # [n_channels × (n_int + n_ext)]
    
    # 3. 计算伪逆
    S_inv = np.linalg.pinv(S, rcond=1e-14)
    
    # 4. 应用到数据
    data = raw.get_data(picks='meg')
    
    # 分解系数
    coeffs = S_inv @ data  # [n_moments × n_times]
    
    # 仅保留内源
    coeffs_int = coeffs[:n_int]
    
    # 重建
    data_clean = S_int @ coeffs_int
    
    # 5. tSSS (时间扩展)
    if st_duration is not None:
        data_clean = _apply_temporal_sss(
            data_clean, S_int, st_duration, st_correlation
        )
    
    return data_clean
```

**时间 SSS (tSSS)**:

在滑动时间窗内应用 SSS，去除时间相关的伪迹

```python
def _apply_temporal_sss(data, S_int, duration, correlation_limit):
    """
    时间信号空间分离
    
    原理:
        - 脑信号随时间缓慢变化
        - 外部干扰可能快速变化
        - 在时间窗内检测异常子空间
    """
    window_samples = int(duration * sfreq)
    
    for start in range(0, data.shape[1], window_samples):
        end = start + window_samples
        segment = data[:, start:end]
        
        # 在此窗口内重新分离内外源
        # 并检测外源中的异常成分
        ...
    
    return data
```

**计算复杂度**: O(n³ + n²t)，n=通道数，t=时间点

---

### 4. 回归方法

**算法位置**: `mne/preprocessing/_regress.py` (391 行)

**原理**: 线性回归去除伪迹

$$
\mathbf{X}_{clean} = \mathbf{X} - \hat{\beta} \mathbf{R}
$$

其中 $\mathbf{R}$ 是参考通道（EOG, ECG），$\hat{\beta}$ 是回归系数

**实现**:
```python
def regress_artifact(inst, picks=None, picks_artifact='eog'):
    """
    基于回归的伪迹去除
    
    优势:
        - 简单直接
        - 不需要分解
        - 保留信号结构
    
    劣势:
        - 假设线性关系
        - 可能过度校正
    """
    # 获取数据
    X = inst.get_data(picks=picks)  # [n_channels × n_times]
    R = inst.get_data(picks=picks_artifact)  # [n_ref × n_times]
    
    # 计算回归系数 (最小二乘)
    # β = (R R^T)^{-1} R X^T
    beta = np.linalg.lstsq(R.T, X.T, rcond=None)[0].T
    
    # 去除伪迹
    X_clean = X - beta @ R
    
    return X_clean, beta
```

**EOGRegression 类**:
```python
class EOGRegression:
    """专用于 EOG 伪迹的回归"""
    
    def fit(self, epochs):
        # 从 epochs 学习回归系数
        self.betas_ = ...
    
    def apply(self, inst):
        # 应用学习的系数
        return inst_clean
```

---

## 信号增强算法

### 1. 电流源密度 (CSD)

**算法位置**: `mne/preprocessing/_csd.py` (324 行)

**数学基础**: 球面样条表面拉普拉斯算子

**拉普拉斯算子**:
$$
\nabla^2 V(\mathbf{r}) = -\frac{1}{\sigma} \nabla \cdot \mathbf{J}(\mathbf{r})
$$

**球面样条实现**:
```python
def compute_current_source_density(inst, lambda2=1e-5, stiffness=4):
    """
    CSD 变换
    
    参数:
        lambda2: 正则化参数（平滑度）
        stiffness: 样条刚度（阶数）
    
    效果:
        - 增强局部信号
        - 减少容积传导
        - "无参考"变换
    """
    # 1. 获取电极位置
    pos = np.array([ch['loc'][:3] for ch in inst.info['chs']])
    
    # 2. 拟合头模型（球）
    sphere_origin, sphere_radius = fit_sphere_to_headshape(inst.info)
    
    # 3. 计算 G 矩阵（距离加权）
    G = _calc_g(pos, sphere_radius, stiffness)  # Legendre 多项式
    
    # 4. 计算 H 矩阵（表面拉普拉斯）
    H = _calc_h(pos, sphere_radius, stiffness)
    
    # 5. 正则化求逆
    G_reg = G + lambda2 * np.eye(len(G))
    G_inv = np.linalg.inv(G_reg)
    
    # 6. CSD 变换矩阵
    CSD_matrix = H @ G_inv / (sphere_radius ** 2)
    
    # 7. 应用到数据
    data = inst.get_data()
    data_csd = CSD_matrix @ data
    
    return data_csd
```

**勒让德多项式**:
```python
def _calc_g(pos, radius, order):
    """计算样条基函数"""
    n = len(pos)
    G = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                G[i, j] = 0
            else:
                # 计算角度
                cos_theta = np.dot(pos[i], pos[j]) / (radius ** 2)
                
                # Legendre 多项式求和
                for l in range(1, order + 1):
                    P_l = legendre(l)(cos_theta)
                    G[i, j] += (2*l + 1) / (l * (l+1)) ** stiffness * P_l
    
    return G
```

**应用场景**:
- 提高 EEG 空间分辨率
- P300, N400 等 ERP 成分分析
- 去除参考电极影响

---

### 2. Xdawn 空间滤波

**算法位置**: `mne/preprocessing/xdawn.py` (534 行)

**用途**: 增强事件相关电位 (ERP)

**算法原理**:

最大化信噪比:
$$
\text{SNR} = \frac{w^T \Sigma_s w}{w^T \Sigma_n w}
$$

其中 $\Sigma_s$ 是信号协方差，$\Sigma_n$ 是噪声协方差

**实现**:
```python
class Xdawn:
    """Xdawn 空间滤波"""
    
    def fit(self, epochs, y):
        """
        学习 Xdawn 滤波器
        
        步骤:
            1. 重构连续信号（处理事件重叠）
            2. 最小二乘估计每个条件的诱发响应
            3. 计算信号/噪声协方差矩阵
            4. 广义特征值分解
        """
        # 1. 最小二乘诱发响应估计
        evokeds = _least_square_evoked(
            epochs.get_data(), 
            epochs.events, 
            epochs.tmin, 
            epochs.info['sfreq']
        )
        
        # 2. 计算协方差
        # 信号协方差
        signal_cov = np.zeros((n_channels, n_channels))
        for evoked in evokeds:
            signal_cov += evoked @ evoked.T
        
        # 数据协方差（总协方差）
        data = epochs.get_data()
        data_cov = np.cov(data.reshape(n_channels, -1))
        
        # 3. 广义特征值问题
        # signal_cov @ w = λ @ data_cov @ w
        eigenvalues, eigenvectors = eigh(signal_cov, data_cov)
        
        # 4. 选取最大特征值对应的向量
        idx = np.argsort(eigenvalues)[::-1]
        self.filters_ = eigenvectors[:, idx[:self.n_components]]
        
        return self
    
    def transform(self, epochs):
        """应用 Xdawn 滤波器"""
        data = epochs.get_data()
        data_filtered = np.dot(self.filters_.T, data.T).T
        return data_filtered
```

**应用**:
- P300 脑机接口
- oddball 范式分析
- 低信噪比 ERP 检测

---

## 模态专用算法

### fNIRS 预处理

**模块位置**: `mne/preprocessing/nirs/`

#### 1. 光密度变换

**文件**: `nirs/_optical_density.py`

```python
def optical_density(raw_intensity):
    """
    光强 → 光密度
    
    公式: OD = -log(I / I_0)
    
    其中:
        I: 测量的光强
        I_0: 参考光强（通常为最大值）
    """
    data = raw_intensity.get_data()
    
    # 计算光密度
    od_data = -np.log(data / data.max(axis=1, keepdims=True))
    
    return RawArray(od_data, raw_intensity.info)
```

#### 2. Beer-Lambert 定律

**文件**: `nirs/_beer_lambert_law.py`

```python
def beer_lambert_law(raw_od, ppf=0.1):
    """
    光密度 → 血红蛋白浓度变化
    
    公式:
        ΔC = (ε^T ε)^{-1} ε^T ΔOD / (ppf × d)
    
    其中:
        ε: 消光系数矩阵 [2 × 2] (HbO, HbR at λ1, λ2)
        d: 光程距离
        ppf: 部分光程因子
    """
    # 消光系数（单位: cm^{-1} / (moles/liter)）
    epsilon = np.array([
        [2.526, 1.221],  # 760 nm: [HbO, HbR]
        [0.734, 2.107]   # 850 nm
    ])
    
    # 源-探测器距离
    distances = source_detector_distances(raw_od.info)
    
    # 计算浓度变化
    od_data = raw_od.get_data()
    
    # 分通道对计算
    hbo_data = []
    hbr_data = []
    
    for ch_pair in channel_pairs:
        od_pair = od_data[ch_pair]  # [2 × n_times]
        
        # 求解线性方程
        conc = np.linalg.lstsq(epsilon, od_pair, rcond=None)[0]
        conc /= (ppf * distances[ch_pair[0]])
        
        hbo_data.append(conc[0])
        hbr_data.append(conc[1])
    
    # 创建新的 Raw 对象
    info_haemo = _create_haemo_info(raw_od.info)
    data_haemo = np.vstack([hbo_data, hbr_data])
    
    return RawArray(data_haemo, info_haemo)
```

#### 3. 头皮耦合指数 (SCI)

**文件**: `nirs/_scalp_coupling_index.py`

**用途**: 检测不良光极

```python
def scalp_coupling_index(raw_od, h_freq=0.5, h_trans_bandwidth=0.3):
    """
    计算 SCI
    
    原理:
        良好耦合: 心跳信号明显（~0.5-2 Hz）
        不良耦合: 缺乏生理信号
    
    指标:
        SCI = 心跳频段功率 / 总功率
    """
    # 滤波到心跳频段
    raw_cardiac = raw_od.copy().filter(
        l_freq=0.5, h_freq=2.5, picks='fnirs'
    )
    
    # 计算功率
    cardiac_power = np.var(raw_cardiac.get_data(), axis=1)
    total_power = np.var(raw_od.get_data(), axis=1)
    
    # SCI
    sci = cardiac_power / total_power
    
    # 阈值判断（通常 SCI < 0.5 为坏通道）
    bad_channels = np.where(sci < 0.5)[0]
    
    return sci, bad_channels
```

---

### 眼动追踪预处理

**模块位置**: `mne/preprocessing/eyetracking/`

#### 瞳孔插值

**文件**: `eyetracking/_pupillometry.py`

```python
def interpolate_blinks(raw_eyetrack, buffer=0.05):
    """
    眨眼期间瞳孔插值
    
    步骤:
        1. 检测眨眼（瞳孔尺寸 = 0 或异常小）
        2. 扩展眨眼边界（buffer）
        3. 三次样条插值
    """
    pupil_data = raw_eyetrack.get_data(picks='pupil')
    
    # 检测眨眼
    blink_mask = (pupil_data < 0.1 * pupil_data.mean()) | (pupil_data == 0)
    
    # 扩展边界
    buffer_samples = int(buffer * raw_eyetrack.info['sfreq'])
    from scipy.ndimage import binary_dilation
    blink_mask_extended = binary_dilation(
        blink_mask, iterations=buffer_samples
    )
    
    # 插值
    time = np.arange(len(pupil_data))
    valid_indices = ~blink_mask_extended
    
    from scipy.interpolate import interp1d
    interpolator = interp1d(
        time[valid_indices], 
        pupil_data[valid_indices],
        kind='cubic', 
        fill_value='extrapolate'
    )
    
    pupil_interpolated = interpolator(time)
    pupil_interpolated[valid_indices] = pupil_data[valid_indices]  # 保留原始
    
    return pupil_interpolated
```

---

## 应用场景

### 场景 1: 标准 EEG 预处理流程

```python
import mne
from mne.preprocessing import ICA

# 1. 加载数据
raw = mne.io.read_raw_brainvision('data.vhdr', preload=True)

# 2. 设置 montage
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# 3. 滤波
raw.filter(l_freq=1.0, h_freq=40.0)
raw.notch_filter(freqs=50)  # 欧洲工频

# 4. 重参考
raw.set_eeg_reference('average', projection=True)

# 5. ICA 去伪迹
ica = ICA(n_components=20, method='picard', random_state=42)
ica.fit(raw)

# 自动检测 EOG/ECG 成分
eog_indices, eog_scores = ica.find_bads_eog(raw)
ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
ica.exclude = eog_indices + ecg_indices

# 应用 ICA
raw_clean = ica.apply(raw.copy())

# 6. 插值坏通道
raw_clean.info['bads'] = ['Fp1', 'O2']
raw_clean.interpolate_bads()

print("预处理完成！")
```

---

### 场景 2: MEG Maxwell 滤波

```python
from mne.preprocessing import maxwell_filter

# 加载 MEG 数据
raw = mne.io.read_raw_fif('meg_data.fif')

# Maxwell 滤波（含头部运动补偿）
raw_sss = maxwell_filter(
    raw,
    origin='auto',          # 自动检测头中心
    int_order=8,            # 内源阶数
    ext_order=3,            # 外源阶数
    st_duration=10.0,       # tSSS 时间窗口（秒）
    st_correlation=0.98,    # 相关性阈值
    destination=None,       # 不进行头位对齐
    coord_frame='head',
    verbose=True
)

# 坏通道自动检测
from mne.preprocessing import find_bad_channels_maxwell
noisy_chs, flat_chs, scores = find_bad_channels_maxwell(
    raw, origin='auto', return_scores=True
)

print(f"检测到坏导: {noisy_chs + flat_chs}")
```

---

### 场景 3: fNIRS 完整流程

```python
from mne.preprocessing.nirs import (
    optical_density,
    beer_lambert_law,
    scalp_coupling_index
)

# 1. 加载原始光强数据
raw_intensity = mne.io.read_raw_nirx('nirx_data')

# 2. 转换为光密度
raw_od = optical_density(raw_intensity)

# 3. 检测坏通道（SCI）
sci = scalp_coupling_index(raw_od)
raw_od.info['bads'] = list(sci[sci < 0.5].index)

# 4. 转换为血红蛋白浓度
raw_haemo = beer_lambert_law(raw_od, ppf=0.1)

# 5. 滤波（去除生理噪声）
raw_haemo.filter(
    l_freq=0.01,   # 高通：去漂移
    h_freq=0.5,    # 低通：去心跳、呼吸
    picks='fnirs'
)

# 6. 可视化
raw_haemo.plot(duration=60, scalings=dict(hbo=1e-5, hbr=1e-5))
```

---

### 场景 4: 自动化批处理

```python
def preprocess_subject(subject_id):
    """单被试预处理流程"""
    
    # 加载
    raw = mne.io.read_raw_fif(f'sub-{subject_id}_raw.fif', preload=True)
    
    # 标准流程
    raw.filter(1, 40)
    raw.set_eeg_reference('average')
    
    # ICA
    ica = ICA(n_components=15, random_state=42)
    ica.fit(raw, decim=3)
    
    # 自动去伪迹
    eog_idx, _ = ica.find_bads_eog(raw, threshold=2.0)
    ecg_idx, _ = ica.find_bads_ecg(raw, threshold='auto')
    ica.exclude = eog_idx + ecg_idx
    
    raw = ica.apply(raw)
    
    # 保存
    raw.save(f'sub-{subject_id}_clean_raw.fif', overwrite=True)
    
    return raw

# 批处理
for sub_id in ['01', '02', '03', '04', '05']:
    try:
        preprocess_subject(sub_id)
        print(f"✓ Subject {sub_id} 完成")
    except Exception as e:
        print(f"✗ Subject {sub_id} 失败: {e}")
```

---

## 总结

### 核心算法汇总

| 算法 | 位置 | 复杂度 | 适用场景 |
|------|------|--------|---------|
| **滤波** |
| FIR | `filter.py:filter_data()` | O(N×M) | 线性相位需求 |
| IIR | `filter.py` | O(N) | 实时处理 |
| Notch | `filter.py:notch_filter()` | O(N) | 工频噪声 |
| **ICA** |
| FastICA | `ica.py:_fit_fastica()` | O(n²p) | 标准ICA |
| Infomax | `infomax_.py` | O(n²p) | 鲁棒性需求 |
| Picard | `ica.py` | O(n²p) | 快速收敛 |
| **SSP** |
| ECG/EOG SSP | `ssp.py` | O(n³) | 实时系统 |
| **Maxwell** |
| SSS | `maxwell.py` | O(n³) | MEG噪声抑制 |
| tSSS | `maxwell.py` | O(n³t) | 头部运动 |
| **回归** |
| 线性回归 | `_regress.py` | O(n²t) | 简单去伪迹 |
| **增强** |
| CSD | `_csd.py` | O(n³) | EEG空间分辨率 |
| Xdawn | `xdawn.py` | O(n³) | ERP增强 |

### 选择指南

**ICA vs SSP**:
- ICA: 更强大，需要足够数据，离线分析
- SSP: 更快，适合实时，需要明确的伪迹模式

**Maxwell vs ICA** (MEG):
- Maxwell: 物理建模，保留信号结构
- ICA: 数据驱动，可能过度分解

**CSD vs 重参考** (EEG):
- CSD: 无参考，高空间分辨率，适合 ERP
- 重参考: 简单，适合后续源定位

### 下一步

预处理完成后，进入 **模块3: 事件提取与Epoching**。
