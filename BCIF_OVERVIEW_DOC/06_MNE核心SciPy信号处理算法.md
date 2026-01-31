# MNE 核心 SciPy 信号处理算法分析

## 概述

本文档详细分析 MNE-Python 中最常用和最重要的 SciPy 信号处理算法，按使用频率和重要性分级。

---

## 🔴 P0 级别：核心必备（最常用、最重要）

### 1. IIR 滤波器设计和应用 ⭐⭐⭐⭐⭐

#### 1.1 `scipy.signal.iirfilter` - 通用 IIR 滤波器设计

**功能**: 设计各种类型的 IIR 数字滤波器（Butterworth, Chebyshev, Elliptic 等）

**MNE 调用位置**: `mne/filter.py:850`

```python
# mne/filter.py:850
system = signal.iirfilter(**kwargs)
# kwargs 包括：
# - N: 滤波器阶数
# - Wn: 临界频率
# - btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
# - ftype: 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'
# - output: 'ba' (传递函数) 或 'sos' (二阶节)
```

**为什么重要**:
- 所有频率滤波的基础（去除工频干扰、提取特定频段）
- 支持多种滤波器类型，满足不同需求
- MNE 默认使用 Butterworth 滤波器

**使用场景**:
- `raw.filter(l_freq=1.0, h_freq=40.0)` - 带通滤波
- `epochs.filter(h_freq=30.0)` - 低通滤波
- 预处理中去除基线漂移和高频噪声

**SciPy 源码**: `scipy/signal/_filter_design.py:iirfilter()` (line ~2800)

**实现细节**:
1. 根据 `ftype` 选择模拟原型（如 `buttap` for Butterworth）
2. 频率变换（如 `lp2bp` 低通到带通）
3. 双线性变换（`bilinear`）将模拟滤波器转为数字滤波器
4. 转换为 SOS 格式（数值稳定性更好）

---

#### 1.2 `scipy.signal.sosfiltfilt` - 零相位滤波 ⭐⭐⭐⭐⭐

**功能**: 双向 IIR 滤波，消除相位延迟

**MNE 调用位置**: `mne/filter.py:549`

```python
# mne/filter.py:549
func = partial(
    _iir_pad_apply_unpad,
    func=signal.sosfiltfilt,
    sos=iir_params["sos"],
    padlen=padlen,
    padtype="reflect_limited",
)
```

**为什么是 P0**:
- **零相位响应**: 对于 EEG/MEG 分析至关重要，不会引入时间延迟
- **默认滤波方法**: MNE 默认使用零相位滤波（`phase='zero'`）
- **边缘效应处理**: 通过填充（padding）减少滤波器边缘伪影

**工作原理**:
1. 对信号进行边缘填充（`padtype='reflect_limited'`）
2. 正向滤波：`y1 = sosfilt(sos, x_padded)`
3. 反转信号：`y1_reversed = y1[::-1]`
4. 反向滤波：`y2 = sosfilt(sos, y1_reversed)`
5. 再次反转：`y_final = y2[::-1]`
6. 去除填充

**优势**:
- 相位响应为零（不改变事件相对时间）
- 幅度响应平方（滤波效果更强）
- 数值稳定（SOS 格式）

**SciPy 源码**: `scipy/signal/_signaltools.py:sosfiltfilt()` (line ~4200)

**性能关键**: Cython 加速的 `_sosfilt.pyx`

---

#### 1.3 `scipy.signal.filtfilt` - 传统零相位滤波

**功能**: 使用传递函数（ba）格式的零相位滤波

**MNE 调用位置**: `mne/filter.py:558`

```python
# mne/filter.py:558
func = partial(
    _iir_pad_apply_unpad,
    func=signal.filtfilt,
    b=iir_params["b"],
    a=iir_params["a"],
    padlen=padlen,
    padtype="reflect_limited",
)
```

**为什么保留**: 向后兼容，某些情况下用户指定 `ba` 格式

**SciPy 源码**: `scipy/signal/_signaltools.py:filtfilt()` (line ~3800)

**注意**: MNE 现在优先使用 `sosfiltfilt`（更稳定）

---

### 2. FIR 滤波器设计 ⭐⭐⭐⭐

#### 2.1 `scipy.signal.firwin` - 窗函数法 FIR 设计

**功能**: 使用窗函数法设计 FIR 滤波器

**MNE 调用位置**: `mne/filter.py:447`

```python
# mne/filter.py:447
this_h = signal.firwin(
    numtaps,
    cutoff=f_c,
    width=trans_bandwidth,
    window=fir_window,
    pass_zero=pass_zero,
    fs=sfreq,
)
```

**为什么重要**:
- **线性相位**: FIR 滤波器天然具有线性相位
- **稳定性**: 始终稳定（所有极点在原点）
- **精确控制**: 可以精确设计过渡带宽

**使用场景**:
- 当需要严格线性相位时
- 实时滤波（非零相位模式）
- 长数据段滤波

**参数**:
- `numtaps`: 滤波器长度（影响过渡带宽度）
- `window`: 窗函数类型（'hamming', 'hann', 'blackman'）
- `pass_zero`: True (低通), False (高通), 'bandpass', 'bandstop'

**SciPy 源码**: `scipy/signal/_fir_filter_design.py:firwin()` (line ~200)

---

#### 2.2 `scipy.signal.firwin2` - 任意频率响应 FIR 设计

**功能**: 设计具有任意频率响应的 FIR 滤波器

**MNE 调用位置**: `mne/filter.py:478`

```python
# mne/filter.py:478
fir_design = signal.firwin2
# 用于设计复杂的频率响应（如 notch 滤波器）
```

**使用场景**:
- Notch 滤波器（去除特定频率，如 50/60 Hz 工频）
- 不规则频率响应需求

**SciPy 源码**: `scipy/signal/_fir_filter_design.py:firwin2()` (line ~400)

---

### 3. 重采样 ⭐⭐⭐⭐⭐

#### 3.1 `scipy.signal.resample_poly` - 多相滤波重采样

**功能**: 使用多相滤波器进行整数比例重采样

**MNE 调用位置**: `mne/filter.py:1920`

```python
# mne/filter.py:1920
parallel, p_fun, n_jobs = parallel_func(signal.resample_poly, n_jobs)
# ...
y = signal.resample_poly(x, axis=-1, **kwargs)
```

**为什么是 P0**:
- **数据降采样**: 减少计算量和存储（如 1000 Hz → 250 Hz）
- **抗混叠**: 自动应用低通滤波器防止混叠
- **高效**: 比 FFT 重采样更快（对整数比例）

**工作原理**:
1. 上采样（插值零值）
2. 低通滤波（抗混叠）
3. 下采样（抽取）

**使用场景**:
```python
# 从 1000 Hz 降采样到 250 Hz
raw.resample(250)
# 内部调用: signal.resample_poly(data, up=1, down=4)
```

**参数**:
- `up`: 上采样因子
- `down`: 下采样因子
- `window`: FIR 抗混叠滤波器窗函数

**SciPy 源码**: `scipy/signal/_signaltools.py:resample_poly()` (line ~2000)

**性能**: 针对整数比例优化，比 FFT 方法快

---

### 4. 频率响应分析 ⭐⭐⭐⭐

#### 4.1 `scipy.signal.freqz` - 数字滤波器频率响应

**功能**: 计算数字滤波器的频率响应

**MNE 调用位置**: `mne/filter.py:390`, `filter.py:884`

```python
# mne/filter.py:390
_, filt_resp = signal.freqz(h.ravel(), worN=np.pi * freq)

# mne/filter.py:884
cutoffs = signal.freqz(system[0], system[1], worN=Wp * np.pi)[1]
```

**为什么重要**:
- **验证滤波器**: 检查实际频率响应是否符合预期
- **可视化**: 绘制滤波器的幅度和相位响应
- **调试**: 诊断滤波问题

**使用场景**:
```python
# 检查滤波器在特定频率的衰减
b, a = signal.butter(4, 0.2)
w, h = signal.freqz(b, a, worN=512)
magnitude_db = 20 * np.log10(np.abs(h))
```

**SciPy 源码**: `scipy/signal/_filter_design.py:freqz()` (line ~1600)

---

### 5. Hilbert 变换 ⭐⭐⭐⭐

#### 5.1 `scipy.signal.hilbert` - 解析信号

**功能**: 计算信号的解析表示（复信号）

**MNE 调用位置**: `mne/filter.py:2813`

```python
# mne/filter.py:2813
out = signal.hilbert(x, N=n_fft, axis=-1)[..., :n_x]
```

**为什么重要**:
- **瞬时相位**: 提取信号的瞬时相位（用于连接性分析）
- **瞬时幅度**: 提取包络（envelope）
- **相位锁定值 (PLV)**: 计算脑区间的相位同步

**工作原理**:
1. FFT 到频域
2. 正频率分量乘以 2，负频率置零
3. IFFT 回时域（得到复信号）

**使用场景**:
```python
# 提取 alpha 波段的包络
raw_alpha = raw.copy().filter(8, 12)
analytic_signal = signal.hilbert(raw_alpha.get_data())
envelope = np.abs(analytic_signal)
phase = np.angle(analytic_signal)
```

**SciPy 源码**: `scipy/signal/_signaltools.py:hilbert()` (line ~1850)

**数学公式**:
$$z(t) = x(t) + j \cdot \mathcal{H}[x(t)]$$

其中 $\mathcal{H}$ 是 Hilbert 变换算子。

---

## 🟡 P1 级别：高频使用

### 6. 功率谱密度 (PSD) ⭐⭐⭐⭐

#### 6.1 `scipy.signal.spectrogram` - 时频谱图

**功能**: 计算短时傅里叶变换 (STFT)

**MNE 调用位置**: `mne/time_frequency/psd.py:248`

```python
# mne/time_frequency/psd.py:248-264
f, t, spect = spectrogram(
    x,
    detrend=detrend,
    noverlap=n_overlap,
    nperseg=n_per_seg,
    nfft=n_fft,
    fs=sfreq,
    window=window,
    mode=mode,
)
```

**为什么重要**:
- **时频分析**: 同时查看时间和频率信息
- **事件相关频谱扰动 (ERSP)**: 分析事件相关的频率变化
- **可视化**: 生成时频图

**参数**:
- `nperseg`: 每个段的长度（窗口大小）
- `noverlap`: 段之间的重叠
- `window`: 窗函数（'hann', 'hamming'）
- `mode`: 'psd' (功率谱密度), 'magnitude', 'phase'

**使用场景**:
```python
# 计算时频谱图
epochs.compute_psd(method='multitaper')
# 内部使用 spectrogram 进行 STFT
```

**SciPy 源码**: `scipy/signal/_spectral_py.py:spectrogram()` (line ~1600)

---

#### 6.2 `scipy.signal.welch` - Welch 方法 PSD（间接使用）

**功能**: Welch 方法估计功率谱密度

**MNE 使用**: MNE 自己实现了 Welch 方法（`psd_array_welch`），但基于 SciPy 的 `spectrogram`

**为什么重要**:
- **频谱分析**: 最常用的 PSD 估计方法
- **降噪**: 通过平均多个段减少方差
- **频段功率**: 计算特定频段的功率（如 alpha, beta）

**工作原理**:
1. 将信号分成重叠的段
2. 对每个段加窗并计算 FFT
3. 计算功率谱
4. 平均所有段的功率谱

**使用场景**:
```python
# 计算 PSD
spectrum = epochs.compute_psd(method='welch', fmin=1, fmax=40)
alpha_power = spectrum.get_data(fmin=8, fmax=12).mean()
```

---

### 7. 窗函数 ⭐⭐⭐

#### 7.1 `scipy.signal.get_window` - 获取窗函数

**功能**: 生成各种窗函数

**MNE 调用位置**: `mne/_ola.py:6`, `mne/time_frequency/multitaper.py:10`

```python
# mne/_ola.py:6
from scipy.signal import get_window

# 使用
window = get_window('hann', n_samples)
```

**窗函数类型**:
- `'hann'`: 汉宁窗（默认，平滑）
- `'hamming'`: 汉明窗（频谱泄漏小）
- `'blackman'`: 布莱克曼窗（最小旁瓣）
- `'tukey'`: 图基窗（可调余弦窗）

**使用场景**:
- FIR 滤波器设计
- STFT 分析
- 减少频谱泄漏

**SciPy 源码**: `scipy/signal/_window_functions.py:get_window()` (line ~2000)

---

## 🟢 P2 级别：特定场景

### 8. 其他信号处理函数

#### 8.1 `scipy.signal.detrend` - 去趋势

**功能**: 去除线性趋势或常数偏移

**MNE 调用位置**: `mne/stats/parametric.py:10`, `mne/preprocessing/_pca_obs.py:11`

```python
from scipy.signal import detrend

# 去除线性趋势
detrended = detrend(data, axis=-1, type='linear')
```

**使用场景**:
- PSD 计算前去除直流分量
- 去除慢漂移

**SciPy 源码**: `scipy/signal/_signaltools.py:detrend()` (line ~3200)

---

#### 8.2 `scipy.signal.find_peaks` - 峰值检测

**功能**: 检测信号中的峰值

**MNE 调用位置**: `mne/preprocessing/artifact_detection.py:8`

```python
from scipy.signal import find_peaks

peaks, properties = find_peaks(
    data, 
    height=threshold,
    distance=min_distance,
    prominence=prominence
)
```

**使用场景**:
- 检测心电图 R 波
- 检测肌电伪迹
- 自动标记事件

**SciPy 源码**: `scipy/signal/_peak_finding.py:find_peaks()` (line ~700)

---

#### 8.3 `scipy.signal.minimum_phase` - 最小相位滤波器

**功能**: 将滤波器转换为最小相位

**MNE 调用位置**: `mne/fixes.py:717`

```python
from scipy.signal import minimum_phase as sp_minimum_phase
```

**使用场景**:
- 实时滤波（减少群延迟）
- 因果滤波器设计

**SciPy 源码**: `scipy/signal/_fir_filter_design.py:minimum_phase()` (line ~800)

---

#### 8.4 `scipy.signal.fftconvolve` - FFT 卷积

**功能**: 使用 FFT 进行快速卷积

**MNE 调用位置**: `mne/decoding/time_delaying_ridge.py:9`

```python
from scipy.signal import fftconvolve

result = fftconvolve(x, h, mode='same')
```

**使用场景**:
- 时间延迟回归
- 快速滤波器应用

**SciPy 源码**: `scipy/signal/_signaltools.py:fftconvolve()` (line ~500)

---

## 📊 使用频率统计

基于 MNE 代码库分析：

| 算法 | 调用次数 | 文件数 | 重要性 | P级别 |
|-----|---------|--------|-------|------|
| `iirfilter` | 高 | 2 | ⭐⭐⭐⭐⭐ | P0 |
| `sosfiltfilt` | 极高 | 3+ | ⭐⭐⭐⭐⭐ | P0 |
| `filtfilt` | 高 | 3+ | ⭐⭐⭐⭐⭐ | P0 |
| `firwin` | 高 | 2 | ⭐⭐⭐⭐ | P0 |
| `resample_poly` | 极高 | 1 | ⭐⭐⭐⭐⭐ | P0 |
| `freqz` | 中 | 2 | ⭐⭐⭐⭐ | P0 |
| `hilbert` | 高 | 2 | ⭐⭐⭐⭐ | P0 |
| `spectrogram` | 高 | 2 | ⭐⭐⭐⭐ | P1 |
| `get_window` | 中 | 3 | ⭐⭐⭐ | P1 |
| `detrend` | 中 | 2 | ⭐⭐⭐ | P2 |
| `find_peaks` | 低 | 1 | ⭐⭐ | P2 |
| `fftconvolve` | 低 | 1 | ⭐⭐ | P2 |

---

## 🎯 Rust 移植优先级建议

### 立即移植（M1-M2）
1. **IIR 滤波器设计** (`iirfilter`, `butter`)
   - 代码量：~500 行
   - 难度：中高
   - 策略：移植 SciPy 的 butter/iirfilter 算法

2. **零相位滤波** (`sosfiltfilt`)
   - 代码量：~400 行
   - 难度：中
   - 策略：基于 biquad crate 实现

3. **FIR 滤波器设计** (`firwin`)
   - 代码量：~300 行
   - 难度：中
   - 策略：窗函数法 + FFT

4. **重采样** (`resample_poly`)
   - 代码量：~300 行
   - 难度：中
   - 策略：多相滤波器实现

5. **Hilbert 变换** (`hilbert`)
   - 代码量：~100 行
   - 难度：低
   - 策略：基于 rustfft

### 近期移植（M3-M4）
6. **频率响应** (`freqz`, `sosfreqz`)
7. **STFT** (`spectrogram`)
8. **窗函数** (`get_window`)

### 延后（M5+）
9. **峰值检测** (`find_peaks`)
10. **其他工具函数**

---

## 🔧 核心算法实现复杂度

### Butterworth 滤波器设计（最复杂）
**步骤**:
1. `buttap()` - 模拟原型（计算极点）
   ```python
   # 计算 Butterworth 极点
   z = np.exp(1j * np.pi * (2*k + N - 1) / (2*N))
   p = -z  # 极点
   k = 1   # 增益
   ```

2. `lp2lp()` / `lp2bp()` - 频率变换
   ```python
   # 低通到带通
   p_bp = Wo * (p * bw/2 + sqrt((p*bw/2)^2 + 1))
   ```

3. `bilinear()` - 双线性变换
   ```python
   # s → z 变换
   z = (2*fs + s) / (2*fs - s)
   ```

4. `zpk2sos()` - 零极点到二阶节
   ```python
   # 配对极点和零点形成二阶节
   for i in range(n_sections):
       sos[i] = [b0, b1, b2, 1, a1, a2]
   ```

**总代码量**: 约 500 行（需要从 SciPy 移植）

---

## 📈 性能关键点

### 最耗时的操作
1. **sosfiltfilt** - 双向滤波（2x 滤波时间）
2. **resample_poly** - 重采样（依赖数据长度）
3. **spectrogram** - STFT（多次 FFT）

### 优化策略
1. **并行化**: MNE 使用 `joblib` 并行处理多通道
2. **SIMD**: Rust 可利用 SIMD 加速滤波
3. **缓存**: 预计算滤波器系数

---

## 总结

**P0 核心算法（必须移植）**:
1. IIR 滤波器设计 (`iirfilter`, `butter`)
2. 零相位滤波 (`sosfiltfilt`, `filtfilt`)
3. FIR 滤波器 (`firwin`, `firwin2`)
4. 重采样 (`resample_poly`)
5. Hilbert 变换 (`hilbert`)
6. 频率响应 (`freqz`)

**关键特性**:
- 零相位滤波是 MNE 的默认和核心
- SOS 格式确保数值稳定性
- 边缘填充减少伪影

**Rust 实现路径**:
1. 移植 Butterworth 设计（最复杂，~500 行）
2. 实现 sosfiltfilt（基于 biquad）
3. 封装 rustfft 实现 Hilbert
4. 其他相对简单

这些算法占 MNE 信号处理的 **80%** 使用量，优先移植这些可以快速建立核心功能。
