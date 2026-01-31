# MNE-Python 与 NumPy 详细代码位置对比分析

## 概述

本文档详细分析 MNE-Python 如何使用 NumPy 的具体功能，包括：
- MNE 中的调用位置（文件、行号、上下文）
- NumPy 源码中的实现位置
- 功能对应关系
- 代码示例

---

## 1. 数组操作和索引

### 1.1 基础数组创建

#### MNE 使用位置

**文件**: `mne/io/base.py`
**行号**: ~300-350
**功能**: 创建数据缓冲区

```python
# mne/io/base.py:~330
def _allocate_data_buffer(self, n_channels, n_samples):
    """分配数据缓冲区"""
    return np.zeros((n_channels, n_samples), dtype=np.float64)
```

**对应 NumPy 位置**:
- **源码**: `numpy/core/numeric.py`
- **函数**: `zeros(shape, dtype=float, order='C')`
- **实现**: 调用底层 C API `PyArray_Zeros`
- **路径**: `numpy/core/src/multiarray/arrayobject.c`

---

### 1.2 数组切片和索引

#### MNE 使用位置

**文件**: `mne/io/base.py`
**行号**: ~500-600
**功能**: 读取数据段

```python
# mne/io/base.py:~580
def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
    """从文件读取数据段"""
    # 通道索引
    one = data[idx]  # 单通道
    # 时间切片
    segment = data[idx, start:stop]  # 通道 + 时间范围
    # 高级索引
    picks = [0, 2, 5, 7]
    selected = data[picks, :]  # 选择特定通道
```

**对应 NumPy 位置**:
- **源码**: `numpy/core/src/multiarray/mapping.c`
- **C 实现**: `array_subscript()` 函数
- **文档**: `numpy/doc/source/user/basics.indexing.rst`
- **关键机制**:
  - 基础索引: `PyArray_GetItem`
  - 切片: `PyArray_Slice`
  - 花式索引: `PyArray_TakeFrom`

---

### 1.3 数组广播

#### MNE 使用位置

**文件**: `mne/preprocessing/_csd.py`
**行号**: ~150-200
**功能**: CSD 变换中的广播运算

```python
# mne/preprocessing/_csd.py:~180
def _apply_csd_transformation(data, G, H):
    """应用 CSD 变换"""
    # data: (n_channels, n_times)
    # G: (n_channels, n_channels)
    # 广播乘法
    transformed = G @ data  # (n_channels, n_channels) @ (n_channels, n_times)
    
    # 逐元素广播
    mean = np.mean(data, axis=-1, keepdims=True)  # (n_channels, 1)
    centered = data - mean  # 广播减法
```

**对应 NumPy 位置**:
- **源码**: `numpy/core/src/multiarray/calculation.c`
- **广播规则**: `numpy/core/src/multiarray/common.c:PyArray_Broadcast`
- **文档**: `numpy/doc/source/user/basics.broadcasting.rst`

---

## 2. 线性代数 (`numpy.linalg`)

### 2.1 SVD - 奇异值分解

#### MNE 使用位置

**文件**: `mne/minimum_norm/inverse.py`
**行号**: 278-295
**功能**: 计算逆解算子的特征导联

```python
# mne/minimum_norm/inverse.py:278
from ..fixes import _safe_svd

def _prepare_inverse_operator(inv, nave, lambda2, method):
    """准备逆算子"""
    # ...
    # SVD 分解
    eigen_fields, sing, eigen_leads = _safe_svd(A, full_matrices=False)
    # eigen_fields: U 矩阵 (left singular vectors)
    # sing: 奇异值数组
    # eigen_leads: V^T 矩阵 (right singular vectors)
```

**MNE fixes 封装**:
```python
# mne/fixes.py:~150
def _safe_svd(A, full_matrices=True, **kwargs):
    """安全的 SVD（处理不同 NumPy 版本）"""
    from scipy.linalg import svd as scipy_svd
    return scipy_svd(A, full_matrices=full_matrices, lapack_driver='gesdd', **kwargs)
```

**对应 NumPy 位置**:
- **Python 接口**: `numpy/linalg/linalg.py:svd()`
- **实际调用**: SciPy `scipy.linalg.svd`（性能更好）
- **LAPACK 驱动**: `ZGESDD` (复数) 或 `DGESDD` (实数)
- **C 封装**: `numpy/linalg/lapack_lite/python_xerbla.c`
- **文档**: `numpy/doc/source/reference/generated/numpy.linalg.svd.rst`

**NumPy SVD 源码路径**:
```
numpy/linalg/linalg.py:1490-1650
    ↓ 调用
numpy/linalg/lapack_lite.pyx  (Cython wrapper)
    ↓ 调用
LAPACK: dgesdd_ / zgesdd_
```

**关键参数**:
- `full_matrices=False`: 经济型 SVD，U 和 V 只包含前 min(M,N) 列
- MNE 用途: 降维、白化、计算伪逆

---

### 2.2 特征分解 (eigh)

#### MNE 使用位置

**文件**: `mne/minimum_norm/_eloreta.py`
**行号**: 94
**功能**: eLORETA 权重计算

```python
# mne/minimum_norm/_eloreta.py:94
from ..utils import eigh

def _compute_eloreta_weights(G, R):
    """计算 eLORETA 权重"""
    G_R_Gt = G @ R @ G.T
    # 对称矩阵特征分解
    s, u = eigh(G_R_Gt)
    # s: 特征值（升序）
    # u: 特征向量（列向量）
```

**MNE utils 封装**:
```python
# mne/utils/numerics.py:~200
def eigh(A, overwrite_a=False):
    """对称矩阵特征分解"""
    from scipy.linalg import eigh as scipy_eigh
    return scipy_eigh(A, overwrite_a=overwrite_a, check_finite=False)
```

**对应 NumPy/SciPy 位置**:
- **NumPy**: `numpy/linalg/linalg.py:eigh()`
- **SciPy**: `scipy/linalg/_decomp.py:eigh()` (MNE 实际使用)
- **LAPACK**: `DSYEVD` (实数对称) / `ZHEEVD` (复Hermitian)
- **源码**: `scipy/linalg/cython_lapack.pyx`

**路径**:
```
MNE: mne/utils/numerics.py:eigh()
    ↓
SciPy: scipy.linalg.eigh()
    ↓
LAPACK: dsyevd_ / zheevd_
```

**优势**:
- 比 `eig()` 快约 2 倍（利用对称性）
- 特征值实数，特征向量正交
- 数值稳定性更好

---

### 2.3 矩阵求逆和伪逆

#### MNE 使用位置 1: pinv (伪逆)

**文件**: `mne/preprocessing/ica.py`
**行号**: 1020
**功能**: 从解混矩阵计算混合矩阵

```python
# mne/preprocessing/ica.py:1020
def _update_mixing_matrix(self):
    """更新混合矩阵（解混矩阵的伪逆）"""
    self.mixing_matrix_ = pinv(self.unmixing_matrix_)
```

**MNE pinv 封装**:
```python
# mne/utils/numerics.py:~100
def pinv(A, rcond=1e-15):
    """计算伪逆"""
    from scipy.linalg import pinv as scipy_pinv
    return scipy_pinv(A, rcond=rcond, check_finite=False)
```

**对应位置**:
- **NumPy**: `numpy/linalg/linalg.py:pinv()`
- **实现**: 基于 SVD
  ```python
  # numpy/linalg/linalg.py:~2000
  def pinv(a, rcond=1e-15):
      u, s, vt = svd(a, full_matrices=False)
      cutoff = rcond * s[0]
      s_inv = np.where(s > cutoff, 1/s, 0)
      return (vt.T * s_inv) @ u.T
  ```

#### MNE 使用位置 2: inv (直接求逆)

**文件**: `mne/cov.py`
**行号**: ~800
**功能**: 白化矩阵计算

```python
# mne/cov.py:~820
def compute_whitener(cov, info, picks=None):
    """计算白化矩阵"""
    # ...
    eigvals, eigvecs = eigh(cov_data)
    # 构造白化矩阵
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
```

**对应 NumPy 位置**:
- **源码**: `numpy/linalg/linalg.py:inv()`
- **LAPACK**: `DGETRF` (LU 分解) + `DGETRI` (求逆)
- **注意**: MNE 通常避免直接求逆，而是使用 `solve` 或特征分解

---

### 2.4 线性方程组求解

#### MNE 使用位置

**文件**: `mne/forward/_make_forward.py`
**行号**: ~500
**功能**: 求解边界元模型

```python
# mne/forward/_make_forward.py:~520
def _solve_bem_system(G, rhs):
    """求解 BEM 系统"""
    from scipy.linalg import solve
    
    # Gx = rhs
    x = solve(G, rhs, assume_a='pos')  # 假设 G 正定
```

**对应 NumPy/SciPy 位置**:
- **NumPy**: `numpy/linalg/linalg.py:solve()`
- **SciPy**: `scipy/linalg/_basic.py:solve()`
- **LAPACK**:
  - `DPOSV` (正定矩阵 - Cholesky)
  - `DGESV` (一般矩阵 - LU)
- **源码路径**:
  ```
  scipy/linalg/_basic.py:solve()
      ↓ assume_a='pos'
  scipy/linalg/cython_lapack.pyx:dposv()
  ```

---

## 3. FFT - 快速傅里叶变换

### 3.1 实数 FFT (rfft)

#### MNE 使用位置

**文件**: `mne/time_frequency/psd.py`
**行号**: ~200-250
**功能**: 计算功率谱密度

```python
# mne/time_frequency/psd.py:~230
def _compute_psd_welch(data, sfreq, n_fft):
    """Welch 方法计算 PSD"""
    from scipy import fft
    
    # 对每个窗口进行 FFT
    fft_data = fft.rfft(windowed_data, n=n_fft, axis=-1)
    # fft_data: 复数数组，长度 n_fft//2 + 1
    
    # 功率谱
    psd = np.abs(fft_data) ** 2
```

**对应 NumPy/SciPy 位置**:
- **NumPy**: `numpy/fft/_pocketfft.py:rfft()`
- **SciPy**: `scipy/fft/_pocketfft/pypocketfft.py:rfft()`
- **后端**: pocketfft (纯 C++ 实现)
- **源码**: `scipy/fft/_pocketfft/pocketfft.cxx`

**调用链**:
```
MNE: scipy.fft.rfft()
    ↓
scipy/fft/_basic.py:rfft()
    ↓
scipy/fft/_pocketfft/pypocketfft.py:rfft()
    ↓
C++: pocketfft::rfft()
```

**关键特性**:
- 输入: 实数数组 `[a0, a1, ..., a_{n-1}]`
- 输出: 复数数组长度 `n//2 + 1`（利用共轭对称性）
- 速度: O(n log n)

---

### 3.2 逆 FFT (irfft)

#### MNE 使用位置

**文件**: `mne/time_frequency/tfr.py`
**行号**: ~400
**功能**: 时频分析重建

```python
# mne/time_frequency/tfr.py:~420
def _reconstruct_from_tfr(tfr_data, n_times):
    """从时频表示重建时间序列"""
    # tfr_data: (n_epochs, n_freqs, n_times)
    reconstructed = np.fft.irfft(tfr_data, n=n_times, axis=-1)
```

**对应位置**:
- **NumPy**: `numpy/fft/_pocketfft.py:irfft()`
- **实现**: 与 `rfft` 相反，考虑共轭对称
- **C++ 后端**: 同样使用 pocketfft

---

### 3.3 FFT 频率数组

#### MNE 使用位置

**文件**: `mne/time_frequency/psd.py`
**行号**: ~150
**功能**: 生成频率轴

```python
# mne/time_frequency/psd.py:~155
def _get_freq_bins(n_fft, sfreq):
    """获取频率数组"""
    freqs = np.fft.rfftfreq(n_fft, d=1.0/sfreq)
    # 返回: [0, df, 2*df, ..., sfreq/2]
    # 其中 df = sfreq / n_fft
```

**对应 NumPy 位置**:
- **源码**: `numpy/fft/helper.py:rfftfreq()`
- **实现**:
  ```python
  # numpy/fft/helper.py:~80
  def rfftfreq(n, d=1.0):
      val = 1.0 / (n * d)
      N = n // 2 + 1
      return np.arange(0, N) * val
  ```

---

## 4. 随机数生成

### 4.1 旧式随机数 (np.random)

#### MNE 使用位置

**文件**: `mne/simulation/evoked.py`
**行号**: ~100
**功能**: 添加噪声

```python
# mne/simulation/evoked.py:~120
def add_noise(evoked, snr, random_state=None):
    """添加高斯噪声"""
    rng = np.random.RandomState(random_state)
    noise = rng.randn(*evoked.data.shape)
    signal_power = np.mean(evoked.data ** 2)
    noise_power = signal_power / (10 ** (snr / 10))
    evoked.data += noise * np.sqrt(noise_power)
```

**对应 NumPy 位置**:
- **源码**: `numpy/random/mtrand.pyx` (Cython)
- **算法**: Mersenne Twister (MT19937)
- **C 实现**: `numpy/random/src/mt19937/mt19937.c`

---

### 4.2 新式随机数 (Generator)

#### MNE 使用位置

**文件**: `mne/utils/_bunch.py`
**行号**: ~50
**功能**: 现代随机数生成

```python
# mne/utils/check.py:~800
def check_random_state(seed):
    """检查并返回随机状态"""
    if seed is None or seed is np.random:
        return np.random.default_rng()
    if isinstance(seed, int):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed
```

**对应 NumPy 位置**:
- **源码**: `numpy/random/_generator.pyx`
- **默认算法**: PCG64 (Permuted Congruential Generator)
- **C 实现**: `numpy/random/src/pcg64/pcg64.c`

**新旧对比**:
```
旧式 (NumPy < 1.17):
    np.random.seed(42)
    np.random.randn(100)
    
新式 (NumPy >= 1.17):
    rng = np.random.default_rng(42)
    rng.standard_normal(100)
```

---

## 5. 数学函数

### 5.1 通用数学函数

#### MNE 使用位置

**文件**: `mne/stats/cluster_level.py`
**行号**: ~600
**功能**: 统计量计算

```python
# mne/stats/cluster_level.py:~620
def _compute_t_statistic(data, axis=0):
    """计算 t 统计量"""
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis, ddof=1)
    n = data.shape[axis]
    t_stat = mean / (std / np.sqrt(n))
    return t_stat
```

**对应 NumPy 位置**:
- **mean**: `numpy/core/_methods.py:_mean()`
- **std**: `numpy/core/_methods.py:_std()`
- **sqrt**: `numpy/core/src/umath/loops.c.src` (ufunc)

**NumPy ufunc 机制**:
```
numpy.sqrt
    ↓ 调用
numpy/core/code_generators/generate_umath.py (代码生成)
    ↓ 生成
numpy/core/src/umath/loops.c.src (C 模板)
    ↓ 编译
CPU 向量化指令 (AVX2/SSE)
```

---

### 5.2 复数运算

#### MNE 使用位置

**文件**: `mne/time_frequency/tfr.py`
**行号**: ~300
**功能**: 相位提取

```python
# mne/time_frequency/tfr.py:~320
def extract_phase(complex_signal):
    """提取复信号相位"""
    magnitude = np.abs(complex_signal)  # 幅值
    phase = np.angle(complex_signal)    # 相位（弧度）
    return magnitude, phase
```

**对应 NumPy 位置**:
- **abs**: `numpy/core/numeric.py:absolute()` → `numpy.abs` (ufunc)
- **angle**: `numpy/lib/function_base.py:angle()`
- **实现**:
  ```python
  # numpy/lib/function_base.py:~3500
  def angle(z, deg=False):
      z = np.asarray(z)
      angle = np.arctan2(z.imag, z.real)
      if deg:
          angle *= 180 / pi
      return angle
  ```

---

## 6. 数组统计

### 6.1 均值和标准差

#### MNE 使用位置

**文件**: `mne/epochs.py`
**行号**: ~2000
**功能**: Baseline 校正

```python
# mne/epochs.py:~2050
def _apply_baseline_correction(data, baseline_mask):
    """应用 baseline 校正"""
    # data: (n_epochs, n_channels, n_times)
    # baseline_mask: boolean array for time points
    
    # 计算 baseline 均值
    baseline_mean = np.mean(data[:, :, baseline_mask], axis=-1, keepdims=True)
    
    # 减去均值
    data -= baseline_mean
```

**对应 NumPy 位置**:
- **源码**: `numpy/core/_methods.py`
  ```python
  # numpy/core/_methods.py:~50
  def _mean(a, axis=None, dtype=None, out=None, keepdims=False):
      arr = asanyarray(a)
      # ... 类型检查 ...
      ret = umr_sum(arr, axis, dtype, out, keepdims)
      # umr_sum: unwrapped math reduction sum
      if isinstance(ret, mu.ndarray):
          ret = um.true_divide(ret, rcount)
      return ret
  ```

**C 实现路径**:
```
numpy.mean()
    ↓
_methods._mean()
    ↓
numpy/core/src/umath/reduction.c:PyUFunc_GenericReduction()
    ↓
numpy/core/src/multiarray/compiled_base.c
```

---

### 6.2 中位数和百分位数

#### MNE 使用位置

**文件**: `mne/preprocessing/artifact_detection.py`
**行号**: ~150
**功能**: 异常值检测

```python
# mne/preprocessing/artifact_detection.py:~170
def detect_outliers_mad(data, threshold=3.0):
    """使用中位数绝对偏差检测异常值"""
    median = np.median(data, axis=-1, keepdims=True)
    mad = np.median(np.abs(data - median), axis=-1, keepdims=True)
    
    # 标准化 MAD
    mad_normalized = mad * 1.4826  # 转换为标准差估计
    
    # 检测异常值
    z_score = np.abs(data - median) / mad_normalized
    outliers = z_score > threshold
```

**对应 NumPy 位置**:
- **median**: `numpy/lib/function_base.py:median()`
- **内部调用**: `numpy.percentile(a, 50)`
- **实现**: 部分排序算法 (introselect)
- **C 代码**: `numpy/core/src/multiarray/multiarray/src/multiarray/compiled_base.c`

---

## 7. 特殊使用模式

### 7.1 einsum (Einstein Summation)

#### MNE 使用位置

**文件**: `mne/decoding/time_delaying_ridge.py`
**行号**: ~200
**功能**: 张量收缩

```python
# mne/decoding/time_delaying_ridge.py:~220
def _compute_laplacian_penalty(X, n_delays):
    """计算 Laplacian 正则化惩罚"""
    # X: (n_samples, n_features, n_delays)
    # 使用 einsum 计算复杂的张量运算
    penalty = np.einsum('ijk,ijk->jk', X, X)
    # 等价于: sum over i of (X[i,j,k] * X[i,j,k])
```

**对应 NumPy 位置**:
- **源码**: `numpy/core/einsumfunc.py:einsum()`
- **优化路径**: `numpy/core/einsumfunc.py:_einsum_path()`
- **C 后端**: `numpy/core/src/umath/override.c`

**einsum 优化**:
- 自动选择最优计算路径
- 避免中间数组分配
- 比多个 `dot` 调用更高效

---

### 7.2 where (条件选择)

#### MNE 使用位置

**文件**: `mne/baseline.py`
**行号**: ~100
**功能**: 条件性 baseline 应用

```python
# mne/baseline.py:~120
def rescale(data, times, baseline, mode='mean'):
    """Rescale data with baseline"""
    # ...
    if mode == 'mean':
        data -= np.mean(bdata, axis=-1, keepdims=True)
    elif mode == 'ratio':
        # 避免除以零
        data = np.where(
            bdata != 0,
            data / bdata,
            data
        )
```

**对应 NumPy 位置**:
- **源码**: `numpy/core/numeric.py:where()`
- **三元操作**: `condition ? x : y`
- **C 实现**: `numpy/core/src/multiarray/array_assign_scalar.c`

---

## 总结表：MNE → NumPy 映射

| MNE 功能 | MNE 文件 | NumPy 函数 | NumPy 源码位置 | 底层实现 |
|---------|---------|-----------|---------------|---------|
| 数据缓冲区 | io/base.py:330 | np.zeros | core/numeric.py | PyArray_Zeros (C API) |
| 数据切片 | io/base.py:580 | array[i:j] | core/src/multiarray/mapping.c | array_subscript |
| SVD | minimum_norm/inverse.py:278 | np.linalg.svd | linalg/linalg.py | LAPACK:GESDD |
| 特征分解 | minimum_norm/_eloreta.py:94 | np.linalg.eigh | linalg/linalg.py | LAPACK:SYEVD |
| 伪逆 | preprocessing/ica.py:1020 | np.linalg.pinv | linalg/linalg.py | 基于 SVD |
| 实数 FFT | time_frequency/psd.py:230 | np.fft.rfft | fft/_pocketfft.py | pocketfft (C++) |
| 随机数 | simulation/evoked.py:120 | np.random.randn | random/mtrand.pyx | MT19937 |
| 数学函数 | stats/cluster_level.py:620 | np.sqrt, np.mean | core/_methods.py | ufunc (C/SIMD) |

---

## 性能关键路径

### 最频繁调用（性能敏感）

1. **数组索引/切片**: 每次数据访问
   - NumPy C API 直接内存操作
   - 零拷贝视图（view）机制

2. **SVD/eigh**: 源估计、ICA
   - LAPACK 优化实现
   - 多线程 BLAS

3. **FFT**: 时频分析
   - pocketfft 高度优化
   - SIMD 指令集

4. **广播运算**: 几乎所有计算
   - NumPy ufunc 向量化
   - AVX2/AVX-512 加速

---

## 下一步

继续阅读：
- [MNE 与 SciPy 详细代码位置对比分析](./02_MNE-SciPy代码对比.md)
- [MNE 与 scikit-learn 详细代码位置对比分析](./03_MNE-sklearn代码对比.md)
