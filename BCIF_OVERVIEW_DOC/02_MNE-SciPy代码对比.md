# MNE-Python 与 SciPy 详细代码位置对比分析

## 概述

本文档详细分析 MNE-Python 如何使用 SciPy 的具体功能，重点关注：
- 信号处理 (`scipy.signal`)
- 稀疏矩阵 (`scipy.sparse`)
- 优化算法 (`scipy.optimize`)
- 统计函数 (`scipy.stats`)
- 线性代数 (`scipy.linalg`)

---

## 1. 信号处理 (`scipy.signal`)

### 1.1 IIR 滤波器设计和应用

#### MNE 使用位置 1: Butterworth 滤波器设计

**文件**: `mne/filter.py`
**行号**: 700-900
**功能**: 构造 IIR 滤波器参数

```python
# mne/filter.py:~850
from scipy import signal

def construct_iir_filter(iir_params, f_pass=None, f_stop=None, 
                        sfreq=None, btype='low', output='sos'):
    """构造 IIR 滤波器"""
    
    # 使用 scipy.signal.butter 设计 Butterworth 滤波器
    if iir_params['ftype'] == 'butter':
        # 归一化频率
        Wp = f_pass / (sfreq / 2.0)  # 通带边缘
        
        # 设计滤波器 - 返回二阶节（SOS）表示
        sos = signal.butter(
            N=iir_params['order'],  # 滤波器阶数
            Wn=Wp,                  # 归一化频率
            btype=btype,            # 滤波器类型
            analog=False,           # 数字滤波器
            output='sos'            # 二阶节输出
        )
        iir_params['sos'] = sos
```

**对应 SciPy 位置**:
- **源码**: `scipy/signal/_filter_design.py:butter()`
- **行号**: ~3500-3650
- **实现路径**:
  ```
  scipy.signal.butter()
      ↓ 调用
  scipy.signal.iirfilter()  (通用 IIR 设计)
      ↓ 调用
  scipy.signal.buttap()  (模拟原型)
      ↓ 调用
  scipy.signal.lp2lp/lp2hp/lp2bp()  (频率变换)
      ↓ 调用
  scipy.signal.bilinear()  (双线性变换到数字域)
  ```

**源码详细**:
```python
# scipy/signal/_filter_design.py:3500
def butter(N, Wn, btype='low', analog=False, output='ba', fs=None):
    """Butterworth数字和模拟滤波器设计"""
    return iirfilter(N, Wn, btype=btype, analog=analog,
                     output=output, ftype='butter', fs=fs)

# scipy/signal/_filter_design.py:2800
def iirfilter(N, Wn, rp=None, rs=None, btype='band', analog=False,
              ftype='butter', output='ba', fs=None):
    """IIR数字和模拟滤波器设计"""
    # ... 参数验证 ...
    
    # 获取模拟原型
    z, p, k = _design_method(N, ...)  # 零点、极点、增益
    
    # 频率变换
    z, p, k = _transform(z, p, k, Wn, btype)
    
    # 双线性变换到数字域
    z, p, k = bilinear_zpk(z, p, k, fs=fs)
    
    # 转换为输出格式
    if output == 'sos':
        return zpk2sos(z, p, k)
```

**关键文件路径**:
- `scipy/signal/_filter_design.py:3500` - butter()
- `scipy/signal/_filter_design.py:2800` - iirfilter()
- `scipy/signal/_filter_design.py:1200` - buttap() (模拟原型)
- `scipy/signal/_filter_design.py:500` - bilinear() (双线性变换)
- `scipy/signal/_filter_design.py:800` - zpk2sos() (零极点到SOS)

---

#### MNE 使用位置 2: sosfiltfilt (零相位滤波)

**文件**: `mne/filter.py`
**行号**: 540-580
**功能**: 双向滤波（零相位响应）

```python
# mne/filter.py:549
from scipy import signal

def _iir_filter(x, iir_params, picks, n_jobs, copy, phase="zero"):
    """应用 IIR 滤波"""
    # ...
    if phase in ("zero", "zero-double"):
        padlen = min(iir_params["padlen"], x.shape[-1] - 1)
        
        if "sos" in iir_params:
            # 使用二阶节表示的零相位滤波
            fun = partial(
                _iir_pad_apply_unpad,
                func=signal.sosfiltfilt,  # 核心：双向滤波
                sos=iir_params["sos"],
                padlen=padlen,
                padtype="reflect_limited",
            )
        else:
            # 使用传递函数表示
            fun = partial(
                _iir_pad_apply_unpad,
                func=signal.filtfilt,  # 传统双向滤波
                b=iir_params["b"],
                a=iir_params["a"],
                padlen=padlen,
                padtype="reflect_limited",
            )
```

**对应 SciPy 位置**:
- **源码**: `scipy/signal/_signaltools.py:sosfiltfilt()`
- **行号**: ~4200-4350
- **调用链**:
  ```
  scipy.signal.sosfiltfilt(sos, x, padlen=padlen)
      ↓
  scipy.signal._sosfilt.sosfilt()  (正向滤波)
      ↓
  _sosfilt_cython()  (Cython 加速)
      ↓
  反转数组
      ↓
  scipy.signal._sosfilt.sosfilt()  (反向滤波)
      ↓
  反转数组（恢复）
  ```

**源码详细**:
```python
# scipy/signal/_signaltools.py:4200
def sosfiltfilt(sos, x, axis=-1, padtype='odd', padlen=None):
    """零相位数字滤波（使用二阶节）"""
    # ... 参数检查 ...
    
    # 边缘填充
    if padlen is not None:
        x = _pad_signal(x, padlen, padtype, axis)
    
    # 正向滤波
    y = sosfilt(sos, x, axis=axis)
    
    # 反转
    y = np.flip(y, axis=axis)
    
    # 反向滤波
    y = sosfilt(sos, y, axis=axis)
    
    # 再次反转恢复
    y = np.flip(y, axis=axis)
    
    # 移除填充
    if padlen is not None:
        y = _remove_padding(y, padlen, axis)
    
    return y
```

**关键特性**:
- **零相位**: 前向+后向滤波消除相位延迟
- **二阶节**: 数值稳定性优于传递函数（ba）
- **填充**: `padtype='reflect_limited'` 减少边缘效应

**性能关键路径**:
- `scipy/signal/_sosfilt.pyx` - Cython 实现
- `scipy/signal/_sosfilt_impl.c` - C 实现（直接型II）

---

#### MNE 使用位置 3: 滤波器频率响应

**文件**: `mne/filter.py`
**行号**: ~1500
**功能**: 计算滤波器频率响应

```python
# mne/filter.py:~1550
from scipy import signal

def _compute_filter_response(iir_params, sfreq, n_freq=1000):
    """计算滤波器的频率响应"""
    if 'sos' in iir_params:
        # 使用 sosfreqz 计算 SOS 滤波器响应
        w, h = signal.sosfreqz(
            iir_params['sos'],
            worN=n_freq,
            fs=sfreq
        )
        # w: 频率数组（Hz）
        # h: 复数频率响应
    else:
        # 使用 freqz 计算传递函数响应
        w, h = signal.freqz(
            iir_params['b'],
            iir_params['a'],
            worN=n_freq,
            fs=sfreq
        )
    
    # 计算幅度和相位
    magnitude = np.abs(h)
    phase = np.angle(h)
    return w, magnitude, phase
```

**对应 SciPy 位置**:
- **sosfreqz**: `scipy/signal/_filter_design.py:sosfreqz()`
- **freqz**: `scipy/signal/_filter_design.py:freqz()`
- **行号**: ~1800-2000

**实现**:
```python
# scipy/signal/_filter_design.py:1850
def sosfreqz(sos, worN=512, whole=False, fs=2*pi):
    """计算数字滤波器的频率响应（SOS格式）"""
    # 对每个二阶节计算响应
    h = np.ones(len(w), dtype=complex)
    for section in sos:
        b = section[:3]  # 分子系数
        a = section[3:]  # 分母系数
        # 计算该节的响应
        h *= _frequency_response(b, a, w)
    return w, h

def _frequency_response(b, a, w):
    """单个节的频率响应"""
    # H(e^jw) = B(e^jw) / A(e^jw)
    zm1 = np.exp(-1j * w)  # e^{-jw}
    h = np.polyval(b[::-1], zm1) / np.polyval(a[::-1], zm1)
    return h
```

---

### 1.2 FIR 滤波器设计

#### MNE 使用位置

**文件**: `mne/filter.py`
**行号**: ~1000-1200
**功能**: FIR 滤波器设计（使用窗函数法）

```python
# mne/filter.py:~1100
from scipy import signal

def _construct_fir_filter(sfreq, freq, filter_length='auto', 
                          phase='zero', fir_window='hamming'):
    """构造 FIR 滤波器"""
    # 计算滤波器长度
    if filter_length == 'auto':
        filter_length = _estimate_filter_length(freq, sfreq)
    
    # 设计 FIR 滤波器（使用 firwin）
    h = signal.firwin(
        numtaps=filter_length,
        cutoff=freq,
        window=fir_window,
        pass_zero='lowpass',  # 或 'bandpass'
        fs=sfreq
    )
    return h
```

**对应 SciPy 位置**:
- **源码**: `scipy/signal/_fir_filter_design.py:firwin()`
- **行号**: ~200-400
- **实现**:
  ```python
  # scipy/signal/_fir_filter_design.py:250
  def firwin(numtaps, cutoff, window='hamming', pass_zero=True, fs=None):
      """FIR 滤波器设计（窗函数法）"""
      # 归一化频率
      cutoff = np.atleast_1d(cutoff) / (fs / 2.0)
      
      # 创建理想频率响应
      bands = _create_bands(cutoff, pass_zero)
      
      # 生成窗函数
      win = get_window(window, numtaps, fftbins=False)
      
      # 使用 firwin2 设计
      h = firwin2(numtaps, bands, [1, 0], window=win, fs=fs)
      return h
  ```

---

### 1.3 重采样 (Resample)

#### MNE 使用位置

**文件**: `mne/io/base.py`
**行号**: ~1800-1900
**功能**: 数据重采样

```python
# mne/io/base.py:~1850
from scipy import signal

def resample(self, sfreq, npad='auto'):
    """重采样数据"""
    # 计算新的采样点数
    ratio = sfreq / self.info['sfreq']
    n_samples_new = int(np.round(len(self.times) * ratio))
    
    # 使用 scipy.signal.resample
    data_new = signal.resample(
        self._data,
        num=n_samples_new,
        axis=-1,
        window='boxcar'  # 或 'hamming'
    )
    
    return data_new
```

**对应 SciPy 位置**:
- **源码**: `scipy/signal/_signaltools.py:resample()`
- **行号**: ~2500-2700
- **实现**（基于 FFT）:
  ```python
  # scipy/signal/_signaltools.py:2550
  def resample(x, num, axis=0, window=None):
      """使用 FFT 重采样信号"""
      # FFT
      X = fft.fft(x, axis=axis)
      
      # 创建新的频率数组
      if num > len(x):
          # 上采样：频域填充零
          X_new = _zero_pad_spectrum(X, num, axis)
      else:
          # 下采样：频域截断
          X_new = _truncate_spectrum(X, num, axis)
      
      # 应用窗函数（如果提供）
      if window is not None:
          X_new *= window
      
      # IFFT 回到时域
      y = fft.ifft(X_new, axis=axis)
      
      return y.real
  ```

**关键特性**:
- 频域重采样（避免时域插值伪影）
- 自动抗混叠（通过频域截断）
- 支持复信号

---

### 1.4 Hilbert 变换

#### MNE 使用位置

**文件**: `mne/time_frequency/tfr.py`
**行号**: ~500
**功能**: 解析信号（幅度+相位）

```python
# mne/time_frequency/tfr.py:~520
from scipy import signal

def _compute_analytic_signal(data, axis=-1):
    """计算解析信号"""
    # 使用 Hilbert 变换
    analytic = signal.hilbert(data, axis=axis)
    # 返回复数信号: x(t) + j*H[x(t)]
    
    # 提取瞬时幅度和相位
    amplitude = np.abs(analytic)
    phase = np.angle(analytic)
    
    return amplitude, phase
```

**对应 SciPy 位置**:
- **源码**: `scipy/signal/_signaltools.py:hilbert()`
- **行号**: ~1800-1900
- **实现**:
  ```python
  # scipy/signal/_signaltools.py:1850
  def hilbert(x, N=None, axis=-1):
      """Hilbert 变换（解析信号）"""
      # FFT
      Xf = fft.fft(x, n=N, axis=axis)
      
      # 构造 Hilbert 变换的频域算子
      h = np.zeros(N)
      h[0] = 1              # DC 分量
      h[1:N//2] = 2         # 正频率加倍
      h[N//2] = 1           # Nyquist
      # 负频率置零（单边谱）
      
      # 频域相乘
      Xf *= h
      
      # IFFT 回到时域
      x_analytic = fft.ifft(Xf, axis=axis)
      
      return x_analytic
  ```

---

## 2. 稀疏矩阵 (`scipy.sparse`)

### 2.1 CSR 矩阵（压缩稀疏行）

#### MNE 使用位置

**文件**: `mne/forward/_make_forward.py`
**行号**: ~800-900
**功能**: 构造前向解算子（稀疏表示）

```python
# mne/forward/_make_forward.py:~850
from scipy import sparse

def _create_forward_operator_sparse(G, idx_active):
    """创建稀疏前向算子"""
    # G: 增益矩阵 (n_channels, n_dipoles*3)
    # 大部分源位置不活跃，矩阵稀疏
    
    # 转换为 CSR 格式（高效的行切片）
    G_sparse = sparse.csr_matrix(G)
    
    # 稀疏矩阵乘法
    result = G_sparse @ source_vector
    
    return result
```

**对应 SciPy 位置**:
- **源码**: `scipy/sparse/_compressed.py`
- **类定义**: `scipy/sparse/_compressed.py:_cs_matrix` (基类)
- **CSR 实现**: `scipy/sparse/_compressed.py:csr_matrix`
- **行号**: ~100-500

**CSR 数据结构**:
```python
# scipy/sparse/_compressed.py:200
class csr_matrix(_cs_matrix):
    """压缩稀疏行矩阵"""
    # 存储格式:
    #   data: 非零元素值
    #   indices: 列索引
    #   indptr: 行指针
    
    # 示例矩阵:
    # [[1, 0, 2],
    #  [0, 0, 3],
    #  [4, 5, 6]]
    
    # 存储:
    # data = [1, 2, 3, 4, 5, 6]
    # indices = [0, 2, 2, 0, 1, 2]  # 列索引
    # indptr = [0, 2, 3, 6]         # 每行起始位置
```

**性能关键**:
- **矩阵乘法**: `scipy/sparse/_sparsetools.pyx:csr_matvec()` (C++)
- **切片**: `scipy/sparse/_compressed.py:_get_submatrix()`

---

### 2.2 稀疏线性方程组求解

#### MNE 使用位置

**文件**: `mne/inverse_sparse/mxne_inverse.py`
**行号**: ~300
**功能**: MxNE 算法中的稀疏求解

```python
# mne/inverse_sparse/mxne_inverse.py:~320
from scipy.sparse.linalg import cg, LinearOperator

def _solve_mxne_system(G, M, alpha):
    """求解 MxNE 线性系统"""
    # (G^T G + alpha*I) x = G^T M
    
    # 构造稀疏线性算子
    def matvec(x):
        return G.T @ (G @ x) + alpha * x
    
    A = LinearOperator(
        shape=(G.shape[1], G.shape[1]),
        matvec=matvec
    )
    
    # 共轭梯度法求解
    x, info = cg(A, G.T @ M, tol=1e-6, maxiter=1000)
    
    return x
```

**对应 SciPy 位置**:
- **cg (共轭梯度)**: `scipy/sparse/linalg/_isolve/iterative.py:cg()`
- **LinearOperator**: `scipy/sparse/linalg/_interface.py:LinearOperator`
- **行号**: ~150-300

**实现**:
```python
# scipy/sparse/linalg/_isolve/iterative.py:200
def cg(A, b, x0=None, tol=1e-5, maxiter=None):
    """共轭梯度法"""
    x = x0.copy() if x0 is not None else np.zeros_like(b)
    r = b - A @ x  # 初始残差
    p = r.copy()   # 搜索方向
    
    for i in range(maxiter):
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x += alpha * p
        r_new = r - alpha * Ap
        
        if np.linalg.norm(r_new) < tol:
            break
        
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
    
    return x, info
```

---

## 3. 优化算法 (`scipy.optimize`)

### 3.1 非线性最小二乘

#### MNE 使用位置

**文件**: `mne/dipole.py`
**行号**: ~500-600
**功能**: 偶极子拟合

```python
# mne/dipole.py:~550
from scipy.optimize import leastsq, least_squares

def _fit_dipole(G, M, r0):
    """拟合偶极子位置和方向"""
    
    def residual(x):
        """残差函数"""
        # x = [rx, ry, rz, qx, qy, qz]  (位置 + 偶极矩)
        r_dipole = x[:3]
        q_dipole = x[3:]
        
        # 计算预测的磁场
        G_dipole = _compute_gain(r_dipole)
        M_pred = G_dipole @ q_dipole
        
        # 残差
        return M - M_pred
    
    # 使用 Levenberg-Marquardt 算法
    result = least_squares(
        residual,
        x0=np.concatenate([r0, [0, 0, 1]]),
        method='lm',
        ftol=1e-8,
        xtol=1e-8
    )
    
    return result.x[:3], result.x[3:]  # 位置, 偶极矩
```

**对应 SciPy 位置**:
- **least_squares**: `scipy/optimize/_lsq/least_squares.py`
- **lm (LM算法)**: `scipy/optimize/_lsq/lm.py`
- **行号**: ~500-800

**Levenberg-Marquardt 实现**:
```python
# scipy/optimize/_lsq/lm.py:50
def levenberg_marquardt(fun, x0, jac, ftol, xtol, gtol, max_nfev):
    """Levenberg-Marquardt 算法"""
    x = x0
    f = fun(x)
    J = jac(x)  # Jacobian 矩阵
    
    lambda_param = 1e-3  # LM 参数
    
    for iteration in range(max_nfev):
        # 构造法方程: (J^T J + lambda*I) h = -J^T f
        JtJ = J.T @ J
        Jtf = J.T @ f
        
        # 求解
        h = solve(JtJ + lambda_param * np.eye(len(x)), -Jtf)
        
        # 更新
        x_new = x + h
        f_new = fun(x_new)
        
        # 调整 lambda（信赖域）
        if np.linalg.norm(f_new) < np.linalg.norm(f):
            lambda_param /= 10
            x, f = x_new, f_new
        else:
            lambda_param *= 10
        
        # 收敛检查
        if np.linalg.norm(h) < xtol:
            break
    
    return x
```

---

### 3.2 约束优化

#### MNE 使用位置

**文件**: `mne/inverse_sparse/mxne_inverse.py`
**行号**: ~400
**功能**: MxNE 的 L21 范数优化

```python
# mne/inverse_sparse/mxne_inverse.py:~420
from scipy.optimize import fmin_l_bfgs_b

def _mxne_optim(M, G, alpha, lipschitz_constant):
    """MxNE 优化（混合范数）"""
    n_dipoles = G.shape[1] // 3
    
    def objective(x):
        """目标函数: ||Gx - M||^2 + alpha*||x||_{2,1}"""
        residual = G @ x - M
        data_fit = 0.5 * np.sum(residual ** 2)
        
        # L21 正则化
        x_reshaped = x.reshape(n_dipoles, 3)
        l21_norm = alpha * np.sum(np.linalg.norm(x_reshaped, axis=1))
        
        return data_fit + l21_norm
    
    def gradient(x):
        """目标函数的梯度"""
        residual = G @ x - M
        grad_data = G.T @ residual
        
        # L21 的次梯度
        x_reshaped = x.reshape(n_dipoles, 3)
        norms = np.linalg.norm(x_reshaped, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除零
        grad_l21 = alpha * (x_reshaped / norms).ravel()
        
        return grad_data + grad_l21
    
    # L-BFGS-B 优化（有界BFGS）
    x0 = np.zeros(G.shape[1])
    bounds = [(0, None)] * len(x0)  # 非负约束
    
    result = fmin_l_bfgs_b(
        func=objective,
        x0=x0,
        fprime=gradient,
        bounds=bounds,
        maxiter=1000,
        pgtol=1e-5
    )
    
    return result[0]  # 最优解
```

**对应 SciPy 位置**:
- **fmin_l_bfgs_b**: `scipy/optimize/_lbfgsb_py.py`
- **Fortran 实现**: `scipy/optimize/lbfgsb_src/lbfgsb.f`
- **行号**: Python wrapper ~100-300, Fortran ~5000 行

---

## 4. 统计函数 (`scipy.stats`)

### 4.1 统计分布

#### MNE 使用位置

**文件**: `mne/stats/cluster_level.py`
**行号**: ~700-800
**功能**: 簇级别置换检验

```python
# mne/stats/cluster_level.py:~750
from scipy import stats

def _cluster_permutation_test(X, n_permutations=1000, threshold=None):
    """簇级别置换检验"""
    n_samples, n_features = X.shape
    
    # 计算 t 统计量
    t_obs = stats.ttest_1samp(X, 0, axis=0)[0]
    
    # 阈值
    if threshold is None:
        # 使用 t 分布的临界值
        threshold = stats.t.ppf(0.975, df=n_samples-1)
    
    # ... 置换测试 ...
    
    # p 值计算
    p_values = (null_distribution >= obs_stat).mean()
    
    return t_obs, p_values
```

**对应 SciPy 位置**:
- **ttest_1samp**: `scipy/stats/_stats_py.py:ttest_1samp()`
- **t.ppf**: `scipy/stats/_continuous_distns.py:t_gen.ppf()`
- **行号**: ~5000-5200 (ttest), ~3000-3200 (t 分布)

**t 检验实现**:
```python
# scipy/stats/_stats_py.py:5100
def ttest_1samp(a, popmean, axis=0):
    """单样本 t 检验"""
    a = np.asarray(a)
    n = a.shape[axis]
    
    # 样本均值和标准差
    mean = np.mean(a, axis=axis)
    std = np.std(a, axis=axis, ddof=1)
    
    # t 统计量
    t = (mean - popmean) / (std / np.sqrt(n))
    
    # p 值（双侧）
    df = n - 1
    p = 2 * t_distribution.sf(np.abs(t), df)
    
    return TtestResult(statistic=t, pvalue=p)
```

---

### 4.2 F 检验

#### MNE 使用位置

**文件**: `mne/stats/parametric.py`
**行号**: ~200
**功能**: ANOVA F 检验

```python
# mne/stats/parametric.py:~220
from scipy.stats import f as f_dist

def _compute_f_statistic(data_groups):
    """计算 F 统计量"""
    # data_groups: list of arrays for each group
    
    # 组间方差
    group_means = [np.mean(g, axis=0) for g in data_groups]
    grand_mean = np.mean(np.concatenate(data_groups), axis=0)
    
    n_groups = len(data_groups)
    n_total = sum(len(g) for g in data_groups)
    
    # 组间平方和
    ss_between = sum(
        len(g) * (m - grand_mean)**2 
        for g, m in zip(data_groups, group_means)
    )
    
    # 组内平方和
    ss_within = sum(
        np.sum((g - m)**2, axis=0)
        for g, m in zip(data_groups, group_means)
    )
    
    # F 统计量
    df_between = n_groups - 1
    df_within = n_total - n_groups
    
    F = (ss_between / df_between) / (ss_within / df_within)
    
    # p 值
    p_value = f_dist.sf(F, df_between, df_within)
    
    return F, p_value
```

**对应 SciPy 位置**:
- **F 分布**: `scipy/stats/_continuous_distns.py:f_gen`
- **sf (生存函数)**: `scipy/stats/_distn_infrastructure.py:rv_continuous.sf()`
- **行号**: ~1500-1700

---

## 5. 线性代数 (`scipy.linalg`)

### 5.1 特征分解 (eigh - Hermitian)

#### MNE 使用位置

**文件**: `mne/minimum_norm/_eloreta.py`
**行号**: 94
**功能**: eLORETA 权重计算

```python
# mne/minimum_norm/_eloreta.py:94
from scipy.linalg import eigh

def _compute_eloreta(G, R, noise_cov):
    """计算 eLORETA 解"""
    # G: 增益矩阵
    # R: 源协方差矩阵
    # noise_cov: 噪声协方差
    
    # 构造中间矩阵
    G_R_Gt = G @ R @ G.T + noise_cov
    
    # 对称矩阵的特征分解
    eigenvalues, eigenvectors = eigh(G_R_Gt)
    # eigenvalues: 升序排列
    # eigenvectors: 列向量
    
    # 计算逆矩阵（通过特征分解）
    G_R_Gt_inv = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T
    
    # eLORETA 解算子
    W = R @ G.T @ G_R_Gt_inv
    
    return W
```

**对应 SciPy 位置**:
- **源码**: `scipy/linalg/_decomp.py:eigh()`
- **行号**: ~400-600
- **LAPACK 调用**: `DSYEVD` (实对称) / `ZHEEVD` (复Hermitian)

**eigh 实现**:
```python
# scipy/linalg/_decomp.py:450
def eigh(a, b=None, lower=True, eigvals_only=False, 
         overwrite_a=False, turbo=True, check_finite=True):
    """对称/Hermitian 矩阵特征分解"""
    if check_finite:
        a1 = np.asarray_chkfinite(a)
    else:
        a1 = np.asarray(a)
    
    # 选择 LAPACK 驱动
    if b is None:
        # 标准特征值问题
        driver = 'evd'  # 使用 SYEVD/HEEVD (快速)
    else:
        # 广义特征值问题
        driver = 'gvd'  # 使用 SYGVD/HEGVD
    
    # 调用 LAPACK
    w, v = _wrapper[driver](a1, lower=lower, 
                            overwrite_a=overwrite_a)
    
    # w: 特征值（实数，升序）
    # v: 特征向量（列）
    
    if eigvals_only:
        return w
    else:
        return w, v
```

**LAPACK 层**:
- **Cython 封装**: `scipy/linalg/cython_lapack.pyx`
- **Fortran**: `LAPACK/dsyevd.f`, `LAPACK/zheevd.f`

---

### 5.2 SVD (奇异值分解)

#### MNE 使用位置

**文件**: `mne/minimum_norm/inverse.py`
**行号**: 278-295
**功能**: 计算特征导联

```python
# mne/minimum_norm/inverse.py:285
from scipy.linalg import svd

def _prepare_inverse_operator(inv, nave, lambda2):
    """准备逆算子"""
    # ...
    
    # SVD 分解增益矩阵
    U, s, Vt = svd(A, full_matrices=False, lapack_driver='gesdd')
    # U: 左奇异向量 (n_channels, k)
    # s: 奇异值 (k,)
    # Vt: 右奇异向量转置 (k, n_sources)
    # k = min(n_channels, n_sources)
    
    # 特征导联
    eigen_leads = Vt  # (k, n_sources)
    eigen_fields = U  # (n_channels, k)
    
    # 正则化
    inv_s = s / (s**2 + lambda2)
    
    return eigen_fields, inv_s, eigen_leads
```

**对应 SciPy 位置**:
- **源码**: `scipy/linalg/_decomp_svd.py:svd()`
- **行号**: ~20-200
- **LAPACK**: `DGESDD` (分治法) 或 `DGESVD` (QR迭代)

**SVD 实现**:
```python
# scipy/linalg/_decomp_svd.py:50
def svd(a, full_matrices=True, compute_uv=True, overwrite_a=False,
        check_finite=True, lapack_driver='gesdd'):
    """奇异值分解"""
    a1 = _asarray_validated(a, check_finite=check_finite)
    
    # 选择 LAPACK 驱动
    if lapack_driver == 'gesdd':
        # 分治算法（通常更快，但需要更多内存）
        lapack_func = _lapack.gesdd
    elif lapack_driver == 'gesvd':
        # QR 迭代（更稳定）
        lapack_func = _lapack.gesvd
    
    # 调用 LAPACK
    u, s, vt, info = lapack_func(
        a1,
        full_matrices=full_matrices,
        compute_uv=compute_uv,
        overwrite_a=overwrite_a
    )
    
    if info > 0:
        raise LinAlgError("SVD did not converge")
    
    if compute_uv:
        return u, s, vt
    else:
        return s
```

**LAPACK 路径**:
```
scipy.linalg.svd()
    ↓
scipy/linalg/cython_lapack.pyx:gesdd()
    ↓
LAPACK/dgesdd.f (Fortran)
    ↓
BLAS 3 (矩阵乘法等)
```

---

### 5.3 Cholesky 分解

#### MNE 使用位置

**文件**: `mne/cov.py`
**行号**: ~900
**功能**: 白化变换

```python
# mne/cov.py:~920
from scipy.linalg import cholesky

def _compute_whitener(cov_matrix, rank=None):
    """计算白化矩阵"""
    # cov_matrix: 协方差矩阵（正定）
    
    # Cholesky 分解: C = L L^T
    L = cholesky(cov_matrix, lower=True)
    
    # 白化矩阵: W = L^{-1}
    # 使用三角求解（比直接求逆更快）
    W = solve_triangular(L, np.eye(len(L)), lower=True)
    
    # 或者：W = inv(L)
    from scipy.linalg import inv
    W = inv(L)
    
    return W
```

**对应 SciPy 位置**:
- **cholesky**: `scipy/linalg/_decomp_cholesky.py:cholesky()`
- **solve_triangular**: `scipy/linalg/_basic.py:solve_triangular()`
- **行号**: ~50-150 (cholesky), ~300-400 (solve_triangular)

**Cholesky 实现**:
```python
# scipy/linalg/_decomp_cholesky.py:70
def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
    """Cholesky 分解"""
    a1 = _asarray_validated(a, check_finite=check_finite)
    
    # 检查对称性
    if not _is_hermitian(a1):
        raise LinAlgError("Matrix must be Hermitian/symmetric")
    
    # 调用 LAPACK POTRF
    c, info = _lapack.potrf(a1, lower=lower, 
                            overwrite_a=overwrite_a)
    
    if info > 0:
        raise LinAlgError("Matrix is not positive definite")
    
    # 只保留下三角（或上三角）
    if lower:
        return np.tril(c)
    else:
        return np.triu(c)
```

**LAPACK**:
- `DPOTRF` - Cholesky 分解
- `DTRTRS` - 三角系统求解

---

## 总结表：MNE → SciPy 映射

| MNE 功能 | MNE 文件 | SciPy 函数 | SciPy 源码位置 | 底层实现 |
|---------|---------|-----------|---------------|---------|
| Butterworth 滤波器 | filter.py:850 | signal.butter | signal/_filter_design.py:3500 | LAPACK (双线性变换) |
| 零相位滤波 | filter.py:549 | signal.sosfiltfilt | signal/_signaltools.py:4200 | _sosfilt.pyx (Cython) |
| FIR 设计 | filter.py:1100 | signal.firwin | signal/_fir_filter_design.py:250 | 窗函数法 |
| 重采样 | io/base.py:1850 | signal.resample | signal/_signaltools.py:2550 | FFT 插值 |
| Hilbert 变换 | time_frequency/tfr.py:520 | signal.hilbert | signal/_signaltools.py:1850 | FFT (单边谱) |
| 稀疏矩阵乘法 | forward/_make_forward.py:850 | sparse.csr_matrix | sparse/_compressed.py:200 | _sparsetools.pyx |
| 共轭梯度 | inverse_sparse/mxne_inverse.py:320 | sparse.linalg.cg | sparse/linalg/_isolve/iterative.py:200 | 迭代求解 |
| 非线性最小二乘 | dipole.py:550 | optimize.least_squares | optimize/_lsq/least_squares.py | LM 算法 |
| L-BFGS-B | inverse_sparse/mxne_inverse.py:420 | optimize.fmin_l_bfgs_b | optimize/_lbfgsb_py.py | Fortran LBFGSB |
| t 检验 | stats/cluster_level.py:750 | stats.ttest_1samp | stats/_stats_py.py:5100 | t 分布 |
| F 检验 | stats/parametric.py:220 | stats.f.sf | stats/_continuous_distns.py | F 分布 |
| Hermitian 特征分解 | minimum_norm/_eloreta.py:94 | linalg.eigh | linalg/_decomp.py:450 | LAPACK:SYEVD |
| SVD | minimum_norm/inverse.py:285 | linalg.svd | linalg/_decomp_svd.py:50 | LAPACK:GESDD |
| Cholesky 分解 | cov.py:920 | linalg.cholesky | linalg/_decomp_cholesky.py:70 | LAPACK:POTRF |

---

## 性能关键路径

### 1. 信号处理（最频繁）
- **sosfiltfilt**: 每次数据滤波
  - Cython: `_sosfilt.pyx`
  - 直接型 II 实现

### 2. 稀疏矩阵（源估计）
- **csr_matrix @ vector**: 前向解计算
  - C++: `_sparsetools.cxx`
  - 并行化（OpenMP）

### 3. 线性代数（逆解）
- **SVD/eigh**: 计算逆算子
  - LAPACK: GESDD, SYEVD
  - 多线程 BLAS

---

## 下一步

继续阅读：
- [MNE 与 scikit-learn 详细代码位置对比分析](./03_MNE-sklearn代码对比.md)
- [Rust 替代方案详细对比](./04_Rust替代方案.md)
