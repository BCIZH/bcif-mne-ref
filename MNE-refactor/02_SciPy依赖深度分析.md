# MNE-Python SciPy 依赖深度分析

## 概述

SciPy 是 MNE-Python 的**第二核心依赖**，提供科学计算的高级功能。MNE 特别依赖其信号处理、线性代数、稀疏矩阵和优化模块。本文档详细分析 MNE 如何使用 SciPy，以及 Rust 替代方案。

---

## SciPy 使用统计

### 模块导入频率

| 模块 | 使用频率 | 关键程度 | Rust 替代难度 |
|------|---------|---------|--------------|
| `scipy.signal` | ⭐⭐⭐⭐⭐ | 必需 | 中-高 |
| `scipy.linalg` | ⭐⭐⭐⭐ | 必需 | 中 |
| `scipy.sparse` | ⭐⭐⭐⭐ | 必需 | 中 |
| `scipy.stats` | ⭐⭐⭐ | 必需 | 中 |
| `scipy.optimize` | ⭐⭐⭐ | 必需 | 中-高 |
| `scipy.fft` | ⭐⭐⭐ | 必需 | 中 |
| `scipy.interpolate` | ⭐⭐ | 可选 | 中 |
| `scipy.ndimage` | ⭐⭐ | 可选 | 中 |
| `scipy.spatial` | ⭐⭐ | 可选 | 低 |

---

## 1. scipy.signal - 信号处理 (最重要)

### 1.1 滤波器设计

#### Python/SciPy 模式
```python
from scipy.signal import butter, iirfilter, iirdesign, cheby1, cheby2, ellip, bessel

# Butterworth 滤波器（最常用）
sos = butter(
    N=4,  # 阶数
    Wn=[l_freq, h_freq],  # 截止频率
    btype='bandpass',  # 类型: lowpass, highpass, bandpass, bandstop
    fs=sfreq,  # 采样率
    output='sos'  # 二阶节（更稳定）
)

# 通用 IIR 滤波器
b, a = iirfilter(
    N=5,
    Wn=critical_freq,
    btype='highpass',
    ftype='butter',  # 也可以是 'cheby1', 'cheby2', 'ellip', 'bessel'
    fs=sfreq,
    output='ba'
)

# 设计满足规范的滤波器
b, a = iirdesign(
    wp=pass_freq,  # 通带边缘频率
    ws=stop_freq,  # 阻带边缘频率
    gpass=3,  # 通带最大损失(dB)
    gstop=40,  # 阻带最小衰减(dB)
    ftype='ellip',
    fs=sfreq
)

# MNE 典型用法：带通滤波
l_freq, h_freq = 1.0, 40.0
sfreq = 500.0
sos = butter(5, [l_freq, h_freq], btype='bandpass', fs=sfreq, output='sos')
```

#### Rust 等价
```rust
use biquad::{Biquad, Coefficients, Type, Q_BUTTERWORTH_F64};

// 简单的二阶滤波器（Biquad）
let coeffs = Coefficients::<f64>::from_params(
    Type::BandPass,
    sfreq.hz(),
    center_freq.hz(),
    Q_BUTTERWORTH_F64,
).unwrap();

let mut biquad = DirectForm2Transposed::<f64>::new(coeffs);

// 对于高阶滤波器，需要级联多个二阶节
// 或者自己实现 Butterworth 设计算法

// 更完整的方案：使用 bacon_sci 或自实现
pub fn butter_bandpass(
    order: usize,
    lowcut: f64,
    highcut: f64,
    fs: f64,
) -> Vec<Biquad<f64>> {
    // 需要实现：
    // 1. 模拟原型设计（Butterworth 多项式根）
    // 2. 双线性变换到数字域
    // 3. 分解为二阶节（SOS）
    
    // 这是一个重大工作量！
    todo!("实现完整的 Butterworth 滤波器设计")
}
```

**关键差异和挑战**:
1. **Rust 生态缺少完整的滤波器设计库**
2. `biquad` crate 只支持简单的二阶滤波器
3. 需要自己实现：
   - Butterworth/Chebyshev 设计算法
   - 双线性变换
   - SOS 分解
4. **建议**：移植 SciPy 的 C/Fortran 代码，或使用 FFI 调用现有库

---

### 1.2 滤波应用

#### Python/SciPy 模式
```python
from scipy.signal import sosfilt, sosfiltfilt, lfilter, filtfilt

# 单向滤波（因果）
filtered = sosfilt(sos, data, axis=-1)
filtered = lfilter(b, a, data, axis=-1)

# 双向滤波（零相位）- MNE 默认
filtered = sosfiltfilt(sos, data, axis=-1)
filtered = filtfilt(b, a, data, axis=-1)

# 带初始条件的滤波（避免瞬态）
zi = sosfilt_zi(sos)
y, zf = sosfilt(sos, data, zi=zi * data[0])

# MNE 典型管道
def filter_data(data, l_freq, h_freq, sfreq):
    """MNE 风格的滤波"""
    sos = butter(5, [l_freq, h_freq], btype='bandpass', fs=sfreq, output='sos')
    # 零相位双向滤波
    return sosfiltfilt(sos, data, axis=-1)
```

#### Rust 等价
```rust
use ndarray::prelude::*;

// 单向滤波（级联二阶节）
pub fn sosfilt(sos: &[Biquad<f64>], data: &Array1<f64>) -> Array1<f64> {
    let mut output = data.clone();
    for section in sos {
        output = output.mapv(|x| section.run(x));
    }
    output
}

// 双向滤波（零相位）
pub fn sosfiltfilt(sos: &[Biquad<f64>], data: &Array1<f64>) -> Array1<f64> {
    // 正向滤波
    let mut forward = sosfilt(sos, data);
    
    // 反转
    forward = forward.slice(s![..;-1]).to_owned();
    
    // 反向滤波
    let backward = sosfilt(sos, &forward);
    
    // 再次反转
    backward.slice(s![..;-1]).to_owned()
}

// 注意：这是简化版本，实际需要处理：
// - 多维数组和轴参数
// - 边界效应（padding）
// - 初始条件
```

**挑战**:
- 需要手动实现 `filtfilt` 的 padding 逻辑
- SciPy 使用 Gustafsson 方法处理边界
- 多维数组支持需要仔细处理

---

### 1.3 频谱分析

#### Python/SciPy 模式
```python
from scipy.signal import welch, spectrogram, stft, istft, csd

# Welch 功率谱密度（最常用）
freqs, psd = welch(
    data,
    fs=sfreq,
    window='hann',
    nperseg=256,
    noverlap=128,
    nfft=512,
    axis=-1
)

# 短时傅里叶变换
f, t, Zxx = stft(
    data,
    fs=sfreq,
    window='hann',
    nperseg=256,
    noverlap=192
)

# 逆 STFT
_, reconstructed = istft(Zxx, fs=sfreq, nperseg=256, noverlap=192)

# 交叉谱密度
freqs, Pxy = csd(x, y, fs=sfreq, nperseg=256)

# Spectrogram（时频表示）
f, t, Sxx = spectrogram(
    data,
    fs=sfreq,
    window='hann',
    nperseg=256,
    noverlap=192,
    scaling='spectrum'
)

# MNE 典型用法：计算 PSD
def compute_psd(data, sfreq, fmin=0, fmax=np.inf):
    freqs, psd = welch(data, fs=sfreq, nperseg=2**int(np.log2(sfreq * 2)))
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[..., mask]
```

#### Rust 等价
```rust
use rustfft::FftPlanner;
use ndarray::prelude::*;

// 手动实现 Welch 方法
pub fn welch(
    data: &Array1<f64>,
    fs: f64,
    nperseg: usize,
    noverlap: usize,
) -> (Array1<f64>, Array1<f64>) {
    let step = nperseg - noverlap;
    let n_segments = (data.len() - noverlap) / step;
    
    // 窗函数
    let window = hann_window(nperseg);
    
    // 初始化累加器
    let nfft = nperseg;
    let nfreqs = nfft / 2 + 1;
    let mut psd_sum = Array1::<f64>::zeros(nfreqs);
    
    // 处理每个段
    for i in 0..n_segments {
        let start = i * step;
        let end = start + nperseg;
        let segment = data.slice(s![start..end]);
        
        // 加窗
        let windowed: Array1<f64> = &segment * &window;
        
        // FFT
        let fft_result = rfft(&windowed, nfft);
        
        // 功率谱
        let power = fft_result.mapv(|c| c.norm_sqr());
        
        psd_sum = psd_sum + power;
    }
    
    // 平均并归一化
    let psd = psd_sum / (n_segments as f64);
    let scale = 1.0 / (fs * window.mapv(|x| x * x).sum());
    let psd = psd * scale;
    
    // 频率数组
    let freqs = Array1::from_vec(
        (0..nfreqs).map(|i| i as f64 * fs / nfft as f64).collect()
    );
    
    (freqs, psd)
}

// STFT（短时傅里叶变换）
pub fn stft(
    data: &Array1<f64>,
    nperseg: usize,
    noverlap: usize,
    fs: f64,
) -> (Array1<f64>, Array1<f64>, Array2<Complex<f64>>) {
    let step = nperseg - noverlap;
    let n_segments = (data.len() - noverlap) / step;
    let nfreqs = nperseg / 2 + 1;
    
    let window = hann_window(nperseg);
    let mut stft_result = Array2::<Complex<f64>>::zeros((nfreqs, n_segments));
    
    for (i, mut col) in stft_result.axis_iter_mut(Axis(1)).enumerate() {
        let start = i * step;
        let end = start + nperseg;
        let segment = data.slice(s![start..end]);
        let windowed = &segment * &window;
        let fft = rfft(&windowed, nperseg);
        col.assign(&fft);
    }
    
    let freqs = Array1::from_vec(
        (0..nfreqs).map(|i| i as f64 * fs / nperseg as f64).collect()
    );
    let times = Array1::from_vec(
        (0..n_segments).map(|i| (i * step) as f64 / fs).collect()
    );
    
    (freqs, times, stft_result)
}
```

**挑战**:
- Welch/STFT 实现量大但直接
- 窗函数需要单独实现或用 `apodize` crate
- ISTFT 更复杂（需要 overlap-add）
- 多维支持需要额外工作

**建议**:
- 考虑 `spectrum-analyzer` crate（功能有限）
- 或创建自己的 DSP 工具箱

---

### 1.4 其他信号处理函数

#### Python/SciPy 模式
```python
from scipy.signal import hilbert, resample, resample_poly, detrend, get_window

# Hilbert 变换（解析信号）
analytic = hilbert(data, axis=-1)
amplitude = np.abs(analytic)
phase = np.angle(analytic)

# 重采样
resampled = resample(data, num=new_length, axis=-1)
resampled = resample_poly(data, up=2, down=3, axis=-1)  # 更高效

# 去趋势
detrended = detrend(data, axis=-1, type='linear')
detrended = detrend(data, type='constant')  # 去均值

# 窗函数
window = get_window('hann', 256)
window = get_window(('kaiser', 8.6), 512)

# 峰值检测
from scipy.signal import find_peaks
peaks, properties = find_peaks(
    data,
    height=threshold,
    distance=min_distance,
    prominence=0.5
)

# MNE 用法：包络提取
def extract_envelope(data, sfreq):
    """提取信号包络（希尔伯特变换）"""
    analytic = hilbert(data, axis=-1)
    return np.abs(analytic)
```

#### Rust 等价
```rust
use rustfft::num_complex::Complex;

// Hilbert 变换
pub fn hilbert(data: &Array1<f64>) -> Array1<Complex<f64>> {
    let n = data.len();
    
    // FFT
    let fft = fft(data);
    
    // 构造 Hilbert 滤波器
    let mut h = Array1::<f64>::zeros(n);
    if n % 2 == 0 {
        h[0] = 1.0;
        h[n / 2] = 1.0;
        h.slice_mut(s![1..n/2]).fill(2.0);
    } else {
        h[0] = 1.0;
        h.slice_mut(s![1..(n+1)/2]).fill(2.0);
    }
    
    // 应用滤波器并逆变换
    let filtered: Array1<Complex<f64>> = fft.iter()
        .zip(h.iter())
        .map(|(f, &h_val)| f * h_val)
        .collect();
    
    ifft(&filtered)
}

// 重采样（使用 FFT 方法）
pub fn resample(data: &Array1<f64>, new_len: usize) -> Array1<f64> {
    let fft = rfft(data, data.len());
    
    // 频域插值/抽取
    let new_fft = if new_len > data.len() {
        // 零填充
        let mut padded = Array1::<Complex<f64>>::zeros(new_len / 2 + 1);
        padded.slice_mut(s![..fft.len()]).assign(&fft);
        padded
    } else {
        // 截断
        fft.slice(s![..new_len/2+1]).to_owned()
    };
    
    irfft(&new_fft, new_len)
}

// 去趋势
pub fn detrend_linear(data: &Array1<f64>) -> Array1<f64> {
    let n = data.len() as f64;
    let x: Array1<f64> = Array1::range(0.0, n, 1.0);
    
    // 最小二乘拟合 y = a + b*x
    let x_mean = x.mean().unwrap();
    let y_mean = data.mean().unwrap();
    
    let numerator: f64 = Zip::from(&x)
        .and(data)
        .map_collect(|&xi, &yi| (xi - x_mean) * (yi - y_mean))
        .sum();
    
    let denominator: f64 = x.mapv(|xi| (xi - x_mean).powi(2)).sum();
    
    let slope = numerator / denominator;
    let intercept = y_mean - slope * x_mean;
    
    // 移除趋势
    Zip::from(&x)
        .and(data)
        .map_collect(|&xi, &yi| yi - (intercept + slope * xi))
}

pub fn detrend_constant(data: &Array1<f64>) -> Array1<f64> {
    let mean = data.mean().unwrap();
    data - mean
}

// 峰值检测（简化版）
pub fn find_peaks(data: &Array1<f64>, height: f64) -> Vec<usize> {
    let mut peaks = Vec::new();
    
    for i in 1..data.len()-1 {
        if data[i] > data[i-1] && data[i] > data[i+1] && data[i] >= height {
            peaks.push(i);
        }
    }
    
    peaks
}
```

**可用 Crates**:
- `apodize` - 窗函数
- `peak-detector` - 简单峰值检测
- 大部分需要自实现

---

## 2. scipy.linalg - 线性代数

### 2.1 核心函数

#### Python/SciPy 模式
```python
from scipy.linalg import svd, eigh, eig, pinv, inv, solve, qr, cholesky, orth

# SVD（奇异值分解）- MNE 中大量使用
U, s, Vt = svd(matrix, full_matrices=False, lapack_driver='gesdd')

# 对称矩阵特征分解（更快更稳定）
eigenvalues, eigenvectors = eigh(symmetric_matrix, lower=True)

# 一般矩阵特征分解
eigenvalues, eigenvectors = eig(matrix)

# 伪逆（MNE 常用于投影）
pinv_mat = pinv(matrix, rcond=1e-15)

# 求解线性系统 Ax = b
x = solve(A, b, assume_a='pos')  # 正定矩阵更快

# QR 分解
Q, R = qr(matrix, mode='economic')

# Cholesky 分解（正定矩阵）
L = cholesky(positive_definite, lower=True)

# 正交化
orthogonal = orth(matrix, rcond=None)

# MNE 典型用法：白化变换
def whiten_data(data, cov):
    """使用协方差矩阵白化数据"""
    # 特征分解
    eigenvalues, eigenvectors = eigh(cov)
    
    # 构造白化矩阵
    W = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    
    return W @ data
```

#### Rust 等价
```rust
use ndarray_linalg::*;

// SVD
let (u, s, vt) = matrix.svd(false, true)?;  // (compute_u, compute_vt)

// 对称特征分解
let (eigenvalues, eigenvectors) = symmetric_matrix.eigh(UPLO::Lower)?;

// 一般特征分解
let (eigenvalues, eigenvectors) = matrix.eig()?;

// 伪逆
let pinv_mat = matrix.pinv(1e-15)?;

// 求解线性系统
let x = A.solve_into(b)?;

// QR 分解
let qr = matrix.qr()?;
let q = qr.q();
let r = qr.r();

// Cholesky 分解
let chol = positive_definite.cholesky(UPLO::Lower)?;

// 正交化（需要手动实现）
pub fn orth(matrix: &Array2<f64>, rcond: Option<f64>) -> Result<Array2<f64>> {
    let (u, s, _) = matrix.svd(true, false)?;
    
    let rcond = rcond.unwrap_or_else(|| {
        let max_dim = matrix.nrows().max(matrix.ncols());
        f64::EPSILON * max_dim as f64
    });
    
    let threshold = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * rcond;
    let rank = s.iter().filter(|&&si| si > threshold).count();
    
    Ok(u.unwrap().slice(s![.., ..rank]).to_owned())
}

// 白化变换
fn whiten_data(data: &Array2<f64>, cov: &Array2<f64>) -> Result<Array2<f64>> {
    let (eigenvalues, eigenvectors) = cov.eigh(UPLO::Lower)?;
    
    let diag = Array2::from_diag(&eigenvalues.mapv(|x| 1.0 / x.sqrt()));
    let W = eigenvectors.dot(&diag).dot(&eigenvectors.t());
    
    Ok(W.dot(data))
}
```

**优势**:
- `ndarray-linalg` 与 SciPy 非常接近
- 基于 LAPACK，性能相当
- API 略有不同但可理解

**注意**:
- 需要链接 LAPACK（OpenBLAS 或 Intel MKL）
- 错误处理通过 `Result`
- 部分高级功能需自实现

---

### 2.2 BLAS 操作

#### Python/SciPy 模式
```python
from scipy.linalg.blas import dgemm, dgemv, ddot

# 矩阵乘法 C = alpha * A @ B + beta * C
C = dgemm(alpha=1.0, a=A, b=B, beta=0.0, c=C)

# 矩阵向量乘法 y = alpha * A @ x + beta * y
y = dgemv(alpha=1.0, a=A, x=x, beta=0.0, y=y)

# 点积
result = ddot(x, y)
```

#### Rust 等价
```rust
use ndarray_linalg::blas::*;

// 矩阵乘法（ndarray 自动调用 BLAS）
let C = 1.0 * A.dot(&B);  // 编译器优化会使用 BLAS

// 或显式使用 BLAS
use blas::dgemm;
unsafe {
    dgemm(
        b'N', b'N',  // 转置标志
        m, n, k,
        1.0,  // alpha
        A.as_ptr(), lda,
        B.as_ptr(), ldb,
        0.0,  // beta
        C.as_mut_ptr(), ldc,
    );
}

// 通常直接用 ndarray 的 dot 就够了
let C = A.dot(&B);  // 自动使用 BLAS
```

---

## 3. scipy.sparse - 稀疏矩阵

### 3.1 稀疏矩阵创建和操作

#### Python/SciPy 模式
```python
from scipy.sparse import csr_array, csc_array, coo_array, lil_array
from scipy.sparse import eye, diags

# 创建稀疏矩阵
csr = csr_array((data, (row_indices, col_indices)), shape=(n, m))
csc = csc_array(dense_matrix)
lil = lil_array((1000, 1000))  # 用于构造

# 单位矩阵
I = eye(1000, format='csr')

# 对角矩阵
D = diags([1, 2, 3], offsets=[0, 1, -1], shape=(n, n))

# 稀疏矩阵运算
result = sparse_A + sparse_B
result = sparse_A.dot(sparse_B)
result = sparse_A @ dense_vector

# 转换格式
csr = lil.tocsr()
dense = csr.toarray()

# MNE 用法：邻接矩阵
from scipy.sparse import csr_array

def make_adjacency_matrix(n_vertices, edges):
    """创建图的邻接矩阵"""
    row = []
    col = []
    data = []
    
    for i, j in edges:
        row.extend([i, j])
        col.extend([j, i])
        data.extend([1, 1])
    
    return csr_array((data, (row, col)), shape=(n_vertices, n_vertices))
```

#### Rust 等价
```rust
use sprs::{CsMat, CsVec, TriMat};

// CSR 格式
let mut triplets = TriMat::new((n, m));
for (i, j, value) in data {
    triplets.add_triplet(i, j, value);
}
let csr: CsMat<f64> = triplets.to_csr();

// 单位矩阵
let I = CsMat::<f64>::eye(1000);

// 稀疏矩阵运算
let result = &sparse_A + &sparse_B;
let result = &sparse_A * &sparse_B;
let result = &sparse_A * &dense_vector;

// 转换
let dense: Array2<f64> = csr.to_dense();

// 邻接矩阵
fn make_adjacency_matrix(n_vertices: usize, edges: &[(usize, usize)]) -> CsMat<f64> {
    let mut triplets = TriMat::new((n_vertices, n_vertices));
    
    for &(i, j) in edges {
        triplets.add_triplet(i, j, 1.0);
        triplets.add_triplet(j, i, 1.0);
    }
    
    triplets.to_csr()
}
```

**可用 Crates**:
- `sprs` - 主要稀疏矩阵库（推荐）
- `sparse` - 另一个选择
- `nalgebra-sparse` - nalgebra 的稀疏扩展

---

### 3.2 稀疏图算法

#### Python/SciPy 模式
```python
from scipy.sparse.csgraph import dijkstra, connected_components, shortest_path
from scipy.sparse.csgraph import laplacian

# Dijkstra 最短路径（MNE 用于源空间距离）
distances = dijkstra(adjacency, indices=[0], directed=False)

# 连通分量
n_components, labels = connected_components(adjacency, directed=False)

# Laplacian 矩阵
L = laplacian(adjacency, normed=True)

# MNE 用法：计算皮层距离
def compute_source_space_distances(src):
    """计算源空间中的测地距离"""
    adjacency = src['dist']  # 稀疏邻接矩阵
    
    # 计算所有点到所有点的距离
    dist_matrix = dijkstra(adjacency, directed=False, limit=np.inf)
    
    return dist_matrix
```

#### Rust 等价
```rust
use petgraph::algo::dijkstra;
use petgraph::graph::UnGraph;
use petgraph::unionfind::UnionFind;

// Dijkstra
fn dijkstra_sparse(adjacency: &CsMat<f64>, source: usize) -> Vec<f64> {
    // 转换为 petgraph
    let mut graph = UnGraph::<(), f64>::new_undirected();
    let nodes: Vec<_> = (0..adjacency.rows()).map(|_| graph.add_node(())).collect();
    
    for (i, vec) in adjacency.outer_iterator().enumerate() {
        for (j, &weight) in vec.iter() {
            graph.add_edge(nodes[i], nodes[j], weight);
        }
    }
    
    // 运行 Dijkstra
    let distances = dijkstra(&graph, nodes[source], None, |e| *e.weight());
    
    // 转换回 Vec
    nodes.iter().map(|&node| *distances.get(&node).unwrap_or(&f64::INFINITY)).collect()
}

// 连通分量
fn connected_components(adjacency: &CsMat<f64>) -> (usize, Vec<usize>) {
    let n = adjacency.rows();
    let mut uf = UnionFind::new(n);
    
    for (i, vec) in adjacency.outer_iterator().enumerate() {
        for (j, _) in vec.iter() {
            uf.union(i, j);
        }
    }
    
    // 标签化
    let mut labels = vec![0; n];
    let mut label_map = std::collections::HashMap::new();
    let mut next_label = 0;
    
    for i in 0..n {
        let root = uf.find(i);
        let label = label_map.entry(root).or_insert_with(|| {
            next_label += 1;
            next_label - 1
        });
        labels[i] = *label;
    }
    
    (next_label, labels)
}
```

**可用 Crates**:
- `petgraph` - 图算法（推荐）
- `pathfinding` - 路径查找算法
- `sprs` 部分支持图操作

---

## 4. scipy.stats - 统计

#### Python/SciPy 模式
```python
from scipy.stats import t, f, zscore, pearsonr, spearmanr, kendalltau
from scipy.stats import normaltest, shapiro, kurtosis, skew

# T 分布
t_stat, p_value = t.sf(t_score, df=n-1)  # 生存函数（单尾）

# F 分布
f_stat, p_value = f.cdf(f_score, dfn=df1, dfd=df2)

# Z-score 标准化
z = zscore(data, axis=0, ddof=1)

# 相关系数
r, p = pearsonr(x, y)
rho, p = spearmanr(x, y)

# 正态性检验
stat, p = normaltest(data)
stat, p = shapiro(data)

# 描述统计
kurt = kurtosis(data, axis=0)
skewness = skew(data, axis=0)

# MNE 用法：置换检验
from scipy.stats import ttest_1samp

def permutation_t_test(data, n_permutations=1000):
    """单样本 t 检验的置换版本"""
    t_obs, _ = ttest_1samp(data, 0)
    
    # 置换
    n = len(data)
    t_dist = []
    for _ in range(n_permutations):
        signs = np.random.choice([-1, 1], size=n)
        perm_data = data * signs
        t_perm, _ = ttest_1samp(perm_data, 0)
        t_dist.append(t_perm)
    
    # p 值
    p = np.mean(np.abs(t_dist) >= np.abs(t_obs))
    return t_obs, p
```

#### Rust 等价
```rust
use statrs::distribution::{StudentsT, FisherSnedecor, ContinuousCDF};
use statrs::statistics::{Statistics, OrderStatistics};

// T 分布
let t_dist = StudentsT::new(0.0, 1.0, df as f64).unwrap();
let p_value = 1.0 - t_dist.cdf(t_score);  // 单尾

// F 分布
let f_dist = FisherSnedecor::new(df1 as f64, df2 as f64).unwrap();
let p_value = f_dist.cdf(f_score);

// Z-score（手动实现）
pub fn zscore(data: &Array2<f64>, axis: Axis, ddof: usize) -> Array2<f64> {
    let mean = data.mean_axis(axis).unwrap().insert_axis(axis);
    let std = data.std_axis(axis, ddof as f64).insert_axis(axis);
    (data - &mean) / &std
}

// Pearson 相关系数
pub fn pearsonr(x: &Array1<f64>, y: &Array1<f64>) -> (f64, f64) {
    let n = x.len() as f64;
    let x_mean = x.mean().unwrap();
    let y_mean = y.mean().unwrap();
    
    let numerator: f64 = Zip::from(x).and(y)
        .map_collect(|&xi, &yi| (xi - x_mean) * (yi - y_mean))
        .sum();
    
    let x_std: f64 = x.mapv(|xi| (xi - x_mean).powi(2)).sum().sqrt();
    let y_std: f64 = y.mapv(|yi| (yi - y_mean).powi(2)).sum().sqrt();
    
    let r = numerator / (x_std * y_std);
    
    // T 统计量用于 p 值
    let t = r * ((n - 2.0) / (1.0 - r * r)).sqrt();
    let t_dist = StudentsT::new(0.0, 1.0, n - 2.0).unwrap();
    let p = 2.0 * (1.0 - t_dist.cdf(t.abs()));
    
    (r, p)
}

// 峰度
pub fn kurtosis(data: &Array1<f64>) -> f64 {
    let mean = data.mean().unwrap();
    let std = data.std(1.0);
    let n = data.len() as f64;
    
    let m4: f64 = data.mapv(|x| ((x - mean) / std).powi(4)).sum() / n;
    m4 - 3.0  // 超额峰度
}
```

**可用 Crates**:
- `statrs` - 统计分布和测试（推荐）
- `statistical` - 描述统计
- `ndarray-stats` - 与 ndarray 集成

---

## 5. scipy.optimize - 优化

#### Python/SciPy 模式
```python
from scipy.optimize import minimize, fmin_cobyla, least_squares

# 无约束优化
result = minimize(
    fun=objective_function,
    x0=initial_guess,
    method='L-BFGS-B',
    jac=gradient_function,  # 可选
    bounds=bounds,
)

# COBYLA（约束优化）- MNE 用于偶极子拟合
result = fmin_cobyla(
    func=objective,
    x0=initial,
    cons=[constraint1, constraint2],
)

# 非线性最小二乘
result = least_squares(
    fun=residual_function,
    x0=initial,
    method='trf',  # Trust Region Reflective
    bounds=(lower, upper),
)

# MNE 用法：偶极子位置拟合
def fit_dipole(measured, forward_model, initial_pos):
    """拟合偶极子位置和方向"""
    
    def objective(params):
        pos = params[:3]
        ori = params[3:6]
        predicted = compute_forward(pos, ori, forward_model)
        return np.linalg.norm(measured - predicted)
    
    result = minimize(
        objective,
        x0=np.concatenate([initial_pos, [0, 0, 1]]),
        method='L-BFGS-B',
    )
    
    return result.x[:3], result.x[3:6]
```

#### Rust 等价
```rust
use argmin::core::{CostFunction, Executor, Gradient};
use argmin::solver::linesearch::MoreThuente;
use argmin::solver::quasinewton::LBFGS;

// 定义优化问题
struct DipoleFitting {
    measured: Array1<f64>,
    forward_model: ForwardModel,
}

impl CostFunction for DipoleFitting {
    type Param = Array1<f64>;
    type Output = f64;
    
    fn cost(&self, params: &Self::Param) -> Result<Self::Output> {
        let pos = params.slice(s![..3]);
        let ori = params.slice(s![3..6]);
        
        let predicted = compute_forward(&pos, &ori, &self.forward_model);
        let residual = &self.measured - &predicted;
        
        Ok(residual.norm_l2())
    }
}

// 可选：提供梯度
impl Gradient for DipoleFitting {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;
    
    fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient> {
        // 数值梯度或解析梯度
        // ...
    }
}

// 运行优化
fn fit_dipole(measured: Array1<f64>, initial: Array1<f64>) -> Result<Array1<f64>> {
    let problem = DipoleFitting { measured, forward_model };
    
    let linesearch = MoreThuente::new();
    let solver = LBFGS::new(linesearch, 7);
    
    let res = Executor::new(problem, solver)
        .configure(|state| state.param(initial).max_iters(100))
        .run()?;
    
    Ok(res.state().get_best_param().unwrap().clone())
}
```

**可用 Crates**:
- `argmin` - 优化框架（推荐）
- `optimization` - 另一个选择
- `lbfgsb` - L-BFGS-B 的 Rust 绑定

**挑战**:
- API 比 SciPy 更底层
- 需要实现 trait
- 部分高级算法需要 C/Fortran 绑定

---

## 6. scipy.interpolate - 插值

#### Python/SciPy 模式
```python
from scipy.interpolate import interp1d, RectBivariateSpline, griddata

# 1D 插值
f = interp1d(x, y, kind='cubic')
y_new = f(x_new)

# 2D 插值（MNE 用于通道插值）
interp = RectBivariateSpline(x, y, z)
z_new = interp(x_new, y_new)

# 散点插值
values = griddata(points, values, xi, method='cubic')

# MNE 用法：坏通道插值
def interpolate_bad_channels(data, good_positions, bad_positions):
    """使用空间插值修复坏通道"""
    from scipy.interpolate import Rbf
    
    # 对每个时间点插值
    interpolated = np.zeros((len(bad_positions), data.shape[1]))
    
    for t in range(data.shape[1]):
        # 径向基函数插值
        rbf = Rbf(
            good_positions[:, 0],
            good_positions[:, 1],
            good_positions[:, 2],
            data[:, t],
            function='thin_plate'
        )
        
        interpolated[:, t] = rbf(
            bad_positions[:, 0],
            bad_positions[:, 1],
            bad_positions[:, 2]
        )
    
    return interpolated
```

#### Rust 等价
```rust
// 1D 插值
use linear_interpolation::LinearInterpolation;

let interp = LinearInterpolation::new(x.to_vec(), y.to_vec());
let y_new = x_new.mapv(|xi| interp.interpolate(xi));

// 三次样条（需要更高级的库）
// 可能需要自己实现或使用 C 绑定

// 散点插值 - 目前 Rust 生态较弱
// 可能需要绑定到 C/C++ 库
```

**可用 Crates**:
- `linear-interpolation` - 线性插值
- `interp` - 基础插值
- **问题**: 高级插值（三次样条、RBF）库较少

**建议**:
- 简单插值可用现有 crate
- 复杂插值可能需要 FFI 到 C 库（如 GSL）

---

## 7. scipy.ndimage - 图像处理

#### Python/SciPy 模式
```python
from scipy.ndimage import gaussian_filter, binary_dilation, label

# 高斯平滑
smoothed = gaussian_filter(data, sigma=2.0)

# 形态学操作
dilated = binary_dilation(mask, iterations=2)

# 连通区域标记
labeled, n_features = label(binary_image)

# MNE 用法：空间平滑（源估计）
def smooth_source_estimate(stc, adjacency, n_iter=1):
    """在皮层表面上平滑源估计"""
    from scipy.ndimage import convolve1d
    
    # 简单的邻域平均
    data = stc.data
    for _ in range(n_iter):
        # 使用邻接矩阵进行卷积平滑
        smoothed = adjacency.dot(data) / adjacency.sum(axis=1)
        data = smoothed
    
    return data
```

#### Rust 等价
```rust
use ndarray::prelude::*;

// 高斯滤波（1D，可扩展到多维）
pub fn gaussian_filter_1d(data: &Array1<f64>, sigma: f64) -> Array1<f64> {
    let kernel_size = (6.0 * sigma).ceil() as usize | 1;  // 奇数
    let kernel = gaussian_kernel(kernel_size, sigma);
    
    convolve(data, &kernel)
}

fn gaussian_kernel(size: usize, sigma: f64) -> Array1<f64> {
    let half = (size / 2) as f64;
    let mut kernel = Array1::zeros(size);
    
    for i in 0..size {
        let x = i as f64 - half;
        kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
    }
    
    let sum = kernel.sum();
    kernel / sum
}

// 卷积
fn convolve(data: &Array1<f64>, kernel: &Array1<f64>) -> Array1<f64> {
    let n = data.len();
    let k = kernel.len();
    let half = k / 2;
    
    let mut result = Array1::zeros(n);
    
    for i in 0..n {
        let mut sum = 0.0;
        for (j, &k_val) in kernel.iter().enumerate() {
            let idx = (i + j).saturating_sub(half);
            if idx < n {
                sum += data[idx] * k_val;
            }
        }
        result[i] = sum;
    }
    
    result
}
```

**可用 Crates**:
- `imageproc` - 图像处理操作
- `ndarray-image` - ndarray 图像工具
- **限制**: 专注于 2D 图像，3D/nD 支持较少

---

## 8. scipy.fft - 快速傅里叶变换

#### Python/SciPy 模式
```python
from scipy.fft import fft, ifft, rfft, irfft, fftfreq, rfftfreq, next_fast_len

# 实数 FFT（最常用）
fft_data = rfft(data, n=n_fft, axis=-1)
freqs = rfftfreq(n_samples, d=1.0/sfreq)

# 复数 FFT
complex_fft = fft(data, axis=-1)
inverse = ifft(complex_fft, axis=-1)

# 找到快速 FFT 长度（2 的幂次）
n_fast = next_fast_len(n_samples)

# FFT 归一化
fft_norm = rfft(data, norm='ortho')  # 正交归一化
```

#### Rust 等价
```rust
// 参见 NumPy 分析中的 FFT 部分
// scipy.fft 和 numpy.fft 功能类似

// next_fast_len
fn next_fast_len(n: usize) -> usize {
    // 找到 >= n 的最小 2^k * 3^m * 5^p
    // 简化版：找下一个 2 的幂次
    n.next_power_of_two()
}
```

---

## 9. scipy.spatial - 空间数据结构

#### Python/SciPy 模式
```python
from scipy.spatial import distance_matrix, cKDTree
from scipy.spatial.distance import cdist, pdist, squareform

# 距离矩阵
dist = distance_matrix(points1, points2)
dist = cdist(points1, points2, metric='euclidean')

# 成对距离
pairwise = pdist(points, metric='euclidean')
dist_matrix = squareform(pairwise)

# KD 树（快速最近邻）
tree = cKDTree(points)
distances, indices = tree.query(query_points, k=5)

# MNE 用法：找最近的传感器
def find_nearest_sensors(target_pos, sensor_positions, n_neighbors=3):
    """找到最近的传感器"""
    tree = cKDTree(sensor_positions)
    distances, indices = tree.query(target_pos, k=n_neighbors)
    return indices, distances
```

#### Rust 等价
```rust
use kiddo::KdTree;
use ndarray::prelude::*;

// 欧几里得距离矩阵
fn distance_matrix(points1: &Array2<f64>, points2: &Array2<f64>) -> Array2<f64> {
    let n1 = points1.nrows();
    let n2 = points2.nrows();
    let mut dist = Array2::zeros((n1, n2));
    
    for i in 0..n1 {
        for j in 0..n2 {
            let diff = points1.row(i).to_owned() - points2.row(j).to_owned();
            dist[[i, j]] = diff.mapv(|x| x * x).sum().sqrt();
        }
    }
    
    dist
}

// KD 树
fn find_nearest_sensors(
    target_pos: &[f64; 3],
    sensor_positions: &Array2<f64>,
    n_neighbors: usize
) -> (Vec<usize>, Vec<f64>) {
    let mut tree = KdTree::new();
    
    for (i, row) in sensor_positions.axis_iter(Axis(0)).enumerate() {
        let pos: [f64; 3] = [row[0], row[1], row[2]];
        tree.add(&pos, i).unwrap();
    }
    
    let results = tree.nearest(target_pos, n_neighbors).unwrap();
    
    let indices: Vec<usize> = results.iter().map(|r| *r.item).collect();
    let distances: Vec<f64> = results.iter().map(|r| r.distance.sqrt()).collect();
    
    (indices, distances)
}
```

**可用 Crates**:
- `kiddo` - KD 树（推荐）
- `kdtree` - 另一个 KD 树实现
- `rstar` - R*树（空间索引）

---

## Rust 实现建议

### 推荐 Crate 组合

```toml
[dependencies]
# 核心数组
ndarray = { version = "0.15", features = ["rayon"] }
ndarray-stats = "0.5"
ndarray-linalg = { version = "0.16", features = ["openblas-static"] }

# 稀疏矩阵
sprs = "0.11"

# 信号处理
rustfft = "6.0"
biquad = "0.4"
apodize = "1.0"

# 统计
statrs = "0.16"

# 优化
argmin = "0.8"
argmin-math = { version = "0.3", features = ["ndarray_latest-serde"] }

# 图算法
petgraph = "0.6"

# 空间
kiddo = "2.0"

# 数学
num-complex = "0.4"
num-traits = "0.2"

# 并行
rayon = "1.7"
```

### 需要自己实现的模块

#### 1. **高级信号处理**
```rust
// src/signal/mod.rs

pub mod filters {
    // Butterworth, Chebyshev, Elliptic 滤波器设计
    pub fn butter(order: usize, cutoff: &[f64], fs: f64, btype: FilterType) -> SOS;
    pub fn cheby1(...) -> SOS;
}

pub mod spectral {
    // Welch, STFT, 谱图
    pub fn welch(...) -> (Array1<f64>, Array1<f64>);
    pub fn stft(...) -> (Array1<f64>, Array1<f64>, Array2<Complex<f64>>);
    pub fn spectrogram(...) -> (Array1<f64>, Array1<f64>, Array2<f64>);
}

pub mod transform {
    // Hilbert, 重采样
    pub fn hilbert(data: &Array1<f64>) -> Array1<Complex<f64>>;
    pub fn resample(data: &Array1<f64>, new_len: usize) -> Array1<f64>;
}
```

#### 2. **插值库**
```rust
// src/interpolate/mod.rs

pub struct Interp1D {
    x: Array1<f64>,
    y: Array1<f64>,
    kind: InterpKind,
}

pub enum InterpKind {
    Linear,
    Cubic,
    Spline,
}

impl Interp1D {
    pub fn new(x: Array1<f64>, y: Array1<f64>, kind: InterpKind) -> Self;
    pub fn interpolate(&self, x_new: &Array1<f64>) -> Array1<f64>;
}
```

---

## 难度评估和优先级

| 模块 | 难度 | 优先级 | 建议 |
|------|------|--------|------|
| scipy.signal.butter/iirfilter | ⭐⭐⭐⭐ | P0 | 移植或 FFI |
| scipy.signal.sosfiltfilt | ⭐⭐⭐ | P0 | 可实现 |
| scipy.signal.welch/stft | ⭐⭐⭐ | P0 | 可实现 |
| scipy.signal.hilbert | ⭐⭐ | P1 | 可实现 |
| scipy.linalg.* | ⭐⭐ | P0 | ndarray-linalg ✅ |
| scipy.sparse.* | ⭐⭐ | P0 | sprs ✅ |
| scipy.sparse.csgraph | ⭐⭐ | P1 | petgraph ✅ |
| scipy.stats.* | ⭐⭐ | P1 | statrs ✅ |
| scipy.optimize | ⭐⭐⭐ | P1 | argmin ✅ |
| scipy.interpolate | ⭐⭐⭐ | P2 | 部分自实现 |

---

## 总结

### 关键要点

1. **scipy.signal 是最大挑战**
   - 滤波器设计需要深入实现
   - Welch/STFT 可以实现但工作量大
   - 考虑移植或 FFI

2. **scipy.linalg 已解决**
   - ndarray-linalg 提供完整支持
   - API 接近，易于移植

3. **scipy.sparse 已解决**
   - sprs 提供完整稀疏矩阵支持
   - petgraph 处理图算法

4. **scipy.stats 大部分可用**
   - statrs 覆盖主要分布和测试
   - 部分高级功能需自实现

5. **scipy.optimize 可用但 API 不同**
   - argmin 功能强大
   - 需要适应 trait 风格

### 实现策略

1. **阶段 1**: 使用现有 crate（linalg, sparse, stats）
2. **阶段 2**: 实现核心信号处理（filtfilt, welch）
3. **阶段 3**: 补充滤波器设计（考虑 FFI）
4. **阶段 4**: 高级功能（插值、优化）

继续阅读：[03_scikit-learn依赖分析.md](03_scikit-learn依赖分析.md)
