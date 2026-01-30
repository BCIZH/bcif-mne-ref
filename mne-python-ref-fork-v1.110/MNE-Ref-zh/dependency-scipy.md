# MNE-Python SciPy 依赖详细分析

> **核心依赖**: `scipy >= 1.11`  
> **使用频率**: 🔥🔥🔥🔥🔥 (核心算法层)  
> **角色**: 信号处理、稀疏矩阵、统计检验、优化、空间计算

---

## 目录

1. [SciPy 模块使用统计](#scipy-模块使用统计)
2. [scipy.signal - 信号处理](#scipysignal---信号处理)
3. [scipy.sparse - 稀疏矩阵](#scipysparse---稀疏矩阵)
4. [scipy.linalg - 线性代数](#scipylinalg---线性代数)
5. [scipy.stats - 统计检验](#scipystats---统计检验)
6. [scipy.spatial - 空间计算](#scipyspatial---空间计算)
7. [scipy.optimize - 优化算法](#scipyoptimize---优化算法)
8. [scipy.interpolate - 插值](#scipyinterpolate---插值)
9. [scipy.ndimage - 图像处理](#scipyndimage---图像处理)
10. [SciPy vs NumPy 对比](#scipy-vs-numpy-对比)

---

## SciPy 模块使用统计

| SciPy 模块 | 使用文件数 | 关键功能 | 核心应用 |
|-----------|-----------|---------|---------|
| **scipy.signal** | ~60 | 滤波、重采样、窗函数 | 预处理、时频分析 |
| **scipy.sparse** | ~80 | CSR/COO 稀疏矩阵、图算法 | 正向模型、邻接矩阵 |
| **scipy.linalg** | ~100 | SVD、特征值、BLAS/LAPACK | 源定位、ICA、降维 |
| **scipy.stats** | ~50 | t/F 检验、分布函数 | 统计推断、permutation test |
| **scipy.spatial** | ~40 | Delaunay、KDTree、距离 | 通道插值、3D 计算 |
| **scipy.optimize** | ~30 | 最小二乘、约束优化 | 偶极子拟合、配准 |
| **scipy.interpolate** | ~25 | 1D/2D 插值 | 通道插值、重采样 |
| **scipy.ndimage** | ~15 | 形态学、标签 | 聚类分析、ROI |
| **scipy.fft** | ~10 | FFT 函数 | 与 NumPy.fft 互补 |

---

## scipy.signal - 信号处理

### 1. 滤波器设计

**位置**: `mne/filter.py`

#### 1.1 IIR 滤波器

```python
from scipy.signal import iirfilter, iirdesign, butter, cheby1, filtfilt

def construct_iir_filter(iir_params, f_pass, f_stop, sfreq):
    """构建 IIR 滤波器"""
    # 方法 1: 直接设计
    b, a = iirfilter(
        N=order,                    # 滤波器阶数
        Wn=f_pass / (sfreq / 2),    # 归一化截止频率
        btype='lowpass',             # 'lowpass', 'highpass', 'bandpass'
        ftype='butter',              # 'butter', 'cheby1', 'ellip'
        output='ba'                  # 'ba', 'sos'
    )
    
    # 方法 2: 自动设计满足规格
    b, a = iirdesign(
        wp=f_pass / (sfreq / 2),     # 通带边缘
        ws=f_stop / (sfreq / 2),     # 阻带边缘
        gpass=3,                      # 通带最大衰减 (dB)
        gstop=40,                     # 阻带最小衰减 (dB)
        ftype='butter'
    )
    
    return b, a
```

**应用**: `Raw.filter()`, `Epochs.filter()` 的 IIR 模式

---

#### 1.2 滤波应用

```python
from scipy.signal import filtfilt, lfilter, sosfiltfilt

def apply_filter(data, b, a, method='fir'):
    """应用滤波器"""
    if method == 'iir':
        # 零相位 IIR 滤波 (前向-后向)
        data_filtered = filtfilt(b, a, data, axis=-1)
        
    elif method == 'iir_forward':
        # 单向滤波 (因果)
        data_filtered = lfilter(b, a, data, axis=-1)
        
    elif method == 'sos':
        # Second-Order Sections (更稳定)
        data_filtered = sosfiltfilt(sos, data, axis=-1)
    
    return data_filtered
```

**filtfilt vs lfilter**:
- `filtfilt`: 零相位延迟，但非因果
- `lfilter`: 有相位延迟，因果滤波

---

### 2. 窗函数

**位置**: `mne/_ola.py`, `mne/preprocessing/stim.py`

```python
from scipy.signal import get_window
from scipy.signal.windows import hann, hamming, blackman

# 方法 1: 使用 get_window
window = get_window('hann', n_samples)

# 方法 2: 直接调用
hann_window = hann(n_samples)
hamming_window = hamming(n_samples)
```

**应用**: STFT、Overlap-Add、artifact 修复

---

### 3. 重采样

**位置**: `mne/filter.py`

```python
from scipy.signal import resample, resample_poly

def resample_data(data, up, down):
    """多相滤波器重采样"""
    # 优于 FFT-based resample
    data_resampled = resample_poly(
        data, 
        up=up,          # 上采样因子
        down=down,      # 下采样因子
        axis=-1,
        window='hamming'
    )
    
    return data_resampled
```

**应用**: `Raw.resample()`, `Epochs.resample()`

---

### 4. 频域分析

**位置**: `mne/viz/misc.py`

```python
from scipy.signal import freqz, group_delay

def compute_filter_response(b, a, worN=8192):
    """计算滤波器频率响应"""
    # 幅频响应
    w, h = freqz(b, a, worN=worN)
    freqs = w * sfreq / (2 * np.pi)
    magnitude = 20 * np.log10(np.abs(h))
    
    # 群延迟
    w, gd = group_delay((b, a), w=worN)
    
    return freqs, magnitude, gd
```

**应用**: `mne.viz.plot_filter()` 滤波器可视化

---

## scipy.sparse - 稀疏矩阵

### 1. 稀疏矩阵格式

**位置**: `mne/forward/`, `mne/stats/`, `mne/channels/`

```python
from scipy.sparse import (
    csr_array,      # Compressed Sparse Row (最常用)
    csc_array,      # Compressed Sparse Column
    coo_array,      # Coordinate format (构建时使用)
    lil_array,      # List of Lists (动态构建)
)

# 创建稀疏矩阵
data = [1, 2, 3, 4]
row = [0, 0, 1, 2]
col = [0, 2, 2, 0]

# COO 格式 (构建)
adjacency_coo = coo_array((data, (row, col)), shape=(3, 3))

# 转换为 CSR (运算)
adjacency_csr = adjacency_coo.tocsr()
```

---

### 2. 正向模型的稀疏表示

**位置**: `mne/forward/forward.py`

```python
from scipy import sparse

def _read_forward_solution(fid):
    """读取稀疏正向矩阵"""
    # 正向矩阵通常很大但稀疏
    # shape: (n_sensors, 3 * n_sources)
    
    # 从 FIFF 读取为稀疏格式
    fwd_matrix = sparse.csr_array(data)
    
    # 稀疏矩阵乘法
    leadfield = fwd_matrix @ source_ori  # 高效!
    
    return leadfield
```

**优势**: 
- 内存占用: 稠密矩阵 GB → 稀疏矩阵 MB
- 计算速度: 跳过零元素

---

### 3. 邻接矩阵

**位置**: `mne/channels/channels.py`, `mne/stats/cluster_level.py`

```python
from scipy.sparse import csr_array
from scipy.spatial import Delaunay

def find_ch_adjacency(info, ch_type='eeg'):
    """计算通道邻接矩阵"""
    # 获取通道位置
    pos = _get_channel_positions(info, picks)
    
    # Delaunay 三角剖分
    tri = Delaunay(pos[:, :2])  # 2D 投影
    
    # 构建邻接矩阵 (稀疏)
    n_channels = len(pos)
    adjacency = lil_array((n_channels, n_channels), dtype=int)
    
    for simplex in tri.simplices:
        # 三角形的三条边
        for i in range(3):
            v1, v2 = simplex[i], simplex[(i + 1) % 3]
            adjacency[v1, v2] = 1
            adjacency[v2, v1] = 1  # 对称
    
    # 转换为 CSR
    adjacency = adjacency.tocsr()
    
    return adjacency
```

**应用**: Cluster permutation test 的空间邻接

---

### 4. 稀疏图算法

**位置**: `mne/source_space/_source_space.py`, `mne/stats/cluster_level.py`

```python
from scipy.sparse.csgraph import (
    connected_components,  # 连通分量
    dijkstra,              # 最短路径
)

# 聚类标记
def _find_clusters(stat_map, threshold, adjacency):
    """使用图论找聚类"""
    # 超过阈值的点
    above_threshold = stat_map > threshold
    
    # 子图邻接矩阵
    sub_adjacency = adjacency[above_threshold][:, above_threshold]
    
    # 连通分量 = 聚类
    n_clusters, labels = connected_components(
        sub_adjacency, 
        directed=False
    )
    
    return n_clusters, labels

# 最短路径
def _compute_source_distances(src):
    """计算源空间距离矩阵"""
    # src['dist']: 稀疏距离矩阵 (邻居之间)
    
    # Dijkstra 算法 (全源最短路径)
    dist_matrix = dijkstra(
        src['dist'], 
        directed=False, 
        return_predecessors=False
    )
    
    return dist_matrix
```

---

## scipy.linalg - 线性代数

### 1. 与 numpy.linalg 的差异

| 功能 | NumPy | SciPy | MNE 选择 |
|------|-------|-------|---------|
| **SVD** | `np.linalg.svd` | `scipy.linalg.svd` | SciPy (更多选项) |
| **特征值** | `np.linalg.eigh` | `scipy.linalg.eigh` | SciPy (广义特征值) |
| **矩阵求逆** | `np.linalg.inv` | `scipy.linalg.inv` | SciPy (更稳定) |
| **BLAS** | ❌ | ✅ `get_blas_funcs` | SciPy (性能优化) |
| **LAPACK** | 间接 | ✅ `get_lapack_funcs` | SciPy (直接调用) |

---

### 2. BLAS/LAPACK 优化

**位置**: `mne/utils/linalg.py`

```python
from scipy import linalg

def _get_blas_funcs(dtype, names):
    """获取优化的 BLAS 函数"""
    return linalg.get_blas_funcs(
        names,                           # ['gemm', 'symm', ...]
        (np.empty(0, dtype),)
    )

# 示例: 矩阵乘法
gemm = _get_blas_funcs(np.float64, ['gemm'])[0]
C = gemm(alpha=1.0, a=A, b=B, beta=0.0, c=C, 
         trans_a=False, trans_b=False)
# C = alpha * A @ B + beta * C
```

**性能提升**: 2-5x 相比 `np.dot`

---

### 3. 广义特征值问题

**位置**: `mne/decoding/csp.py`, `mne/decoding/_ged.py`

```python
from scipy.linalg import eigh

def solve_gep(A, B):
    """求解广义特征值问题: A v = lambda B v"""
    # eigh: 对称/Hermitian 矩阵
    eigvals, eigvecs = eigh(A, B)
    
    # 降序排序 (最大特征值优先)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    return eigvals, eigvecs
```

**应用**:
- **CSP (Common Spatial Pattern)**: 类间/类内协方差
- **GED (Generalized Eigenvalue Decomposition)**: 信号/噪声协方差

---

### 4. 矩阵分解

**位置**: `mne/preprocessing/xdawn.py`

```python
from scipy.linalg import (
    pinv,       # 伪逆
    svd,        # 奇异值分解
    qr,         # QR 分解
    cholesky,   # Cholesky 分解
)

# 伪逆 (Moore-Penrose)
def compute_pseudoinverse(A, rcond=1e-15):
    """计算伪逆"""
    A_pinv = pinv(A, rcond=rcond)
    
    # 等价于 (但 pinv 更稳定):
    # U, s, Vh = svd(A, full_matrices=False)
    # A_pinv = Vh.T @ np.diag(1/s) @ U.T
    
    return A_pinv
```

---

## scipy.stats - 统计检验

### 1. t 检验

**位置**: `mne/stats/parametric.py`

```python
from scipy.stats import t as t_dist

def ttest_1samp_no_p(X, sigma=0, method='relative'):
    """单样本 t 检验 (不计算 p 值)"""
    # X: shape (n_samples, ...)
    
    n_samples = X.shape[0]
    
    # 均值
    X_mean = X.mean(axis=0)
    
    # 标准误
    if method == 'relative':
        X_std = X.std(axis=0, ddof=1)
        denom = X_std + sigma * np.abs(X_mean)
    else:  # 'absolute'
        X_std = X.std(axis=0, ddof=1)
        denom = X_std + sigma
    
    # t 统计量
    t_vals = np.sqrt(n_samples) * X_mean / denom
    
    # p 值 (如果需要)
    # p_vals = 2 * t_dist.sf(np.abs(t_vals), n_samples - 1)
    
    return t_vals
```

**应用**: ERP 分析、组间比较

---

### 2. F 检验 (ANOVA)

**位置**: `mne/stats/parametric.py`

```python
from scipy.stats import f as f_dist

def f_oneway(*args):
    """单因素方差分析"""
    # args: (group1, group2, ..., groupN)
    # 每个 group shape: (n_samples, ...)
    
    n_groups = len(args)
    n_samples_per_group = [a.shape[0] for a in args]
    n_total = sum(n_samples_per_group)
    
    # 总均值
    grand_mean = np.concatenate(args, axis=0).mean(axis=0)
    
    # 组间平方和 (Between-group SS)
    ss_between = sum(
        n * (group.mean(axis=0) - grand_mean) ** 2
        for n, group in zip(n_samples_per_group, args)
    )
    
    # 组内平方和 (Within-group SS)
    ss_within = sum(
        ((group - group.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
        for group in args
    )
    
    # F 统计量
    df_between = n_groups - 1
    df_within = n_total - n_groups
    
    f_vals = (ss_between / df_between) / (ss_within / df_within)
    
    # p 值
    # p_vals = f_dist.sf(f_vals, df_between, df_within)
    
    return f_vals
```

---

### 3. 其他分布函数

**位置**: `mne/minimum_norm/inverse.py`, `mne/filter.py`

```python
from scipy.stats import chi2

# 卡方分布 (置信区间)
def compute_noise_cov_confidence(cov, n_samples):
    """噪声协方差置信区间"""
    df = n_samples - 1
    
    # 95% 置信区间
    lower = df / chi2.ppf(0.975, df)
    upper = df / chi2.ppf(0.025, df)
    
    cov_lower = cov * lower
    cov_upper = cov * upper
    
    return cov_lower, cov_upper
```

---

## scipy.spatial - 空间计算

### 1. Delaunay 三角剖分

**位置**: `mne/channels/channels.py`, `mne/viz/topomap.py`

```python
from scipy.spatial import Delaunay

def compute_delaunay_triangulation(pos):
    """计算 Delaunay 三角剖分"""
    # pos: shape (n_points, 2 or 3)
    
    tri = Delaunay(pos[:, :2])  # 仅使用 x, y
    
    # tri.simplices: 三角形顶点索引, shape (n_triangles, 3)
    # tri.neighbors: 邻居三角形索引
    
    return tri
```

**应用**: 
- 地形图插值 (topomap)
- 通道邻接矩阵

---

### 2. 凸包 (Convex Hull)

**位置**: `mne/surface.py`, `mne/viz/_3d.py`

```python
from scipy.spatial import ConvexHull

def compute_head_surface_hull(points):
    """计算头部表面凸包"""
    hull = ConvexHull(points)
    
    # hull.vertices: 凸包顶点索引
    # hull.simplices: 凸包面 (三角形)
    
    return hull
```

---

### 3. KDTree (近邻搜索)

**位置**: `mne/surface.py`

```python
from scipy.spatial import KDTree

def find_nearest_neighbors(points, query_points, k=3):
    """KDTree 近邻搜索"""
    tree = KDTree(points)
    
    # k 近邻
    distances, indices = tree.query(query_points, k=k)
    
    return distances, indices
```

**应用**: 源空间点匹配

---

### 4. 距离计算

**位置**: `mne/channels/layout.py`, `mne/viz/montage.py`

```python
from scipy.spatial.distance import (
    pdist,        # Pairwise distances
    cdist,        # Cross distances
    squareform,   # 向量 ↔ 矩阵转换
)

# 成对距离
distances_vec = pdist(pos, metric='euclidean')  # shape: (n*(n-1)/2,)
distances_mat = squareform(distances_vec)       # shape: (n, n)

# 交叉距离
distances = cdist(pos1, pos2, metric='euclidean')  # shape: (n1, n2)
```

---

## scipy.optimize - 优化算法

### 1. 最小二乘

**位置**: `mne/coreg.py`

```python
from scipy.optimize import leastsq

def fit_matched_points(src_pts, tgt_pts):
    """ICP 配准: 最小化点到点距离"""
    
    def objective(params):
        # params: [tx, ty, tz, rx, ry, rz, scale]
        trans = _params_to_transform(params)
        src_transformed = apply_trans(trans, src_pts)
        
        # 残差
        residuals = (src_transformed - tgt_pts).ravel()
        return residuals
    
    # Levenberg-Marquardt 算法
    params_opt, _ = leastsq(objective, params_init)
    
    return params_opt
```

---

### 2. 约束优化

**位置**: `mne/dipole.py`, `mne/bem.py`

```python
from scipy.optimize import fmin_cobyla

def fit_dipole_position(data, leadfield):
    """拟合偶极子位置"""
    
    def objective(pos):
        # 计算拟合残差
        lf = compute_leadfield(pos)
        residual = np.linalg.norm(data - lf @ dipole_moment)
        return residual
    
    def constraint_inside_head(pos):
        # 约束: 偶极子必须在头部内
        return head_radius - np.linalg.norm(pos)
    
    # COBYLA (约束优化)
    pos_opt = fmin_cobyla(
        objective, 
        x0=pos_init,
        cons=[constraint_inside_head]
    )
    
    return pos_opt
```

---

## scipy.interpolate - 插值

### 1. 1D 插值

**位置**: `mne/preprocessing/stim.py`, `mne/preprocessing/realign.py`

```python
from scipy.interpolate import interp1d

def interpolate_bad_segments(data, times, bad_mask):
    """插值坏数据段"""
    good_mask = ~bad_mask
    
    # 1D 线性插值
    f = interp1d(
        times[good_mask], 
        data[good_mask],
        kind='linear',      # 'linear', 'cubic', 'nearest'
        axis=0,
        fill_value='extrapolate'
    )
    
    data_interp = f(times[bad_mask])
    data[bad_mask] = data_interp
    
    return data
```

---

### 2. 2D 插值

**位置**: `mne/channels/interpolation.py`

```python
from scipy.interpolate import RectBivariateSpline

def interpolate_topomap(x, y, z, xi, yi):
    """2D 地形图插值"""
    # 双立方插值
    interp = RectBivariateSpline(x, y, z, kx=3, ky=3)
    zi = interp(xi, yi)
    
    return zi
```

---

## scipy.ndimage - 图像处理

### 1. 形态学操作

**位置**: `mne/surface.py`, `mne/preprocessing/artifact_detection.py`

```python
from scipy.ndimage import (
    binary_dilation,   # 二值膨胀
    binary_erosion,    # 二值腐蚀
    label,             # 连通区域标记
)

# 扩展 ROI
roi_dilated = binary_dilation(roi_mask, iterations=2)

# 标记连通区域
labeled_array, n_features = label(binary_image)
```

---

### 2. 距离变换

**位置**: `mne/preprocessing/artifact_detection.py`

```python
from scipy.ndimage import distance_transform_edt

def compute_distance_to_artifact(artifact_mask):
    """计算到 artifact 的欧氏距离"""
    distances = distance_transform_edt(~artifact_mask)
    return distances
```

---

## SciPy vs NumPy 对比

| 功能 | NumPy | SciPy | MNE 策略 |
|------|-------|-------|---------|
| **FFT** | `np.fft` | `scipy.fft` | 两者混用 |
| **linalg** | `np.linalg` | `scipy.linalg` | 优先 SciPy (更强大) |
| **random** | `np.random` | ❌ | NumPy |
| **signal** | ❌ | `scipy.signal` | SciPy 独有 |
| **sparse** | ❌ | `scipy.sparse` | SciPy 独有 |
| **stats** | ❌ | `scipy.stats` | SciPy 独有 |
| **spatial** | ❌ | `scipy.spatial` | SciPy 独有 |
| **optimize** | ❌ | `scipy.optimize` | SciPy 独有 |

---

## 总结

### SciPy 在 MNE 中的重要性

| 维度 | 评分 | 说明 |
|------|------|------|
| **信号处理** | ⭐⭐⭐⭐⭐ | 滤波、重采样不可或缺 |
| **稀疏矩阵** | ⭐⭐⭐⭐⭐ | 正向模型、邻接矩阵 |
| **线性代数** | ⭐⭐⭐⭐⭐ | BLAS/LAPACK 性能关键 |
| **统计检验** | ⭐⭐⭐⭐⭐ | 推断统计核心 |
| **空间计算** | ⭐⭐⭐⭐ | 3D 可视化、插值 |
| **优化** | ⭐⭐⭐ | 配准、偶极子拟合 |

---

**返回**: [依赖分析总览](dependency-analysis-overview.md)  
**上一步**: [NumPy 依赖分析](dependency-numpy.md)  
**下一步**: [scikit-learn 依赖分析](dependency-sklearn.md)
