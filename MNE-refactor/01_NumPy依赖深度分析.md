# MNE-Python NumPy 依赖深度分析

## 概述

NumPy 是 MNE-Python 的**绝对核心依赖**，几乎每个模块都直接使用。本文档详细分析 MNE 如何使用 NumPy，以及如何用 Rust `ndarray` 替代。

---

## NumPy 使用统计

### 导入频率
- **直接导入**: `import numpy as np` - 200+ 文件
- **子模块导入**: 
  - `np.linalg` - 100+ 使用
  - `np.fft` - 50+ 使用
  - `np.random` - 80+ 使用
  - `np.polynomial` - 10+ 使用

### 核心功能分布

| 功能类别 | 使用频率 | 关键程度 | Rust 替代难度 |
|---------|---------|---------|--------------|
| 数组创建/操作 | ⭐⭐⭐⭐⭐ | 必需 | 低 (ndarray) |
| 索引和切片 | ⭐⭐⭐⭐⭐ | 必需 | 低 (ndarray) |
| 广播和向量化 | ⭐⭐⭐⭐⭐ | 必需 | 低 (ndarray) |
| 线性代数 | ⭐⭐⭐⭐⭐ | 必需 | 中 (ndarray-linalg) |
| FFT | ⭐⭐⭐⭐ | 必需 | 中 (rustfft + 封装) |
| 随机数 | ⭐⭐⭐⭐ | 必需 | 低 (rand crate) |
| 数学函数 | ⭐⭐⭐⭐⭐ | 必需 | 低 (num-traits) |
| 多项式 | ⭐⭐ | 可选 | 中 (自实现) |

---

## 详细功能分析

### 1. 数组创建和基础操作

#### Python/NumPy 模式
```python
import numpy as np

# 创建数组
data = np.zeros((n_channels, n_times))
data = np.ones((100, 200))
data = np.empty((n_epochs, n_channels, n_times))
data = np.full((10, 10), fill_value=np.nan)

# 从列表/序列创建
arr = np.array([1, 2, 3, 4])
arr = np.asarray(data)  # 如果已是数组则不复制

# 范围和序列
x = np.arange(0, 10, 0.1)
y = np.linspace(0, 1, 100)
z = np.logspace(0, 3, 50)

# 网格
xx, yy = np.meshgrid(x, y)

# 对角矩阵
diag = np.diag([1, 2, 3])
I = np.eye(100)

# 随机数组（旧式）
random_data = np.random.randn(100, 200)
```

#### Rust ndarray 等价
```rust
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

// 创建数组
let data = Array2::<f64>::zeros((n_channels, n_times));
let data = Array2::<f64>::ones((100, 200));
let mut data = Array3::<f64>::uninit((n_epochs, n_channels, n_times));

// 填充特定值
let data = Array2::<f64>::from_elem((10, 10), f64::NAN);

// 从Vec创建
let arr = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
let arr = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])?;

// 范围（使用迭代器）
let x: Array1<f64> = Array::range(0.0, 10.0, 0.1);
let y: Array1<f64> = Array::linspace(0.0, 1.0, 100);

// 对角矩阵
let diag = Array2::from_diag(&arr1d![1.0, 2.0, 3.0]);
let I = Array2::<f64>::eye(100);

// 随机数组
let random_data: Array2<f64> = Array::random((100, 200), StandardNormal);
```

**关键差异**:
1. Rust 需要显式类型参数 `<f64>`
2. 错误处理通过 `Result` (如 `from_shape_vec`)
3. `uninit` 更高效但需要手动初始化
4. 迭代器风格更 Rust化

---

### 2. 数组索引和切片

#### Python/NumPy 模式
```python
# 基础索引
element = data[i, j]
row = data[i, :]
col = data[:, j]
block = data[i:j, k:l]

# 负索引
last = data[-1]
last_row = data[-1, :]

# 布尔索引
mask = data > 0.5
filtered = data[mask]

# 花式索引
indices = [0, 2, 5, 7]
selected = data[indices]
selected_2d = data[np.ix_(row_idx, col_idx)]

# 省略号
sliced = data[..., 0]  # 最后一维的第一个元素

# 常见模式：通道选择
picks = [0, 2, 5]
epoch_data = epochs[:, picks, :]  # 所有epoch，选择通道，所有时间点

# 条件选择
good_epochs = epochs[epoch_indices]
```

#### Rust ndarray 等价
```rust
// 基础索引
let element = data[[i, j]];
let row = data.row(i);
let col = data.column(j);
let block = data.slice(s![i..j, k..l]);

// 负索引（需要计算）
let last_idx = data.nrows() - 1;
let last = data.row(last_idx);

// 布尔索引（需要手动过滤）
let mask: Array2<bool> = data.mapv(|x| x > 0.5);
let filtered: Vec<f64> = data.iter()
    .zip(mask.iter())
    .filter(|(_, &m)| m)
    .map(|(&v, _)| v)
    .collect();

// 花式索引（需要手动构建）
let indices = vec![0, 2, 5, 7];
let selected = data.select(Axis(0), &indices);

// 切片宏 s![]
let sliced = data.slice(s![.., .., 0]);

// 通道选择
let picks = vec![0, 2, 5];
let epoch_data = epochs.select(Axis(1), &picks);

// 可变切片
let mut slice = data.slice_mut(s![i..j, ..]);
slice.fill(0.0);
```

**关键差异**:
1. Rust 用 `[[i, j]]` 代替 `[i, j]`
2. 切片用 `s![]` 宏，语法类似但需要显式
3. 布尔索引需要手动实现或用 `select`
4. 花式索引用 `select` 方法
5. 可变/不可变借用规则更严格

---

### 3. 数组操作和变换

#### Python/NumPy 模式
```python
# 形状操作
reshaped = data.reshape((n_total, -1))
flattened = data.ravel()
transposed = data.T
swapped = np.swapaxes(data, 0, 1)

# 堆叠和拼接
stacked = np.vstack([arr1, arr2])
hstacked = np.hstack([arr1, arr2])
concatenated = np.concatenate([arr1, arr2], axis=0)

# 重复和tile
repeated = np.repeat(arr, repeats=3, axis=0)
tiled = np.tile(arr, (2, 3))

# 扩展维度
expanded = arr[:, np.newaxis, :]
expanded = np.expand_dims(arr, axis=1)

# 压缩维度
squeezed = np.squeeze(arr)

# 常见模式：EEG数据转置
data_t = data.T  # (n_times, n_channels)
```

#### Rust ndarray 等价
```rust
// 形状操作
let reshaped = data.into_shape((n_total, Ix::dyn(-1)))?;
let flattened = data.into_shape((data.len(),))?;
let transposed = data.t();
let swapped = data.permuted_axes([1, 0, 2]);

// 堆叠和拼接
let stacked = ndarray::stack(Axis(0), &[arr1.view(), arr2.view()])?;
let concatenated = ndarray::concatenate(Axis(0), &[arr1.view(), arr2.view()])?;

// 重复（需要手动实现或用迭代器）
let repeated = arr.broadcast((3, arr.len())).unwrap().to_owned();

// 扩展维度
let expanded = arr.insert_axis(Axis(1));

// 压缩维度（移除长度为1的维度）
// 需要手动检查和重塑

// 转置
let data_t = data.t().to_owned();  // 或 reversed_axes()
```

**关键差异**:
1. Rust 的 `into_shape` 消费原数组
2. `view()` 用于借用，`to_owned()` 用于复制
3. `broadcast` 替代 `repeat` 和 `tile`
4. 维度操作更显式

---

### 4. 广播和向量化运算

#### Python/NumPy 模式
```python
# 基础算术（自动广播）
result = data + 100  # 标量加法
result = data * 2.0
result = data1 + data2  # 逐元素

# 广播不同形状
centered = data - np.mean(data, axis=0)  # (n, m) - (m,)
normalized = data / std[:, np.newaxis]   # (n, m) / (n, 1)

# 比较运算
mask = data > threshold
mask = (data > low) & (data < high)

# 逻辑运算
combined = mask1 & mask2
inverted = ~mask

# 常见模式：Z-score标准化
z_data = (data - data.mean(axis=0)) / data.std(axis=0)

# 条件选择
cleaned = np.where(mask, data, 0.0)
clamped = np.clip(data, -1, 1)
```

#### Rust ndarray 等价
```rust
// 基础算术
let result = &data + 100.0;  // 注意引用
let result = &data * 2.0;
let result = &data1 + &data2;

// 广播
let mean = data.mean_axis(Axis(0)).unwrap();
let centered = &data - &mean.insert_axis(Axis(0));

// 比较运算（返回bool数组）
let mask = data.mapv(|x| x > threshold);
let mask = data.mapv(|x| x > low && x < high);

// 逻辑运算
let combined = Zip::from(&mask1).and(&mask2)
    .map_collect(|&a, &b| a && b);

// Z-score标准化
let mean = data.mean_axis(Axis(0)).unwrap();
let std = data.std_axis(Axis(0), 0.0);
let z_data = (&data - &mean) / &std;

// 条件选择
let cleaned = Zip::from(&data).and(&mask)
    .map_collect(|&d, &m| if m { d } else { 0.0 });

// Clamp
let clamped = data.mapv(|x| x.clamp(-1.0, 1.0));
```

**关键差异**:
1. Rust 运算符需要引用 `&`
2. `mapv` 用于逐元素操作
3. `Zip` 用于多数组并行操作
4. 广播需要显式 `insert_axis`
5. `mean_axis` 等统计函数

---

### 5. 数学函数

#### Python/NumPy 模式
```python
# 基础数学
abs_data = np.abs(data)
sqrt_data = np.sqrt(data)
exp_data = np.exp(data)
log_data = np.log(data)
log10_data = np.log10(data)

# 三角函数
sin_data = np.sin(data)
cos_data = np.cos(data)
arctan2_data = np.arctan2(y, x)

# 复数
complex_data = data_real + 1j * data_imag
magnitude = np.abs(complex_data)
phase = np.angle(complex_data)
conjugate = np.conj(complex_data)

# 舍入
rounded = np.round(data, decimals=2)
floored = np.floor(data)
ceiled = np.ceil(data)

# 常见模式：功率计算
power = np.abs(complex_signal) ** 2
power_db = 10 * np.log10(power)
```

#### Rust ndarray 等价
```rust
use num_complex::Complex64;

// 基础数学
let abs_data = data.mapv(|x| x.abs());
let sqrt_data = data.mapv(|x| x.sqrt());
let exp_data = data.mapv(|x| x.exp());
let log_data = data.mapv(|x| x.ln());
let log10_data = data.mapv(|x| x.log10());

// 三角函数
let sin_data = data.mapv(|x| x.sin());
let cos_data = data.mapv(|x| x.cos());
let arctan2_data = Zip::from(&y).and(&x)
    .map_collect(|&y_val, &x_val| y_val.atan2(x_val));

// 复数
let complex_data: Array1<Complex64> = Zip::from(&data_real)
    .and(&data_imag)
    .map_collect(|&r, &i| Complex64::new(r, i));
let magnitude = complex_data.mapv(|c| c.norm());
let phase = complex_data.mapv(|c| c.arg());
let conjugate = complex_data.mapv(|c| c.conj());

// 舍入
let rounded = data.mapv(|x| (x * 100.0).round() / 100.0);
let floored = data.mapv(|x| x.floor());
let ceiled = data.mapv(|x| x.ceil());

// 功率计算
let power = complex_signal.mapv(|c| c.norm_sqr());
let power_db = power.mapv(|p| 10.0 * p.log10());
```

**关键差异**:
1. 使用 `mapv` 应用函数
2. 复数用 `num_complex` crate
3. 方法名略有不同 (`abs` vs `norm`, `ln` vs `log`)
4. 需要显式类型 `Complex64`

---

### 6. 聚合和统计

#### Python/NumPy 模式
```python
# 基础聚合
total = np.sum(data)
mean = np.mean(data, axis=0)
std = np.std(data, axis=0, ddof=1)  # 样本标准差
var = np.var(data)
min_val = np.min(data)
max_val = np.max(data)

# 位置
argmin = np.argmin(data, axis=1)
argmax = np.argmax(data)

# 中位数和分位数
median = np.median(data, axis=0)
q25 = np.percentile(data, 25, axis=0)
q75 = np.quantile(data, 0.75)

# 累积
cumsum = np.cumsum(data, axis=-1)
cumprod = np.cumprod(data)

# 常见模式：每个epoch的RMS
rms = np.sqrt(np.mean(data ** 2, axis=-1))

# 全局归一化
global_mean = np.mean(data)
global_std = np.std(data, ddof=1)
normalized = (data - global_mean) / global_std
```

#### Rust ndarray 等价
```rust
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use ndarray_stats::DeviationExt;

// 基础聚合
let total = data.sum();
let mean = data.mean_axis(Axis(0)).unwrap();
let std = data.std_axis(Axis(0), 1.0);  // ddof=1
let var = data.var_axis(Axis(0), 1.0);
let min_val = *data.min().unwrap();
let max_val = *data.max().unwrap();

// 位置（需要 ndarray_stats）
let argmin = data.argmin().unwrap();
let argmax_axis = data.map_axis(Axis(1), |row| {
    row.argmax().unwrap()
});

// 中位数和分位数
let median = data.median_axis_skipnan(Axis(0)).unwrap();
// 百分位需要手动实现或使用 statrs

// 累积
let cumsum = data.accumulate_axis_inplace(Axis(1), |&prev, curr| {
    *curr += prev;
});

// RMS
let squared = data.mapv(|x| x * x);
let mean_sq = squared.mean_axis(Axis(1)).unwrap();
let rms = mean_sq.mapv(|x| x.sqrt());

// 全局归一化
let global_mean = data.mean().unwrap();
let global_std = data.std(1.0);
let normalized = (&data - global_mean) / global_std;
```

**注意**:
- 需要 `ndarray-stats` crate
- 部分功能（百分位）需要额外实现
- `unwrap()` 处理 `Option` 返回

---

### 7. 线性代数 (`np.linalg`)

#### Python/NumPy 模式
```python
import numpy.linalg as LA

# 向量范数
norm = LA.norm(vector)
norm_2d = LA.norm(matrix, axis=1)
frobenius = LA.norm(matrix, ord='fro')

# 矩阵求逆
inv = LA.inv(matrix)
try:
    inv = LA.inv(singular_matrix)
except LA.LinAlgError:
    # 处理奇异矩阵
    pass

# 求解线性方程组 Ax = b
x = LA.solve(A, b)

# 特征值分解
eigenvalues, eigenvectors = LA.eigh(symmetric_matrix)
eigenvalues, eigenvectors = LA.eig(general_matrix)

# 奇异值分解
U, s, Vt = LA.svd(matrix, full_matrices=False)

# 矩阵秩
rank = LA.matrix_rank(matrix)

# QR 分解
Q, R = LA.qr(matrix)

# Cholesky 分解
L = LA.cholesky(positive_definite)

# 常见模式：协方差矩阵
cov = np.cov(data, rowvar=False)
eigenvalues, eigenvectors = LA.eigh(cov)

# 白化变换
whitened = data @ eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues))
```

#### Rust ndarray 等价
```rust
use ndarray::prelude::*;
use ndarray_linalg::*;

// 向量范数
let norm = vector.norm_l2();
let norm_2d: Array1<f64> = matrix.map_axis(Axis(1), |row| row.norm_l2());
let frobenius = matrix.norm_fro();

// 矩阵求逆
let inv = matrix.inv()?;  // 返回 Result

// 求解线性方程组
let x = A.solve_into(b)?;

// 特征值分解
let (eigenvalues, eigenvectors) = symmetric_matrix.eigh(UPLO::Lower)?;
let (eigenvalues, eigenvectors) = general_matrix.eig()?;

// 奇异值分解
let result = matrix.svd(false, true)?;  // (Some(U), s, Some(Vt))
let (u, s, vt) = result;

// 矩阵秩
let rank = matrix.rank(1e-10)?;

// QR 分解
let qr = matrix.qr()?;
let Q = qr.q();
let R = qr.r();

// Cholesky 分解
let chol = positive_definite.cholesky(UPLO::Lower)?;

// 协方差矩阵（需要手动实现或用 ndarray-stats）
fn covariance(data: &Array2<f64>) -> Result<Array2<f64>> {
    let mean = data.mean_axis(Axis(0)).unwrap();
    let centered = data - &mean;
    let n = data.nrows() as f64 - 1.0;
    Ok(centered.t().dot(&centered) / n)
}

let cov = covariance(&data)?;
let (eigenvalues, eigenvectors) = cov.eigh(UPLO::Lower)?;

// 白化变换
let diag = Array2::from_diag(&eigenvalues.mapv(|x| 1.0 / x.sqrt()));
let whitened = data.dot(&eigenvectors).dot(&diag);
```

**关键差异**:
1. 需要 `ndarray-linalg` crate (链接 LAPACK)
2. 所有操作返回 `Result` 用于错误处理
3. 方法名略有不同
4. 协方差需要手动实现或用 `ndarray-stats`
5. `UPLO` 枚举指定上/下三角

---

### 8. FFT (`np.fft`)

#### Python/NumPy 模式
```python
import numpy.fft as fft

# 实数FFT（最常用）
fft_data = fft.rfft(time_data, n=n_fft, axis=-1)
freqs = fft.rfftfreq(n_samples, d=1.0/sfreq)

# 逆FFT
time_data = fft.irfft(fft_data, n=n_time, axis=-1)

# 复数FFT
complex_fft = fft.fft(complex_data)
inverse = fft.ifft(complex_fft)

# 多维FFT
fft_2d = fft.fft2(image)
fft_nd = fft.fftn(volume)

# FFT移位（将零频率移到中心）
shifted = fft.fftshift(fft_data)
freqs_shifted = fft.fftshift(freqs)

# 常见模式：功率谱密度
fft_result = fft.rfft(data, axis=-1)
psd = np.abs(fft_result) ** 2 / n_samples
freqs = fft.rfftfreq(n_samples, 1.0 / sfreq)
```

#### Rust 等价
```rust
use rustfft::{FftPlanner, num_complex::Complex};
use ndarray::prelude::*;

// 实数FFT封装
fn rfft(data: &Array1<f64>) -> (Array1<Complex<f64>>, usize) {
    let n = data.len();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    
    // 转换为复数
    let mut buffer: Vec<Complex<f64>> = data.iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    
    fft.process(&mut buffer);
    
    // 只返回正频率部分
    let n_freq = n / 2 + 1;
    (Array1::from_vec(buffer[..n_freq].to_vec()), n)
}

fn irfft(fft_data: &Array1<Complex<f64>>, n_time: usize) -> Array1<f64> {
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_time);
    
    // 重建完整频谱（共轭对称）
    let mut buffer = vec![Complex::new(0.0, 0.0); n_time];
    for (i, &val) in fft_data.iter().enumerate() {
        buffer[i] = val;
        if i > 0 && i < n_time - i {
            buffer[n_time - i] = val.conj();
        }
    }
    
    ifft.process(&mut buffer);
    
    // 提取实部并归一化
    Array1::from_vec(buffer.iter()
        .map(|c| c.re / n_time as f64)
        .collect())
}

// 频率数组
fn rfftfreq(n: usize, d: f64) -> Array1<f64> {
    let n_freq = n / 2 + 1;
    Array1::from_vec((0..n_freq)
        .map(|i| i as f64 / (n as f64 * d))
        .collect())
}

// 使用示例
let fft_data = rfft(&time_data);
let freqs = rfftfreq(n_samples, 1.0 / sfreq);

// PSD
let psd: Array1<f64> = fft_data.0.mapv(|c| c.norm_sqr() / n_samples as f64);
```

**关键差异**:
1. `rustfft` 需要手动管理计划器和缓冲区
2. 实数FFT需要手动处理共轭对称
3. 需要自己实现 `rfft`/`irfft` 封装
4. 归一化需要手动处理
5. 性能优异，但API更底层

**更高级的封装**:
```rust
// 考虑创建自己的FFT模块
pub mod fft {
    use rustfft::FftPlanner;
    use ndarray::prelude::*;
    use num_complex::Complex;
    
    pub struct RealFFT {
        planner: FftPlanner<f64>,
    }
    
    impl RealFFT {
        pub fn new() -> Self {
            Self {
                planner: FftPlanner::new(),
            }
        }
        
        pub fn rfft(&mut self, data: &Array1<f64>) -> Array1<Complex<f64>> {
            // 实现...
        }
        
        pub fn irfft(&mut self, fft_data: &Array1<Complex<f64>>, n: usize) -> Array1<f64> {
            // 实现...
        }
    }
}
```

---

### 9. 随机数生成 (`np.random`)

#### Python/NumPy 模式
```python
import numpy as np

# 旧式（全局状态）
np.random.seed(42)
data = np.random.randn(100, 200)
uniform = np.random.rand(10)
integers = np.random.randint(0, 10, size=100)
choice = np.random.choice([1, 2, 3], size=10)

# 新式（推荐）
rng = np.random.default_rng(seed=42)
data = rng.standard_normal(size=(100, 200))
uniform = rng.random(size=10)
integers = rng.integers(0, 10, size=100)
choice = rng.choice([1, 2, 3], size=10)

# 特定分布
gamma = rng.gamma(shape=2.0, scale=1.0, size=100)
poisson = rng.poisson(lam=5.0, size=100)

# 打乱
rng.shuffle(array)  # 原地打乱
permuted = rng.permutation(array)

# 常见模式：添加噪声
noisy_data = clean_data + rng.normal(0, noise_level, size=clean_data.shape)
```

#### Rust 等价
```rust
use rand::prelude::*;
use rand_distr::{StandardNormal, Uniform, Gamma, Poisson};
use ndarray::prelude::*;
use ndarray_rand::RandomExt;

// 创建RNG
let mut rng = StdRng::seed_from_u64(42);

// 随机数组
let data: Array2<f64> = Array::random_using((100, 200), StandardNormal, &mut rng);
let uniform: Array1<f64> = Array::random_using(10, Uniform::new(0.0, 1.0), &mut rng);
let integers: Array1<i32> = Array::random_using(100, Uniform::new(0, 10), &mut rng);

// 选择（需要手动实现）
fn choice<T: Clone>(
    items: &[T], 
    size: usize, 
    rng: &mut impl Rng
) -> Vec<T> {
    let dist = Uniform::new(0, items.len());
    (0..size)
        .map(|_| items[dist.sample(rng)].clone())
        .collect()
}

// 特定分布
let gamma_dist = Gamma::new(2.0, 1.0).unwrap();
let gamma: Array1<f64> = Array::random_using(100, gamma_dist, &mut rng);

let poisson_dist = Poisson::new(5.0).unwrap();
let poisson: Array1<u32> = Array::random_using(100, poisson_dist, &mut rng);

// 打乱
use rand::seq::SliceRandom;
let mut vec = array.to_vec();
vec.shuffle(&mut rng);

// 添加噪声
let noise: Array2<f64> = Array::random_using(
    clean_data.dim(), 
    StandardNormal, 
    &mut rng
);
let noisy_data = clean_data + &noise * noise_level;
```

**关键差异**:
1. 使用 `rand` + `rand_distr` + `ndarray_rand`
2. RNG 需要显式传递（无全局状态）
3. 分布需要先创建再采样
4. 更好的类型安全和性能

---

### 10. 其他常用功能

#### 数据类型转换
```python
# NumPy
int_arr = float_arr.astype(np.int32)
float_arr = int_arr.astype(np.float64)

# Rust
let int_arr: Array1<i32> = float_arr.mapv(|x| x as i32);
let float_arr: Array1<f64> = int_arr.mapv(|x| x as f64);
```

#### NaN 处理
```python
# NumPy
is_nan = np.isnan(data)
no_nan = data[~np.isnan(data)]
filled = np.nan_to_num(data, nan=0.0)

# Rust
let is_nan = data.mapv(|x| x.is_nan());
let no_nan: Vec<f64> = data.iter()
    .filter(|x| !x.is_nan())
    .copied()
    .collect();
let filled = data.mapv(|x| if x.is_nan() { 0.0 } else { x });
```

#### 唯一值
```python
# NumPy
unique_vals = np.unique(data)
unique_counts = np.unique(data, return_counts=True)

# Rust
use std::collections::HashSet;
let unique: HashSet<_> = data.iter().copied().collect();
let unique_vec: Vec<_> = unique.into_iter().collect();

// 或使用 itertools
use itertools::Itertools;
let unique: Vec<_> = data.iter().unique().copied().collect();
```

---

## MNE 中 NumPy 的关键使用场景

### 场景 1: 数据存储和访问

**MNE 模式**:
```python
class BaseRaw:
    def __init__(self):
        self._data = None  # (n_channels, n_times)
    
    def _read_segment_file(self, data, idx, fi, start, stop):
        """读取数据段"""
        # 直接索引NumPy数组
        one = data[idx]
        mult = data[idx, start:stop]
```

**Rust 等价**:
```rust
pub struct Raw {
    data: Option<Array2<f64>>,  // (n_channels, n_times)
}

impl Raw {
    fn read_segment(&self, idx: usize, start: usize, stop: usize) -> ArrayView1<f64> {
        self.data.as_ref()
            .unwrap()
            .slice(s![idx, start..stop])
    }
}
```

### 场景 2: 预处理管道

**MNE 模式**:
```python
def preprocess_data(raw):
    # 1. 获取数据
    data = raw.get_data()  # NumPy array
    
    # 2. 去均值
    data -= np.mean(data, axis=-1, keepdims=True)
    
    # 3. 滤波
    from scipy.signal import sosfiltfilt
    data = sosfiltfilt(sos, data, axis=-1)
    
    # 4. 降采样
    data = data[::decim]
    
    return data
```

**Rust 等价**:
```rust
fn preprocess_data(raw: &Raw, sos: &SOS, decim: usize) -> Array2<f64> {
    // 1. 获取数据
    let mut data = raw.get_data().to_owned();
    
    // 2. 去均值
    let mean = data.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
    data = &data - &mean;
    
    // 3. 滤波
    let filtered = sosfiltfilt(&sos, &data);
    
    // 4. 降采样
    let decimated = filtered.slice_axis(Axis(1), Slice::from(..).step_by(decim));
    
    decimated.to_owned()
}
```

### 场景 3: Epoching

**MNE 模式**:
```python
def make_epochs(data, events, tmin, tmax, sfreq):
    n_times = int((tmax - tmin) * sfreq)
    n_epochs = len(events)
    n_channels = data.shape[0]
    
    epochs = np.zeros((n_epochs, n_channels, n_times))
    
    for i, event_sample in enumerate(events[:, 0]):
        start = event_sample + int(tmin * sfreq)
        stop = start + n_times
        epochs[i] = data[:, start:stop]
    
    return epochs
```

**Rust 等价**:
```rust
fn make_epochs(
    data: &Array2<f64>, 
    events: &Array2<i32>,
    tmin: f64,
    tmax: f64,
    sfreq: f64
) -> Array3<f64> {
    let n_times = ((tmax - tmin) * sfreq) as usize;
    let n_epochs = events.nrows();
    let n_channels = data.nrows();
    
    let mut epochs = Array3::zeros((n_epochs, n_channels, n_times));
    
    for (i, event_sample) in events.column(0).iter().enumerate() {
        let start = (*event_sample as f64 + tmin * sfreq) as usize;
        let stop = start + n_times;
        epochs.slice_mut(s![i, .., ..])
            .assign(&data.slice(s![.., start..stop]));
    }
    
    epochs
}
```

---

## Rust 实现建议

### 推荐 Crate 组合

```toml
[dependencies]
# 核心数组
ndarray = { version = "0.15", features = ["rayon"] }
ndarray-stats = "0.5"
ndarray-rand = "0.14"

# 线性代数
ndarray-linalg = { version = "0.16", features = ["openblas-static"] }
# 或
nalgebra = "0.32"

# FFT
rustfft = "6.0"

# 随机数
rand = "0.8"
rand_distr = "0.4"

# 数学
num-complex = "0.4"
num-traits = "0.2"

# 并行
rayon = "1.7"

# 工具
itertools = "0.11"
```

### 封装建议

创建自己的数组操作模块：

```rust
// src/core/array_ops.rs
pub mod array_ops {
    use ndarray::prelude::*;
    
    /// Z-score 标准化
    pub fn zscore(data: &Array2<f64>, axis: Axis, ddof: usize) -> Array2<f64> {
        let mean = data.mean_axis(axis).unwrap().insert_axis(axis);
        let std = data.std_axis(axis, ddof as f64).insert_axis(axis);
        (data - &mean) / &std
    }
    
    /// RMS (均方根)
    pub fn rms(data: &Array2<f64>, axis: Axis) -> Array1<f64> {
        data.map_axis(axis, |view| {
            (view.mapv(|x| x * x).mean().unwrap()).sqrt()
        })
    }
    
    // ... 更多常用操作
}
```

---

## 性能优化技巧

### 1. 避免不必要的复制
```rust
// ❌ 差
let result = data.clone() + &other;

// ✅ 好
let result = &data + &other;
```

### 2. 使用并行迭代器
```rust
use rayon::prelude::*;

// 并行处理每一行
let processed: Vec<_> = data.axis_iter(Axis(0))
    .into_par_iter()
    .map(|row| process_row(row))
    .collect();
```

### 3. 原地操作
```rust
// 原地修改避免分配
data.mapv_inplace(|x| x * 2.0);
Zip::from(&mut data).and(&other).for_each(|d, &o| *d += o);
```

### 4. 使用 View 避免复制
```rust
fn process(data: ArrayView2<f64>) { /* ... */ }

// 传递 view 而非 owned
process(array.view());
```

---

## 总结

### NumPy → Rust 迁移难度评估

| 功能 | 难度 | 主要挑战 |
|------|------|---------|
| 基础数组操作 | ⭐ | API 略有不同 |
| 索引切片 | ⭐⭐ | 语法差异 |
| 广播 | ⭐⭐ | 需要显式 |
| 数学函数 | ⭐ | 用 `mapv` |
| 聚合统计 | ⭐⭐ | 需要额外 crate |
| 线性代数 | ⭐⭐⭐ | 需要 LAPACK |
| FFT | ⭐⭐⭐⭐ | 需要手动封装 |
| 随机数 | ⭐⭐ | 不同的API |

### 关键要点

1. **`ndarray` 是 NumPy 的优秀替代**，但 API 有学习曲线
2. **借用检查器**需要适应，但带来安全性
3. **显式类型**增加代码量，但提高可靠性
4. **性能通常更好**，特别是并行场景
5. **生态成熟度**：数组操作成熟，部分高级功能需自实现

### 下一步

继续阅读：[02_SciPy依赖深度分析.md](02_SciPy依赖深度分析.md)
