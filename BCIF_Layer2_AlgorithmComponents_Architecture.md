# BCIF Layer 2: Algorithm Components 架构设计

> **Status**: ✅ 设计决策已完成
> **Version**: 0.1.0
> **Date**: 2026-02-02
> **Purpose**: Layer 2 算法组件层的详细架构与设计规范

---

## 1. Layer 2 在整体架构中的位置

```
┌────────────────────��────────────────────────────────────────────┐
│ Layer 3: Pipeline Orchestration (编排层) ★核心价值★              │
│   BatchPipeline, StreamPipeline, ProcessContext                 │
└─────────────────��───────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Algorithm Components (算法组件) ★ 当前设计层 ★          │
│                                                                 │
│   ┌─────────────────────────┐  ┌─────────────────────────┐     │
│   │ bcif-dsp (~500 lines)   │  │ bcif-algo (~400 lines)  │     │
│   ├─────────────────────────┤  ├─────────────────────────┤     │
│   │ 纯变换，无学习/拟合      │  │ 需要 fit/transform      │     │
│   │                         │  │                         │     │
│   │ • 滤波 (idsp)           │  │ • ICA (petal)           │     │
│   │ • 重采样 (rubato)       │  │ • PCA (faer)            │     │
│   │ • FFT/PSD (realfft)     │  │ • CSP (手写)            │     │
│   │ • 窗函数                │  │                         │     │
│   │ • 基线校正              │  │                         │     │
│   └─────────────────────────┘  └─────────────────────────┘     │
│                                                                 │
│   核心原则：【直接使用底层 crate，薄封装】                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Core Numerics (基础数值)                                │
│   ndarray, faer, num-complex, bcif-core                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 设计决策汇总

### 2.1 决策总览

| # | 决策项 | 决策结果 | 理由 |
|---|--------|----------|------|
| 1 | API 风格 | 混合风格 | 离线用函数，实时用对象 |
| 2 | 滤波器设计 | 离线 filtfilt + 实时单向 | 零相位 + 状态管理 |
| 3 | 重采样 | 只支持离线 | 实时场景采样率固定 |
| 4 | PSD 计算 | 只实现 Welch | 覆盖 95%+ 场景 |
| 5 | 窗函数 | Hann + Hamming + Blackman | 覆盖 99% 场景 |
| 6 | 基线校正 | 包含 | EEG 标准步骤 |
| 7 | bcif-algo 算法 | ICA + PCA + CSP | 覆盖 BCI 需求 |
| 8 | bcif-algo API | sklearn 风格 | fit/transform 模式 |
| 9 | 统一 Trait | 不定义 | 保持简单 |
| 10 | 模型序列化 | 使用 serde | BCI 场景必需 |

---

## 3. bcif-dsp 模块设计

### 3.1 模块结构

```
bcif-dsp/ (~500 行)
├── Cargo.toml
└── src/
    ├── lib.rs           // 模块导出
    ├── filter.rs        // 滤波器 (~130 行)
    ├── resample.rs      // 重采样 (~50 行)
    ├── spectral.rs      // FFT/PSD (~100 行)
    ├── window.rs        // 窗函数 (~30 行)
    └── baseline.rs      // 基线校正 (~20 行)
```

### 3.2 滤波器设计 (filter.rs)

#### 3.2.1 支持的滤波器类型

| 类型 | 函数名 | 用途 |
|------|--------|------|
| 带通 | `filter_bandpass` | 保留特定频段 (如 1-40 Hz) |
| 高通 | `filter_highpass` | 去除直流漂移 (如 >0.1 Hz) |
| 低通 | `filter_lowpass` | 抗混叠 (如 <100 Hz) |
| 陷波 | `filter_notch` | 去除工频干扰 (50/60 Hz) |
| 带阻 | `filter_bandstop` | 去除特定频段 |

#### 3.2.2 离线滤波 API

```rust
/// 带通滤波（零相位，使用 filtfilt）
pub fn filter_bandpass(
    data: &mut Array2<f64>,
    info: &SignalInfo,
    low: f64,
    high: f64,
    order: usize,  // 默认 4
) -> Result<(), Error>;

/// 高通滤波
pub fn filter_highpass(
    data: &mut Array2<f64>,
    info: &SignalInfo,
    freq: f64,
    order: usize,
) -> Result<(), Error>;

/// 低通滤波
pub fn filter_lowpass(
    data: &mut Array2<f64>,
    info: &SignalInfo,
    freq: f64,
    order: usize,
) -> Result<(), Error>;

/// 陷波滤波（去除工频）
pub fn filter_notch(
    data: &mut Array2<f64>,
    info: &SignalInfo,
    freq: f64,       // 50.0 或 60.0
    width: f64,      // 陷波宽度，默认 2.0 Hz
) -> Result<(), Error>;
```

#### 3.2.3 实时滤波 API

```rust
/// 实时带通滤波器（有状态）
pub struct StreamBandpassFilter {
    biquads: Vec<Biquad<f64>>,
    zi: Array2<f64>,  // 滤波器状态 (n_sections * 2, n_channels)
}

impl StreamBandpassFilter {
    pub fn new(low: f64, high: f64, sample_rate: f64, order: usize) -> Self;

    /// 处理一个 chunk（更新内部状态）
    pub fn process(&mut self, chunk: &mut Array2<f64>) -> Result<(), Error>;

    /// 重置滤波器状态
    pub fn reset(&mut self);
}

impl StreamProcessor for StreamBandpassFilter {
    fn process_chunk(&mut self, chunk: &mut Array2<f64>, ctx: &mut ProcessContext) -> Result<(), Error>;
    fn latency_samples(&self) -> usize;
    fn reset(&mut self);
    fn name(&self) -> &str;
}
```

#### 3.2.4 内部实现

```rust
// 内部模块：共享核心实现
mod internal {
    use idsp::iir::Biquad;

    /// 设计 Butterworth 滤波器
    pub fn design_butterworth(
        filter_type: FilterType,
        freq: f64,       // 或 (low, high) for bandpass
        sample_rate: f64,
        order: usize,
    ) -> Vec<Biquad<f64>>;

    /// 单向滤波（sosfilt）
    pub fn sosfilt(
        data: &mut [f64],
        biquads: &[Biquad<f64>],
        zi: Option<&mut [f64]>,
    );

    /// 零相位滤波（filtfilt）
    pub fn filtfilt(
        data: &mut [f64],
        biquads: &[Biquad<f64>],
    );
}
```

### 3.3 重采样设计 (resample.rs)

```rust
use rubato::{SincFixedIn, InterpolationParameters, WindowFunction};

/// 重采样（只支持离线）
pub fn resample(
    data: &Array2<f64>,
    info: &SignalInfo,
    new_sample_rate: f64,
) -> Result<(Array2<f64>, SignalInfo), Error> {
    // 使用 rubato::SincFixedIn
    // 返回新数据和更新后的 SignalInfo
}
```

### 3.4 频谱分析设计 (spectral.rs)

```rust
use realfft::RealFftPlanner;

/// Welch 方法计算功率谱密度
pub fn psd_welch(
    data: &Array2<f64>,
    info: &SignalInfo,
    n_fft: usize,           // FFT 点数，默认 256
    n_overlap: Option<usize>, // 重叠样本数，默认 n_fft / 2
    window: WindowType,     // 窗函数类型，默认 Hann
) -> Result<(Array2<f64>, Array1<f64>), Error>;
// 返回：(psd, freqs)
// psd shape: (n_channels, n_freqs)
// freqs shape: (n_freqs,)

/// 计算特定频段的功率
pub fn band_power(
    psd: &Array2<f64>,
    freqs: &Array1<f64>,
    fmin: f64,
    fmax: f64,
) -> Array1<f64>;
// 返回每个通道在 [fmin, fmax] 频段的功率
```

### 3.5 窗函数设计 (window.rs)

```rust
/// 窗函数类型
#[derive(Clone, Copy, Debug, Default)]
pub enum WindowType {
    #[default]
    Hann,
    Hamming,
    Blackman,
}

/// 生成窗函数
pub fn window(window_type: WindowType, length: usize) -> Array1<f64> {
    match window_type {
        WindowType::Hann => hann(length),
        WindowType::Hamming => hamming(length),
        WindowType::Blackman => blackman(length),
    }
}

fn hann(n: usize) -> Array1<f64> {
    // w[i] = 0.5 * (1 - cos(2π * i / (n-1)))
}

fn hamming(n: usize) -> Array1<f64> {
    // w[i] = 0.54 - 0.46 * cos(2π * i / (n-1))
}

fn blackman(n: usize) -> Array1<f64> {
    // w[i] = 0.42 - 0.5 * cos(2π * i / (n-1)) + 0.08 * cos(4π * i / (n-1))
}
```

### 3.6 基线校正设计 (baseline.rs)

```rust
/// 基线校正（用于 Epochs 数据）
pub fn baseline_correction(
    data: &mut Array3<f64>,  // (n_epochs, n_channels, n_times)
    info: &SignalInfo,
    baseline: (f64, f64),    // 基线时间范围 (秒)，如 (-0.2, 0.0)
    tmin: f64,               // epoch 起始时间
) -> Result<(), Error> {
    // 1. 计算基线时间段对应的样本索引
    // 2. 计算每个 epoch 每个通道的基线均值
    // 3. 减去基线均值
}
```

---

## 4. bcif-algo 模块设计

### 4.1 模块结构

```
bcif-algo/ (~400 行)
├── Cargo.toml
└── src/
    ├── lib.rs           // 模块导出
    ├── ica.rs           // ICA (~80 行)
    ├── pca.rs           // PCA (~100 行)
    └── csp.rs           // CSP (~180 行)
```

### 4.2 ICA 设计 (ica.rs)

```rust
use petal_decomposition::FastIca;
use serde::{Serialize, Deserialize};

/// 独立成分分析
#[derive(Clone, Serialize, Deserialize)]
pub struct ICA {
    n_components: usize,
    unmixing_matrix: Option<Array2<f64>>,
    mixing_matrix: Option<Array2<f64>>,
    mean: Option<Array1<f64>>,
}

impl ICA {
    /// 创建 ICA 实例
    pub fn new(n_components: usize) -> Self;

    /// 从数据中学习 unmixing matrix
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<(), Error>;

    /// 将数据变换为独立成分
    pub fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, Error>;

    /// fit + transform
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Result<Array2<f64>, Error>;

    /// 从独立成分重建原始数据
    pub fn inverse_transform(&self, sources: &Array2<f64>) -> Result<Array2<f64>, Error>;

    /// 获取 unmixing matrix
    pub fn unmixing_matrix(&self) -> Option<&Array2<f64>>;

    /// 获取 mixing matrix
    pub fn mixing_matrix(&self) -> Option<&Array2<f64>>;
}
```

### 4.3 PCA 设计 (pca.rs)

```rust
use faer::Svd;
use serde::{Serialize, Deserialize};

/// 主成分分析
#[derive(Clone, Serialize, Deserialize)]
pub struct PCA {
    n_components: usize,
    components: Option<Array2<f64>>,      // 主成分 (n_components, n_features)
    explained_variance: Option<Array1<f64>>,
    mean: Option<Array1<f64>>,
}

impl PCA {
    /// 创建 PCA 实例
    pub fn new(n_components: usize) -> Self;

    /// 从数据中学习主成分
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<(), Error>;

    /// 将数据投影到主成分空间
    pub fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, Error>;

    /// fit + transform
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Result<Array2<f64>, Error>;

    /// 从主成分空间重建数据
    pub fn inverse_transform(&self, transformed: &Array2<f64>) -> Result<Array2<f64>, Error>;

    /// 获取解释方差比例
    pub fn explained_variance_ratio(&self) -> Option<Array1<f64>>;
}
```

### 4.4 CSP 设计 (csp.rs)

```rust
use serde::{Serialize, Deserialize};

/// 共空间模式 (Common Spatial Pattern)
/// 用于 BCI 中的运动想象分类
#[derive(Clone, Serialize, Deserialize)]
pub struct CSP {
    n_components: usize,
    filters: Option<Array2<f64>>,   // 空间滤波器 (n_components, n_channels)
    patterns: Option<Array2<f64>>,  // 空间模式 (用于可视化)
}

impl CSP {
    /// 创建 CSP 实例
    /// n_components: 每类取的成分数（总共 2 * n_components）
    pub fn new(n_components: usize) -> Self;

    /// 从两类数据中学习空间滤波器
    /// data_class1: (n_epochs, n_channels, n_times)
    /// data_class2: (n_epochs, n_channels, n_times)
    pub fn fit(
        &mut self,
        data_class1: &Array3<f64>,
        data_class2: &Array3<f64>,
    ) -> Result<(), Error>;

    /// 提取 CSP 特征
    /// data: (n_epochs, n_channels, n_times)
    /// 返回: (n_epochs, 2 * n_components) - 每个成分的对数方差
    pub fn transform(&self, data: &Array3<f64>) -> Result<Array2<f64>, Error>;

    /// fit + transform
    pub fn fit_transform(
        &mut self,
        data_class1: &Array3<f64>,
        data_class2: &Array3<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), Error>;

    /// 获取空间滤波器
    pub fn filters(&self) -> Option<&Array2<f64>>;

    /// 获取空间模式（用于可视化）
    pub fn patterns(&self) -> Option<&Array2<f64>>;
}
```

### 4.5 CSP 算法实现要点

```rust
// CSP 核心算法（手写实现）
impl CSP {
    fn fit_internal(
        &mut self,
        data_class1: &Array3<f64>,
        data_class2: &Array3<f64>,
    ) -> Result<(), Error> {
        // 1. 计算每类的平均协方差矩阵
        let cov1 = compute_average_covariance(data_class1);
        let cov2 = compute_average_covariance(data_class2);

        // 2. 计算复合协方差矩阵
        let cov_total = &cov1 + &cov2;

        // 3. 白化变换
        // 使用 faer 进行特征分解
        let whitening = compute_whitening(&cov_total)?;

        // 4. 对白化后的 cov1 进行特征分解
        let cov1_white = whitening.dot(&cov1).dot(&whitening.t());
        let (eigenvalues, eigenvectors) = eigendecomposition(&cov1_white)?;

        // 5. 选择最大和最小特征值对应的特征向量
        let filters = select_components(&eigenvectors, &eigenvalues, self.n_components);

        // 6. 计算最终的空间滤波器
        self.filters = Some(filters.dot(&whitening));

        Ok(())
    }
}

/// 计算平均协方差矩阵
fn compute_average_covariance(data: &Array3<f64>) -> Array2<f64> {
    // data: (n_epochs, n_channels, n_times)
    // 对每个 epoch 计算协方差，然后平均
}

/// 计算白化矩阵
fn compute_whitening(cov: &Array2<f64>) -> Result<Array2<f64>, Error> {
    // 使用 faer 进行特征分解
    // 返回 D^(-1/2) * V^T
}
```

---

## 5. 依赖关系

### 5.1 Cargo.toml

```toml
# bcif-dsp/Cargo.toml
[package]
name = "bcif-dsp"
version = "0.1.0"
edition = "2021"

[dependencies]
bcif-core = { path = "../bcif-core" }
ndarray = "0.15"
idsp = "0.15"
rubato = "0.14"
realfft = "3.3"
num-complex = "0.4"

# bcif-algo/Cargo.toml
[package]
name = "bcif-algo"
version = "0.1.0"
edition = "2021"

[dependencies]
bcif-core = { path = "../bcif-core" }
ndarray = "0.15"
faer = "0.19"
petal-decomposition = "0.7"
serde = { version = "1.0", features = ["derive"] }
```

### 5.2 Crate 使用映射

| 功能 | Crate | 使用方式 | 手写量 |
|------|-------|----------|--------|
| IIR 滤波 | idsp | Biquad 滤波器 | ~80 行胶水 |
| 重采样 | rubato | SincFixedIn | ~50 行胶水 |
| FFT | realfft | RealFftPlanner | ~60 行胶水 |
| ICA | petal | FastIca | ~50 行胶水 |
| PCA/CSP | faer | SVD/特征分解 | ~280 行 |
| 序列化 | serde | derive 宏 | 0 |

---

## 6. 使用示例

### 6.1 离线处理示例

```rust
use bcif_io::read_edf;
use bcif_dsp::{filter_bandpass, filter_notch, psd_welch, WindowType};
use bcif_algo::ICA;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 读取数据
    let (mut data, info) = read_edf("subject01.edf")?;

    // 2. 预处理
    filter_bandpass(&mut data, &info, 1.0, 40.0, 4)?;
    filter_notch(&mut data, &info, 50.0, 2.0)?;

    // 3. ICA 去伪影
    let mut ica = ICA::new(info.n_channels());
    let sources = ica.fit_transform(&data)?;

    // 4. 手动标记并去除伪影成分（假设成分 0 是眨眼）
    let mut sources_cleaned = sources.clone();
    sources_cleaned.row_mut(0).fill(0.0);
    let data_cleaned = ica.inverse_transform(&sources_cleaned)?;

    // 5. 计算功率谱
    let (psd, freqs) = psd_welch(&data_cleaned, &info, 256, None, WindowType::Hann)?;

    println!("PSD shape: {:?}", psd.dim());
    println!("Frequency range: {:.1} - {:.1} Hz", freqs[0], freqs[freqs.len()-1]);

    Ok(())
}
```

### 6.2 BCI 训练示例

```rust
use bcif_dsp::{filter_bandpass, baseline_correction};
use bcif_algo::CSP;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 加载两类运动想象数据
    let (epochs_left, info) = load_epochs("left_hand.npy")?;   // (n_epochs, n_ch, n_times)
    let (epochs_right, _) = load_epochs("right_hand.npy")?;

    // 2. 预处理：8-30 Hz 带通滤波（运动想象频段）
    let mut epochs_left = epochs_left;
    let mut epochs_right = epochs_right;
    // ... 对每个 epoch 滤波

    // 3. 训练 CSP
    let mut csp = CSP::new(3);  // 每类取 3 个成分
    csp.fit(&epochs_left, &epochs_right)?;

    // 4. 提取特征
    let features_left = csp.transform(&epochs_left)?;   // (n_epochs, 6)
    let features_right = csp.transform(&epochs_right)?;

    // 5. 保存模型
    let model_json = serde_json::to_string(&csp)?;
    std::fs::write("csp_model.json", model_json)?;

    println!("CSP 训练完成，特征维度: {:?}", features_left.dim());

    Ok(())
}
```

### 6.3 实时处理示例

```rust
use bcif_dsp::StreamBandpassFilter;
use bcif_pipeline::StreamProcessor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 创建实时滤波器
    let mut filter = StreamBandpassFilter::new(8.0, 30.0, 256.0, 4);

    // 2. 加载预训练的 CSP 模型
    let csp_json = std::fs::read_to_string("csp_model.json")?;
    let csp: CSP = serde_json::from_str(&csp_json)?;

    // 3. 实时处理循环
    loop {
        // 获取新数据 chunk
        let mut chunk = get_new_chunk();  // (n_channels, chunk_size)

        // 实时滤波
        filter.process(&mut chunk)?;

        // 提取特征（需要累积足够数据）
        // ...
    }
}
```

---

## 7. 与其他层的接口

### 7.1 Layer 1 → Layer 2

```rust
// 输入数据格式（来自 Layer 1 规范）
// 数据: Array2<f64>, shape = (n_channels, n_times), C-order
// 元数据: SignalInfo { sample_rate, channels }
// 关联方式: 分离传递
```

### 7.2 Layer 2 → Layer 3

```rust
// bcif-dsp 函数可直接在 Pipeline 中使用
// bcif-algo 算法可封装为 BatchProcessor

impl BatchProcessor for IcaProcessor {
    fn process(&mut self, data: &mut Array2<f64>, ctx: &mut ProcessContext) -> Result<(), Error> {
        // 使用 ICA 处理数据
    }

    fn name(&self) -> &str { "ICA" }
}
```

---

## 8. 代码量估算

| 模块 | 核心代码 | 胶水代码 | 总计 |
|------|----------|----------|------|
| bcif-dsp/filter.rs | ~50 行 | ~80 行 | ~130 行 |
| bcif-dsp/resample.rs | 0 | ~50 行 | ~50 行 |
| bcif-dsp/spectral.rs | ~60 行 | ~40 行 | ~100 行 |
| bcif-dsp/window.rs | ~30 行 | 0 | ~30 行 |
| bcif-dsp/baseline.rs | ~20 行 | 0 | ~20 行 |
| bcif-dsp/lib.rs | - | ~70 行 | ~70 行 |
| **bcif-dsp 总计** | **~160 行** | **~240 行** | **~400 行** |
| bcif-algo/ica.rs | ~30 行 | ~50 行 | ~80 行 |
| bcif-algo/pca.rs | ~60 行 | ~40 行 | ~100 行 |
| bcif-algo/csp.rs | ~150 行 | ~30 行 | ~180 行 |
| bcif-algo/lib.rs | - | ~40 行 | ~40 行 |
| **bcif-algo 总计** | **~240 行** | **~160 行** | **~400 行** |
| **Layer 2 总计** | **~400 行** | **~400 行** | **~800 行** |

---

## 9. 下一步

Layer 2 设计决策已全部完成，下一步可以：

1. 设计 Layer 3 (bcif-pipeline) - BatchPipeline、StreamPipeline 的详细设计
2. 开始实现 - 按 Phase 顺序实现各模块
3. 编写测试 - 单元测试和集成测试

---

## 10. 参考资料

| 资源 | 位置 |
|------|------|
| idsp 文档 | crates.io/crates/idsp |
| rubato 文档 | crates.io/crates/rubato |
| realfft 文档 | crates.io/crates/realfft |
