# BCIF 框架架构设计规划

> **Status**: 架构设计阶段
> **Version**: 1.0.0
> **Date**: 2026-02-01
> **Purpose**: BCIF 框架的可执行架构方案

---

## 1. 核心定位

**BCIF 不是 MNE 的 Rust 克隆，而是一个集成层**

```
BCIF 的价值 = 领域知识的编码 + 成熟 crates 的组合
```

用户不需要理解 idsp 的 Biquad 系数如何计算，只需要说"我要 1-40Hz 带通滤波"。

### 1.1 核心原则

1. **组合优于重写** - 最大化利用现有成熟 Rust crates
2. **最小化手写** - 只在 Gap 处编写代码
3. **现代化设计** - Rust-native API，非 MNE 克隆
4. **纯 Rust 栈** - 避免 C 库依赖（选择 faer 而非 OpenBLAS）

---

## 2. 分层架构

去掉 "Layer 2.5"，明确 BCIF 的核心价值在 Layer 3：

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: Language Bindings (语言绑定)                            │
│   PyO3, WASM, CLI, C FFI                                        │
│   【第二阶段，MVP 不包含】                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Pipeline Orchestration (编排层) ★ BCIF 核心价值 ★       │
│                                                                 │
│   ┌─────────────────┐  ┌─────────────────┐                     │
│   │ BatchPipeline   │  │ StreamPipeline  │                     │
│   │ (离线处理)       │  │ (实时处理)       │                     │
│   └─────────────────┘  └─────────────────┘                     │
│                                                                 │
│   统一的 Processor trait，不同的执行模式                          │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Algorithm Components (算法组件)                         │
│                                                                 │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      │
│   │ idsp   │ │ rubato │ │realfft │ │ petal  │ │ statrs │      │
│   │ 滤波   │ │ 重采样 │ │ FFT    │ │ ICA    │ │ 统计   │      │
│   └────────┘ └────────┘ └────────┘ └────────┘ └────────┘      │
│                                                                 │
│   【直接使用，薄封装】                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Core Numerics (基础数值)                                │
│                                                                 │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│   │ ndarray      │ │ faer         │ │ num-complex  │           │
│   │ 多维数组     │ │ 线性代数     │ │ 复数运算     │           │
│   └──────────────┘ └──────────────┘ └──────────────┘           │
│                                                                 │
│   【直接使用，不封装】                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 0: Data I/O (数据进出)                                     │
│                                                                 │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                  │
│   │ edflib │ │ xdf    │ │ lsl    │ │ ADC→μV │                  │
│   │ EDF    │ │ XDF    │ │ 实时流 │ │ 转换   │                  │
│   └────────┘ └────────┘ └────────┘ └────────┘                  │
│                                                                 │
│   【薄封装，返回 ndarray】                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 从 MNE/mne-lsl 借鉴的设计模式

### 3.1 借鉴：Trait 组合（来自 MNE 的 Mixin 模式）

MNE 使用 12 个 Mixin 组合功能，但过于复杂。BCIF 简化为 3-5 个核心 trait：

```rust
// 核心 trait（限制���量）
// Core traits (limited number)
pub trait HasMetadata {
    fn info(&self) -> &SignalInfo;
    fn info_mut(&mut self) -> &mut SignalInfo;
}

pub trait Filterable: HasMetadata {
    fn filter(&mut self, l_freq: f64, h_freq: f64) -> Result<(), Error>;
    fn notch_filter(&mut self, freq: f64) -> Result<(), Error>;
}

pub trait Referenceable: HasMetadata {
    fn set_reference(&mut self, ref_type: ReferenceType) -> Result<(), Error>;
}

pub trait ChannelSelectable: HasMetadata {
    fn pick(&mut self, picks: &[usize]) -> Result<(), Error>;
    fn drop_channels(&mut self, names: &[&str]) -> Result<(), Error>;
}
```

### 3.2 借鉴：共享元数据结构（来自 MNE 的 Info）

MNE 的 Info 有 50+ 字段，过于复杂。BCIF 最小化设计：

```rust
/// Signal metadata (minimal design)
/// 信号元数据（最小化设计）
pub struct SignalInfo {
    /// Sampling rate (Hz)
    /// 采样率 (Hz)
    pub sample_rate: f64,

    /// Channel information
    /// 通道信息
    pub channels: Vec<ChannelInfo>,
}

pub struct ChannelInfo {
    pub name: String,
    pub ch_type: ChannelType,
    pub unit: Unit,  // μV, mV, etc.
}

#[derive(Clone, Copy)]
pub enum ChannelType {
    Eeg, Eog, Emg, Ecg, Stim, Misc,
}
```

### 3.3 借鉴：离线/在线统一接口（来自 mne-lsl）

mne-lsl 的关键洞察：离线和在线处理共享相同的 Info 对象和操作接口。

**区别在于执行模式，而非 API：**

| 维度 | 离线 (Batch) | 在线 (Stream) |
|------|-------------|---------------|
| 数据来源 | 文件 | LSL 流 / 环形缓冲区 |
| 滤波模式 | 非因果 (filtfilt) | 因果 (有状态) |
| 处理单位 | 整个数组 | Chunk |
| 状态管理 | 无 | 滤波器状态跨 chunk 保持 |

### 3.4 借鉴：环形缓冲区 + 后台线程（来自 mne-lsl）

mne-lsl 的实时架构：

```
LSL 网络流 → StreamInlet → 后台线程 (_acquire) → 环形缓冲区 → 用户 get_data()
```

BCIF 等效设计：

```rust
pub struct StreamPipeline {
    /// Ring buffer
    /// 环形缓冲区
    buffer: RingBuffer<f64>,

    /// Processor chain with state
    /// 处理器链（带状态）
    processors: Vec<Box<dyn StreamProcessor>>,

    /// Background acquisition thread
    /// 后台采集线程
    acquisition_handle: Option<JoinHandle<()>>,

    /// New sample counter
    /// 新样本计数
    n_new_samples: AtomicUsize,
}
```

---

## 4. 不照搬 MNE 的地方

### 4.1 数据结构：函数式 > 面向对象

**MNE 风格（不采用）：**
```python
raw = Raw(data, info)
raw.filter(1, 40)
raw.resample(256)
epochs = raw.epoch(events, tmin, tmax)
```

**BCIF 风格（采用）��**
```rust
// Data and metadata separated, functional processing
// 数据和元数据分离，函数式处理
let (mut data, mut info) = read_edf("file.edf")?;

// Use Pipeline for orchestration
// 使用 Pipeline 编排
let mut pipeline = BatchPipeline::new(&mut info)
    .bandpass(1.0, 40.0)
    .resample(256.0)
    .build();

pipeline.process(&mut data)?;

// Or call functions directly
// 或者直接调用函数
filter_bandpass(&mut data, 1.0, 40.0, info.sample_rate);
```

### 4.2 不使用 Raw/Epochs 包装类

**MNE 的问题：** Raw 类有 50+ 方法，继承 12 个 Mixin，过于复杂。

**BCIF 的选择：** 直接使用 ndarray，不包装。

```rust
// Continuous data: directly Array2
// 连续数据：直接是 Array2
type ContinuousData = Array2<f64>;  // (n_channels, n_times)

// Epoched data: directly Array3
// 分段数据：直接是 Array3
type EpochsData = Array3<f64>;  // (n_epochs, n_channels, n_times)

// Metadata: independent struct
// 元数据：独立结构
struct SignalInfo { ... }
```

### 4.3 不使用延迟投影模式

MNE 有 `_do_delayed_proj` 标志，导致复杂的分支逻辑。BCIF 采用即时处理。

### 4.4 不使用 dict-like 元数据

MNE 的 Info 继承自 ValidatedDict，混合了字典接口和验证逻辑。BCIF 使用纯 struct。

---

## 5. 模块边界设计

### 5.1 划分原则

| 模块 | 职责 | 划分原则 |
|------|------|----------|
| bcif-core | 核心类型、错误、元数据 | 无外部依赖（除 ndarray） |
| bcif-io | 文件读写、LSL 流 | 数据进出系统边界 |
| bcif-dsp | 信号变换 | **纯变换，无学习/拟合** |
| bcif-algo | 统计/ML 算法 | **需要 fit/transform 的算法** |
| bcif-pipeline | 编排层 | 组合 dsp/algo，提供高层 API |

### 5.2 bcif-dsp vs bcif-algo 的边界

**bcif-dsp（信号变换）：**
- 滤波（bandpass, notch, highpass, lowpass）
- FFT / IFFT / PSD
- 重采样
- 窗函数
- 基线校正

**bcif-algo（学习算法）：**
- ICA（需要 fit）
- PCA（需要 fit）
- CSP（需要 fit，用于 BCI）
- CCA（需要 fit，用于 SSVEP）
- LDA（需要 fit，分类器）

**判断标准：** 是否需要从数据中"学习"参数？

---

## 6. 实时处理设计（参考 mne-lsl）

### 6.1 核心抽象

```rust
/// Batch processor (offline)
/// 批处理器（离线）
pub trait BatchProcessor: Send + Sync {
    fn process(&mut self, data: &mut Array2<f64>, ctx: &mut ProcessContext) -> Result<(), Error>;
    fn name(&self) -> &str;
}

/// Stream processor (real-time)
/// 流处理器（实时）
pub trait StreamProcessor: Send + Sync {
    /// Process one chunk
    /// 处理一个 chunk
    fn process_chunk(&mut self, chunk: &mut Array2<f64>, ctx: &mut ProcessContext) -> Result<(), Error>;

    /// Declare latency in samples
    /// 声明延迟（样本数）
    fn latency_samples(&self) -> usize;

    /// Reset state
    /// 重置状态
    fn reset(&mut self);

    fn name(&self) -> &str;
}

/// Processing context (metadata passing)
/// 处理上下文（元数据传递）
pub struct ProcessContext {
    pub sample_rate: f64,
    pub channel_names: Vec<String>,
    pub channel_types: Vec<ChannelType>,
    // Extensible...
    // 可扩展...
}
```

### 6.2 滤波器状态管理

参考 mne-lsl 的设计，实时滤波需要跨 chunk 保持状态：

```rust
pub struct StreamFilter {
    /// SOS coefficients
    /// SOS 系数
    sos: Vec<[f64; 6]>,

    /// Filter state per channel
    /// 滤波器状态（每个通道）
    zi: Array2<f64>,  // (n_sections * 2, n_channels)

    /// Channels to apply
    /// 应用的通道
    picks: Vec<usize>,
}

impl StreamProcessor for StreamFilter {
    fn process_chunk(&mut self, chunk: &mut Array2<f64>, ctx: &mut ProcessContext) -> Result<(), Error> {
        // Use sosfilt and update zi state
        // 使用 sosfilt 并更新 zi 状态
        // ...
    }

    fn reset(&mut self) {
        // Reset zi to initial conditions
        // 重置 zi 为初始条件
    }
}
```

### 6.3 环形缓冲区

```rust
pub struct RingBuffer<T> {
    data: Array2<T>,      // (buffer_size, n_channels)
    timestamps: Vec<f64>, // Timestamps / 时间戳
    write_pos: usize,     // Write position / 写入位置
    n_samples: usize,     // Current sample count / 当前样本数
}

impl<T: Clone> RingBuffer<T> {
    pub fn push(&mut self, chunk: ArrayView2<T>, timestamps: &[f64]);
    pub fn get_last(&self, n_samples: usize) -> ArrayView2<T>;
    pub fn clear(&mut self);
}
```

### 6.4 StreamPipeline 设计

```rust
pub struct StreamPipeline {
    buffer: RingBuffer<f64>,
    processors: Vec<Box<dyn StreamProcessor>>,
    ctx: ProcessContext,

    // Background acquisition (optional)
    // 后台采集（可选）
    inlet: Option<LslInlet>,
    acquisition_thread: Option<JoinHandle<()>>,
    n_new_samples: Arc<AtomicUsize>,
}

impl StreamPipeline {
    /// Create from LSL stream
    /// 从 LSL 流创建
    pub fn from_lsl(name: &str, bufsize_sec: f64) -> Result<Self, Error>;

    /// Manual data push
    /// 手动推送数据
    pub fn push(&mut self, chunk: ArrayView2<f64>, timestamps: &[f64]) -> Result<(), Error>;

    /// Get latest data
    /// 获取最新数据
    pub fn get_data(&self, winsize_sec: Option<f64>) -> (Array2<f64>, Vec<f64>);

    /// Number of new samples
    /// 新样本数
    pub fn n_new_samples(&self) -> usize;
}
```

---

## 7. Workspace 结构

```
bcif/
├── Cargo.toml                    # Workspace configuration / Workspace 配置
├── bcif-core/                    # Core types (~500 lines) / 核心类型 (~500行)
│   ├── src/lib.rs
│   ├── src/info.rs               # SignalInfo, ChannelInfo
│   ├── src/error.rs              # Error types (thiserror) / 错误类型
│   └── src/types.rs              # ChannelType, Unit, ReferenceType
├── bcif-io/                      # Data I/O (~800 lines) / 数据 I/O (~800行)
│   ├── src/lib.rs
│   ├── src/edf.rs                # EDF parsing / EDF 解析
│   ├── src/xdf.rs                # XDF parsing / XDF 解析
│   └── src/lsl.rs                # LSL stream wrapper / LSL 流封装
├── bcif-dsp/                     # Signal transforms (~600 lines) / 信号变换 (~600行)
│   ├── src/lib.rs
│   ├── src/filter.rs             # Filtering (wraps idsp) / 滤波 (封装 idsp)
│   ├── src/resample.rs           # Resampling (wraps rubato) / 重采样 (封装 rubato)
│   ├── src/spectral.rs           # FFT/PSD (wraps realfft) / FFT/PSD (封装 realfft)
│   └── src/window.rs             # Window functions / 窗函数
├── bcif-algo/                    # Learning algorithms (~400 lines) / 学习算法 (~400行)
│   ├── src/lib.rs
│   ├── src/ica.rs                # ICA (wraps petal) / ICA (封装 petal)
│   ├── src/pca.rs                # PCA (uses faer) / PCA (使用 faer)
│   └── src/csp.rs                # CSP (hand-written, BCI core) / CSP (手写，BCI 核心)
├── bcif-pipeline/                # Orchestration layer (~1000 lines) ★ CORE ★
│   │                             # 编排层 (~1000行) ★ 核心 ★
│   ├── src/lib.rs
│   ├── src/context.rs            # ProcessContext
│   ├── src/batch/
│   │   ├── mod.rs
│   │   ├── processor.rs          # BatchProcessor trait
│   │   └── builder.rs            # BatchPipeline builder
│   └── src/stream/
│       ├── mod.rs
│       ├── processor.rs          # StreamProcessor trait
│       ├── buffer.rs             # RingBuffer
│       └── pipeline.rs           # StreamPipeline
└── examples/
    ├── batch_processing.rs       # Offline processing example / 离线处理示例
    └── realtime_processing.rs    # Real-time processing example / 实时处��示例
```

**Estimated code:** ~3300 lines (increased from original estimate due to real-time processing)

**代码量估算：** ~3300 行（比原估计增加，因为加入了实时处理）

---

## 8. API 设计示例

### 8.1 离线处理（BatchPipeline）

```rust
use bcif_io::read_edf;
use bcif_pipeline::BatchPipeline;
use bcif_dsp::psd_welch;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Read data / 读取数据
    let (mut data, info) = read_edf("subject01.edf")?;

    // 2. Build processing pipeline / 构建处理流水线
    let mut pipeline = BatchPipeline::new(info.sample_rate)
        .bandpass(1.0, 40.0)
        .notch(50.0)
        .resample(256.0)
        .build();

    // 3. Process / 处理
    let mut ctx = ProcessContext::from(&info);
    pipeline.process(&mut data, &mut ctx)?;

    // 4. Feature extraction / 特征提取
    let psd = psd_welch(&data, ctx.sample_rate, 256)?;

    Ok(())
}
```

### 8.2 实时处理（StreamPipeline）

```rust
use bcif_pipeline::StreamPipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Connect to LSL stream / 连接 LSL 流
    let mut stream = StreamPipeline::from_lsl("EEG", 2.0)?  // 2-second buffer / 2秒缓冲
        .bandpass(1.0, 40.0)
        .notch(50.0)
        .build();

    stream.connect()?;

    // 2. Real-time processing loop / 实时处理循环
    loop {
        if stream.n_new_samples() > 0 {
            let (data, timestamps) = stream.get_data(Some(1.0));  // Last 1 second / 最近1秒

            // Process data... / 处理数据...
            let alpha_power = compute_alpha_power(&data);
            println!("Alpha power: {:.2}", alpha_power);
        }

        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}
```

---

## 9. Crate 映射表

### 9.1 直接使用 vs 需要手写

| 功能 | Python 库 | Rust Crate | 状态 | 手写量 |
|------|----------|------------|------|--------|
| 数组容器 | numpy.ndarray | **ndarray** | ✅ 直接用 | 0 |
| 线性代数 | scipy.linalg | **faer** | ✅ 直接用 | 0 |
| 实数 FFT | scipy.fft.rfft | **realfft** | ✅ 直接用 | 0 |
| 复数 FFT | scipy.fft.fft | **rustfft** | ✅ 直接用 | 0 |
| IIR 滤波 | scipy.signal.butter | **idsp** | ✅ 直接用 | ~100行胶水 |
| 重采样 | scipy.signal.resample | **rubato** | ✅ 直接用 | ~50行胶水 |
| ICA | sklearn.FastICA | **petal-decomposition** | ✅ 直接用 | ~50行胶水 |
| PCA | sklearn.PCA | **faer** (直接 SVD) | ✅ 直接用 | ~80行 |
| 稀疏矩阵 | scipy.sparse | **sprs** | ✅ 直接用 | 0 |
| 优化 | scipy.optimize | **argmin** | ✅ 直接用 | 0 |
| 统计 | scipy.stats | **statrs** | ✅ 直接用 | 0 |
| EDF 解析 | mne.io.read_raw_edf | **edflib** | ✅ 有 crate | ~100行胶水 |
| XDF 解析 | pyxdf | **xdf** | ✅ 有 crate | ~50行胶水 |
| LSL 流 | pylsl | **lsl** | ✅ 官方绑定 | ~50行胶水 |

### 9.2 不需要手写的（直接调用）

| 功能 | 直接用 | 不要手写 |
|------|-------|---------|
| FFT/PSD | `realfft::RealFftPlanner` | ❌ 不要实现 FFT |
| 滤波 | `idsp::iir::Biquad` | ❌ 不要实现 Butterworth |
| 重采样 | `rubato::SincFixedIn` | ❌ 不要实现 Sinc 插值 |
| ICA | `petal_decomposition::FastIca` | ❌ 不要实现 FastICA |
| SVD/PCA | `faer::Svd` | ❌ 不要实现矩阵分解 |
| 统计 | `statrs::distribution::*` | ❌ 不要实现分布函数 |

---

## 10. 实施路线图

### Phase 1: 骨架搭建 (Skeleton Setup)

- [x] 创建 workspace 结构
- [ ] 定义 bcif-core 类型（SignalInfo, Error, ChannelType）
- [ ] 定义 ProcessContext
- [ ] 定义 BatchProcessor / StreamProcessor trait

### Phase 2: 离线处理 (Offline Processing)

- [ ] bcif-io: EDF 读取
- [ ] bcif-dsp: 滤波、重采样、FFT
- [ ] bcif-pipeline: BatchPipeline builder
- [ ] 端到端测试：读取 EDF → 滤波 → PSD

### Phase 3: 实时处理 (Real-time Processing)

- [ ] bcif-io: LSL 流封装
- [ ] bcif-pipeline: RingBuffer
- [ ] bcif-pipeline: StreamPipeline
- [ ] 端到端测试：LSL 流 → 实时滤波 → 输出

### Phase 4: 算法扩展 (Algorithm Extension)

- [ ] bcif-algo: ICA
- [ ] bcif-algo: PCA
- [ ] bcif-algo: CSP（BCI 核心）

---

## 11. 关键设计决策总结

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 数据结构 | 直接用 ndarray，不包装 | 减少复杂度，函数式风格 |
| 元数据 | 独立 struct，不用 dict | 类型安全，Rust 风格 |
| 离线/在线 | 统一 trait，不同实现 | 借鉴 mne-lsl 的设计 |
| 滤波状态 | StreamProcessor 内部管理 | 跨 chunk 状态保持 |
| 模块边界 | dsp=变换, algo=学习 | 清晰的职责划分 |
| Layer 编号 | 0-4，去掉 2.5 | 明确 BCIF 核心在 Layer 3 |
| no_std | 第一版不支持 | 降低复杂度 |

---

## 12. 关键参考文档

| 文件 | 用途 |
|------|------|
| `BCIF_Core_Pipeline.md` | 五层数据流架构参考 |
| `BCIF_Agent_Prompt.md` | 编码规范和任务模板 |
| `Rust-dependency-docs/` | 各 crate 详细文档 |
| `Rust_Guideline/Rust_AI_Coding_Guideline_Std.md` | Rust 编码规范 |
| `BCIF_OVERVIEW_DOC/04_Rust替代方案详细分析.md` | Crate 选型依据 |
| `BCIF_OVERVIEW_DOC/05_代码移植优先级.md` | 功能优先级 |

---

*Document Version: 1.0.0*
*Last Updated: 2026-02-01*
*Status: 架构设计完成，进入实施阶段*
