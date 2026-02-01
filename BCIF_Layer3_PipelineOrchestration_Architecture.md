# BCIF Layer 3: Pipeline Orchestration 架构设计

> **Status**: ✅ 设计决策已完成
> **Version**: 0.1.0
> **Date**: 2026-02-02
> **Purpose**: Layer 3 编排层的详细架构与设计规范

---

## 1. Layer 3 在整体架构中的位置

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: Language Bindings (语言绑定) 【第二阶段】               │
│   PyO3, WASM, CLI, C FFI                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌───────────────────────────────────���─────────────────────────────┐
│ Layer 3: Pipeline Orchestration (编排层) ★ 当前设计层 ★          │
│                                                                 │
│   ┌─────────────────┐  ┌─────────────────┐                     │
│   │ BatchPipeline   │  │ StreamPipeline  │                     │
│   │ (离线处理)       │  │ (实时处理)       │                     │
│   └─────────────────┘  └─────────────────┘                     │
│                                                                 │
│   ★ BCIF 核心价值：将底层组件编排成完整处理流水线 ★               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Algorithm Components (算法组件)                         │
│   bcif-dsp (滤波/FFT), bcif-algo (ICA/PCA/CSP)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Core Numerics (基础数值)                                │
│   ndarray, faer, num-complex, bcif-core                         │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 0: Data I/O (数据进出)                                     │
│   edflib, xdf, lsl, ADC→μV 转换                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 设计决策汇总

### 2.1 决策总览

| # | 决策项 | 决策结果 | 理由 |
|---|--------|----------|------|
| 1 | Pipeline 类型 | BatchPipeline + StreamPipeline | 离线/实时分离，StreamPipeline 支持多数据源 |
| 2 | Processor Trait | 分离设计 | BatchProcessor 和 StreamProcessor 独立 |
| 3 | ProcessContext | 完整功能 | 基础元数据 + 可变 + 事件标记 + 处理历史 |
| 4 | RingBuffer 布局 | (n_channels, buffer_size) | 与处理数据形状一致 |
| 5 | Builder 模式 | 混合 | 链式快捷方法 + add() |
| 6 | 并行策略 | MVP 不并行 | 实时场景不需要，后续按需添加 |
| 7 | 内置处理器 | 核心集 | 滤波 + 通道选择 + 重参考 |
| 8 | 错误处理 | 统一 Result | 符合 Rust 惯例 |

---

## 3. Pipeline 类型设计

### 3.1 BatchPipeline（离线处理）

**用途**：离线数据分析、模型训练

**特点**：
- 输入：文件（EDF/XDF）或内存数组
- 滤波：零相位（filtfilt），无延迟
- 处理：一次性处理整个数据集

```rust
/// 批处理流水线（离线）
pub struct BatchPipeline {
    /// 处理器链
    processors: Vec<Box<dyn BatchProcessor>>,
    /// 处理上下文
    ctx: ProcessContext,
}

impl BatchPipeline {
    /// 创建 Builder
    pub fn new(sample_rate: f64) -> BatchPipelineBuilder;

    /// 处理数据（原地修改）
    pub fn process(&mut self, data: &mut Array2<f64>) -> Result<(), Error>;

    /// 获取处理上下文
    pub fn context(&self) -> &ProcessContext;

    /// 获取可变处理上下文
    pub fn context_mut(&mut self) -> &mut ProcessContext;
}
```

### 3.2 StreamPipeline（实时处理）

**用途**：实时 BCI、在线监测

**特点**：
- 输入：LSL 流、文件（模拟实时）、内存数组
- 滤波：单向有状态，有延迟
- 处理：按 chunk 持续处理

```rust
/// 流处理流水线（实时）
pub struct StreamPipeline {
    /// 数据源（LSL 或文件）
    source: Box<dyn DataSource>,
    /// 环形缓冲区
    buffer: RingBuffer<f64>,
    /// 处理器链
    processors: Vec<Box<dyn StreamProcessor>>,
    /// 处理上下文
    ctx: ProcessContext,
    /// 新样本计数
    n_new_samples: usize,
}

impl StreamPipeline {
    /// 从 LSL 流创建（实时场景）
    pub fn from_lsl(name: &str, bufsize_sec: f64) -> Result<StreamPipelineBuilder, Error>;

    /// 从数组创建（测试场景）
    pub fn from_array(
        data: Array2<f64>,
        sample_rate: f64,
        chunk_size: usize,
    ) -> StreamPipelineBuilder;

    /// 从 EDF 文件创建（测试场景）
    pub fn from_edf(path: &str, chunk_size: usize) -> Result<StreamPipelineBuilder, Error>;

    /// 连接数据源
    pub fn connect(&mut self) -> Result<(), Error>;

    /// 断开连接
    pub fn disconnect(&mut self) -> Result<(), Error>;

    /// 获取最新数据
    pub fn get_data(&self, winsize_sec: Option<f64>) -> (Array2<f64>, Vec<f64>);

    /// 新样本数量
    pub fn n_new_samples(&self) -> usize;

    /// 获取下一个处理后的 chunk（用于迭代）
    pub fn next(&mut self) -> Result<Option<Array2<f64>>, Error>;
}
```

### 3.3 数据源抽象（内部）

```rust
/// 数据源 trait（内部使用）
trait DataSource: Send {
    /// 获取下一个 chunk
    fn next_chunk(&mut self) -> Option<(Array2<f64>, Vec<f64>)>;
    /// 采样率
    fn sample_rate(&self) -> f64;
    /// 通道数
    fn n_channels(&self) -> usize;
    /// 是否已连接
    fn is_connected(&self) -> bool;
}

/// LSL 数据源
struct LslSource {
    inlet: LslInlet,
    sample_rate: f64,
    n_channels: usize,
}

/// 文件/数组数据源（模拟实时）
struct ArraySource {
    data: Array2<f64>,
    timestamps: Vec<f64>,
    chunk_size: usize,
    current_pos: usize,
    sample_rate: f64,
}
```

---

## 4. Processor Trait 设计

### 4.1 BatchProcessor（离线处理器）

```rust
/// 批处理器 trait
pub trait BatchProcessor: Send + Sync {
    /// 处理数据（原地修改）
    fn process(
        &mut self,
        data: &mut Array2<f64>,
        ctx: &mut ProcessContext,
    ) -> Result<(), Error>;

    /// 处理器名称
    fn name(&self) -> &str;
}
```

### 4.2 StreamProcessor（实时处理器）

```rust
/// 流处理器 trait
pub trait StreamProcessor: Send + Sync {
    /// 处理一个 chunk（原地修改）
    fn process_chunk(
        &mut self,
        chunk: &mut Array2<f64>,
        ctx: &mut ProcessContext,
    ) -> Result<(), Error>;

    /// 声明延迟（样本数）
    fn latency_samples(&self) -> usize;

    /// 重置状态
    fn reset(&mut self);

    /// 处理器名称
    fn name(&self) -> &str;
}
```

### 4.3 离线 vs 实时处理器对比

| 维度 | BatchProcessor | StreamProcessor |
|------|----------------|-----------------|
| 数据可见性 | 完整数据 | 只有当前 chunk |
| 滤波方式 | filtfilt（零相位） | 单向（有状态） |
| 状态管理 | 无状态 | 需要跨 chunk 保持状态 |
| 延迟 | 无 | 需要声明 latency_samples |
| 重置 | 不需要 | 需要 reset() 方法 |

---

## 5. ProcessContext 设计

```rust
/// 处理上下文
pub struct ProcessContext {
    // === 基础元数据（可变） ===
    /// 采样率 (Hz) - 可被重采样处理器修改
    pub sample_rate: f64,

    /// 通道信息 - 可被通道选择处理器修改
    pub channels: Vec<ChannelInfo>,

    // === 事件标记 ===
    /// 事件列表（用于 Epochs 切分）
    pub events: Vec<Event>,

    // === 处理历史（只读） ===
    /// 已应用的处理步骤
    history: Vec<ProcessingStep>,
}

/// 事件标记
#[derive(Clone, Debug)]
pub struct Event {
    /// 事件发生的样本索引
    pub sample: usize,
    /// 事件代码
    pub code: i32,
    /// 事件描述（可选）
    pub description: Option<String>,
}

/// 处理步骤记录
#[derive(Clone, Debug)]
pub struct ProcessingStep {
    /// 处理器名称
    pub name: String,
    /// 处理参数（JSON 格式）
    pub params: String,
    /// 处理时间戳
    pub timestamp: std::time::SystemTime,
}

impl ProcessContext {
    /// 从 SignalInfo 创建
    pub fn from_info(info: &SignalInfo) -> Self;

    /// 获取通道数
    pub fn n_channels(&self) -> usize;

    /// 按名称查找通道索引
    pub fn channel_index(&self, name: &str) -> Option<usize>;

    /// 按类型筛选通道索引
    pub fn channels_by_type(&self, ch_type: ChannelType) -> Vec<usize>;

    /// 记录处理步骤（内部使用）
    pub(crate) fn record_step(&mut self, name: &str, params: &str);

    /// 获取处理历史
    pub fn history(&self) -> &[ProcessingStep];
}
```

---

## 6. RingBuffer 设计

```rust
/// 环形缓冲区
pub struct RingBuffer<T> {
    /// 数据存储 (n_channels, buffer_size)
    data: Array2<T>,
    /// 时间戳
    timestamps: Vec<f64>,
    /// 写入位置
    write_pos: usize,
    /// 当前样本数（可能小于 buffer_size）
    n_samples: usize,
    /// 缓冲区容量
    capacity: usize,
}

impl<T: Clone + Default> RingBuffer<T> {
    /// 创建新的环形缓冲区
    pub fn new(n_channels: usize, capacity: usize) -> Self;

    /// 推入新数据
    pub fn push(&mut self, chunk: ArrayView2<T>, timestamps: &[f64]);

    /// 获取最近 n 个样本
    pub fn get_last(&self, n_samples: usize) -> (Array2<T>, Vec<f64>);

    /// 获取所有有效数据
    pub fn get_all(&self) -> (Array2<T>, Vec<f64>);

    /// 当前样本数
    pub fn len(&self) -> usize;

    /// 是否为空
    pub fn is_empty(&self) -> bool;

    /// 清空缓冲区
    pub fn clear(&mut self);
}
```

---

## 7. Builder 模式设计

### 7.1 BatchPipelineBuilder

```rust
/// BatchPipeline 构建器
pub struct BatchPipelineBuilder {
    sample_rate: f64,
    channels: Option<Vec<ChannelInfo>>,
    processors: Vec<Box<dyn BatchProcessor>>,
}

impl BatchPipelineBuilder {
    /// 创建新的 Builder
    pub fn new(sample_rate: f64) -> Self;

    /// 设置通道信息
    pub fn with_channels(self, channels: Vec<ChannelInfo>) -> Self;

    // === 滤波器快捷方法 ===

    /// 带通滤波
    pub fn bandpass(self, low: f64, high: f64) -> Self;

    /// 高通滤波
    pub fn highpass(self, freq: f64) -> Self;

    /// 低通滤波
    pub fn lowpass(self, freq: f64) -> Self;

    /// 陷波滤波
    pub fn notch(self, freq: f64) -> Self;

    // === 通道操作快捷方法 ===

    /// 选择通道
    pub fn pick_channels(self, names: &[&str]) -> Self;

    /// 设置参考
    pub fn set_reference(self, ref_type: ReferenceType) -> Self;

    // === 通用方法 ===

    /// 添加自定义处理器
    pub fn add<P: BatchProcessor + 'static>(self, processor: P) -> Self;

    /// 构建 Pipeline
    pub fn build(self) -> BatchPipeline;
}
```

### 7.2 StreamPipelineBuilder

```rust
/// StreamPipeline 构建器
pub struct StreamPipelineBuilder {
    source: Box<dyn DataSource>,
    bufsize_samples: usize,
    processors: Vec<Box<dyn StreamProcessor>>,
}

impl StreamPipelineBuilder {
    // === 滤波器快捷方法 ===

    /// 带通滤波（有状态）
    pub fn bandpass(self, low: f64, high: f64) -> Self;

    /// 高通滤波（有状态）
    pub fn highpass(self, freq: f64) -> Self;

    /// 低通滤波（有状态）
    pub fn lowpass(self, freq: f64) -> Self;

    /// 陷波滤波（有状态）
    pub fn notch(self, freq: f64) -> Self;

    // === 通道操作快捷方法 ===

    /// 选择通道
    pub fn pick_channels(self, names: &[&str]) -> Self;

    /// 设置参考
    pub fn set_reference(self, ref_type: ReferenceType) -> Self;

    // === 通用方法 ===

    /// 添加自定义处理器
    pub fn add<P: StreamProcessor + 'static>(self, processor: P) -> Self;

    /// 构建 Pipeline
    pub fn build(self) -> StreamPipeline;
}
```

---

## 8. 内置处理器

### 8.1 BatchPipeline 处理器

| 处理器 | 快捷方法 | 功能 |
|--------|----------|------|
| BatchBandpassFilter | `.bandpass(low, high)` | 带通滤波（零相位） |
| BatchHighpassFilter | `.highpass(freq)` | 高通滤波 |
| BatchLowpassFilter | `.lowpass(freq)` | 低通滤波 |
| BatchNotchFilter | `.notch(freq)` | 陷波滤波 |
| ChannelPicker | `.pick_channels(names)` | 通道选择 |
| Rereferencer | `.set_reference(ref_type)` | 重参考 |

### 8.2 StreamPipeline 处理器

| 处理器 | 快捷方法 | 功能 |
|--------|----------|------|
| StreamBandpassFilter | `.bandpass(low, high)` | 带通滤波（有状态） |
| StreamHighpassFilter | `.highpass(freq)` | 高通滤波（有状态） |
| StreamLowpassFilter | `.lowpass(freq)` | 低通滤波（有状态） |
| StreamNotchFilter | `.notch(freq)` | 陷波滤波（有状态） |
| ChannelPicker | `.pick_channels(names)` | 通道选择 |
| Rereferencer | `.set_reference(ref_type)` | 重参考 |

### 8.3 重参考类型

```rust
/// 重参考类型
#[derive(Clone, Debug)]
pub enum ReferenceType {
    /// 平均参考：每个通道减去所有通道的均值
    Average,
    /// 单通道参考：每个通道减去指定通道
    Channel(String),
    /// 双极参考：通道对相减
    Bipolar(Vec<(String, String)>),
}
```

---

## 9. 使用示例

### 9.1 离线处理示例

```rust
use bcif_io::read_edf;
use bcif_pipeline::{BatchPipeline, ReferenceType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 读取数据
    let (mut data, info) = read_edf("subject01.edf")?;

    // 2. 构建处理流水线
    let mut pipeline = BatchPipeline::new(info.sample_rate)
        .with_channels(info.channels.clone())
        .pick_channels(&["Fp1", "Fp2", "C3", "C4", "O1", "O2"])
        .set_reference(ReferenceType::Average)
        .bandpass(1.0, 40.0)
        .notch(50.0)
        .build();

    // 3. 处理数据
    pipeline.process(&mut data)?;

    // 4. 查看处理历史
    for step in pipeline.context().history() {
        println!("Applied: {} with params: {}", step.name, step.params);
    }

    Ok(())
}
```

### 9.2 实时处理示例（LSL）

```rust
use bcif_pipeline::{StreamPipeline, ReferenceType};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 连接 LSL 流
    let mut stream = StreamPipeline::from_lsl("EEG", 2.0)?  // 2 秒缓冲
        .pick_channels(&["C3", "Cz", "C4"])
        .set_reference(ReferenceType::Average)
        .bandpass(8.0, 30.0)  // 运动想象频段
        .build();

    stream.connect()?;

    // 2. 实时处理循环
    loop {
        if stream.n_new_samples() >= 64 {  // 每 64 个样本处理一次
            let (data, timestamps) = stream.get_data(Some(1.0));  // 获取最近 1 秒

            // 处理数据...
            let features = extract_features(&data)?;
            let prediction = classify(&features)?;

            println!("Prediction: {}", prediction);
        }

        std::thread::sleep(Duration::from_millis(50));
    }
}
```

### 9.3 模拟实时测试示例

```rust
use bcif_io::read_edf;
use bcif_pipeline::StreamPipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 读取离线数据
    let (data, info) = read_edf("test_data.edf")?;

    // 2. 用离线数据模拟实时处理
    let mut stream = StreamPipeline::from_array(data, info.sample_rate, 64)
        .bandpass(8.0, 30.0)
        .build();

    // 3. 迭代处理每个 chunk（和真实实时完全相同的代码）
    let mut chunk_count = 0;
    while let Some(chunk) = stream.next()? {
        chunk_count += 1;

        // 处理 chunk...
        let features = extract_features(&chunk)?;
        let prediction = classify(&features)?;

        println!("Chunk {}: prediction = {}", chunk_count, prediction);
    }

    println!("Processed {} chunks", chunk_count);
    Ok(())
}
```

---

## 10. 模块结构

```
bcif-pipeline/ (~1000 行)
├── Cargo.toml
└── src/
    ├── lib.rs                    // 模块导出
    ├── context.rs                // ProcessContext (~100 行)
    ├── error.rs                  // 错误类型 (~50 行)
    │
    ├── batch/                    // 离线处理
    │   ├── mod.rs
    │   ├── pipeline.rs           // BatchPipeline (~150 行)
    │   ├── builder.rs            // BatchPipelineBuilder (~100 行)
    │   └── processors/           // 内置处理器
    │       ├── mod.rs
    │       ├── filter.rs         // 滤波器 (~80 行)
    │       ├── channel.rs        // 通道操作 (~60 行)
    │       └── reference.rs      // 重参考 (~50 行)
    │
    ├── stream/                   // 实时处理
    │   ├── mod.rs
    │   ├── pipeline.rs           // StreamPipeline (~200 行)
    │   ├── builder.rs            // StreamPipelineBuilder (~100 行)
    │   ├── buffer.rs             // RingBuffer (~80 行)
    │   ├── source.rs             // DataSource trait + 实现 (~100 行)
    │   └── processors/           // 内置处理器
    │       ├── mod.rs
    │       ├── filter.rs         // 有状态滤波器 (~100 行)
    │       ├── channel.rs        // 通道操作 (~40 行)
    │       └── reference.rs      // 重参考 (~40 行)
    │
    └── traits.rs                 // BatchProcessor, StreamProcessor (~50 行)
```

---

## 11. 依赖关系

### 11.1 Cargo.toml

```toml
[package]
name = "bcif-pipeline"
version = "0.1.0"
edition = "2021"

[dependencies]
bcif-core = { path = "../bcif-core" }
bcif-dsp = { path = "../bcif-dsp" }
bcif-io = { path = "../bcif-io" }
ndarray = "0.15"
thiserror = "1.0"
```

### 11.2 Crate 依赖图

```
                    ┌─────────────────────┐
                    │   bcif-pipeline     │
                    │  (~1000 lines)      │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    bcif-io      │ │    bcif-dsp     │ │   bcif-algo     │
│  (数据 I/O)     │ │  (信号处理)     │ │  (学习算法)     │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │     bcif-core       │
                    │   (核心类型)        │
                    └─────────────────────┘
```

---

## 12. 与其他层的接口

### 12.1 Layer 0 → Layer 3

```rust
// 从 bcif-io 读取数据
let (data, info) = bcif_io::read_edf("file.edf")?;

// 创建 Pipeline
let mut pipeline = BatchPipeline::new(info.sample_rate)
    .with_channels(info.channels)
    // ...
    .build();
```

### 12.2 Layer 2 → Layer 3

```rust
// bcif-dsp 的滤波函数被 Pipeline 内部使用
// 用户也可以直接使用 bcif-dsp

use bcif_dsp::{filter_bandpass, psd_welch};

// 方式 1：通过 Pipeline
let mut pipeline = BatchPipeline::new(256.0)
    .bandpass(1.0, 40.0)  // 内部调用 bcif_dsp::filter_bandpass
    .build();

// 方式 2：直接调用
filter_bandpass(&mut data, &info, 1.0, 40.0, 4)?;
```

### 12.3 Layer 3 → 应用层

```rust
// Pipeline 输出可直接用于特征提取和分类
let mut pipeline = BatchPipeline::new(256.0)
    .bandpass(1.0, 40.0)
    .build();

pipeline.process(&mut data)?;

// 特征提取
let (psd, freqs) = bcif_dsp::psd_welch(&data, pipeline.context(), 256, None, WindowType::Hann)?;

// 分类
let features = extract_band_powers(&psd, &freqs)?;
let prediction = classifier.predict(&features)?;
```

---

## 13. 代码量估算

| 模块 | 文件 | 行数 |
|------|------|------|
| 核心 | context.rs, error.rs, traits.rs | ~200 行 |
| BatchPipeline | pipeline.rs, builder.rs | ~250 行 |
| Batch 处理器 | filter.rs, channel.rs, reference.rs | ~190 行 |
| StreamPipeline | pipeline.rs, builder.rs, buffer.rs, source.rs | ~480 行 |
| Stream 处理器 | filter.rs, channel.rs, reference.rs | ~180 行 |
| **总计** | | **~1300 行** |

---

## 14. 下一步

Layer 3 设计决策已全部完成，下一步可以：

1. **开始实现** - 按以下顺序：
   - bcif-core（核心类型）
   - bcif-dsp（信号处理）
   - bcif-pipeline（编排层）
   - bcif-io（数据 I/O）

2. **编写测试** - 单元测试和集成测试

3. **验证设计** - 用实际 EEG 数据验证处理流程

---

## 15. 参考资料

| 资源 | 位置 |
|------|------|
| 整体架构 | `Preview_BCIF_Arch_TBD.md` |
| Layer 0 设计 | `BCIF_Layer0_DataIO_Architecture.md` |
| Layer 1 设计 | `BCIF_Layer1_CoreNumerics_Architecture.md` |
| Layer 2 设计 | `BCIF_Layer2_AlgorithmComponents_Architecture.md` |
| MNE Pipeline 参考 | `mne-python-ref-fork-v1.110/mne/` |
| mne-lsl 参考 | `mne-python-ref-fork-v1.110/mne_lsl/` |

---

*Document Version: 0.1.0*
*Last Updated: 2026-02-02*
*Status: ✅ 设计决策已完成*
