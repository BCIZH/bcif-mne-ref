# BCIF Layer 1: Core Numerics 架构设计

> **Status**: ✅ 设计决策已完成
> **Version**: 0.2.0
> **Date**: 2026-02-02
> **Purpose**: Layer 1 基础数值层的详细架构与设计规范

---

## 1. Layer 1 在整体架构中的位置

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Pipeline Orchestration (编排层) ★核心价值★              │
│   BatchPipeline, StreamPipeline, ProcessContext                 │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Algorithm Components (算法组件)                         │
│   bcif-dsp (滤波/FFT), bcif-algo (ICA/PCA)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Core Numerics (基础数值) ★ 当前设计层 ★                  │
│                                                                 │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│   │ ndarray      │ │ faer         │ │ num-complex  │           │
│   │ 多维数组     │ │ 线性代数     │ │ 复数运算     │           │
│   └──────────────┘ └──────────────┘ └──────────────┘           │
│                                                                 │
│   ┌──────────────────────────────────────────────────┐         │
│   │ bcif-core: 共享类型定义                           │         │
│   │ SignalInfo, ChannelInfo, ChannelType, Error      │         │
│   └──────────────────────────────────────────────────┘         │
│                                                                 │
│   【核心原则：直接使用底层 crate，最小封装】                      │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 0: Data I/O (数据进出)                                     │
│   edflib, xdf, lsl, ADC→μV 转换                                 │
│   【生产者：输出 Array2<f64> + SignalInfo】                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 设计决策汇总

### 2.1 决策总览

| # | 决策项 | 决策结果 | 理由 |
|---|--------|----------|------|
| 1 | SignalInfo 定位 | bcif-core | 所有模块共享，避免循环依赖 |
| 2 | bcif-core 内容范围 | 最小核心 (~200 行) | YAGNI 原则，需要时再扩展 |
| 3 | 数据精度 | 统一 f64 | 计算安全、行业标准 |
| 4 | 内存布局 | C-order (行优先) | 与 MNE/NumPy 一致，滤波高效 |
| 5 | faer 转换策略 | 显式转换 + 内部封装 | 用户无感知，开销可忽略 |
| 6 | 数据形状 | (n_channels, n_times) | EEG 领域标准 |
| 7 | 元数据关联 | 分离传递 | 灵活、与 Pipeline 设计一致 |

---

## 3. 决策 1: bcif-core 定位与内容

### 3.1 为什么需要 bcif-core

```
问题：SignalInfo 应该定义在哪里？

如果定义在 bcif-io:
├── bcif-dsp 需要依赖 bcif-io ✗
├── bcif-algo 需要依赖 bcif-io ✗
└── 不合理：DSP/算法模块不应依赖 IO 模块

解决方案：创建 bcif-core
├── 存放所有模块共享的"核心类型"
├── 零业务逻辑，只有类型定义
├── 所有其他 crate 都依赖它
└── 是整个 BCIF 框架的"公共语言"
```

### 3.2 bcif-core 内容（最小核心）

**决策：只包含必须共享的类型，约 200 行代码**

```rust
// bcif-core/src/lib.rs

// === 核心元数据类型 ===

/// 信号元数据
pub struct SignalInfo {
    /// 采样率 (Hz)
    pub sample_rate: f64,
    /// 通道信息列表
    pub channels: Vec<ChannelInfo>,
}

/// 单通道信息
pub struct ChannelInfo {
    /// 通道名称 (如 "Fp1", "O1")
    pub name: String,
    /// 通道类型
    pub ch_type: ChannelType,
    /// 物理单位
    pub unit: Unit,
}

/// 通道类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelType {
    Eeg,   // 脑电
    Eog,   // 眼电
    Emg,   // 肌电
    Ecg,   // 心电
    Stim,  // 刺激/事件标记
    Misc,  // 其他
}

/// 物理单位枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    MicroVolt,  // μV
    MilliVolt,  // mV
    Volt,       // V
    Unknown,    // 未知
}

// === 错误类型 ===

/// BCIF 统一错误类型
#[derive(Debug)]
pub enum Error {
    /// IO 错误
    Io(std::io::Error),
    /// 无效数据
    InvalidData(String),
    /// 形状不匹配
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// 通道不存在
    ChannelNotFound(String),
    /// 参数无效
    InvalidParameter(String),
}

// === 辅助方法 ===

impl SignalInfo {
    /// 创建新的 SignalInfo
    pub fn new(sample_rate: f64, channels: Vec<ChannelInfo>) -> Self {
        Self { sample_rate, channels }
    }

    /// 获取通道数
    pub fn n_channels(&self) -> usize {
        self.channels.len()
    }

    /// 按名称查找通道索引
    pub fn channel_index(&self, name: &str) -> Option<usize> {
        self.channels.iter().position(|ch| ch.name == name)
    }

    /// 按类型筛选通道索引
    pub fn channels_by_type(&self, ch_type: ChannelType) -> Vec<usize> {
        self.channels
            .iter()
            .enumerate()
            .filter(|(_, ch)| ch.ch_type == ch_type)
            .map(|(i, _)| i)
            .collect()
    }
}
```

### 3.3 不包含的内容

```
bcif-core 不包含：
├── 类型别名 (type ContinuousData = Array2<f64>)
│   └── 理由：直接用 Array2<f64> 更透明
│
├── EEG 频段常量 (ALPHA_BAND, BETA_BAND...)
│   └── 理由：放在 bcif-dsp，使用它的地方定义
│
├── 转换辅助函数 (ndarray ↔ faer)
│   └── 理由：放在 bcif-algo 内部，需要时再提升
│
└── 验证函数
    └── 理由：各模块自行验证，避免过早抽象
```

---

## 4. 决策 2: 数据精度

### 4.1 决策：统一使用 f64

```
┌─────────────────────────────────────────────────────────────────┐
│ 数据精度决策：统一 f64                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 存储: f64                                                       │
│ 计算: f64                                                       │
│ 输出: f64                                                       │
│                                                                 │
│ 不采用 f32/f64 混合策略，保持简单一致                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 理由

| 考量 | 分析 |
|------|------|
| **数据来源** | ADS129x 24-bit ADC，f32 刚好够，f64 有余量 |
| **计算安全** | IIR 滤波器、ICA 需要高精度，f32 可能不稳定 |
| **行业标准** | MNE-Python、EEGLAB、FieldTrip 都用 f64 |
| **性能影响** | EEG 数据量小 (32ch × 1kHz = 256KB/s)，可忽略 |
| **简单性** | 不需要在 f32/f64 之间转换，减少 bug |

---

## 5. 决策 3: 内存布局

### 5.1 决策：C-order (行优先)

```
┌─────────────────────────────────────────────────────────────────┐
│ 内存布局决策：C-order (行优先)                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 数据形状: (n_channels, n_times)                                 │
│ 内存布局: C-order (行优先)                                       │
│                                                                 │
│ 结果: 同一通道的时间序列在内存中连续                             │
│                                                                 │
│     Ch0: [t0, t1, t2, t3, ...] ← 内存连续                       │
│     Ch1: [t0, t1, t2, t3, ...]                                  │
│     Ch2: [t0, t1, t2, t3, ...]                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 理由

| 考量 | 分析 |
|------|------|
| **滤波效率** | 按通道滤波是最频繁操作，C-order 缓存友好 |
| **MNE 兼容** | MNE-Python 使用 C-order，便于迁移 |
| **ndarray 默认** | Rust ndarray 默认 C-order |
| **实时场景** | 虽然数据按时间点到达，但滤波仍是热点 |

### 5.3 faer 转换策略

```
┌─────────────────────────────────────────────────────────────────┐
│ bcif-algo 内部的 ndarray ↔ faer 转换                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 问题：faer 偏好 F-order，我们用 C-order                         │
│                                                                 │
│ 解决方案：bcif-algo 内部显式转换                                 │
│                                                                 │
│ pub struct ICA { ... }                                          │
│                                                                 │
│ impl ICA {                                                      │
│     pub fn fit_transform(                                       │
│         &mut self,                                              │
│         data: &Array2<f64>,  // C-order 输入                    │
│     ) -> Result<Array2<f64>, Error> {                           │
│                                                                 │
│         // 1. 转换为 faer Mat (F-order)                         │
│         let mat = ndarray_to_faer(data);                        │
│                                                                 │
│         // 2. 执行 ICA (faer 内部高效)                          │
│         let result_mat = self.ica_internal(&mat)?;              │
│                                                                 │
│         // 3. 转换回 ndarray (C-order)                          │
│         Ok(faer_to_ndarray(&result_mat))                        │
│     }                                                           │
│ }                                                               │
│                                                                 │
│ 性能分析：                                                       │
│ ├── 转换开销：< 1 ms (32ch × 10000 samples)                     │
│ ├── ICA 计算：100-1000 ms                                       │
│ └── 转换占比：< 1%，可忽略                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 决策 4: 数据形状约定

### 6.1 决策：(n_channels, n_times)

```
┌─────────────────────────────────────────────────────────────────┐
│ 数据形状约定                                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 连续数据 (Continuous):                                          │
│ ├── 类型: Array2<f64>                                           │
│ ├── 形状: (n_channels, n_times)                                 │
│ ├── 索引: data[[ch, t]]                                         │
│ └── 示例: 8 通道, 10 秒 @ 256 Hz → (8, 2560)                    │
│                                                                 │
│ 分段数据 (Epochs):                                              │
│ ├── 类型: Array3<f64>                                           │
│ ├── 形状: (n_epochs, n_channels, n_times)                       │
│ ├── 索引: data[[epoch, ch, t]]                                  │
│ └── 示例: 100 段, 8 通道, 1 秒 @ 256 Hz → (100, 8, 256)         │
│                                                                 │
│ 功率谱 (PSD):                                                   │
│ ├── 类型: Array2<f64>                                           │
│ ├── 形状: (n_channels, n_freqs)                                 │
│ └── 索引: psd[[ch, freq_idx]]                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 与 C-order 的配合

```
C-order + (n_channels, n_times):
├── data.row(ch) → 第 ch 通道的所有时间点 (连续内存)
├── data.column(t) → 第 t 时间点的所有通道 (跳跃访问)
│
├── 滤波：for row in data.rows_mut() { filter(&mut row); }
│   └── 高效：顺序访问连续内存
│
└── 空间滤波：data.column(t)
    └── 可接受：数据量小，跳跃访问开销不大
```

---

## 7. 决策 5: 元数据关联方式

### 7.1 决策：分离传递

```
┌─────────────────────────────────────────────────────────────────┐
│ 元数据关联方式：分离传递                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 函数签名示例：                                                   │
│                                                                 │
│ // 原地修改，info 不变                                           │
│ fn filter_inplace(                                              │
│     data: &mut Array2<f64>,                                     │
│     info: &SignalInfo,                                          │
│     low: f64,                                                   │
│     high: f64,                                                  │
│ ) -> Result<(), Error>;                                         │
│                                                                 │
│ // 返回新数据和新 info                                           │
│ fn resample(                                                    │
│     data: &Array2<f64>,                                         │
│     info: &SignalInfo,                                          │
│     new_sample_rate: f64,                                       │
│ ) -> Result<(Array2<f64>, SignalInfo), Error>;                  │
│                                                                 │
│ // 通道选择                                                      │
│ fn pick_channels(                                               │
│     data: &Array2<f64>,                                         │
│     info: &SignalInfo,                                          │
│     names: &[&str],                                             │
│ ) -> Result<(Array2<f64>, SignalInfo), Error>;                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 理由

| 考量 | 分析 |
|------|------|
| **与 Pipeline 一致** | ProcessContext 已包含 SignalInfo |
| **灵活性** | 多个 epoch 可共享同一 info |
| **函数式风格** | 纯函数，易于组合和测试 |
| **Rust 生态** | ndarray 函数都接受 array 参数，不强制封装 |

### 7.3 使用示例

```rust
// 加载数据
let (mut data, mut info) = read_edf("recording.edf")?;

// 处理链
filter_inplace(&mut data, &info, 1.0, 40.0)?;
let (data, info) = resample(&data, &info, 256.0)?;
let (data, info) = pick_channels(&data, &info, &["Fp1", "Fp2", "O1", "O2"])?;

// 最终结果
println!("Shape: {:?}", data.dim());  // (4, ...)
println!("Sample rate: {}", info.sample_rate);  // 256.0
println!("Channels: {:?}", info.channels.iter().map(|c| &c.name).collect::<Vec<_>>());
```

---

## 8. 模块依赖关系

### 8.1 Crate 依赖图

```
                    ┌─────────────────────┐
                    │     bcif-core       │
                    │  (~200 lines)       │
                    ├─────────────────────┤
                    │ • SignalInfo        │
                    │ • ChannelInfo       │
                    │ • ChannelType       │
                    │ • Unit              │
                    │ • Error             │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    bcif-io      │ │    bcif-dsp     │ │   bcif-algo     │
│  (~800 lines)   │ │  (~600 lines)   │ │  (~400 lines)   │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ 依赖:           │ │ 依赖:           │ │ 依赖:           │
│ • bcif-core     │ │ • bcif-core     │ │ • bcif-core     │
│ • ndarray       │ │ • ndarray       │ │ • ndarray       │
│ • edflib        │ │ • idsp          │ │ • faer          │
│ • lsl           │ │ • rubato        │ │ • petal-decomp  │
│                 │ │ • realfft       │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   bcif-pipeline     │
                    │  (~1300 lines)      │
                    ├─────────────────────┤
                    │ 依赖:               │
                    │ • bcif-core         │
                    │ • bcif-io           │
                    │ • bcif-dsp          │
                    │ • bcif-algo         │
                    │ • ndarray           │
                    └─────────────────────┘
```

### 8.2 Cargo.toml 示例

```toml
# bcif-core/Cargo.toml
[package]
name = "bcif-core"
version = "0.1.0"
edition = "2021"

[dependencies]
thiserror = "1.0"  # 错误处理

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

# bcif-algo/Cargo.toml
[package]
name = "bcif-algo"
version = "0.1.0"
edition = "2021"

[dependencies]
bcif-core = { path = "../bcif-core" }
ndarray = "0.15"
faer = "0.19"
```

---

## 9. 数据表示统一规范

```
┌─────────────────────────────────────────────────────────────────┐
│                    BCIF 数据表示统一规范                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 数据类型:     Array2<f64>  (连续数据)                           │
│              Array3<f64>  (分段数据)                            │
│                                                                 │
│ 数据精度:     f64 (统一)                                        │
│                                                                 │
│ 内存布局:     C-order (行优先)                                   │
│                                                                 │
│ 数据形状:     (n_channels, n_times)      连续数据               │
│              (n_epochs, n_channels, n_times)  分段数据          │
│                                                                 │
│ 元数据:       SignalInfo (分离传递)                             │
│                                                                 │
│ 索引约定:     data[[ch, t]]              连续数据               │
│              data[[epoch, ch, t]]       ��段数据                │
│                                                                 │
│ 通道访问:     data.row(ch)              获取单通道 (连续内存)   │
│              data.rows()               遍历所有通道             │
│                                                                 │
│ 时间点访问:   data.column(t)            获取单时间点            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. 与其他层的接口

### 10.1 Layer 0 → Layer 1 接口

```rust
// Layer 0 (bcif-io) 输出
pub fn read_edf(path: &str) -> Result<(Array2<f64>, SignalInfo), Error>;
pub fn read_xdf(path: &str) -> Result<(Array2<f64>, SignalInfo), Error>;

// LSL 流
pub struct LslStream { ... }
impl LslStream {
    pub fn get_data(&mut self) -> (Array2<f64>, Vec<f64>);  // data, timestamps
    pub fn info(&self) -> &SignalInfo;
}
```

### 10.2 Layer 1 → Layer 2 接口

```rust
// Layer 2 (bcif-dsp) 函数签名
pub fn filter_bandpass(
    data: &mut Array2<f64>,
    info: &SignalInfo,
    low: f64,
    high: f64,
) -> Result<(), Error>;

pub fn resample(
    data: &Array2<f64>,
    info: &SignalInfo,
    new_rate: f64,
) -> Result<(Array2<f64>, SignalInfo), Error>;

pub fn compute_psd(
    data: &Array2<f64>,
    info: &SignalInfo,
    fmin: f64,
    fmax: f64,
) -> Result<(Array2<f64>, Array1<f64>), Error>;  // psd, freqs

// Layer 2 (bcif-algo) 函数签名
pub struct ICA { ... }
impl ICA {
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<(), Error>;
    pub fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, Error>;
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Result<Array2<f64>, Error>;
}
```

### 10.3 Layer 2 → Layer 3 接口

```rust
// Layer 3 (bcif-pipeline) 使用 Processor trait
pub trait BatchProcessor: Send + Sync {
    fn process(
        &mut self,
        data: &mut Array2<f64>,
        ctx: &mut ProcessContext,
    ) -> Result<(), Error>;

    fn name(&self) -> &str;
}

pub struct ProcessContext {
    pub info: SignalInfo,
    // 其他上下文...
}
```

---

## 11. 下一步

Layer 1 设计决策已全部完成，下一步可以：

1. **实现 bcif-core** - 创建 Cargo workspace，实现核心类型 (~200 行)
2. **设计 Layer 2 (bcif-dsp)** - 滤波、重采样、FFT 的详细设计
3. **设计 Layer 3 (bcif-pipeline)** - BatchPipeline、StreamPipeline 的详细设计

---

## 12. 参考资料

| 资源 | 位置 |
|------|------|
| ndarray 文档 | `Rust-dependency-docs/ndarray/` |
| faer 文档 | `Rust-dependency-docs/faer/` |
| Layer 0 设计 | `BCIF_Layer0_DataIO_Architecture.md` |
| 整体架构 | `Preview_BCIF_Arch_TBD.md` |
| MNE 数据结构 | `mne-python-ref-fork-v1.110/mne/io/` |

---

*Document Version: 0.2.0*
*Last Updated: 2026-02-02*
*Status: ✅ 设计决策已完成*
