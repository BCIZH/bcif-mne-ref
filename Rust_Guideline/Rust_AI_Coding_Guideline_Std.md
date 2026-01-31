# Rust AI-Friendly Coding Guideline (Standard Environment)
# Rust AI 友好编码规范（标准环境）

> **Edition**: Rust 2021  
> **Target**: Desktop / Server / CLI / Library (with `std`)  
> **Philosophy**: Keep it simple. Readable code > clever code.  
> **核心理念**: 简单至上。可读性 > 炫技。

---

## Table of Contents / 目录

1. [Core Principles / 核心原则](#1-core-principles--核心原则)
2. [Naming Conventions / 命名规范](#2-naming-conventions--命名规范)
3. [Type System / 类型系统](#3-type-system--类型系统)
4. [Ownership & Borrowing / 所有权与借用](#4-ownership--borrowing--所有权与借用)
5. [Error Handling / 错误处理](#5-error-handling--错误处理)
6. [Comments & Documentation / 注释与文档](#6-comments--documentation--注释与文档)
7. [Collections / 集合类型](#7-collections--集合类型)
8. [Control Flow / 控制流](#8-control-flow--控制流)
9. [Functions & Modules / 函数与模块](#9-functions--modules--函数与模块)
10. [Forbidden Patterns / 禁止项](#10-forbidden-patterns--禁止项)
11. [Recommended Crates / 推荐依赖](#11-recommended-crates--推荐依赖)
12. [Code Examples / 代码示例](#12-code-examples--代码示例)

---

## 1. Core Principles / 核心原则

### English

1. **Explicit over implicit** - Always prefer clear, explicit code over clever shortcuts.
2. **Clone when confused** - If borrowing is tricky, use `.clone()`. Optimize later.
3. **One thing per function** - Functions should do one thing and do it well.
4. **No magic numbers** - Use named constants for all numeric literals.
5. **AI-readable patterns** - Stick to common patterns that AI tools understand well.

### 中文

1. **显式优于隐式** - 始终选择清晰、显式的代码，而非巧妙的捷径。
2. **困惑时就克隆** - 如果借用很棘手，使用 `.clone()`，之后再优化。
3. **单一职责** - 函数只做一件事，并把它做好。
4. **禁止魔法数字** - 所有数字字面量必须使用命名常量。
5. **AI 可读模式** - 坚持使用 AI 工具能够良好理解的常见模式。

---

## 2. Naming Conventions / 命名规范

### Summary Table / 总结表

| Entity / 实体 | Style / 风格 | Example / 示例 |
|--------------|-------------|----------------|
| Crate / Module | `snake_case` | `signal_processing` |
| Struct / Enum / Trait | `UpperCamelCase` | `SignalBuffer`, `ProcessingError` |
| Function / Method | `snake_case` | `calculate_fft`, `get_channel_data` |
| Variable / Parameter | `snake_case` | `sample_rate`, `channel_count` |
| Constant / Static | `SCREAMING_SNAKE_CASE` | `MAX_CHANNELS`, `DEFAULT_SAMPLE_RATE` |
| Type Parameter | Single uppercase | `T`, `E`, `K`, `V` |

### Rules / 规则

```rust
// ✅ GOOD: Descriptive names with units
// ✅ 好: 描述性名称，带单位
let sample_rate_hz: f64 = 256.0;
let duration_seconds: f64 = 10.0;
let voltage_microvolts: f64 = 5.5;

// ❌ BAD: Cryptic abbreviations
// ❌ 坏: 神秘的缩写
let sr: f64 = 256.0;
let dur: f64 = 10.0;
let v: f64 = 5.5;
```

### Boolean Naming / 布尔值命名

```rust
// ✅ GOOD: Use is_, has_, can_, should_ prefix
// ✅ 好: 使用 is_, has_, can_, should_ 前缀
let is_valid: bool = true;
let has_data: bool = false;
let can_process: bool = true;

// ❌ BAD: Ambiguous boolean names
// ❌ 坏: 模糊的布尔值名称
let valid: bool = true;
let data: bool = false;
```

---

## 3. Type System / 类型系统

### Always Annotate Types / 始终标注类型

```rust
// ✅ GOOD: Explicit type annotations
// ✅ 好: 显式类型标注
let channel_count: usize = 32;
let sample_rate: f64 = 256.0;
let data: Vec<f64> = Vec::with_capacity(1024);
let name: String = String::from("EEG");

// ❌ BAD: Rely on inference for complex types
// ❌ 坏: 对复杂类型依赖推导
let data = vec![1.0, 2.0, 3.0];  // Type unclear in large codebase
                                  // 在大型代码库中类型不清晰
```

### Use Specific Integer Types / 使用具体整数类型

```rust
// ✅ GOOD: Use specific types for clarity
// ✅ 好: 使用具体类型以清晰
let index: usize = 0;           // For array indexing / 数组索引
let count: u32 = 100;           // For counts / 计数
let signed_value: i32 = -10;    // For signed values / 有符号值
let byte_value: u8 = 255;       // For bytes / 字节

// ❌ BAD: Use i32/u32 for everything
// ❌ 坏: 所有地方都用 i32/u32
```

### Struct Definition / 结构体定义

```rust
// ✅ GOOD: Own your data, no lifetimes in structs
// ✅ 好: 拥有数据，结构体中不使用生命周期

/// Signal buffer for EEG data processing.
/// EEG 数据处理的信号缓冲区。
pub struct SignalBuffer {
    /// Raw sample data in microvolts.
    /// 原始采样数据（微伏）。
    pub data: Vec<f64>,
    
    /// Sampling rate in Hz.
    /// 采样率（赫兹）。
    pub sample_rate: f64,
    
    /// Channel names.
    /// 通道名称。
    pub channel_names: Vec<String>,
}

// ❌ BAD: Lifetimes in struct (avoid unless necessary)
// ❌ 坏: 结构体中使用生命周期（除非必要否则避免）
pub struct SignalBufferBad<'a> {
    pub data: &'a [f64],  // Avoid this pattern
                          // 避免这种模式
}
```

---

## 4. Ownership & Borrowing / 所有权与借用

### The Simple Rule / 简单规则

```rust
// Rule: Pass by reference for reading, pass by value for consuming
// 规则: 读取时传引用，消耗时传值

// ✅ GOOD: Clear ownership patterns
// ✅ 好: 清晰的所有权模式

/// Process signal data without modifying.
/// 处理信号数据，不修改。
fn analyze_signal(data: &[f64]) -> f64 {
    // Borrow for read-only access
    // 借用用于只读访问
    data.iter().sum::<f64>() / data.len() as f64
}

/// Process and modify signal data.
/// 处理并修改信号数据。
fn normalize_signal(data: &mut [f64]) {
    // Mutable borrow for modification
    // 可变借用用于修改
    let max: f64 = data.iter().cloned().fold(f64::MIN, f64::max);
    for sample in data.iter_mut() {
        *sample /= max;
    }
}

/// Consume and transform data.
/// 消耗并转换数据。
fn into_processed(data: Vec<f64>) -> Vec<f64> {
    // Take ownership when transforming
    // 转换时获取所有权
    data.into_iter().map(|x| x * 2.0).collect()
}
```

### When to Clone / 何时使用克隆

```rust
// ✅ GOOD: Clone when ownership is complex
// ✅ 好: 当所有权复杂时使用克隆

fn process_channels(buffer: &SignalBuffer) -> Vec<f64> {
    // Clone to avoid borrow checker issues
    // 克隆以避免借用检查器问题
    let data: Vec<f64> = buffer.data.clone();
    
    // Now we own the data and can process freely
    // 现在我们拥有数据，可以自由处理
    data.into_iter().map(|x| x.abs()).collect()
}

// Note: Optimize clones later if profiling shows it's a bottleneck
// 注意: 如果性能分析显示是瓶颈，之后再优化克隆
```

---

## 5. Error Handling / 错误处理

### Use Result with ? Operator / 使用 Result 和 ? 操作符

```rust
use std::fs;
use std::io;

/// Read signal data from file.
/// 从文件读取信号数据。
///
/// # Errors
/// # 错误
///
/// Returns error if file cannot be read or parsed.
/// 如果文件无法读取或解析则返回错误。
pub fn read_signal_file(path: &str) -> Result<Vec<f64>, io::Error> {
    // ✅ GOOD: Use ? for error propagation
    // ✅ 好: 使用 ? 进行错误传播
    let content: String = fs::read_to_string(path)?;
    
    let data: Vec<f64> = content
        .lines()
        .filter_map(|line| line.parse::<f64>().ok())
        .collect();
    
    Ok(data)
}
```

### Define Clear Error Types / 定义清晰的错误类型

```rust
/// Error types for signal processing.
/// 信号处理的错误类型。
#[derive(Debug, Clone)]
pub enum ProcessingError {
    /// Invalid sample rate (must be positive).
    /// 无效采样率（必须为正）。
    InvalidSampleRate(f64),
    
    /// Empty data buffer.
    /// 空数据缓冲区。
    EmptyData,
    
    /// Channel count mismatch.
    /// 通道数不匹配。
    ChannelMismatch {
        expected: usize,
        actual: usize,
    },
    
    /// File I/O error.
    /// 文件 I/O 错误。
    IoError(String),
}

impl std::fmt::Display for ProcessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSampleRate(rate) => {
                write!(f, "Invalid sample rate: {}", rate)
            }
            Self::EmptyData => {
                write!(f, "Data buffer is empty")
            }
            Self::ChannelMismatch { expected, actual } => {
                write!(f, "Channel mismatch: expected {}, got {}", expected, actual)
            }
            Self::IoError(msg) => {
                write!(f, "I/O error: {}", msg)
            }
        }
    }
}

impl std::error::Error for ProcessingError {}
```

### Never Use unwrap in Production / 生产代码禁止使用 unwrap

```rust
// ❌ BAD: unwrap can panic
// ❌ 坏: unwrap 会 panic
let value = some_option.unwrap();
let result = some_result.unwrap();

// ✅ GOOD: Handle errors explicitly
// ✅ 好: 显式处理错误
let value: i32 = some_option.unwrap_or(0);
let value: i32 = some_option.unwrap_or_default();

// Or propagate with ?
// 或使用 ? 传播
let value: i32 = some_option.ok_or(ProcessingError::EmptyData)?;

// Or use if let for optional handling
// 或使用 if let 进行可选处理
if let Some(value) = some_option {
    println!("Value: {}", value);
}
```

---

## 6. Comments & Documentation / 注释与文档

### Bilingual Comment Style / 双语注释风格

```rust
/// Calculate the power spectral density using Welch's method.
/// 使用 Welch 方法计算功率谱密度。
///
/// # Arguments / 参数
///
/// * `data` - Input signal samples in microvolts.
///            输入信号采样（微伏）。
/// * `sample_rate` - Sampling frequency in Hz.
///                   采样频率（赫兹）。
/// * `window_size` - FFT window size (must be power of 2).
///                   FFT 窗口大小（必须是 2 的幂）。
///
/// # Returns / 返回
///
/// Power spectral density array.
/// 功率谱密度数组。
///
/// # Errors / 错误
///
/// Returns `ProcessingError` if parameters are invalid.
/// 如果参数无效则返回 `ProcessingError`。
///
/// # Example / 示例
///
/// ```rust
/// let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
/// let psd = calculate_psd(&data, 256.0, 256)?;
/// ```
pub fn calculate_psd(
    data: &[f64],
    sample_rate: f64,
    window_size: usize,
) -> Result<Vec<f64>, ProcessingError> {
    // Validate input parameters.
    // 验证输入参数。
    if data.is_empty() {
        return Err(ProcessingError::EmptyData);
    }
    
    if sample_rate <= 0.0 {
        return Err(ProcessingError::InvalidSampleRate(sample_rate));
    }
    
    // Implementation here...
    // 实现代码...
    
    Ok(vec![])
}
```

### Inline Comments / 行内注释

```rust
fn process_epoch(data: &[f64], baseline_samples: usize) -> Vec<f64> {
    // Step 1: Calculate baseline mean.
    // 步骤 1: 计算基线均值。
    let baseline: &[f64] = &data[..baseline_samples];
    let baseline_mean: f64 = baseline.iter().sum::<f64>() / baseline.len() as f64;
    
    // Step 2: Subtract baseline from all samples.
    // 步骤 2: 从所有采样中减去基线。
    let corrected: Vec<f64> = data
        .iter()
        .map(|&sample| sample - baseline_mean)
        .collect();
    
    // Step 3: Return corrected data.
    // 步骤 3: 返回校正后的数据。
    corrected
}
```

### Module-Level Documentation / 模块级文档

```rust
//! # Signal Processing Module
//! # 信号处理模块
//!
//! This module provides functions for EEG signal processing,
//! including filtering, FFT, and artifact removal.
//!
//! 本模块提供 EEG 信号处理函数，包括滤波、FFT 和伪影去除。
//!
//! ## Usage / 使用方法
//!
//! ```rust
//! use signal_processing::filter;
//!
//! let filtered = filter::bandpass(&data, 1.0, 40.0, 256.0)?;
//! ```

pub mod filter;
pub mod fft;
pub mod artifact;
```

---

## 7. Collections / 集合类型

### Vec: The Default Choice / Vec: 默认选择

```rust
// ✅ GOOD: Pre-allocate when size is known
// ✅ 好: 已知大小时预分配

/// Create a signal buffer with pre-allocated capacity.
/// 创建预分配容量的信号缓冲区。
fn create_buffer(channel_count: usize, sample_count: usize) -> Vec<Vec<f64>> {
    let total_samples: usize = channel_count * sample_count;
    
    // Pre-allocate to avoid repeated reallocations.
    // 预分配以避免重复重新分配。
    let mut channels: Vec<Vec<f64>> = Vec::with_capacity(channel_count);
    
    for _ in 0..channel_count {
        let channel: Vec<f64> = Vec::with_capacity(sample_count);
        channels.push(channel);
    }
    
    channels
}
```

### HashMap: Use Entry API / HashMap: 使用 Entry API

```rust
use std::collections::HashMap;

/// Count occurrences of each event type.
/// 统计每种事件类型的出现次数。
fn count_events(events: &[String]) -> HashMap<String, u32> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    
    for event in events {
        // ✅ GOOD: Use entry API
        // ✅ 好: 使用 entry API
        *counts.entry(event.clone()).or_insert(0) += 1;
    }
    
    counts
}

// ❌ BAD: Check then insert pattern
// ❌ 坏: 先检查再插入模式
fn count_events_bad(events: &[String]) -> HashMap<String, u32> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    
    for event in events {
        if counts.contains_key(event) {
            *counts.get_mut(event).unwrap() += 1;
        } else {
            counts.insert(event.clone(), 1);
        }
    }
    
    counts
}
```

### String vs &str / String 与 &str

```rust
// ✅ GOOD: Use String for owned data in structs
// ✅ 好: 在结构体中使用 String 作为拥有数据

pub struct ChannelInfo {
    pub name: String,        // Owned, no lifetime needed
                             // 拥有，不需要生命周期
    pub unit: String,
}

// ✅ GOOD: Use &str for function parameters (read-only)
// ✅ 好: 函数参数使用 &str（只读）

fn find_channel(channels: &[ChannelInfo], name: &str) -> Option<usize> {
    channels.iter().position(|c| c.name == name)
}
```

---

## 8. Control Flow / 控制流

### Prefer match Over if-else Chains / 优先使用 match 而非 if-else 链

```rust
/// Get frequency band name.
/// 获取频段名称。
fn get_band_name(frequency: f64) -> &'static str {
    // ✅ GOOD: Clear pattern matching
    // ✅ 好: 清晰的模式匹配
    match frequency {
        f if f < 4.0 => "Delta",
        f if f < 8.0 => "Theta",
        f if f < 13.0 => "Alpha",
        f if f < 30.0 => "Beta",
        _ => "Gamma",
    }
}
```

### Use if let for Single Pattern / 单一模式使用 if let

```rust
/// Process optional configuration.
/// 处理可选配置。
fn apply_config(config: Option<&Config>) {
    // ✅ GOOD: if let for single pattern
    // ✅ 好: 单一模式使用 if let
    if let Some(cfg) = config {
        println!("Using config: {:?}", cfg);
    }
    
    // ❌ BAD: match for single pattern
    // ❌ 坏: 单一模式使用 match
    match config {
        Some(cfg) => println!("Using config: {:?}", cfg),
        None => {}
    }
}
```

### Iterator Methods / 迭代器方法

```rust
// ✅ GOOD: Use iterator methods for transformations
// ✅ 好: 转换使用迭代器方法

fn normalize_data(data: &[f64]) -> Vec<f64> {
    let max: f64 = data.iter().cloned().fold(f64::MIN, f64::max);
    
    // Clear transformation pipeline.
    // 清晰的转换管道。
    data.iter()
        .map(|&x| x / max)
        .collect()
}

// ✅ GOOD: Use for loop for side effects
// ✅ 好: 副作用使用 for 循环

fn print_channels(channels: &[String]) {
    for (index, name) in channels.iter().enumerate() {
        println!("Channel {}: {}", index, name);
    }
}
```

---

## 9. Functions & Modules / 函数与模块

### Function Signature Guidelines / 函数签名指南

```rust
// ✅ GOOD: Clear, explicit function signature
// ✅ 好: 清晰、显式的函数签名

/// Apply bandpass filter to signal.
/// 对信号应用带通滤波器。
pub fn apply_bandpass_filter(
    data: &[f64],           // Input signal (read-only)
                            // 输入信号（只读）
    low_freq: f64,          // Low cutoff frequency in Hz
                            // 低截止频率（赫兹）
    high_freq: f64,         // High cutoff frequency in Hz
                            // 高截止频率（赫兹）
    sample_rate: f64,       // Sampling rate in Hz
                            // 采样率（赫兹）
) -> Result<Vec<f64>, ProcessingError> {
    // Implementation...
    // 实现...
    Ok(vec![])
}

// ❌ BAD: Too many parameters, unclear purpose
// ❌ 坏: 参数太多，目的不清
pub fn filter(d: &[f64], l: f64, h: f64, s: f64, o: i32, t: &str) -> Vec<f64> {
    vec![]
}
```

### Module Organization / 模块组织

```
src/
├── lib.rs              # Library root / 库根
├── signal/
│   ├── mod.rs          # Signal module / 信号模块
│   ├── filter.rs       # Filtering functions / 滤波函数
│   ├── fft.rs          # FFT functions / FFT 函数
│   └── resample.rs     # Resampling functions / 重采样函数
├── data/
│   ├── mod.rs          # Data module / 数据模块
│   ├── raw.rs          # Raw data structure / 原始数据结构
│   └── epochs.rs       # Epoch structure / Epoch 结构
└── io/
    ├── mod.rs          # I/O module / I/O 模块
    ├── reader.rs       # File readers / 文件读取器
    └── writer.rs       # File writers / 文件写入器
```

### Keep Functions Short / 保持函数简短

```rust
// ✅ GOOD: Small, focused functions
// ✅ 好: 小而专注的函数

/// Calculate mean of data.
/// 计算数据均值。
fn calculate_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Calculate standard deviation.
/// 计算标准差。
fn calculate_std(data: &[f64]) -> f64 {
    let mean: f64 = calculate_mean(data);
    let variance: f64 = data
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

/// Z-score normalize data.
/// Z-score 归一化数据。
fn zscore_normalize(data: &[f64]) -> Vec<f64> {
    let mean: f64 = calculate_mean(data);
    let std: f64 = calculate_std(data);
    
    data.iter()
        .map(|&x| (x - mean) / std)
        .collect()
}
```

---

## 10. Forbidden Patterns / 禁止项

### ❌ DO NOT USE / 禁止使用

| Pattern / 模式 | Reason / 原因 | Alternative / 替代方案 |
|---------------|--------------|----------------------|
| `unwrap()` in production | Can panic / 会 panic | `?`, `unwrap_or()`, `if let` |
| Complex macros | AI cannot parse / AI 无法解析 | Regular functions / 普通函数 |
| Deep generics `<T: A + B + C>` | Hard to read / 难以阅读 | Concrete types / 具体类型 |
| Lifetime in structs `<'a>` | Increases complexity / 增加复杂度 | Own data with `String`, `Vec` |
| `unsafe` blocks | Memory safety risk / 内存安全风险 | Safe abstractions / 安全抽象 |
| `async`/`await` (for beginners) | Complex state machine / 复杂状态机 | Sync code or threads / 同步代码或线程 |
| `Rc`/`Arc`/`RefCell` | Complex ownership / 复杂所有权 | Simplify data flow / 简化数据流 |

### Code Examples of Forbidden Patterns / 禁止模式代码示例

```rust
// ❌ BAD: Complex macro
// ❌ 坏: 复杂宏
macro_rules! create_processor {
    ($name:ident, $($field:ident : $type:ty),*) => {
        struct $name { $($field: $type),* }
    };
}

// ✅ GOOD: Just define the struct directly
// ✅ 好: 直接定义结构体
struct SignalProcessor {
    sample_rate: f64,
    channel_count: usize,
}

// ❌ BAD: Deep trait bounds
// ❌ 坏: 深层 trait 约束
fn process<T: Iterator<Item = f64> + Clone + Send + Sync>(data: T) {}

// ✅ GOOD: Use concrete types
// ✅ 好: 使用具体类型
fn process(data: &[f64]) -> Vec<f64> {
    data.to_vec()
}
```

---

## 11. Recommended Crates / 推荐依赖

### Core Dependencies for BCIF / BCIF 核心依赖

```toml
[dependencies]
# Array operations / 数组操作
ndarray = { version = "0.16", features = ["rayon"] }

# Linear algebra (pure Rust) / 线性代数（纯 Rust）
faer = { version = "0.19", features = ["rayon"] }

# FFT for real signals / 实数信号 FFT
realfft = "3.3"

# Signal filtering / 信号滤波
idsp = "0.15"

# Resampling / 重采样
rubato = "0.18"

# FastICA / 独立成分分析
petal-decomposition = "0.7"

# Statistics / 统计
statrs = "0.18"

# Error handling (application) / 错误处理（应用程序）
anyhow = "1.0"

# Error handling (library) / 错误处理（库）
thiserror = "2.0"

# Serialization / 序列化
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

---

## 12. Code Examples / 代码示例

### Complete Example: Signal Processing Pipeline
### 完整示例: 信号处理管道

```rust
//! EEG Signal Processing Example
//! EEG 信号处理示例

use std::error::Error;

// ============================================
// Constants / 常量
// ============================================

/// Default sampling rate in Hz.
/// 默认采样率（赫兹）。
const DEFAULT_SAMPLE_RATE: f64 = 256.0;

/// Maximum number of channels.
/// 最大通道数。
const MAX_CHANNELS: usize = 64;

// ============================================
// Error Types / 错误类型
// ============================================

/// Processing error types.
/// 处理错误类型。
#[derive(Debug, Clone)]
pub enum ProcessingError {
    EmptyData,
    InvalidSampleRate(f64),
    ChannelMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for ProcessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyData => write!(f, "Empty data"),
            Self::InvalidSampleRate(r) => write!(f, "Invalid rate: {}", r),
            Self::ChannelMismatch { expected, actual } => {
                write!(f, "Channel mismatch: {} vs {}", expected, actual)
            }
        }
    }
}

impl std::error::Error for ProcessingError {}

// ============================================
// Data Structures / 数据结构
// ============================================

/// EEG signal buffer.
/// EEG 信号缓冲区。
pub struct EegBuffer {
    /// Sample data [channels][samples].
    /// 采样数据 [通道][采样点]。
    pub data: Vec<Vec<f64>>,
    
    /// Sampling rate in Hz.
    /// 采样率（赫兹）。
    pub sample_rate: f64,
    
    /// Channel names.
    /// 通道名称。
    pub channel_names: Vec<String>,
}

impl EegBuffer {
    /// Create a new EEG buffer.
    /// 创建新的 EEG 缓冲区。
    pub fn new(
        channel_count: usize,
        sample_rate: f64,
    ) -> Result<Self, ProcessingError> {
        // Validate sample rate.
        // 验证采样率。
        if sample_rate <= 0.0 {
            return Err(ProcessingError::InvalidSampleRate(sample_rate));
        }
        
        // Initialize channels.
        // 初始化通道。
        let data: Vec<Vec<f64>> = vec![Vec::new(); channel_count];
        
        // Generate default channel names.
        // 生成默认通道名称。
        let channel_names: Vec<String> = (0..channel_count)
            .map(|i| format!("CH{}", i + 1))
            .collect();
        
        Ok(Self {
            data,
            sample_rate,
            channel_names,
        })
    }
    
    /// Get number of channels.
    /// 获取通道数。
    pub fn channel_count(&self) -> usize {
        self.data.len()
    }
    
    /// Get number of samples per channel.
    /// 获取每通道采样点数。
    pub fn sample_count(&self) -> usize {
        self.data.first().map(|c| c.len()).unwrap_or(0)
    }
}

// ============================================
// Processing Functions / 处理函数
// ============================================

/// Calculate mean of signal.
/// 计算信号均值。
pub fn calculate_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Apply baseline correction.
/// 应用基线校正。
pub fn baseline_correct(
    data: &[f64],
    baseline_samples: usize,
) -> Result<Vec<f64>, ProcessingError> {
    // Check for empty data.
    // 检查空数据。
    if data.is_empty() {
        return Err(ProcessingError::EmptyData);
    }
    
    // Calculate baseline mean.
    // 计算基线均值。
    let baseline_end: usize = baseline_samples.min(data.len());
    let baseline: &[f64] = &data[..baseline_end];
    let baseline_mean: f64 = calculate_mean(baseline);
    
    // Subtract baseline.
    // 减去基线。
    let corrected: Vec<f64> = data
        .iter()
        .map(|&sample| sample - baseline_mean)
        .collect();
    
    Ok(corrected)
}

/// Downsample signal by integer factor.
/// 按整数因子降采样信号。
pub fn downsample(data: &[f64], factor: usize) -> Vec<f64> {
    if factor == 0 {
        return data.to_vec();
    }
    
    // Take every Nth sample.
    // 取每 N 个采样点。
    data.iter()
        .step_by(factor)
        .cloned()
        .collect()
}

// ============================================
// Main Entry Point / 主入口
// ============================================

fn main() -> Result<(), Box<dyn Error>> {
    // Create buffer.
    // 创建缓冲区。
    let mut buffer: EegBuffer = EegBuffer::new(32, DEFAULT_SAMPLE_RATE)?;
    
    // Add some test data.
    // 添加测试数据。
    for channel in buffer.data.iter_mut() {
        for i in 0..1000 {
            let sample: f64 = (i as f64 * 0.1).sin();
            channel.push(sample);
        }
    }
    
    // Process first channel.
    // 处理第一通道。
    if let Some(channel_data) = buffer.data.first() {
        // Apply baseline correction.
        // 应用基线校正。
        let corrected: Vec<f64> = baseline_correct(channel_data, 100)?;
        
        // Downsample.
        // 降采样。
        let downsampled: Vec<f64> = downsample(&corrected, 2);
        
        println!("Original samples: {}", channel_data.len());
        println!("Downsampled samples: {}", downsampled.len());
    }
    
    Ok(())
}
```

---

## Quick Reference Card / 快速参考卡

```
┌─────────────────────────────────────────────────────────────────┐
│                    RUST AI CODING CHECKLIST                     │
│                    RUST AI 编码检查清单                          │
├─────────────────────────────────────────────────────────────────┤
│ ✅ Types annotated          ✅ 类型已标注                        │
│ ✅ No unwrap()              ✅ 无 unwrap()                       │
│ ✅ Bilingual comments       ✅ 双语注释                          │
│ ✅ Named constants          ✅ 命名常量                          │
│ ✅ Error types defined      ✅ 错误类型已定义                     │
│ ✅ Functions < 50 lines     ✅ 函数 < 50 行                      │
│ ✅ No complex macros        ✅ 无复杂宏                          │
│ ✅ No unsafe blocks         ✅ 无 unsafe 块                      │
│ ✅ Structs own data         ✅ 结构体拥有数据                     │
│ ✅ cargo fmt applied        ✅ 已应用 cargo fmt                  │
│ ✅ cargo clippy clean       ✅ cargo clippy 无警告               │
└─────────────────────────────────────────────────────────────────┘
```

---

*Document Version: 1.0.0 / 文档版本: 1.0.0*  
*Last Updated: 2026-02-01 / 最后更新: 2026-02-01*  
*For BCIF Project / 用于 BCIF 项目*
