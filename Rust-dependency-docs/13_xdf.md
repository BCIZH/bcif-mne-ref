# xdf - XDF 文件解析

> **Crate**: [xdf](https://crates.io/crates/xdf)  
> **GitHub**: https://github.com/Garfield100/xdf_rs

> **类型**: 纯 Rust 实现  
> **License**: MIT&Apache-2.0

---

## 1. 概述 / Overview

**xdf** 是 Rust 生态中解析 XDF (Extensible Data Format) 文件的标准实现。XDF 是 Lab Streaming Layer (LSL) 生态系统的默认存储格式，用于记录多路同步数据流。

**xdf** is the standard Rust implementation for parsing XDF (Extensible Data Format) files. XDF is the default storage format for the Lab Streaming Layer (LSL) ecosystem, used for recording multi-stream synchronized data.

### 适用场景 / Use Cases

| 场景 | 适用性 | 说明 |
|------|--------|------|
| XDF 文件读取 | ✅ 推荐 | 完整支持 XDF 1.0 格式 |
| 多流数据解析 | ✅ 推荐 | 支持多种数据类型的流 |
| 时间戳同步 | ✅ 推荐 | 内置时钟偏移校正 |
| 嵌入式系统 | ⚠️ 需评估 | 纯 Rust，但可能需要 std |
| WASM | ⚠️ 需验证 | 纯 Rust 实现，理论可行 |

---

## 2. 安装 / Installation

### Cargo.toml

```toml
[dependencies]
xdf = "0.3"  # 检查最新版本
```

---

## 3. XDF 格式简介 / XDF Format Overview

XDF 是一种灵活的容器格式，设计用于存储多路时间序列数据：

```
┌─────────────────────────────────────────────────────────────┐
│ XDF 文件结构                                                 │
├─────────────────────────────────────────────────────────────┤
│ Magic Number: "XDF:"                                         │
├─────────────────────────────────────────────────────────────┤
│ FileHeader (XML)                                            │
│   - version                                                  │
│   - datetime                                                 │
├─────────────────────────────────────────────────────────────┤
│ Stream 1: Header + Data Chunks                              │
│   - StreamHeader (XML): name, type, channel_count, srate... │
│   - DataChunk[]                                             │
│   - ClockOffset[]                                           │
├─────────────────────────────────────────────────────────────┤
│ Stream 2: Header + Data Chunks                              │
│   ...                                                        │
├─────────────────────────────────────────────────────────────┤
│ Stream N: Header + Data Chunks                              │
│   ...                                                        │
├─────────────────────────────────────────────────────────────┤
│ Boundary Chunks (同步标记)                                   │
└─────────────────────────────────────────────────────────────┘
```

### 支持的数据类型

| 类型 | 描述 | Rust 类型 |
|------|------|-----------|
| `float32` | 32位浮点 | `f32` |
| `float64` | 64位浮点 | `f64` |
| `int8` | 8位整数 | `i8` |
| `int16` | 16位整数 | `i16` |
| `int32` | 32位整数 | `i32` |
| `int64` | 64位整数 | `i64` |
| `string` | 字符串 | `String` |

---

## 4. 核心 API / Core API

### 4.1 读取 XDF 文件

```rust
use xdf::{XdfReader, XdfError};

fn read_xdf_file(path: &str) -> Result<(), XdfError> {
    // 打开 XDF 文件
    let xdf = XdfReader::open(path)?;
    
    // 获取文件信息
    println!("文件版本: {}", xdf.version());
    println!("流数量: {}", xdf.streams().len());
    
    // 遍历所有流
    for stream in xdf.streams() {
        println!("\n流 ID: {}", stream.id());
        println!("  名称: {}", stream.name());
        println!("  类型: {}", stream.stream_type());
        println!("  通道数: {}", stream.channel_count());
        println!("  采样率: {} Hz", stream.nominal_srate());
        println!("  数据格式: {:?}", stream.channel_format());
        println!("  样本数: {}", stream.sample_count());
        
        // 打印通道信息
        for (i, channel) in stream.channels().iter().enumerate() {
            println!("  通道 {}: {}", i, channel.label().unwrap_or("unnamed"));
        }
    }
    
    Ok(())
}
```

### 4.2 读取流数据

```rust
use xdf::{XdfReader, StreamData};
use ndarray::Array2;

fn read_stream_data(path: &str, stream_name: &str) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let xdf = XdfReader::open(path)?;
    
    // 查找指定名称的流
    let stream = xdf.streams()
        .iter()
        .find(|s| s.name() == stream_name)
        .ok_or("Stream not found")?;
    
    let n_channels = stream.channel_count();
    let n_samples = stream.sample_count();
    
    // 获取数据（根据格式转换）
    let mut data = Array2::<f64>::zeros((n_channels, n_samples));
    
    match stream.data() {
        StreamData::Float64(samples) => {
            for (i, sample) in samples.iter().enumerate() {
                for (ch, &value) in sample.iter().enumerate() {
                    data[[ch, i]] = value;
                }
            }
        }
        StreamData::Float32(samples) => {
            for (i, sample) in samples.iter().enumerate() {
                for (ch, &value) in sample.iter().enumerate() {
                    data[[ch, i]] = value as f64;
                }
            }
        }
        StreamData::Int16(samples) => {
            for (i, sample) in samples.iter().enumerate() {
                for (ch, &value) in sample.iter().enumerate() {
                    data[[ch, i]] = value as f64;
                }
            }
        }
        _ => return Err("Unsupported data format".into()),
    }
    
    Ok(data)
}
```

### 4.3 获取时间戳

```rust
use xdf::XdfReader;

fn get_timestamps(path: &str, stream_name: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let xdf = XdfReader::open(path)?;
    
    let stream = xdf.streams()
        .iter()
        .find(|s| s.name() == stream_name)
        .ok_or("Stream not found")?;
    
    // 获取时间戳（已应用时钟偏移校正）
    let timestamps = stream.timestamps().to_vec();
    
    println!("第一个时间戳: {:.6} s", timestamps.first().unwrap_or(&0.0));
    println!("最后时间戳: {:.6} s", timestamps.last().unwrap_or(&0.0));
    println!("时长: {:.2} s", timestamps.last().unwrap_or(&0.0) - timestamps.first().unwrap_or(&0.0));
    
    Ok(timestamps)
}
```

---

## 5. BCIF 集成示例 / BCIF Integration

### 5.1 完整的 XDF 读取接口

```rust
use xdf::XdfReader;
use ndarray::Array2;
use std::collections::HashMap;

/// XDF 流信息
#[derive(Debug, Clone)]
pub struct XdfStreamInfo {
    pub name: String,
    pub stream_type: String,
    pub channel_count: usize,
    pub sample_rate: f64,
    pub channel_names: Vec<String>,
}

/// XDF 文件数据
pub struct XdfData {
    pub streams: HashMap<String, (Array2<f64>, Vec<f64>, XdfStreamInfo)>,
}

/// 读取 XDF 文件，返回所有数值流
pub fn read_xdf(path: &str) -> Result<XdfData, Box<dyn std::error::Error>> {
    let xdf = XdfReader::open(path)?;
    let mut streams = HashMap::new();
    
    for stream in xdf.streams() {
        // 跳过 marker/string 流
        if matches!(stream.channel_format(), xdf::ChannelFormat::String) {
            continue;
        }
        
        let info = XdfStreamInfo {
            name: stream.name().to_string(),
            stream_type: stream.stream_type().to_string(),
            channel_count: stream.channel_count(),
            sample_rate: stream.nominal_srate(),
            channel_names: stream.channels()
                .iter()
                .map(|c| c.label().unwrap_or("").to_string())
                .collect(),
        };
        
        let n_channels = stream.channel_count();
        let n_samples = stream.sample_count();
        let mut data = Array2::<f64>::zeros((n_channels, n_samples));
        
        // 转换数据到 f64
        match stream.data() {
            xdf::StreamData::Float64(samples) => {
                for (i, sample) in samples.iter().enumerate() {
                    for (ch, &value) in sample.iter().enumerate() {
                        data[[ch, i]] = value;
                    }
                }
            }
            xdf::StreamData::Float32(samples) => {
                for (i, sample) in samples.iter().enumerate() {
                    for (ch, &value) in sample.iter().enumerate() {
                        data[[ch, i]] = value as f64;
                    }
                }
            }
            xdf::StreamData::Int32(samples) => {
                for (i, sample) in samples.iter().enumerate() {
                    for (ch, &value) in sample.iter().enumerate() {
                        data[[ch, i]] = value as f64;
                    }
                }
            }
            xdf::StreamData::Int16(samples) => {
                for (i, sample) in samples.iter().enumerate() {
                    for (ch, &value) in sample.iter().enumerate() {
                        data[[ch, i]] = value as f64;
                    }
                }
            }
            _ => continue,
        }
        
        let timestamps = stream.timestamps().to_vec();
        streams.insert(info.name.clone(), (data, timestamps, info));
    }
    
    Ok(XdfData { streams })
}
```

### 5.2 提取 EEG 流

```rust
/// 从 XDF 文件提取 EEG 数据
pub fn extract_eeg_stream(path: &str) -> Result<(Array2<f64>, Vec<f64>, XdfStreamInfo), Box<dyn std::error::Error>> {
    let xdf_data = read_xdf(path)?;
    
    // 查找 EEG 类型的流
    for (name, (data, timestamps, info)) in xdf_data.streams {
        if info.stream_type.to_lowercase() == "eeg" {
            return Ok((data, timestamps, info));
        }
    }
    
    Err("No EEG stream found in XDF file".into())
}
```

### 5.3 读取 Marker 流

```rust
use xdf::{XdfReader, StreamData};

/// Marker 事件
#[derive(Debug, Clone)]
pub struct Marker {
    pub timestamp: f64,
    pub value: String,
}

/// 从 XDF 文件提取 Marker 事件
pub fn extract_markers(path: &str) -> Result<Vec<Marker>, Box<dyn std::error::Error>> {
    let xdf = XdfReader::open(path)?;
    let mut markers = Vec::new();
    
    for stream in xdf.streams() {
        // 查找 Markers 类型的流
        if stream.stream_type().to_lowercase() != "markers" {
            continue;
        }
        
        if let StreamData::String(samples) = stream.data() {
            let timestamps = stream.timestamps();
            
            for (i, sample) in samples.iter().enumerate() {
                if let Some(&ts) = timestamps.get(i) {
                    markers.push(Marker {
                        timestamp: ts,
                        value: sample[0].clone(),
                    });
                }
            }
        }
    }
    
    // 按时间戳排序
    markers.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());
    
    Ok(markers)
}
```

---

## 6. 时间同步 / Time Synchronization

XDF 格式的一个关键特性是支持多流时间同步：

### 6.1 时钟偏移校正

```rust
use xdf::XdfReader;

fn analyze_clock_offsets(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let xdf = XdfReader::open(path)?;
    
    for stream in xdf.streams() {
        let offsets = stream.clock_offsets();
        
        if !offsets.is_empty() {
            println!("流 '{}' 时钟偏移:", stream.name());
            println!("  偏移数量: {}", offsets.len());
            println!("  第一个偏移: {:.6} s", offsets.first().unwrap().1);
            println!("  最后偏移: {:.6} s", offsets.last().unwrap().1);
            
            // 计算偏移漂移
            if offsets.len() >= 2 {
                let drift = (offsets.last().unwrap().1 - offsets.first().unwrap().1) 
                           / (offsets.last().unwrap().0 - offsets.first().unwrap().0);
                println!("  时钟漂移率: {:.6} s/s", drift);
            }
        }
    }
    
    Ok(())
}
```

### 6.2 多流对齐

```rust
/// 将多个流对齐到统一时间轴
pub fn align_streams(
    stream1: (&Array2<f64>, &[f64]),
    stream2: (&Array2<f64>, &[f64]),
    target_srate: f64,
) -> Result<(Array2<f64>, Array2<f64>), Box<dyn std::error::Error>> {
    let (data1, ts1) = stream1;
    let (data2, ts2) = stream2;
    
    // 找到共同时间范围
    let start_time = ts1.first().unwrap().max(*ts2.first().unwrap());
    let end_time = ts1.last().unwrap().min(*ts2.last().unwrap());
    
    let duration = end_time - start_time;
    let n_samples = (duration * target_srate) as usize;
    
    // 这里需要插值逻辑...
    // 简化示例：实际应用中需要使用适当的插值算法
    
    todo!("实现插值对齐")
}
```

---

## 7. 常见 XDF 数据结构 / Common XDF Patterns

### 7.1 典型 EEG 录制 XDF

```
XDF 文件
├── Stream 1: "EEG" (type: "EEG")
│   ├── 64 通道, 512 Hz
│   ├── 格式: float32
│   └── 通道: Fp1, Fp2, F3, F4, ...
├── Stream 2: "Markers" (type: "Markers")
│   ├── 1 通道, 不规则采样
│   ├── 格式: string
│   └── 值: "stimulus_onset", "response", ...
└── Stream 3: "EyeTracker" (type: "Gaze")
    ├── 2 通道 (x, y), 120 Hz
    └── 格式: float64
```

### 7.2 提取特定时间段

```rust
/// 根据 Marker 提取 Epoch
pub fn extract_epochs_by_markers(
    data: &Array2<f64>,
    timestamps: &[f64],
    markers: &[Marker],
    marker_filter: &str,
    pre_time: f64,   // 事件前时间（秒）
    post_time: f64,  // 事件后时间（秒）
    sample_rate: f64,
) -> Vec<Array2<f64>> {
    let pre_samples = (pre_time * sample_rate) as usize;
    let post_samples = (post_time * sample_rate) as usize;
    let epoch_len = pre_samples + post_samples;
    
    let mut epochs = Vec::new();
    
    for marker in markers {
        if marker.value != marker_filter {
            continue;
        }
        
        // 找到最近的样本索引
        if let Some(idx) = timestamps.iter()
            .position(|&t| t >= marker.timestamp) {
            
            if idx >= pre_samples && idx + post_samples <= timestamps.len() {
                let epoch = data.slice(ndarray::s![.., idx - pre_samples..idx + post_samples]).to_owned();
                epochs.push(epoch);
            }
        }
    }
    
    epochs
}
```

---

## 8. 注意事项 / Considerations

### 8.1 优势

- ✅ **纯 Rust**: 无外部依赖，编译简单
- ✅ **完整解析**: 支持所有 XDF 数据类型
- ✅ **时间同步**: 内置时钟偏移处理
- ✅ **LSL 兼容**: 与 LSL 生态系统无缝集成

### 8.2 限制

- ⚠️ **仅读取**: 目前不支持写入 XDF 文件
- ⚠️ **内存占用**: 大文件需要足够内存
- ⚠️ **API 稳定性**: 相对较新的 crate，API 可能变化

### 8.3 大文件处理建议

```rust
// 对于大文件，考虑分块处理
// 或使用内存映射（如果 crate 支持）
```

---

## 9. 相关资源 / Resources

- [XDF 格式规范](https://github.com/sccn/xdf/wiki/Specifications)
- [LSL 官网](https://labstreaminglayer.org/)
- [pyXDF (Python 参考实现)](https://github.com/xdf-modules/pyxdf)

---

## 10. 版本历史 / Changelog

| 版本 | 日期 | 变更 |
|------|------|------|
| 0.3.x | 2024 | 最新稳定版本 |

> **注**: 请查阅 [crates.io](https://crates.io/crates/xdf) 获取最新版本信息。
