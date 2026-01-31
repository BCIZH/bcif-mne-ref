# edflib - EDF/BDF 文件解析与写入

> **Crate**: [edflib](https://crates.io/crates/edflib)  
> **GitHub**: https://github.com/eugenehp/edflib-rs

> **类型**: C 库封装 (FFI Wrapper)  
> **License**: MIT

---

## 1. 概述 / Overview

**edflib** 是对经典 C 语言库 [edflib](https://www.teuniz.net/edflib/) 的 Rust 封装，是处理 EDF (European Data Format) 和 BDF (BioSemi Data Format) 文件最稳健的选择。

**edflib** is a Rust wrapper around the classic C library [edflib](https://www.teuniz.net/edflib/), providing the most robust solution for reading and writing EDF/BDF files.

### 适用场景 / Use Cases

| 场景 | 适用性 | 说明 |
|------|--------|------|
| EDF 文件读取 | ✅ 推荐 | 完整支持 EDF/EDF+ 格式 |
| BDF 文件读取 | ✅ 推荐 | 完整支持 BDF/BDF+ 格式 |
| EDF 文件写入 | ✅ 推荐 | 创建符合标准的 EDF 文件 |
| 嵌入式系统 | ⚠️ 需评估 | 依赖 C 库，需交叉编译支持 |
| WASM | ❌ 不适用 | C FFI 不兼容 WASM |

---

## 2. 安装 / Installation

### Cargo.toml

```toml
[dependencies]
edflib = "0.4"  # 检查最新版本

# 如果需要底层 FFI 绑定
# edflib-sys = "0.4"
```

### 系统依赖

edflib crate 会自动编译并链接 C 库，通常无需额外安装。但需要：

- C 编译器 (gcc, clang, MSVC)
- CMake (可选，取决于版本)

---

## 3. 核心 API / Core API

### 3.1 读取 EDF 文件

```rust
use edflib::{EdfReader, EdfError};

fn read_edf_file(path: &str) -> Result<(), EdfError> {
    // 打开 EDF 文件
    let reader = EdfReader::open(path)?;
    
    // 获取文件信息
    println!("信号数量: {}", reader.num_signals());
    println!("数据记录数: {}", reader.num_data_records());
    println!("记录时长: {} 秒", reader.data_record_duration());
    
    // 遍历信号通道
    for i in 0..reader.num_signals() {
        let signal = reader.signal(i)?;
        println!("通道 {}: {}", i, signal.label());
        println!("  采样率: {} Hz", signal.sample_rate());
        println!("  物理单位: {}", signal.physical_dimension());
        println!("  物理范围: {} ~ {}", signal.physical_minimum(), signal.physical_maximum());
    }
    
    Ok(())
}
```

### 3.2 读取信号数据

```rust
use edflib::EdfReader;
use ndarray::Array2;

fn read_signal_data(path: &str) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let reader = EdfReader::open(path)?;
    
    let n_channels = reader.num_signals();
    let n_samples = reader.num_data_records() * reader.signal(0)?.samples_per_data_record();
    
    // 分配数据数组
    let mut data = Array2::<f64>::zeros((n_channels, n_samples));
    
    // 读取每个通道的数据
    for ch in 0..n_channels {
        let signal = reader.signal(ch)?;
        let samples = signal.read_physical_samples(0, n_samples)?;
        
        for (i, &sample) in samples.iter().enumerate() {
            data[[ch, i]] = sample;
        }
    }
    
    Ok(data)
}
```

### 3.3 写入 EDF 文件

```rust
use edflib::{EdfWriter, EdfError};

fn write_edf_file(
    path: &str,
    data: &[Vec<f64>],
    sample_rate: f64,
    channel_names: &[&str],
) -> Result<(), EdfError> {
    let n_channels = data.len();
    
    // 创建 EDF 写入器
    let mut writer = EdfWriter::create(path, n_channels)?;
    
    // 设置通道信息
    for (i, name) in channel_names.iter().enumerate() {
        writer.set_label(i, name)?;
        writer.set_physical_dimension(i, "uV")?;
        writer.set_physical_minimum(i, -3200.0)?;
        writer.set_physical_maximum(i, 3200.0)?;
        writer.set_digital_minimum(i, -32768)?;
        writer.set_digital_maximum(i, 32767)?;
        writer.set_sample_rate(i, sample_rate as i32)?;
    }
    
    // 设置患者/记录信息（可选）
    writer.set_patient_name("Subject001")?;
    writer.set_recording_additional("EEG Recording")?;
    
    // 写入数据
    // 注：具体 API 可能因版本而异，参考最新文档
    for samples in data.iter() {
        writer.write_physical_samples(samples)?;
    }
    
    writer.close()?;
    Ok(())
}
```

---

## 4. BCIF 集成示例 / BCIF Integration

### 4.1 与 ndarray 集成

```rust
use edflib::EdfReader;
use ndarray::{Array2, Axis};

/// EDF 文件信息
pub struct EdfInfo {
    pub sample_rate: f64,
    pub channel_names: Vec<String>,
    pub n_samples: usize,
    pub patient_id: String,
    pub recording_date: String,
}

/// 读取 EDF 文件，返回 ndarray 格式数据
pub fn read_edf(path: &str) -> Result<(Array2<f64>, EdfInfo), Box<dyn std::error::Error>> {
    let reader = EdfReader::open(path)?;
    
    let n_channels = reader.num_signals();
    let samples_per_record = reader.signal(0)?.samples_per_data_record();
    let n_records = reader.num_data_records();
    let n_samples = n_records * samples_per_record;
    
    // 收集元数据
    let mut channel_names = Vec::with_capacity(n_channels);
    let mut sample_rate = 0.0;
    
    for i in 0..n_channels {
        let signal = reader.signal(i)?;
        channel_names.push(signal.label().to_string());
        if i == 0 {
            sample_rate = signal.sample_rate();
        }
    }
    
    // 读取数据到 ndarray
    let mut data = Array2::<f64>::zeros((n_channels, n_samples));
    
    for ch in 0..n_channels {
        let signal = reader.signal(ch)?;
        let samples = signal.read_physical_samples(0, n_samples)?;
        data.row_mut(ch).assign(&ndarray::Array1::from_vec(samples));
    }
    
    let info = EdfInfo {
        sample_rate,
        channel_names,
        n_samples,
        patient_id: reader.patient().unwrap_or_default().to_string(),
        recording_date: reader.start_date().unwrap_or_default().to_string(),
    };
    
    Ok((data, info))
}
```

### 4.2 批量处理 EDF 文件

```rust
use std::path::Path;
use rayon::prelude::*;

/// 并行读取多个 EDF 文件
pub fn batch_read_edf<P: AsRef<Path> + Sync>(
    paths: &[P]
) -> Vec<Result<(Array2<f64>, EdfInfo), Box<dyn std::error::Error + Send + Sync>>> {
    paths.par_iter()
        .map(|path| {
            let path_str = path.as_ref().to_str().ok_or("Invalid path")?;
            read_edf(path_str).map_err(|e| e.into())
        })
        .collect()
}
```

---

## 5. EDF/BDF 格式说明 / Format Reference

### 5.1 EDF 文件结构

```
┌─────────────────────────────────────────────────────────────┐
│ Header (256 bytes)                                          │
│   - version (8 bytes): "0       "                           │
│   - patient_id (80 bytes)                                   │
│   - recording_id (80 bytes)                                 │
│   - start_date (8 bytes): dd.mm.yy                          │
│   - start_time (8 bytes): hh.mm.ss                          │
│   - header_bytes (8 bytes)                                  │
│   - reserved (44 bytes)                                     │
│   - num_data_records (8 bytes)                              │
│   - data_record_duration (8 bytes)                          │
│   - num_signals (4 bytes)                                   │
├─────────────────────────────────────────────────────────────┤
│ Signal Headers (256 × num_signals bytes)                    │
│   每个信号:                                                  │
│   - label (16 bytes)                                        │
│   - transducer (80 bytes)                                   │
│   - physical_dimension (8 bytes)                            │
│   - physical_minimum (8 bytes)                              │
│   - physical_maximum (8 bytes)                              │
│   - digital_minimum (8 bytes)                               │
│   - digital_maximum (8 bytes)                               │
│   - prefiltering (80 bytes)                                 │
│   - samples_per_data_record (8 bytes)                       │
│   - reserved (32 bytes)                                     │
├─────────────────────────────────────────────────────────────┤
│ Data Records                                                │
│   每条记录包含所有通道的采样数据                              │
│   EDF: 16-bit signed integers (int16)                       │
│   BDF: 24-bit signed integers (int24)                       │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 数值转换公式

```
物理值 = (数字值 - digital_min) × (physical_max - physical_min) 
         / (digital_max - digital_min) + physical_min
```

### 5.3 EDF vs BDF 对比

| 特性 | EDF | BDF |
|------|-----|-----|
| 数据位宽 | 16-bit | 24-bit |
| 动态范围 | ±32,767 | ±8,388,607 |
| 分辨率 | 较低 | 较高 |
| 文件大小 | 较小 | 较大 |
| 常见用途 | 一般 EEG | BioSemi 设备 |

---

## 6. 注意事项 / Considerations

### 6.1 优势

- ✅ **稳定成熟**: 基于久经考验的 C 库
- ✅ **完整功能**: 支持读写、EDF/BDF、EDF+/BDF+
- ✅ **标准兼容**: 严格遵循 EDF/BDF 规范

### 6.2 限制

- ⚠️ **C 依赖**: 需要 C 编译器，增加构建复杂度
- ⚠️ **跨平台编译**: 交叉编译需要配置 C 工具链
- ⚠️ **WASM 不兼容**: 无法编译到 WebAssembly
- ⚠️ **内存安全**: FFI 边界可能存在风险

### 6.3 替代方案

如果需要纯 Rust 实现：
- 可以考虑自行实现 EDF 解析器（~800行代码）
- 或等待社区开发纯 Rust 方案

---

## 7. 相关资源 / Resources

- [EDF 规范](https://www.edfplus.info/specs/edf.html)
- [EDF+ 规范](https://www.edfplus.info/specs/edfplus.html)
- [BDF 规范](https://www.biosemi.com/faq/file_format.htm)
- [edflib C 库](https://www.teuniz.net/edflib/)

---

## 8. 版本历史 / Changelog

| 版本 | 日期 | 变更 |
|------|------|------|
| 0.4.x | 2024 | 最新稳定版本 |

> **注**: 请查阅 [crates.io](https://crates.io/crates/edflib) 获取最新版本信息。
