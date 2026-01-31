# lsl - Lab Streaming Layer 实时通信

> **Crate**: [lsl](https://crates.io/crates/lsl)  
> **GitHub**: https://github.com/labstreaminglayer/liblsl-rust

> **类型**: 官方 FFI 绑定  
> **License**: MIT  
> **维护者**: Lab Streaming Layer 官方组织

---

## 1. 概述 / Overview

**lsl** 是 Lab Streaming Layer (LSL) 的官方 Rust 绑定，用于高性能实时神经生理数据传输。LSL 是神经科学实验中多设备时间同步的事实标准。

**lsl** is the official Rust binding for Lab Streaming Layer (LSL), designed for high-performance real-time neurophysiological data transmission. LSL is the de facto standard for multi-device time synchronization in neuroscience experiments.

### 适用场景 / Use Cases

| 场景 | 适用性 | 说明 |
|------|--------|------|
| 实时 EEG 采集 | ✅ 推荐 | 亚毫秒级延迟 |
| 多设备同步 | ✅ 推荐 | 自动时钟同步 |
| 网络数据传输 | ✅ 推荐 | 支持局域网广播 |
| 嵌入式系统 | ⚠️ 需评估 | 需要 liblsl 动态库 |
| WASM | ❌ 不适用 | C FFI 不兼容 |

---

## 2. 安装 / Installation

### 2.1 系统依赖

**lsl crate 依赖于本地安装的 liblsl 动态库**：

#### macOS

```bash
# 使用 Homebrew
brew install labstreaminglayer/tap/lsl

# 或下载预编译包
# https://github.com/sccn/liblsl/releases
```

#### Linux (Ubuntu/Debian)

```bash
# 添加 PPA
sudo add-apt-repository ppa:labstreaminglayer/liblsl
sudo apt-get update
sudo apt-get install liblsl-dev

# 或从源码编译
git clone https://github.com/sccn/liblsl.git
cd liblsl
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

#### Windows

```powershell
# 下载预编译 DLL
# https://github.com/sccn/liblsl/releases
# 将 lsl.dll 放入系统 PATH 或项目目录
```

### 2.2 Cargo.toml

```toml
[dependencies]
lsl = "0.4"  # 检查最新版本
```

### 2.3 验证安装

```rust
use lsl::lsl_library_version;

fn main() {
    let version = lsl_library_version();
    println!("liblsl version: {}", version);
}
```

---

## 3. LSL 核心概念 / LSL Core Concepts

```
┌─────────────────────────────────────────────────────────────────┐
│                    LSL 网络架构                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐   │
│   │ EEG 设备    │       │ 眼动仪      │       │ Marker      │   │
│   │ (Outlet)    │       │ (Outlet)    │       │ (Outlet)    │   │
│   └──────┬──────┘       └──────┬──────┘       └──────┬──────┘   │
│          │                     │                     │          │
│          └─────────────────────┼─────────────────────┘          │
│                                │                                │
│                    ┌───────────┴───────────┐                    │
│                    │    LSL 网络层         │                    │
│                    │  (自动发现 + 同步)     │                    │
│                    └───────────┬───────────┘                    │
│                                │                                │
│          ┌─────────────────────┼─────────────────────┐          │
│          │                     │                     │          │
│   ┌──────┴──────┐       ┌──────┴──────┐       ┌──────┴──────┐   │
│   │ 录制程序    │       │ 实时处理    │       │ 可视化      │   │
│   │ (Inlet)     │       │ (Inlet)     │       │ (Inlet)     │   │
│   └─────────────┘       └─────────────┘       └─────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 关键组件

| 组件 | 描述 |
|------|------|
| **StreamInfo** | 流的元数据描述（名称、类型、通道数、采样率等） |
| **StreamOutlet** | 数据发送端（生产者） |
| **StreamInlet** | 数据接收端（消费者） |
| **Resolver** | 流发现机制 |

---

## 4. 核心 API / Core API

### 4.1 创建 StreamOutlet（发送数据）

```rust
use lsl::{StreamInfo, StreamOutlet, ChannelFormat};

fn create_eeg_outlet() -> Result<StreamOutlet, Box<dyn std::error::Error>> {
    // 创建流信息
    let info = StreamInfo::new(
        "MyEEG",           // 流名称
        "EEG",             // 流类型
        8,                 // 通道数
        256.0,             // 采样率 (Hz)
        ChannelFormat::Float32,  // 数据格式
        "myuid12345"       // 唯一标识符
    )?;
    
    // 添加通道元数据（可选）
    let channels = info.desc()?.append_child("channels")?;
    for i in 0..8 {
        let ch = channels.append_child("channel")?;
        ch.append_child_value("label", &format!("Ch{}", i + 1))?;
        ch.append_child_value("unit", "microvolts")?;
        ch.append_child_value("type", "EEG")?;
    }
    
    // 创建 Outlet
    let outlet = StreamOutlet::new(&info, 0, 360)?;
    
    println!("Outlet created. Waiting for consumers...");
    Ok(outlet)
}
```

### 4.2 发送数据

```rust
use lsl::StreamOutlet;
use std::time::{Duration, Instant};

fn send_data(outlet: &StreamOutlet, duration_secs: f64, sample_rate: f64) {
    let n_channels = 8;
    let sample_interval = Duration::from_secs_f64(1.0 / sample_rate);
    let start = Instant::now();
    
    let mut sample = vec![0.0f32; n_channels];
    let mut sample_idx = 0u64;
    
    while start.elapsed().as_secs_f64() < duration_secs {
        // 生成模拟数据
        let t = sample_idx as f64 / sample_rate;
        for (i, s) in sample.iter_mut().enumerate() {
            // 10 Hz 正弦波 + 噪声
            *s = (10.0 * 2.0 * std::f64::consts::PI * t).sin() as f32 
                 + (i as f32 * 0.1);
        }
        
        // 发送样本
        outlet.push_sample(&sample).ok();
        
        sample_idx += 1;
        std::thread::sleep(sample_interval);
    }
}
```

### 4.3 发现流

```rust
use lsl::{resolve_streams, resolve_stream};

fn discover_streams() -> Result<(), Box<dyn std::error::Error>> {
    // 发现所有流
    println!("Searching for streams...");
    let streams = resolve_streams(1.0)?;  // 等待 1 秒
    
    println!("Found {} streams:", streams.len());
    for stream in &streams {
        println!("  - {} ({}): {} channels @ {} Hz",
            stream.name()?,
            stream.stream_type()?,
            stream.channel_count(),
            stream.nominal_srate()
        );
    }
    
    // 按类型查找特定流
    let eeg_streams = resolve_stream("type", "EEG", 1, 5.0)?;
    println!("Found {} EEG streams", eeg_streams.len());
    
    Ok(())
}
```

### 4.4 创建 StreamInlet（接收数据）

```rust
use lsl::{StreamInlet, StreamInfo, resolve_stream};

fn create_eeg_inlet() -> Result<StreamInlet, Box<dyn std::error::Error>> {
    // 查找 EEG 流
    let streams = resolve_stream("type", "EEG", 1, 5.0)?;
    
    if streams.is_empty() {
        return Err("No EEG stream found".into());
    }
    
    let info = &streams[0];
    println!("Connecting to: {} @ {} Hz", info.name()?, info.nominal_srate());
    
    // 创建 Inlet
    let inlet = StreamInlet::new(info, 360, 0, true)?;
    
    // 打开连接
    inlet.open_stream()?;
    
    Ok(inlet)
}
```

### 4.5 接收数据

```rust
use lsl::StreamInlet;
use ndarray::Array2;

fn receive_data(inlet: &StreamInlet, duration_secs: f64) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let n_channels = inlet.info()?.channel_count() as usize;
    let sample_rate = inlet.info()?.nominal_srate();
    let expected_samples = (duration_secs * sample_rate) as usize;
    
    let mut all_data: Vec<Vec<f32>> = Vec::with_capacity(expected_samples);
    let mut timestamps: Vec<f64> = Vec::with_capacity(expected_samples);
    
    let mut sample = vec![0.0f32; n_channels];
    
    let start = std::time::Instant::now();
    while start.elapsed().as_secs_f64() < duration_secs {
        // 拉取单个样本（带时间戳）
        if let Ok(ts) = inlet.pull_sample(&mut sample, 0.2) {
            if ts > 0.0 {
                all_data.push(sample.clone());
                timestamps.push(ts);
            }
        }
    }
    
    // 转换为 ndarray
    let n_samples = all_data.len();
    let mut data = Array2::<f64>::zeros((n_channels, n_samples));
    
    for (i, sample) in all_data.iter().enumerate() {
        for (ch, &value) in sample.iter().enumerate() {
            data[[ch, i]] = value as f64;
        }
    }
    
    Ok(data)
}
```

### 4.6 批量接收（更高效）

```rust
fn receive_chunk(inlet: &StreamInlet, chunk_size: usize) -> Result<(Vec<Vec<f32>>, Vec<f64>), Box<dyn std::error::Error>> {
    let n_channels = inlet.info()?.channel_count() as usize;
    
    let mut buffer = vec![vec![0.0f32; n_channels]; chunk_size];
    let mut timestamps = vec![0.0f64; chunk_size];
    
    // 批量拉取
    let pulled = inlet.pull_chunk(&mut buffer, &mut timestamps, 1.0)?;
    
    // 截取实际接收的数据
    buffer.truncate(pulled);
    timestamps.truncate(pulled);
    
    Ok((buffer, timestamps))
}
```

---

## 5. BCIF 集成示例 / BCIF Integration

### 5.1 实时 EEG 处理管道

```rust
use lsl::{StreamInlet, resolve_stream};
use ndarray::Array1;
use std::sync::mpsc;
use std::thread;

/// 实时处理器 trait
pub trait RealtimeProcessor: Send {
    fn process(&mut self, sample: &[f64], timestamp: f64);
}

/// 启动实时 LSL 处理管道
pub fn start_realtime_pipeline<P: RealtimeProcessor + 'static>(
    stream_type: &str,
    mut processor: P,
) -> Result<mpsc::Sender<()>, Box<dyn std::error::Error>> {
    // 查找流
    let streams = resolve_stream("type", stream_type, 1, 10.0)?;
    if streams.is_empty() {
        return Err(format!("No {} stream found", stream_type).into());
    }
    
    let inlet = StreamInlet::new(&streams[0], 360, 0, true)?;
    inlet.open_stream()?;
    
    let n_channels = inlet.info()?.channel_count() as usize;
    
    // 创建停止信号通道
    let (stop_tx, stop_rx) = mpsc::channel();
    
    // 启动处理线程
    thread::spawn(move || {
        let mut sample = vec![0.0f32; n_channels];
        let mut sample_f64 = vec![0.0f64; n_channels];
        
        loop {
            // 检查停止信号
            if stop_rx.try_recv().is_ok() {
                break;
            }
            
            // 拉取样本
            if let Ok(ts) = inlet.pull_sample(&mut sample, 0.1) {
                if ts > 0.0 {
                    // 转换为 f64
                    for (i, &v) in sample.iter().enumerate() {
                        sample_f64[i] = v as f64;
                    }
                    
                    // 处理
                    processor.process(&sample_f64, ts);
                }
            }
        }
    });
    
    Ok(stop_tx)
}
```

### 5.2 实时滤波示例

```rust
use idsp::iir::*;

/// 实时 IIR 滤波器
pub struct RealtimeFilter {
    filters: Vec<Biquad<f64>>,
    states: Vec<[f64; 2]>,
}

impl RealtimeFilter {
    pub fn new_bandpass(n_channels: usize, sample_rate: f64, low: f64, high: f64) -> Self {
        // 创建 Butterworth 带通滤波器系数
        let coeffs = Biquad::from_normalized_coefficients(
            // 这里需要实际的滤波器设计
            [1.0, 0.0, -1.0],  // 示例系数
            [1.0, -1.5, 0.7],
        );
        
        Self {
            filters: vec![coeffs; n_channels],
            states: vec![[0.0; 2]; n_channels],
        }
    }
    
    pub fn filter_sample(&mut self, sample: &mut [f64]) {
        for (i, (filter, state)) in self.filters.iter().zip(self.states.iter_mut()).enumerate() {
            sample[i] = filter.update(state, sample[i]);
        }
    }
}

impl RealtimeProcessor for RealtimeFilter {
    fn process(&mut self, sample: &[f64], _timestamp: f64) {
        let mut filtered = sample.to_vec();
        self.filter_sample(&mut filtered);
        // 输出或进一步处理...
    }
}
```

### 5.3 发送处理后的数据

```rust
use lsl::{StreamInfo, StreamOutlet, ChannelFormat};

/// 创建处理后数据的输出流
pub struct ProcessedDataOutlet {
    outlet: StreamOutlet,
    n_channels: usize,
}

impl ProcessedDataOutlet {
    pub fn new(name: &str, n_channels: usize, sample_rate: f64) -> Result<Self, Box<dyn std::error::Error>> {
        let info = StreamInfo::new(
            name,
            "ProcessedEEG",
            n_channels as i32,
            sample_rate,
            ChannelFormat::Float64,
            &format!("{}_processed", name),
        )?;
        
        let outlet = StreamOutlet::new(&info, 0, 360)?;
        
        Ok(Self { outlet, n_channels })
    }
    
    pub fn push(&self, sample: &[f64]) {
        self.outlet.push_sample(sample).ok();
    }
}
```

---

## 6. 时间同步 / Time Synchronization

LSL 的核心优势是精确的时间同步：

### 6.1 获取本地时钟

```rust
use lsl::local_clock;

fn get_lsl_time() -> f64 {
    local_clock()  // 返回 LSL 时间戳（秒）
}
```

### 6.2 时间戳校正

```rust
use lsl::StreamInlet;

fn get_time_correction(inlet: &StreamInlet) -> Result<f64, Box<dyn std::error::Error>> {
    // 获取时钟偏移
    let correction = inlet.time_correction(5.0)?;
    
    println!("Time correction: {} seconds", correction);
    
    // 校正后的时间戳 = 原始时间戳 + correction
    Ok(correction)
}
```

### 6.3 多流同步示例

```rust
use std::collections::HashMap;

/// 多流同步接收器
pub struct SyncedReceiver {
    inlets: HashMap<String, StreamInlet>,
    buffers: HashMap<String, Vec<(Vec<f64>, f64)>>,
}

impl SyncedReceiver {
    pub fn new(stream_names: &[&str]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut inlets = HashMap::new();
        let mut buffers = HashMap::new();
        
        for name in stream_names {
            let streams = lsl::resolve_stream("name", name, 1, 5.0)?;
            if let Some(info) = streams.first() {
                let inlet = StreamInlet::new(info, 360, 0, true)?;
                inlet.open_stream()?;
                inlets.insert(name.to_string(), inlet);
                buffers.insert(name.to_string(), Vec::new());
            }
        }
        
        Ok(Self { inlets, buffers })
    }
    
    /// 接收所有流的数据，按时间戳对齐
    pub fn receive_synced(&mut self, duration: f64) {
        // 实现多流同步接收逻辑...
    }
}
```

---

## 7. 常见问题 / Common Issues

### 7.1 找不到 liblsl

```
error: linking with `cc` failed: exit code: 1
  = note: ld: library not found for -llsl
```

**解决方案**：
```bash
# macOS
export LIBRARY_PATH="/opt/homebrew/lib:$LIBRARY_PATH"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"

# Linux
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
sudo ldconfig
```

### 7.2 运行时找不到库

```
dyld: Library not loaded: @rpath/liblsl.dylib
```

**解决方案**：
- 确保 liblsl 在系统库路径中
- 或将 `.dylib`/`.so`/`.dll` 复制到可执行文件同目录

### 7.3 网络发现问题

```rust
// 如果自动发现不工作，可以直接指定地址
// 需要知道发送方的 IP 和端口
```

---

## 8. 注意事项 / Considerations

### 8.1 优势

- ✅ **官方维护**: 由 LSL 官方组织维护
- ✅ **高性能**: 亚毫秒级延迟
- ✅ **时间同步**: 自动时钟同步，跨设备精度高
- ✅ **网络透明**: 自动发现，无需配置 IP
- ✅ **广泛兼容**: 与所有 LSL 生态系统工具兼容

### 8.2 限制

- ⚠️ **系统依赖**: 必须安装 liblsl 动态库
- ⚠️ **跨平台**: 需要为每个平台准备 liblsl
- ⚠️ **嵌入式**: 对于资源受限设备可能不适合
- ⚠️ **WASM 不可用**: 无法编译到 WebAssembly

### 8.3 最佳实践

```rust
// 1. 始终检查流是否可用
// 2. 使用适当的缓冲区大小
// 3. 在独立线程中接收数据
// 4. 定期检查时间校正
// 5. 优雅处理连接断开
```

---

## 9. 相关资源 / Resources

- [LSL 官网](https://labstreaminglayer.org/)
- [liblsl GitHub](https://github.com/sccn/liblsl)
- [LSL 文档](https://labstreaminglayer.readthedocs.io/)
- [LSL 应用列表](https://labstreaminglayer.readthedocs.io/info/supported_devices.html)

---

## 10. 版本对应 / Version Compatibility

| lsl crate | liblsl 版本 |
|-----------|------------|
| 0.4.x | liblsl 1.16+ |
| 0.3.x | liblsl 1.14+ |

> **注**: 请查阅 [crates.io](https://crates.io/crates/lsl) 获取最新版本信息。
