# BCIF Layer 0: Data I/O 架构设计

> **Status**: 架构设计阶段
> **Version**: 0.1.0
> **Date**: 2026-02-01
> **Purpose**: Layer 0 数据采集与输出层的详细架构

---

## 1. 设计背景与约束

### 1.1 硬件约束

| 项目 | 规格 |
|------|------|
| 设备类型 | 自研 EEG 硬件 |
| ADC 芯片 | TI ADS129x 系列 (24-bit) |
| 传输方式 | 有线 (USB/串口)、BLE |
| 协议状态 | **未定义** - 可软硬件协同设计 |

### 1.2 软件约束

| 项目 | 规格 |
|------|------|
| 目标平台 | Windows、macOS、Linux |
| 输出格式 | ndarray、EDF、XDF、LSL 流 |
| 语言 | Rust (纯 Rust 优先) |

---

## 2. 分层架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Layer 0: Data I/O 内部分层                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ L0.4: Output Adapters (输出适配器)                               │   │
│  │                                                                   │   │
│  │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │   │
│  │   │ ndarray  │ │ EDF      │ │ XDF      │ │ LSL      │          │   │
│  │   │ 内存输出 │ │ 文件写入 │ │ 文件写入 │ │ 流发布   │          │   │
│  │   └──────────┘ └──────────┘ └──────────┘ └──────────┘          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↑                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ L0.3: Unit Conversion (单位转换)                                 │   │
│  │                                                                   │   │
│  │   ADC Raw (24-bit signed) → Physical Units (μV)                  │   │
│  │   考虑: 增益(Gain)、参考电压(Vref)、通道校准(Calibration)         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↑                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ L0.2: Protocol Parser (协议解析)                                 │   │
│  │                                                                   │   │
│  │   字节流 → 结构化数据包                                           │   │
│  │   包头解析、数据提取、校验验证、时间戳同步                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↑                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ L0.1: Transport Abstraction (传输抽象)                           │   │
│  │                                                                   │   │
│  │   ┌──────────┐ ┌──────────┐                                    │   │
│  │   │ Serial   │ │ BLE      │                                    │   │
│  │   │ USB/UART │ │ 低功耗   │                                    │   │
│  │   └──────────┘ └──────────┘                                    │   │
│  │                                                                   │   │
│  │   统一 trait: DataTransport                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↑                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ L0.0: Physical Layer (物理层) - 硬件/OS 驱动                      │   │
│  │                                                                   │   │
│  │   USB CDC Driver | Bluetooth Stack | Serial Port                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. L0.1 传输抽象层设计

### 3.1 核心 Trait 设计

```
DataTransport trait
├── connect() → Result<()>           // 建立连接
├── disconnect() → Result<()>        // 断开连接
├── read_bytes() → Result<Vec<u8>>   // 读取原始字节
├── write_bytes() → Result<usize>    // 写入字节（配置命令）
├── is_connected() → bool            // 连接状态
└── transport_info() → TransportInfo // 传输层信息（带宽、延迟等）
```

### 3.2 两种传输方式对比

| 特性 | 有线 (Serial) | BLE |
|------|--------------|-----|
| **带宽** | 高 (1-12 Mbps) | 低 (1 Mbps 理论) |
| **延迟** | 低 (<1ms) | 中高 (20-50ms) |
| **功耗** | N/A | 低 |
| **连接复杂度** | 简单 | 需配对+GATT |
| **跨平台难度** | 低 | 中 |
| **Rust crate** | `serialport` | `bluest` 或 `btleplug` |

### 3.3 跨平台 BLE 方案

#### BLE Crate 选型 (二选一)

| Crate | 特点 | 链接 |
|-------|------|------|
| **bluest** | 更现代的 API，async-first 设计，活跃开发 | [github](https://github.com/alexmoon/bluest) |
| **btleplug** | 成熟稳定，社区广泛使用 | [crates.io](https://crates.io/crates/btleplug) |

```
bluest crate (推荐)
├── 支持: Windows, macOS, Linux
├── 纯 Rust + async/await
├── 更简洁的 API 设计
└── 活跃维护中

btleplug crate (备选)
├── 支持: Windows, macOS, Linux
├── 纯 Rust 实现
├── 社区成熟，文档丰富
└── GATT 操作完整支持
```

**建议**: 优先评估 `bluest`，如遇兼容性问题可切换到 `btleplug`

---

## 4. L0.2 协议解析层设计

### 4.1 数据包格式建议 (待定义)

由于协议未定，以下是**推荐的协议设计**：

```
┌────────┬────────┬────────┬──────────────┬────────┬────────┐
│ SYNC   │ VER    │ TYPE   │ PAYLOAD_LEN  │PAYLOAD │ CRC16  │
│ 2 bytes│ 1 byte │ 1 byte │ 2 bytes      │ N bytes│ 2 bytes│
└────────┴────────┴────────┴──────────────┴────────┴────────┘

SYNC:        0xAA 0x55 (固定同步字)
VER:         协议版本 (0x01)
TYPE:        包类型 (0x01=EEG数据, 0x02=事件标记, 0x03=状态, 0x04=配置响应)
PAYLOAD_LEN: 有效载荷长度 (小端序)
PAYLOAD:     数据内容
CRC16:       CRC-16/CCITT 校验
```

### 4.2 EEG 数据包 Payload 格式

```
┌──────────┬──────────┬─────────────────────────────────────┐
│ COUNTER  │TIMESTAMP │ CHANNEL_DATA                        │
│ 4 bytes  │ 8 bytes  │ N_CH × 3 bytes (24-bit per channel) │
└──────────┴──────────┴─────────────────────────────────────┘

COUNTER:      样本计数器 (用于丢包检测)
TIMESTAMP:    设备端时间戳 (μs, uint64)
CHANNEL_DATA: 各通道 ADC 原始值 (24-bit 有符号, 大端序)
```

### 4.3 协议设计考量

| 考量点 | 建议 | 理由 |
|--------|------|------|
| 字节序 | 小端序 (除 ADC 数据) | ARM Cortex-M 原生序 |
| ADC 字节序 | 大端序 | ADS129x 输出格式 |
| 同步字 | 0xAA55 | 易于检测，不易误判 |
| 校验 | CRC-16 | 平衡可靠性与计算量 |
| 时间戳 | 设备端 μs | 避免传输延迟影响 |
| 丢包检测 | 样本计数器 | 简单有效 |

---

## 5. L0.3 单位转换层设计

### 5.1 ADS129x ADC 转换公式

```
电压 (V) = (ADC_RAW / 2^23) × (Vref / Gain)

其中:
- ADC_RAW: 24-bit 有符号整数 (-8388608 ~ +8388607)
- Vref: 参考电压 (ADS1299 内部 4.5V，或外部 2.4V)
- Gain: PGA 增益 (1, 2, 4, 6, 8, 12, 24)

转换为 μV:
μV = V × 1,000,000
```

### 5.2 典型配置示例

| 参数 | 典型值 | 说明 |
|------|--------|------|
| Vref | 4.5V | ADS1299 内部参考 |
| Gain | 24 | EEG 典型增益 |
| LSB | 0.022 μV | 最小分辨单位 |
| 范围 | ±187.5 mV | 输入范围 (Gain=24) |

### 5.3 转换参数结构

```
AdcConfig
├── resolution_bits: u8      // 24
├── vref_mv: f64             // 4500.0 (mV)
├── gain: u8                 // 24
├── is_signed: bool          // true
├── byte_order: ByteOrder    // BigEndian (ADS129x)
└── channel_calibration: Vec<ChannelCalibration>

ChannelCalibration
├── offset: f64              // 零点偏移 (μV)
└── scale: f64               // 比例因子 (默认 1.0)
```

---

## 6. L0.4 输出适配器设计

### 6.1 输出格式对比

| 格式 | 用途 | Rust crate | 状态 |
|------|------|------------|------|
| ndarray | 内存处理，传递给上层 | `ndarray` | ✅ 直接用 |
| EDF/EDF+ | 离线存储，医疗标准 | `edflib` | ✅ 有 crate |
| XDF | LSL 生态，多流同步 | `xdf` | ✅ 有 crate |
| LSL | 实时流发布 | `lsl` | ✅ 官方绑定 |

### 6.2 输出 Trait 设计

```
DataSink trait
├── write_samples(data: &Array2<f64>, timestamps: &[f64]) → Result<()>
├── flush() → Result<()>
├── close() → Result<()>
└── sink_info() → SinkInfo

具体实现:
├── MemorySink      → 累积到 Vec/ndarray
├── EdfSink         → 写入 EDF 文件
├── XdfSink         → 写入 XDF 文件
└── LslOutletSink   → 发布到 LSL 网络
```

### 6.3 多输出支持

```
MultiSink (组合模式)
├── sinks: Vec<Box<dyn DataSink>>
└── write_samples() → 广播到所有 sink

使用场景:
- 同时保存 EDF 文件 + 发布 LSL 流
- 内存处理 + 实时可视化
```

---

## 7. 数据流完整路径

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         完整数据流                                       │
└─────────────────────────────────────────────────────────────────────────┘

[ADS129x ADC]
     │
     │ SPI (硬件内部)
     ▼
[MCU 固件] ──打包──→ [数据包: SYNC|VER|TYPE|LEN|PAYLOAD|CRC]
     │
     │ USB/BLE (选择其一)
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ L0.1 Transport                                                          │
│                                                                         │
│   SerialTransport / BleTransport                                       │
│   └── read_bytes() → Vec<u8>                                           │
└─────────────────────────────────────────────────────────────────────────┘
     │
     │ 原始字节流
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ L0.2 Protocol Parser                                                    │
│                                                                         │
│   1. 同步字检测 (0xAA55)                                                │
│   2. 包头解析 (VER, TYPE, LEN)                                          │
│   3. Payload 提取                                                       │
│   4. CRC 校验                                                           │
│   5. 丢包检测 (COUNTER 连续性)                                          │
│                                                                         │
│   输出: ParsedPacket { counter, timestamp, raw_channels: Vec<i32> }    │
└─────────────────────────────────────────────────────────────────────────┘
     │
     │ 结构化数据包
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ L0.3 Unit Conversion                                                    │
│                                                                         │
│   raw_channels: Vec<i32> (24-bit ADC)                                  │
│        │                                                                │
│        ▼                                                                │
│   μV = (raw / 2^23) × (Vref / Gain) × 1e6                              │
│        │                                                                │
│        ▼                                                                │
│   channels_uv: Vec<f64>                                                │
└─────────────────────────────────────────────────────────────────────────┘
     │
     │ 物理单位数据 (μV)
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ L0.4 Output Adapters                                                    │
│                                                                         │
│   ┌─────────────┐                                                      │
│   │ MultiSink   │                                                      │
│   └──────┬──────┘                                                      │
│          │                                                              │
│    ┌─────┼─────┬─────────┬──────────┐                                  │
│    ▼     ▼     ▼         ▼          ▼                                  │
│  Memory  EDF   XDF      LSL      (扩展...)                             │
│  Sink    Sink  Sink     Outlet                                         │
└─────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[Layer 1+: 上层处理]
```

---

## 8. 跨平台实现策略

### 8.1 Rust Crate 选型

| 功能 | Crate | 平台支持 | 备注 |
|------|-------|----------|------|
| 串口 | `serialport` | Win/Mac/Linux | ✅ 成熟稳定 |
| BLE | `bluest` 或 `btleplug` | Win/Mac/Linux | ✅ 纯 Rust，二选一 |
| 异步运行时 | `tokio` | 全平台 | ✅ 用于 BLE |
| 字节解析 | `nom` 或 `bytes` | 全平台 | ✅ 协议解析 |
| CRC | `crc` | 全平台 | ✅ 校验计算 |

### 8.2 代码结构

```
bcif-io/
├── src/
│   ├── transport/
│   │   ├── mod.rs           // DataTransport trait
│   │   ├── serial.rs        // 串口实现 (跨平台)
│   │   └── ble.rs           // BLE 实现 (bluest 或 btleplug)
```

---

## 9. BLE MTU 动态适配设计

### 9.1 MTU 协商机制

BLE 默认 MTU 为 23 bytes（有效载荷 20 bytes），但现代设备支持协商更大 MTU（最高 512 bytes）。

```
MTU 协商流程:
┌─────────┐                      ┌─────────┐
│ Central │                      │Peripheral│
│ (PC)    │                      │ (设备)   │
└────┬────┘                      └────┬────┘
     │                                │
     │  1. ATT_MTU_REQ (期望 MTU)     │
     │ ──────────────────────────────>│
     │                                │
     │  2. ATT_MTU_RSP (实际 MTU)     │
     │ <──────────────────────────────│
     │                                │
     │  3. 使用 min(请求, 响应) 作为  │
     │     协商后的 MTU               │
     └────────────────────────────────┘
```

### 9.2 数据包大小计算

| 通道数 | 每样本字节数 | 包头+校验 | 单样本包总大小 |
|--------|-------------|-----------|---------------|
| 8 ch   | 24 bytes    | 16 bytes  | 40 bytes      |
| 16 ch  | 48 bytes    | 16 bytes  | 64 bytes      |
| 32 ch  | 96 bytes    | 16 bytes  | 112 bytes     |

**结论**: 8 通道单样本包 (40 bytes) 超过默认 MTU (20 bytes)，必须协商更大 MTU 或分包。

### 9.3 动态适配策略

```
BleTransport
├── negotiated_mtu: usize           // 协商后的 MTU
├── effective_payload: usize        // 有效载荷 = MTU - 3 (ATT header)
└── packing_strategy: PackingStrategy

PackingStrategy (根据 MTU 动态选择)
├── SingleSample    // MTU >= 单样本包大小，每包 1 样本
├── MultiSample(n)  // MTU 较大，每包 n 个样本
└── SplitSample     // MTU 太小，单样本需分包 (不推荐)
```

### 9.4 固件配合要求

固件需要支持以下配置命令：

```
配置命令: SET_PACKET_SIZE
├── 参数: samples_per_packet (1-N)
└── 响应: ACK/NAK

流程:
1. PC 端完成 MTU 协商，得到 effective_payload
2. PC 计算最优 samples_per_packet = (effective_payload - 包头) / 每样本字节数
3. PC 发送 SET_PACKET_SIZE 命令给固件
4. 固件调整打包策略
```

### 9.5 MTU 适配代码逻辑 (伪代码)

```
fn calculate_packing_strategy(mtu: usize, n_channels: usize) -> PackingStrategy:
    effective_payload = mtu - 3  // ATT header
    header_size = 16             // SYNC + VER + TYPE + LEN + COUNTER + TIMESTAMP + CRC
    bytes_per_sample = n_channels * 3  // 24-bit per channel

    available_for_data = effective_payload - header_size

    if available_for_data < bytes_per_sample:
        return PackingStrategy::SplitSample  // 需要分包，不推荐

    samples_per_packet = available_for_data / bytes_per_sample

    if samples_per_packet == 1:
        return PackingStrategy::SingleSample
    else:
        return PackingStrategy::MultiSample(samples_per_packet)
```

---

## 10. 可配置采样率设计

### 10.1 支持的采样率

| 采样率 | 用途 | 数据率 (8ch) | BLE 可行性 |
|--------|------|-------------|-----------|
| 250 Hz | EEG 常规 | 6 KB/s | ✅ 轻松 |
| 500 Hz | EEG 高精度 | 12 KB/s | ✅ 可行 |
| 1000 Hz | EMG/高频 | 24 KB/s | ⚠️ 需优化 |
| 2000 Hz | 特殊应用 | 48 KB/s | ❌ BLE 困难 |

### 10.2 采样率配置协议

```
配置命令: SET_SAMPLE_RATE
├── 参数: sample_rate_hz (u16)
└── 响应: ACK + 实际采样率

支持的值 (ADS1299):
├── 250 Hz  (0x00FA)
├── 500 Hz  (0x01F4)
├── 1000 Hz (0x03E8)
├── 2000 Hz (0x07D0)
├── 4000 Hz (0x0FA0)
├── 8000 Hz (0x1F40)
└── 16000 Hz (0x3E80)  // SRB 模式
```

### 10.3 采样率与传输方式匹配

```
┌─────────────────────────────────────────────────────────────────┐
│                    采样率 vs 传输方式决策树                       │
└─────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │ 目标采样率?      │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
      ≤500 Hz          500-1000 Hz       >1000 Hz
            │                │                │
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ BLE / Serial  │ │ Serial 推荐   │ │ Serial Only   │
    │ 均可          │ │ BLE 需优化    │ │ 必须          │
    └───────────────┘ └───────────────┘ └───────────────┘
```

### 10.4 运行时采样率切换

```
DeviceConfig
├── current_sample_rate: u16
├── supported_rates: Vec<u16>
└── transport_constraints: TransportConstraints

TransportConstraints
├── max_throughput_bps: usize    // 传输层最大吞吐
├── recommended_rate: u16        // 推荐采样率
└── max_rate: u16                // 最大可支持采样率

切换流程:
1. 用户请求新采样率
2. 检查 transport_constraints.max_rate
3. 如果超出，返回错误或降级到 max_rate
4. 发送 SET_SAMPLE_RATE 命令
5. 更新 ProcessContext.sample_rate
6. 通知上层组件采样率变更
```

### 10.5 采样率变更事件

```
SampleRateChangeEvent
├── old_rate: u16
├── new_rate: u16
├── timestamp: u64
└── reason: ChangeReason

ChangeReason
├── UserRequest      // 用户主动切换
├── TransportLimit   // 传输层限制自动降级
└── DeviceReset      // 设备重置恢复默认
```

---

## 11. 可配置通道数设计

### 11.1 支持的通道配置

| 配置 | 芯片方案 | 每样本字节数 | BLE 单包样本数 (MTU=247) |
|------|----------|-------------|-------------------------|
| 8 ch | ADS1299 ×1 | 24 bytes | 9 samples |
| 16 ch | ADS1299 ×2 | 48 bytes | 4 samples |
| 32 ch | ADS1299 ×4 | 96 bytes | 2 samples |

### 11.2 通道配置协议

```
配置命令: GET_DEVICE_INFO
响应:
├── device_id: [u8; 16]      // 设备唯一标识
├── firmware_version: u16    // 固件版本
├── n_channels: u8           // 通道数 (8/16/32)
├── adc_resolution: u8       // ADC 位数 (24)
└── supported_rates: [u16]   // 支持的采样率列表

配置命令: SET_CHANNEL_MASK
├── 参数: channel_mask (u32 位掩码)
└── 用途: 禁用部分通道以提高传输效率
```

### 11.3 动态通道数处理

```
ChannelConfig
├── total_channels: u8           // 硬件总通道数
├── active_channels: u8          // 当前启用通道数
├── channel_mask: u32            // 通道启用掩码
├── channel_names: Vec<String>   // 通道名称 (Fp1, Fp2, ...)
└── channel_types: Vec<ChannelType>  // 通道类型 (EEG, EOG, EMG, ...)

数据包解析:
1. 读取 channel_mask
2. 计算 active_channels = popcount(channel_mask)
3. 按 mask 顺序解析 ADC 数据
4. 填充到对应通道位置
```

---

## 12. LSL 时钟同步设计

### 12.1 LSL 时间同步原理

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LSL 时钟同步机制                                      │
└─────────────────────────────────────────────────────────────────────────┘

设备端:                          PC 端 (LSL):
┌─────────────┐                  ┌─────────────┐
│ 本地时钟    │                  │ LSL 时钟    │
│ (μs 计数器) │                  │ (lsl_clock) │
└──────┬──────┘                  └──────┬──────┘
       │                                │
       │  设备时间戳 t_device           │
       │ ─────────────────────────────> │
       │                                │
       │                         计算时钟偏移:
       │                         offset = lsl_clock() - t_device
       │                                │
       │                         校正后时间戳:
       │                         t_lsl = t_device + offset
       │                                │
       └────────────────────────────────┘
```

### 12.2 时间戳处理流程

```
时间戳处理 Pipeline:

1. 设备端:
   ├── 每个样本附带 device_timestamp (μs, uint64)
   └── 使用设备本地高精度计时器

2. PC 端接收:
   ├── 记录 pc_receive_time = lsl_local_clock()
   └── 保存原始 device_timestamp

3. 时钟偏移估计:
   ├── 首次连接时进行时钟同步握手
   ├── 发送 SYNC_REQUEST，设备回复 SYNC_RESPONSE
   ├── 计算往返时间 RTT 和时钟偏移
   └── 定期重新同步 (每 10-60 秒)

4. 时间戳校正:
   └── corrected_timestamp = device_timestamp + clock_offset
```

### 12.3 时钟同步协议

```
同步握手:

PC → 设备: SYNC_REQUEST
├── pc_send_time: f64 (LSL 时钟)
└── sequence: u32

设备 → PC: SYNC_RESPONSE
├── pc_send_time: f64 (回传)
├── device_time: u64 (设备时钟)
└── sequence: u32

PC 计算:
├── rtt = pc_receive_time - pc_send_time
├── one_way_delay = rtt / 2
└── clock_offset = pc_receive_time - device_time - one_way_delay
```

### 12.4 多设备同步

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    多设备 LSL 同步架构                                   │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ Device A │     │ Device B │     │ Device C │
  │ (EEG)    │     │ (EMG)    │     │ (Markers)│
  └────┬─────┘     └────┬─────┘     └────┬─────┘
       │                │                │
       │ BLE            │ Serial         │ BLE
       ▼                ▼                ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    BCIF Layer 0                                  │
  │                                                                  │
  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
  │  │ StreamA    │  │ StreamB    │  │ StreamC    │                │
  │  │ offset_a   │  │ offset_b   │  │ offset_c   │                │
  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                │
  │        │               │               │                        │
  │        └───────────────┼───────────────┘                        │
  │                        ▼                                        │
  │              ┌─────────────────┐                                │
  │              │ LSL 统一时钟    │                                │
  │              │ (所有流对齐)    │                                │
  │              └─────────────────┘                                │
  └─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ XDF 录制    │  ← 多流同步保存
                    └─────────────┘
```

### 12.5 时间戳精度要求

| 应用场景 | 精度要求 | 实现方式 |
|----------|----------|----------|
| EEG 分析 | ±1 ms | LSL 默认同步 |
| ERP 研究 | ±0.1 ms | 高频同步 + 硬件触发 |
| BCI 实时 | ±10 ms | 软件同步足够 |
| 多模态融合 | ±1 ms | LSL 多流同步 |

---

## 13. 事件标记 (Event Marker) 设计

### 13.1 事件来源

| 来源 | 触发方式 | 时间精度 | 典型用途 |
|------|----------|----------|----------|
| 外部触发 (TTL) | GPIO 中断 | < 0.1 ms | 刺激同步、ERP 实验 |
| 软件标记 | PC 发送命令 | 1-10 ms | 任务标记、用户交互 |
| 自动检测 | 固件算法 | 取决于算法 | 电极脱落、阻抗变化 |

### 13.2 事件数据包格式

```
事件包类型: TYPE = 0x02 (EVENT)

┌────────┬────────┬────────┬──────────────┬─────────────────────┬────────┐
│ SYNC   │ VER    │ TYPE   │ PAYLOAD_LEN  │ EVENT_PAYLOAD       │ CRC16  │
│ 0xAA55 │ 0x01   │ 0x02   │ 变长         │                     │        │
└────────┴────────┴────────┴──────────────┴─────────────────────┴────────┘

EVENT_PAYLOAD 结构:
┌──────────────┬──────────────┬──────────────┬──────────────────────────┐
│ EVENT_TYPE   │ TIMESTAMP    │ EVENT_CODE   │ EVENT_DATA (可选)        │
│ 1 byte       │ 8 bytes      │ 2 bytes      │ 0-N bytes                │
└──────────────┴──────────────┴──────────────┴──────────────────────────┘

EVENT_TYPE:
├── 0x01: TTL_TRIGGER      // 外部 TTL 触发
├── 0x02: SOFTWARE_MARKER  // 软件标记
├── 0x03: IMPEDANCE_ALERT  // 阻抗告警
├── 0x04: ELECTRODE_OFF    // 电极脱落
├── 0x05: BATTERY_LOW      // 电池低电量
└── 0x06: SYNC_PULSE       // 同步脉冲

EVENT_CODE: 用户定义的事件代码 (0-65535)
EVENT_DATA: 附加数据 (如阻抗值、电池电量等)
```

### 13.3 TTL 触发设计

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TTL 触发时序                                          │
└─────────────────────────────────────────────────────────────────────────┘

外部刺激设备                    EEG 设备 (MCU)
     │                              │
     │  TTL 上升沿                  │
     │ ────────────────────────────>│ GPIO 中断
     │                              │
     │                              │ 记录 device_timestamp
     │                              │ 生成 EVENT 包
     │                              │
     │                              │ 在下一个数据包中发送
     │                              │ 或立即发送独立事件包
     │                              ▼
                              ┌─────────────┐
                              │ EVENT 包    │
                              │ type=TTL    │
                              │ code=用户定义│
                              │ ts=精确时间 │
                              └─────────────┘
```

### 13.4 软件标记协议

```
PC → 设备: SEND_MARKER
├── event_code: u16      // 事件代码
└── event_data: [u8]     // 可选附加数据

设备处理:
1. 接收 SEND_MARKER 命令
2. 记录当前 device_timestamp
3. 生成 EVENT 包 (type=SOFTWARE_MARKER)
4. 发送回 PC

时间戳策略:
├── 方案 A: 设备端时间戳 (推荐，精度高)
└── 方案 B: PC 端时间戳 (简单，但有传输延迟)
```

### 13.5 事件与 EEG 数据同步

```
同步策略:

方案 A: 事件嵌入数据流 (推荐)
┌─────────────────────────────────────────────────────────────────┐
│ 数据流: [EEG][EEG][EVENT][EEG][EEG][EVENT][EEG]...             │
│                                                                 │
│ 优点: 事件与数据在同一流中，天然同步                             │
│ 缺点: 解析稍复杂                                                │
└─────────────────────────────────────────────────────────────────┘

方案 B: 独立事件流
┌─────────────────────────────────────────────────────────────────┐
│ EEG 流:   [EEG][EEG][EEG][EEG][EEG]...                         │
│ Event 流: [EVENT]........[EVENT]...                            │
│                                                                 │
│ 优点: 解析简单                                                  │
│ 缺点: 需要时间戳对齐                                            │
└─────────────────────────────────────────────────────────────────┘

建议: 使用方案 A，通过 TYPE 字段区分 EEG 包和 EVENT 包
```

### 13.6 LSL 事件流发布

```
LSL 多流架构:

┌─────────────┐     ┌─────────────┐
│ EEG Stream  │     │ Marker Stream│
│ type=EEG    │     │ type=Markers │
│ 8-32 ch     │     │ 1 ch (code)  │
│ 250-1000 Hz │     │ irregular    │
└──────┬──────┘     └──────┬───────┘
       │                   │
       └─────────┬─────────┘
                 │
                 ▼
          ┌─────────────┐
          │ XDF 录制    │
          │ 多流同步    │
          └─────────────┘
```

---

## 14. 延迟与实时性设计

### 14.1 延迟目标

| 应用场景 | 延迟目标 | 优先级 |
|----------|----------|--------|
| 实时 BCI (P300/SSVEP/MI) | < 200 ms | 高 |
| 神经反馈 | < 100 ms | 中 |
| 离线分析 | 无要求 | - |

**设计目标**: < 200 ms 端到端延迟 (软实时)

### 14.2 延迟分解

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    端到端延迟分解                                        │
└─────────────────────────────────────────────────────────────────────────┘

ADC 采样 → MCU 处理 → 传输 → PC 接收 → 软件处理 → 应用响应
   │          │         │        │          │           │
   ▼          ▼         ▼        ▼          ▼           ▼
  T1         T2        T3       T4         T5          T6

典型延迟预算 (目标 < 200ms):
┌────────────────────────────────────────────────────────────────┐
│ 阶段              │ 有线      │ BLE       │ 说明              │
├────────────────────────────────────────────────────────────────┤
│ T1: ADC 采样      │ 1-4 ms    │ 1-4 ms    │ 取决于采样率      │
│ T2: MCU 打包      │ < 1 ms    │ < 1 ms    │ 固件处理          │
│ T3: 传输延迟      │ < 5 ms    │ 20-50 ms  │ BLE 连接间隔      │
│ T4: OS 接收       │ 1-10 ms   │ 1-10 ms   │ USB/BLE 驱动      │
│ T5: 软件处理      │ 1-5 ms    │ 1-5 ms    │ 解析+转换         │
│ T6: 应用处理      │ 变化      │ 变化      │ 取决于算法        │
├────────────────────────────────────────────────────────────────┤
│ 总计 (不含 T6)    │ ~20 ms    │ ~70 ms    │                   │
└────────────────────────────────────────────────────────────────┘
```

### 14.3 BLE 连接参数优化

```
BLE 连接参数对延迟的影响:

Connection Interval (CI): 7.5ms - 4000ms
├── 较小 CI → 低延迟，高功耗
└── 较大 CI → 高延迟，低功耗

推荐配置:
┌─────────────────────────────────────────────────────────────────┐
│ 场景              │ CI 范围        │ 预期延迟    │ 功耗       │
├─────────────────────────────────────────────────────────────────┤
│ 实时 BCI          │ 7.5-15 ms      │ 20-40 ms    │ 高         │
│ 一般采集          │ 30-50 ms       │ 50-80 ms    │ 中         │
│ 低功耗采集        │ 100-200 ms     │ 150-250 ms  │ 低         │
└─────────────────────────────────────────────────────────────────┘

配置命令: SET_BLE_PARAMS
├── connection_interval_min: u16 (单位: 1.25ms)
├── connection_interval_max: u16
├── slave_latency: u16
└── supervision_timeout: u16
```

### 14.4 缓冲区策略

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    缓冲区设计                                            │
└─────────────────────────────────────────────────────────────────────────┘

设备端缓冲:
┌─────────────────────────────────────────────────────────────────┐
│ MCU Ring Buffer                                                 │
│                                                                 │
│ 大小: 采样率 × 通道数 × 缓冲时间                                │
│ 例: 250 Hz × 8 ch × 3 bytes × 0.5s = 3 KB                      │
│                                                                 │
│ 策略:                                                           │
│ ├── 正常: 累积到 N 个样本后发送                                 │
│ ├── 低延迟模式: 每个样本立即发送                                │
│ └── 溢出: 丢弃最旧数据，设置溢出标志                            │
└─────────────────────────────────────────────────────────────────┘

PC 端缓冲:
┌─────────────────────────────────────────────────────────────────┐
│ RingBuffer (见第 6 节 StreamPipeline)                           │
│                                                                 │
│ 大小: 采样率 × 通道数 × 缓冲时间                                │
│ 例: 250 Hz × 8 ch × 8 bytes × 10s = 160 KB                     │
│                                                                 │
│ 策略:                                                           │
│ ├── 实时模式: 小缓冲 (1-2s)，快速响应                           │
│ └── 录制模式: 大缓冲 (10-60s)，防止丢数据                       │
└─────────────────────────────────────────────────────────────────┘
```

### 14.5 低延迟模式

```
LowLatencyMode 配置:

启用条件:
├── 传输方式: 有线 或 BLE (优化 CI)
├── 采样率: ≤ 500 Hz
└── 通道数: ≤ 16 ch

配置命令: SET_LATENCY_MODE
├── mode: LatencyMode
│   ├── Normal      // 默认，平衡延迟和效率
│   ├── LowLatency  // 低延迟，每样本发送
│   └── HighThroughput // 高吞吐，大包发送
└── 响应: ACK + 实际模式

软件端配置:
├── buffer_size: 减小到 0.5-1s
├── processing_interval: 减小到 10-50ms
└── callback_mode: 启用实时回调
```

---

## 15. 错误处理与恢复设计

### 15.1 错误分类

```
错误层次:

L0.0 物理层错误
├── USB 断开
├── 蓝牙连接丢失
└── 串口错误

L0.1 传输层错误
├── 读取超时
├── 写入失败
└── MTU 协商失败

L0.2 协议层错误
├── 同步字丢失
├── CRC 校验失败
├── 包长度异常
└── 未知包类型

L0.3 数据层错误
├── ADC 溢出
├── 样本计数器跳变 (丢包)
└── 时间戳异常

L0.4 设备层错误
├── 电极脱落
├── 阻抗过高
├── ���池低电量
└── 固件错误
```

### 15.2 错误码定义

```
ErrorCode 结构:

┌────────────────┬────────────────┬────────────────────────────────┐
│ 类别 (高 8 位) │ 代码 (低 8 位) │ 说明                           │
├────────────────┼────────────────┼────────────────────────────────┤
│ 0x01           │ 0x01           │ TRANSPORT_DISCONNECTED         │
│ 0x01           │ 0x02           │ TRANSPORT_TIMEOUT              │
│ 0x01           │ 0x03           │ TRANSPORT_WRITE_FAILED         │
│ 0x01           │ 0x04           │ BLE_MTU_NEGOTIATION_FAILED     │
├────────────────┼────────────────┼────────────────────────────────┤
│ 0x02           │ 0x01           │ PROTOCOL_SYNC_LOST             │
│ 0x02           │ 0x02           │ PROTOCOL_CRC_ERROR             │
│ 0x02           │ 0x03           │ PROTOCOL_INVALID_LENGTH        │
│ 0x02           │ 0x04           │ PROTOCOL_UNKNOWN_TYPE          │
├────────────────┼────────────────┼────────────────────────────────┤
│ 0x03           │ 0x01           │ DATA_PACKET_LOST               │
│ 0x03           │ 0x02           │ DATA_ADC_OVERFLOW              │
│ 0x03           │ 0x03           │ DATA_TIMESTAMP_JUMP            │
├────────────────┼────────────────┼────────────────────────────────┤
│ 0x04           │ 0x01           │ DEVICE_ELECTRODE_OFF           │
│ 0x04           │ 0x02           │ DEVICE_IMPEDANCE_HIGH          │
│ 0x04           │ 0x03           │ DEVICE_BATTERY_LOW             │
│ 0x04           │ 0x04           │ DEVICE_FIRMWARE_ERROR          │
└────────────────┴────────────────┴────────────────────────────────┘
```

### 15.3 丢包检测与处理

```
丢包检测机制:

1. 样本计数器检测
   ├── 每个 EEG 包含 counter (u32)
   ├── 正常: counter_new = counter_old + samples_per_packet
   ├── 丢包: counter_new > counter_old + samples_per_packet
   └── 计算丢失样本数: lost = counter_new - counter_old - samples_per_packet

2. 丢包处理策略
   ┌─────────────────────────────────────────────────────────────────┐
   │ 策略              │ 处理方式           │ 适用场景              │
   ├─────────────────────────────────────────────────────────────────┤
   │ 插入 NaN          │ 用 NaN 填充丢失段  │ 离线分析              │
   │ 线性插值          │ 插值填充           │ 可视化               │
   │ 跳过              │ 不填充，记录事件   │ 实时 BCI              │
   │ 重传请求          │ 请求设备重发       │ 可靠传输 (有线)       │
   └─────────────────────────────────────────────────────────────────┘

3. 丢包事件记录
   PacketLossEvent
   ├── timestamp: f64
   ├── expected_counter: u32
   ├── received_counter: u32
   ├── lost_samples: u32
   └── recovery_action: RecoveryAction
```

### 15.4 连接恢复策略

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    连接恢复状态机                                        │
└─────────────────────────────────────────────────────────────────────────┘

         ┌─────────────┐
         │ Disconnected│
         └──────┬──────┘
                │ connect()
                ▼
         ┌─────────────┐
    ┌───>│ Connecting  │<───────────────────┐
    │    └──────┬──────┘                    │
    │           │ success                   │ retry
    │           ▼                           │
    │    ┌─────────────┐    timeout/error   │
    │    │ Connected   │────────────────────┤
    │    └──────┬──────┘                    │
    │           │ disconnect                │
    │           ▼                           │
    │    ┌─────────────┐                    │
    └────│ Reconnecting│────────────────────┘
         └─────────────┘
              │ max_retries exceeded
              ▼
         ┌─────────────┐
         │ Failed      │
         └─────────────┘

重连参数:
├── initial_delay: 1s
├── max_delay: 30s
├── backoff_factor: 2.0
├── max_retries: 10
└── jitter: 0-500ms
```

### 15.5 错误回调机制

```
ErrorCallback 设计:

trait ErrorHandler: Send + Sync {
    /// 处理错误
    /// Handle error
    fn on_error(&self, error: &LayerError) -> ErrorAction;

    /// 处理警告
    /// Handle warning
    fn on_warning(&self, warning: &LayerWarning);

    /// 连接状态变化
    /// Connection state changed
    fn on_connection_state(&self, state: ConnectionState);
}

ErrorAction:
├── Continue      // 继续运行，忽略错误
├── Retry         // 重试当前操作
├── Reconnect     // 断开并重连
├── Stop          // 停止采集
└── Panic         // 严重错误，终止程序

使用示例:
├── CRC 错误 → Continue (丢弃该包)
├── 连接断开 → Reconnect
├── 电池低 → Continue + 通知用户
└── 固件错误 → Stop + 报告
```

### 15.6 数据完整性保证

```
数据完整性级别:

Level 0: Best Effort (尽力而为)
├── 丢包不重传
├── 适用: 实时 BCI，可容忍少量丢失
└── 配置: reliability = BestEffort

Level 1: Detected Loss (检测丢失)
├── 检测丢包，记录但不恢复
├── 适用: 一般采集，后处理时标记
└── 配置: reliability = DetectedLoss

Level 2: Guaranteed Delivery (保证送达)
├── 丢包重传 (仅有线支持)
├── 适用: 关键实验，不能丢数据
└── 配置: reliability = Guaranteed

配置命令: SET_RELIABILITY_LEVEL
├── level: ReliabilityLevel
└── 响应: ACK + 实际级别
```

---

## 16. 设备管理与发现

### 16.1 设备发现机制

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    设备发现流程                                          │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │ DeviceScanner   │
                    └────────┬────────┘
                             │
         ┌───────────────────┴───────────────────┐
         ▼                                       ▼
   ┌───────────┐                          ┌───────────┐
   │ Serial    │                          │ BLE       │
   │ Scanner   │                          │ Scanner   │
   └─────┬─────┘                          └─────┬─────┘
         │                                      │
         ▼                                      ▼
   列举 COM/tty                           扫描 BLE 广播
         │                                      │
         └──────────────────┬───────────────────┘
                            ▼
                    ┌─────────────────┐
                    │ DeviceList      │
                    │ Vec<DeviceInfo> │
                    └─────────────────┘
```

### 16.2 设备标识

```
DeviceInfo 结构:

DeviceInfo
├── device_id: [u8; 16]        // 全局唯一标识 (UUID 或 MAC 派生)
├── device_name: String        // 用户可读名称 "BCIF-EEG-001"
├── transport_type: TransportType
│   ├── Serial { port: String }           // "/dev/ttyUSB0" 或 "COM3"
│   ├── Ble { address: BleAddress }       // MAC 地址
│   └── Spp { address: BtAddress }        // 蓝牙地址
├── firmware_version: Version  // 固件版本 "1.2.3"
├── hardware_version: String   // 硬件版本 "v2.0"
├── n_channels: u8             // 通道数
├── battery_level: Option<u8>  // 电池电量 (0-100%)
├── signal_strength: Option<i8> // 信号强度 RSSI (dBm)
└── last_seen: Instant         // 最后发现时间
```

### 16.3 BLE 广播数据设计

```
BLE Advertisement Packet:

┌─────────────────────────────────────────────────────────────────┐
│ Flags (3 bytes)                                                 │
│ └── 0x02 0x01 0x06 (LE General Discoverable)                   │
├─────────────────────────────────────────────────────────────────┤
│ Complete Local Name (变长)                                      │
│ └── "BCIF-EEG-XXXX" (XXXX = 设备序列号后4位)                   │
├─────────────────────────────────────────────────────────────────┤
│ Manufacturer Specific Data (变长)                               │
│ ├── Company ID: 0xFFFF (开发用) 或申请正式 ID                  │
│ ├── Device Type: 0x01 (EEG)                                    │
│ ├── Firmware Version: Major.Minor (2 bytes)                    │
│ ├── Battery Level: 0-100 (1 byte)                              │
│ ├── Channel Count: 8/16/32 (1 byte)                            │
│ └── Status Flags: (1 byte)                                     │
│     ├── bit 0: Acquiring (正在采集)                            │
│     ├── bit 1: Charging (正在充电)                             │
│     ├── bit 2: Error (有错误)                                  │
│     └─�� bit 3-7: Reserved                                      │
├─────────────────────────────────────────────────────────────────┤
│ Service UUIDs (16 bytes)                                        │
│ └── BCIF EEG Service UUID: 自定义 128-bit UUID                 │
└─────────────────────────────────────────────────────────────────┘
```

### 16.4 设备配对与绑定

```
配对流程:

1. BLE 配对 (首次连接)
   ┌─────────┐                      ┌─────────┐
   │ PC      │                      │ Device  │
   └────┬────┘                      └────┬────┘
        │  1. 扫描发现设备               │
        │ <─────────────────────────────│ (广播)
        │                                │
        │  2. 发起连接                   │
        │ ─────────────────────────────> │
        │                                │
        │  3. 配对请求 (Just Works)      │
        │ <────────────────────────────> │
        │                                │
        │  4. 绑定 (保存密钥)            │
        │ <────────────────────────────> │
        │                                │
        │  5. 服务发现                   │
        │ ─────────────────────────────> │
        │                                │
        │  6. 连接完成                   │
        └────────────────────────────────┘

2. 设备记忆
   DeviceRegistry
   ├── known_devices: HashMap<DeviceId, DeviceRecord>
   ├── auto_connect: Vec<DeviceId>  // 自动连接列表
   └── persist_to_file()            // 保存到配置文件

   DeviceRecord
   ├── device_id: [u8; 16]
   ├── device_name: String
   ├── transport_type: TransportType
   ├── bonding_info: Option<BondingInfo>  // BLE 绑定信息
   ├── last_connected: DateTime
   └── user_alias: Option<String>  // 用户自定义别名
```

### 16.5 多设备管理

```
DeviceManager 设计:

DeviceManager
├── scanner: DeviceScanner
├── registry: DeviceRegistry
├── connections: HashMap<DeviceId, Box<dyn DataTransport>>
└── event_tx: Sender<DeviceEvent>

DeviceEvent
├── DeviceDiscovered(DeviceInfo)
├── DeviceConnected(DeviceId)
├── DeviceDisconnected(DeviceId, DisconnectReason)
├── DeviceLost(DeviceId)           // 广播消失
└── DeviceUpdated(DeviceId, DeviceInfo)  // 信息更新

API:
├── scan_start() → Result<()>
├── scan_stop() → Result<()>
├── connect(device_id) → Result<Connection>
├── disconnect(device_id) → Result<()>
├── get_connected_devices() → Vec<DeviceId>
└── subscribe_events() → Receiver<DeviceEvent>
```

---

## 17. 固件升级 (OTA) 设计

### 17.1 OTA 架构

```
┌───────���─────────────────────────────────────────────────────────────────┐
│                    OTA 升级架构                                          │
└─────────────────────────────────────────────────────────────────────────┘

PC 端:                              设备端:
┌─────────────────┐                ┌─────────────────┐
│ OtaManager      │                │ Bootloader      │
├─────────────────┤                ├─────────────────┤
│ - 固件文件解析  │                │ - 接收固件数据  │
│ - 版本检查      │                │ - 写入 Flash    │
│ - 分块传输      │                │ - 校验完整性    │
│ - 进度跟踪      │                │ - 切换分区      │
│ - 错误恢复      │                │ - 回滚机制      │
└────────┬────────┘                └────────┬────────┘
         │                                  │
         │  OTA 协议 (基于现有传输层)        │
         └──────────────────────────────────┘
```

### 17.2 固件文件格式

```
固件包格式 (.bcif_fw):

┌────────────────────────────────────────────────────────────────┐
│ HEADER (64 bytes)                                              │
├────────────────────────────────────────────────────────────────┤
│ Magic: "BCIF" (4 bytes)                                        │
│ Version: Major.Minor.Patch (3 bytes)                           │
│ Hardware Compatibility: (2 bytes)                              │
│ │ ├── bit 0-3: Min hardware version                           │
│ │ └── bit 4-7: Max hardware version                           │
│ Firmware Size: (4 bytes, little-endian)                        │
│ Firmware CRC32: (4 bytes)                                      │
│ Build Timestamp: (8 bytes, Unix timestamp)                     │
│ Signature: (32 bytes, Ed25519 签名)                            │
│ Reserved: (7 bytes)                                            │
├────────────────────────────────────────────────────────────────┤
│ FIRMWARE_DATA (变长)                                           │
│ └── 压缩的固件二进制 (可选 LZ4 压缩)                           │
└────────────────────────────────────────────────────────────────┘
```

### 17.3 OTA 传输协议

```
OTA 命令集:

1. OTA_START
   PC → Device:
   ├── firmware_size: u32
   ├── firmware_crc: u32
   ├── version: [u8; 3]
   └── chunk_size: u16 (建议 512 bytes)

   Device → PC:
   ├── status: OtaStatus (Ready/Busy/Error)
   └── max_chunk_size: u16

2. OTA_DATA
   PC → Device:
   ├── chunk_index: u16
   ├── chunk_data: [u8; N]
   └── chunk_crc: u16

   Device → PC:
   ├── status: OtaStatus (Ok/Retry/Error)
   └── next_expected: u16

3. OTA_VERIFY
   PC → Device: (无参数)

   Device → PC:
   ├── status: OtaStatus (Verified/CrcError/SizeError)
   └── calculated_crc: u32

4. OTA_APPLY
   PC → Device: (无参数)

   Device → PC:
   ├── status: OtaStatus (Applying/Error)
   └── reboot_delay_ms: u16

5. OTA_ABORT
   PC → Device: (无参数)
   Device → PC: status: OtaStatus (Aborted)
```

### 17.4 OTA 状态机

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    OTA 状态机                                            │
└─────────────────────────────────────────────────────────────────────────┘

         ┌─────────────┐
         │ Idle        │
         └──────┬──────┘
                │ OTA_START
                ▼
         ┌─────────────┐
    ┌───>│ Receiving   │<───┐
    │    └──────┬──────┘    │
    │           │ OTA_DATA  │ retry
    │           ▼           │
    │    ┌─────────────┐    │
    │    │ chunk ok?   │────┘
    │    └──────┬──────┘
    │           │ all chunks received
    │           ▼
    │    ┌─────────────┐
    │    │ Verifying   │
    │    └──────┬──────┘
    │           │ OTA_VERIFY
    │           ▼
    │    ┌─────────────┐
    │    │ verified?   │───────────────┐
    │    └──────┬──────┘               │ CRC error
    │           │ ok                   ▼
    │           ▼               ┌─────────────┐
    │    ┌─────────────┐        │ Failed      │
    │    │ Applying    │        └─────────────┘
    │    └──────┬──────┘               ▲
    │           │ OTA_APPLY            │
    │           ▼                      │
    │    ┌─────────────┐               │
    │    │ Rebooting   │───────────────┘
    │    └──────┬──────┘         apply error
    │           │ success
    │           ▼
    │    ┌─────────────┐
    └────│ Idle (new)  │
         └─────────────┘

OTA_ABORT 可在任意状态触发，返回 Idle
```

### 17.5 安全性考虑

```
OTA 安全机制:

1. 固件签名验证
   ├── 使用 Ed25519 签名算法
   ├── 公钥烧录在设备 Bootloader 中
   ├── 私钥由固件发布方保管
   └── 设备验证签名后才写入 Flash

2. 版本回滚保护
   ├── 设备记录当前版本号
   ├── 拒绝降级到更低版本 (可配置)
   └── 紧急回滚需要特殊命令 + 确认

3. 双分区设计 (A/B 分区)
   ┌─────────────────────────────────────────┐
   │ Flash Layout                            │
   ├────────────────────────────────────────��┤
   │ Bootloader (固定, 不可升级)             │
   ├─────────────────────────────────────────┤
   │ Partition A (当前运行)                  │
   ├─────────────────────────────────────────┤
   │ Partition B (OTA 写入)                  │
   ├─────────────────────────────────────────┤
   │ Config/Data (用户数据, 保留)            │
   └─────────────────────────────────────────┘

   升级流程:
   1. 写入新固件到非活动分区
   2. 验证成功后标记为待启动
   3. 重启后 Bootloader 切换分区
   4. 新固件启动失败则自动回滚

4. 传输完整性
   ├── 每个 chunk 有 CRC16 校验
   ├── 整体固件有 CRC32 校验
   └── 支持断点续传
```

---

## 18. 电源管理设计

### 18.1 电源状态

```
PowerState 枚举:

PowerState
├── Running           // 正常运行
├── LowPower          // 低功耗模式 (降低采样率/关闭部分通道)
├── Sleep             // 睡眠模式 (停止采集，保持连接)
├── DeepSleep         // 深度睡眠 (断开连接，定时唤醒)
├── Charging          // 充电中
└── CriticalBattery   // 电量极低，即将关机
```

### 18.2 电池信息

```
BatteryInfo 结构:

BatteryInfo
├── level: u8              // 电量百分比 (0-100)
├── voltage_mv: u16        // 电池电压 (mV)
├── charging: bool         // 是否正在充电
├── charge_current_ma: i16 // 充电电流 (mA, 负值表示放电)
├── temperature_c: i8      // 电池温度 (°C)
├── time_to_empty_min: Option<u16>  // 预计剩余时间 (分钟)
├── time_to_full_min: Option<u16>   // 预计充满时间 (分钟)
├── cycle_count: u16       // 充电循环次数
└── health: BatteryHealth  // 电池健康状态

BatteryHealth
├── Good           // 健康
├── Degraded       // 轻微老化
├── Replace        // 建议更换
└── Unknown        // 未知
```

### 18.3 电源管理协议

```
电源相关命令:

1. GET_BATTERY_INFO
   PC → Device: (无参数)
   Device → PC: BatteryInfo

2. SET_POWER_MODE
   PC → Device:
   ├── mode: PowerMode
   │   ├── HighPerformance  // 最高性能，高功耗
   │   ├── Balanced         // 平衡模式 (默认)
   │   └── PowerSaver       // 省电模式
   └── 响应: ACK

3. ENTER_SLEEP
   PC → Device:
   ├── wake_condition: WakeCondition
   │   ├── OnConnect        // 有连接时唤醒
   │   ├── OnButton         // 按钮唤醒
   │   ├── OnTimer(secs)    // 定时唤醒
   │   └── OnMotion         // 运动检测唤醒 (如有加速度计)
   └── 响应: ACK + 预计睡眠时间

4. SHUTDOWN
   PC → Device:
   ├── confirm: bool (需要确认)
   └── 响应: ACK (然后关机)
```

### 18.4 低电量策略

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    低电量处理策略                                        │
└─────────────────────────────────────────────────────────────────────────┘

电量阈值与动作:

┌─────────────────────────────────────────────────────────────────┐
│ 电量      │ 状态              │ 动作                           │
├─────────────────────────────────────────────────────────────────┤
│ > 20%     │ Normal            │ 正常运行                       │
│ 10-20%    │ LowBattery        │ 发送警告事件，建议充电         │
│ 5-10%     │ VeryLow           │ 自动切换省电模式               │
│           │                   │ - 降低采样率到 250Hz           │
│           │                   │ - 增大 BLE CI 到 100ms         │
│ < 5%      │ Critical          │ 发送紧急事件                   │
│           │                   │ - 停止采集                     │
│           │                   │ - 保存当前数据                 │
│           │                   │ - 准备关机                     │
│ < 2%      │ Shutdown          │ 强制关机保护电池               │
└─────────────────────────────────────────────────────────────────┘

事件通知:
├── BatteryLow(level)       // 电量低警告
├── BatteryVeryLow(level)   // 电量极低
├── BatteryCritical(level)  // 电量危急
├── ChargingStarted         // 开始充电
├── ChargingComplete        // 充电完成
└── ShutdownImminent        // 即将关机
```

### 18.5 功耗优化策略

```
功耗优化配置:

PowerProfile
├── high_performance:
│   ├── sample_rate: 1000 Hz
│   ├── ble_ci: 7.5-15 ms
│   ├── all_channels: true
│   └── estimated_runtime: 2-3 hours
│
├── balanced (默认):
│   ├── sample_rate: 250 Hz
│   ├── ble_ci: 30-50 ms
│   ├── all_channels: true
│   └── estimated_runtime: 6-8 hours
│
└── power_saver:
    ├── sample_rate: 250 Hz
    ├── ble_ci: 100-200 ms
    ├── active_channels: 减少到必要通道
    └── estimated_runtime: 10-12 hours

自动切换逻辑:
├── 电量 < 20% 且未充电 → 切换到 power_saver
├── 开始充电 → 恢复到 balanced
└── 用户可手动覆盖
```

### 18.6 充电状态显示

```
充电状态 LED 指示 (固件实现):

┌─────────────────────────────────────────────────────────────────┐
│ 状态              │ LED 行为                                   │
├─────────────────────────────────────────────────────────────────┤
│ 充电中 (< 100%)   │ 橙色呼吸灯                                 │
│ 充电完成          │ 绿色常亮                                   │
│ 低电量            │ 红色闪烁 (1Hz)                             │
│ 电量危急          │ 红色快闪 (2Hz)                             │
│ 正常运行          │ 绿色闪烁 (0.5Hz)                           │
│ 采集中            │ 蓝色常亮                                   │
│ 错误              │ 红色常亮                                   │
└─────────────────────────────────────────────────────────────────┘

软件端显示:
├── 电池图标 + 百分比
├── 预计剩余时间
├── 充电状态指示
└── 电池健康警告 (如有)
```

---

## 19. 待决策事项

### 19.1 协议设计 (需与固件团队确认)

- [ ] 同步字选择 (0xAA55 或其他)
- [ ] 时间戳精度 (μs 还是 ms)
- [x] ~~事件标记包格式~~ → **已设计** (见第 13 节)
- [ ] 配置命令格式完整定义
- [ ] 时钟同步握手协议细节

### 19.2 蓝牙策略 (部分已决策)

- [x] ~~SPP 是否为必须支持？~~ → **不支持**，仅使用 BLE (跨平台实现复杂)
- [x] ~~BLE 数据包大小限制如何处理？~~ → **动态 MTU 适配** (见第 9 节)
- [ ] BLE MTU 协商失败时的降级策略？

### 19.3 性能目标 (大部分已决策)

- [x] ~~最大采样率？~~ → **可配置** (250/500/1000/2000+ Hz，见第 10 节)
- [x] ~~最大通道数？~~ → **可配置** (8/16/32，见第 11 节)
- [x] ~~时间同步策略？~~ → **LSL 时钟同步** (见第 12 节)
- [x] ~~端到端延迟目标？~~ → **< 200ms 软实时** (见第 14 节)
- [x] ~~事件标记来源？~~ → **TTL + 软件 + 自动检测** (见第 13 节)

### 19.4 可靠性策略 (待确认)

- [ ] 默认可靠性级别？(BestEffort / DetectedLoss / Guaranteed)
- [ ] 采样率切换时是否允许数据丢失？
- [ ] 重连最大重试次数？
- [ ] 丢包填充策略？(NaN / 插值 / 跳过)

### 19.5 设备管理 (待确认)

- [ ] 是否需要支持多设备同时连接？→ 架构已支持，需确认产品需求
- [ ] 多设备时钟同步精度要求？
- [ ] 时钟同步频率？(10s / 30s / 60s)
- [ ] 设备配对方式？(Just Works / Passkey / OOB)

### 19.6 OTA 升级 (待确认)

- [ ] 是否需要固件签名验证？
- [ ] 是否允许版本降级？
- [ ] OTA 传输块大小？(256 / 512 / 1024 bytes)
- [ ] 是否支持断点续传？

### 19.7 电源管理 (待确认)

- [ ] 低电量阈值设置？(10% / 15% / 20%)
- [ ] 自动省电模式触发条件？
- [ ] 电池健康监测是否需要？
- [ ] 是否支持远程关机命令？

---

## 20. 下一步行动

### 20.1 Phase 1: 核心功能

1. **协议定义**: 与固件团队确认数据包格式和配置命令
2. **原型验证**: 先实现串口 + ndarray 输出的最小路径
3. **BLE 集成**: 使用 btleplug 实现 BLE 传输 + MTU 协商
4. **采样率管理**: 实现运行时采样率切换机制
5. **通道配置**: 实现动态通道数和通道掩码

### 20.2 Phase 2: 数据处理

6. **时钟同步**: 实现 LSL 时钟同步机制
7. **事件标记**: 实现 TTL 触发和软件标记
8. **错误处理**: 实现丢包检测和连接恢复
9. **文件输出**: 集成 EDF/XDF 写入
10. **LSL 发布**: 集成 LSL outlet

### 20.3 Phase 3: 设备管理

11. **设备发现**: 实现多传输方式的设备扫描
12. **设备配对**: 实现 BLE 配对和设备记忆
13. **OTA 升级**: 实现固件升级功能
14. **电源管理**: 实现电池监控和低电量策略

---

## 21. 附录：配置命令汇总

### 21.1 基础命令

| 命令 | 代码 | 参数 | 说明 |
|------|------|------|------|
| GET_DEVICE_INFO | 0x01 | - | 获取设备信息 |
| SET_SAMPLE_RATE | 0x02 | sample_rate_hz | 设置采样率 |
| SET_PACKET_SIZE | 0x03 | samples_per_packet | 设置每包样本数 |
| SET_CHANNEL_MASK | 0x04 | channel_mask | 设置通道掩码 |
| START_ACQUISITION | 0x10 | - | 开始采集 |
| STOP_ACQUISITION | 0x11 | - | 停止采集 |

### 21.2 传输配置

| 命令 | 代码 | 参数 | 说明 |
|------|------|------|------|
| SET_BLE_PARAMS | 0x20 | ci_min, ci_max, ... | 设置 BLE 参数 |
| SET_LATENCY_MODE | 0x21 | mode | 设置延迟模式 |
| SET_RELIABILITY_LEVEL | 0x22 | level | 设置可靠性级别 |

### 21.3 事件与同步

| 命令 | 代码 | 参数 | 说明 |
|------|------|------|------|
| SEND_MARKER | 0x30 | event_code, data | 发送软件标记 |
| SYNC_REQUEST | 0x31 | pc_send_time, seq | 时钟同步请求 |

### 21.4 电源管理

| 命令 | 代码 | 参数 | 说明 |
|------|------|------|------|
| GET_BATTERY_INFO | 0x40 | - | 获取电池信息 |
| SET_POWER_MODE | 0x41 | mode | 设置电源模式 |
| ENTER_SLEEP | 0x42 | wake_condition | 进入睡眠 |
| SHUTDOWN | 0x43 | confirm | 关机 |

### 21.5 OTA 升级

| 命令 | 代码 | 参数 | 说明 |
|------|------|------|------|
| OTA_START | 0x50 | size, crc, version | 开始 OTA |
| OTA_DATA | 0x51 | chunk_index, data | 传输数据块 |
| OTA_VERIFY | 0x52 | - | 验证固件 |
| OTA_APPLY | 0x53 | - | 应用固件 |
| OTA_ABORT | 0x54 | - | 中止 OTA |

---

*Document Version: 0.5.0*
*Last Updated: 2026-02-01*
*Status: Layer 0 架构设计完成，待固件团队评审*
