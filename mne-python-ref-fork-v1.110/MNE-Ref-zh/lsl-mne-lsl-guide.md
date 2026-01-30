# Lab Streaming Layer (LSL) 与 MNE-LSL 完整指南

> **用途**: 实时脑电(EEG)、眼电(EOG)、肌电(EMG)数据采集与处理  
> **创建日期**: 2026-01-30  
> **核心技术**: 网络数据流、时间同步、实时分析

---

## 目录

1. [什么是 LSL](#什么是-lsl)
2. [什么是 MNE-LSL](#什么是-mne-lsl)
3. [核心概念](#核心概念)
4. [为什么需要 LSL](#为什么需要-lsl)
5. [系统架构](#系统架构)
6. [支持的设备](#支持的设备)
7. [快速开始](#快速开始)
8. [实战示例](#实战示例)
9. [与 MNE-Python 的关系](#与-mne-python-的关系)
10. [官方资源](#官方资源)

---

## 什么是 LSL

### Lab Streaming Layer (实验室数据流层)

**LSL** 是一个**开源的实时数据流系统**，专门设计用于科学实验中**多设备数据的统一采集**。

#### 核心功能

```
┌─────────────────────────────────────────────────┐
│          Lab Streaming Layer (LSL)              │
│                                                 │
│  🔹 网络传输 (TCP/IP)                           │
│  🔹 时间同步 (亚毫秒级精度)                      │
│  🔹 实时访问 (低延迟)                            │
│  🔹 集中记录 (XDF 格式)                          │
│  🔹 设备发现 (自动)                              │
│  🔹 跨平台 (Windows/Linux/macOS/Android/iOS)    │
└─────────────────────────────────────────────────┘
```

---

### 主要特点

| 特性 | 说明 | 优势 |
|------|------|------|
| **统一数据流** | 所有设备使用相同协议 | 简化集成 |
| **时间同步** | NTP 算法，< 1ms 精度 | 多设备同步 |
| **网络传输** | 基于 TCP | 无线/有线都支持 |
| **自动发现** | 无需手动配置 IP | 即插即用 |
| **类型安全** | 数据类型自动转换 | 防止错误 |
| **故障恢复** | 自动重连和数据缓冲 | 数据不丢失 |

---

### LSL 解决的问题

#### ❌ 传统方式的痛点

```
EEG 设备 ────┐
              ├──▶ 各自的软件 ────┐
眼动仪 ──────┤                   ├──▶ 数据不同步
              │                   │    时间戳不统一
刺激呈现 ────┤                   │    格式不兼容
              └──▶ 手动整合 ──────┘
```

#### ✅ LSL 方式

```
EEG 设备 ────▶ LSL Outlet ────┐
                               ├──▶ LSL 网络 ──▶ 自动同步
眼动仪 ──────▶ LSL Outlet ────┤                 统一时间戳
                               │                 XDF 格式
刺激呈现 ────▶ LSL Outlet ────┘
```

---

## 什么是 MNE-LSL

### MNE-Python + LSL = MNE-LSL

**MNE-LSL** 是 **MNE-Python** 的实时数据流扩展包，将 LSL 的实时能力与 MNE 的强大分析工具结合。

```
┌────────────────────────────────────────────────┐
│              MNE-LSL 架构                       │
│                                                │
│  ┌──────────────────────────────────────┐    │
│  │    高层 API (MNE 风格)               │    │
│  │  • StreamLSL (类似 Raw)              │    │
│  │  • EpochsLSL (实时分段)              │    │
│  │  • PlayerLSL (模拟流)                │    │
│  └──────────────┬───────────────────────┘    │
│                 ▼                             │
│  ┌──────────────────────────────────────┐    │
│  │    低层 API (改进的 pylsl)           │    │
│  │  • StreamOutlet (发送端)             │    │
│  │  • StreamInlet (接收端)              │    │
│  │  • StreamInfo (元数据)               │    │
│  │  • resolve_streams (查找)            │    │
│  └──────────────┬───────────────────────┘    │
│                 ▼                             │
│  ┌──────────────────────────────────────┐    │
│  │    liblsl (C++ 核心库)               │    │
│  └──────────────────────────────────────┘    │
└────────────────────────────────────────────────┘
```

---

### MNE-LSL 的优势

#### 1. **MNE 兼容性** ⭐⭐⭐⭐⭐

```python
# MNE-LSL 使用 MNE 的 API 风格
from mne_lsl.stream import StreamLSL

stream = StreamLSL(bufsize=5)  # 类似 mne.io.Raw
stream.connect()
data = stream.get_data()  # 返回 numpy array (n_channels, n_samples)
info = stream.info  # mne.Info 对象

# 可以直接用于 MNE 函数
stream.filter(l_freq=1, h_freq=40)  # MNE 滤波器
stream.set_eeg_reference('average')  # MNE 重参考
```

---

#### 2. **实时分析能力** ⭐⭐⭐⭐⭐

```python
# 实时 Epochs (事件触发分段)
from mne_lsl.stream import EpochsLSL

epochs_stream = EpochsLSL(
    stream,
    bufsize=20,      # 缓冲 20 个 epoch
    event_id={'stim': 1},
    tmin=-0.2,
    tmax=0.5
)

# 获取最新的 epochs
epochs_data = epochs_stream.get_data()  # (n_epochs, n_channels, n_times)
```

---

#### 3. **环形缓冲区管理** ⭐⭐⭐⭐⭐

```
传统 Raw 文件:
[========================] 固定长度，全部加载

实时 StreamLSL:
     [====写入指针→]
     ↑              ↓
[←读取←←←←←←←←←←←←←←] 环形缓冲区
     自动循环，只保留最新数据
```

---

#### 4. **数据模拟 (PlayerLSL)** ⭐⭐⭐⭐⭐

```python
# 将离线数据变成实时流 (用于开发和测试)
from mne_lsl.player import PlayerLSL
from mne.io import read_raw_fif

raw = read_raw_fif('sample_data.fif', preload=True)

# 创建模拟 LSL 流
player = PlayerLSL(raw, chunk_size=200, name='MockEEG')
player.start()  # 开始发送数据
```

---

## 核心概念

### 1. Stream (数据流)

**定义**: 单个设备的所有通道数据 + 元数据

```python
# 一个 EEG Stream 包含:
- 64 个通道 (Fz, Cz, Pz, ...)
- 采样率: 500 Hz
- 数据类型: float32
- 元数据: 通道名称、位置、单位等
```

---

### 2. Sample (样本)

**定义**: 某一时刻所有通道的单次测量

```python
# 一个 Sample (t=0.002s):
[
  0.000012,  # Fz 通道
  0.000008,  # Cz 通道
  0.000015,  # Pz 通道
  ...        # 其他 61 个通道
]
```

---

### 3. Chunk (数据块)

**定义**: 多个连续 Sample 的集合

```python
# 一个 Chunk (100 samples):
[
  [s1_ch1, s1_ch2, ..., s1_ch64],  # Sample 1
  [s2_ch1, s2_ch2, ..., s2_ch64],  # Sample 2
  ...
  [s100_ch1, s100_ch2, ..., s100_ch64]  # Sample 100
]

# 延迟 vs 吞吐量权衡:
- 小 chunk (1-10 samples): 低延迟 (~20ms)
- 大 chunk (100-1000 samples): 高吞吐量 (减少网络开销)
```

---

### 4. StreamOutlet (发送端)

**定义**: 数据发送者 (通常是硬件设备)

```python
from mne_lsl.lsl import StreamOutlet, StreamInfo

# 创建 Stream 信息
info = StreamInfo(
    name='MyEEG',
    stype='EEG',          # 类型
    n_channels=64,
    sfreq=500,            # 采样率
    dtype='float32',
    source_id='device-123'
)

# 创建 Outlet
outlet = StreamOutlet(info, chunk_size=100)

# 发送数据
outlet.push_sample([0.1, 0.2, ...])  # 单个 sample
outlet.push_chunk([[...], [...]])    # 多个 samples
```

---

### 5. StreamInlet (接收端)

**定义**: 数据接收者 (通常是分析软件)

```python
from mne_lsl.lsl import StreamInlet, resolve_streams

# 查找流
streams = resolve_streams(timeout=5)  # 自动发现网络上的流

# 连接到第一个流
inlet = StreamInlet(streams[0])

# 接收数据
sample, timestamp = inlet.pull_sample()  # 拉取单个 sample
chunk, timestamps = inlet.pull_chunk()   # 拉取多个 samples
```

---

### 6. Metadata (元数据)

**定义**: 描述数据流的 XML 信息

```xml
<info>
  <name>MyEEG</name>
  <type>EEG</type>
  <channel_count>64</channel_count>
  <nominal_srate>500</nominal_srate>
  <channel_format>float32</channel_format>
  <source_id>device-123</source_id>
  <desc>
    <channels>
      <channel>
        <label>Fz</label>
        <unit>microvolts</unit>
        <type>EEG</type>
      </channel>
      ...
    </channels>
  </desc>
</info>
```

---

## 为什么需要 LSL

### 应用场景

#### 1. **脑机接口 (BCI)** ⭐⭐⭐⭐⭐

```
EEG 信号 → 实时解码 → 控制指令 → 外部设备
         (< 100ms 延迟要求)

示例: 想象运动控制轮椅
- EEG 检测运动想象
- LSL 实时传输到分类器
- 控制轮椅移动方向
```

---

#### 2. **神经反馈 (Neurofeedback)** ⭐⭐⭐⭐⭐

```
EEG 信号 → 特征提取 → 可视化反馈 → 用户调整
         (实时显示 alpha 波功率)

示例: 冥想训练
- 监测 alpha 波 (8-12 Hz)
- 实时显示功率变化
- 用户学习控制脑活动
```

---

#### 3. **多模态同步采集** ⭐⭐⭐⭐⭐

```
EEG ─────────┐
             │
眼动追踪 ────┼──▶ LSL 网络 ──▶ 时间同步 ──▶ 统一记录
             │
刺激标记 ────┘
```

**示例: 阅读研究**
- EEG: 大脑活动
- 眼动仪: 注视点
- 刺激: 单词呈现时间
- **关键**: 三者必须精确同步 (< 1ms)

---

#### 4. **实时质量监控** ⭐⭐⭐⭐

```
实验中实时监控:
- 电极阻抗
- 信号质量
- 伪迹检测
- 受试者状态

发现问题 → 立即调整 → 避免数据浪费
```

---

#### 5. **分布式数据采集** ⭐⭐⭐⭐

```
实验室 A (EEG 设备)  ─┐
                      ├──▶ LSL 网络 (局域网)
实验室 B (分析电脑)  ─┘

优势:
- 无需设备和分析软件在同一台电脑
- 降低单台电脑负载
- 可远程监控
```

---

## 系统架构

### 完整生态系统

```
┌─────────────────────────────────────────────────────────┐
│                   LSL 生态系统                           │
└─────────────────────────────────────────────────────────┘

第 1 层: 硬件设备
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│ BioSemi  │ Brain    │ Cognionics│ emotiv   │ Tobii    │
│ (EEG)    │ Products │ (EEG)     │ (EEG)    │ (Eye)    │
└─────┬────┴─────┬────┴─────┬────┴─────┬────┴─────┬────┘
      │          │          │          │          │
      └──────────┴──────────┴──────────┴──────────┘
                            │
第 2 层: LSL Apps (设备驱动)
┌─────────────────────────────────────────────────────────┐
│ BioSemi App, BrainProducts App, Cognionics App, ...     │
│ → 将硬件数据转换为 LSL Outlet                            │
└──────────────────────────┬──────────────────────────────┘
                           │
第 3 层: LSL 网络层 (liblsl)
┌─────────────────────────────────────────────────────────┐
│ • 网络传输 (TCP)                                         │
│ • 时间同步 (NTP)                                         │
│ • 自动发现                                               │
│ • 故障恢复                                               │
└──────────────────────────┬──────────────────────────────┘
                           │
第 4 层: 客户端应用
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│ LabRecorder│ MNE-LSL │ MATLAB   │ Python   │ 自定义   │
│ (记录)    │ (分析)   │ (分析)   │ (pylsl)  │ 应用     │
└──────────┴──────────┴──────────┴──────────┴──────────┘
```

---

### 数据流向

```
┌─────────────────────────────────────────────────────────┐
│ 1. 硬件设备采集原始信号                                   │
│    EEG: 500 Hz, 64 通道                                  │
└───────────────┬─────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────┐
│ 2. LSL App 创建 StreamOutlet                            │
│    • 包装为 LSL 格式                                     │
│    • 添加时间戳                                          │
│    • 推送到网络                                          │
└───────────────┬─────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────┐
│ 3. LSL 网络层                                            │
│    • 广播流信息 (UDP)                                    │
│    • 传输数据 (TCP)                                      │
│    • 同步时钟                                            │
└───────────────┬─────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────┐
│ 4. MNE-LSL StreamInlet                                  │
│    • 接收数据块                                          │
│    • 填充环形缓冲区                                       │
│    • 提供 MNE API                                        │
└───────────────┬─────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────┐
│ 5. 实时分析                                              │
│    • 滤波 (NumPy/SciPy)                                 │
│    • 特征提取                                            │
│    • 分类/解码 (scikit-learn)                           │
│    • 反馈/控制                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 支持的设备

### EEG 设备 (脑电图)

| 厂商 | 设备 | 通道数 | LSL App |
|------|------|--------|---------|
| **BioSemi** | ActiveTwo | 8-256 | ✅ BioSemi |
| **Brain Products** | BrainAmp | 16-128 | ✅ BrainAmpSeries |
| **Brain Products** | LiveAmp | 8-64 | ✅ LiveAmp |
| **Brain Products** | ActiChamp | 32-160 | ✅ ActiChamp |
| **Cognionics** | Quick-20 | 20 | ✅ Cognionics |
| **emotiv** | EPOC/Insight | 5-14 | ✅ emotiv |
| **g.tec** | g.USBamp | 16 | ✅ g.Tec |
| **ANT Neuro** | eego sports | 32-256 | ✅ eegoSports |
| **Neuroscan** | SynAmps | 32-128 | ✅ Neuroscan |
| **EGI** | AmpServer | 32-256 | ✅ EGIAmpServer |

---

### 眼动追踪设备

| 厂商 | 设备 | LSL App |
|------|------|---------|
| **Tobii** | Pro X2/X3 | ✅ TobiiPro |
| **Tobii** | Stream Engine | ✅ TobiiStreamEngine |
| **SR Research** | EyeLink | ✅ EyeLink |
| **Pupil Labs** | Pupil Core | ✅ PupilLabs |
| **SMI** | iView | ✅ SMIEyetracker |
| **EyeTribe** | EyeTribe | ✅ EyeTribe |

---

### 其他传感器

| 类型 | 设备 | LSL App |
|------|------|---------|
| **运动捕捉** | OptiTrack | ✅ OptiTrack |
| **运动捕捉** | Qualisys | ✅ Qualisys |
| **运动捕捉** | PhaseSpace | ✅ PhaseSpace |
| **VR 追踪** | OpenVR (HTC Vive) | ✅ OpenVR |
| **游戏手柄** | Xbox/PS Controller | ✅ GameController |
| **音频** | Microphone | ✅ AudioCapture |
| **串口设备** | Arduino 等 | ✅ SerialPort |

---

## 快速开始

### 安装

#### 1. 安装 MNE-LSL

```bash
# 方法 1: pip (推荐)
pip install mne-lsl

# 方法 2: conda
conda install -c conda-forge mne-lsl

# 验证安装
python -c "import mne_lsl; print(mne_lsl.__version__)"
```

---

#### 2. 安装依赖

```bash
# MNE-LSL 自动安装以下依赖:
- mne >= 1.6           # MNE-Python
- numpy >= 1.21        # 数组计算
- scipy                # 信号处理
- pyqtgraph            # 实时可视化
- qtpy                 # Qt 界面
- psutil               # 系统监控
```

---

### 第一个 LSL 程序

#### 场景: 模拟 EEG 流并接收

```python
# ========== 步骤 1: 创建模拟数据流 (发送端) ==========
from mne_lsl.player import PlayerLSL
from mne.io import read_raw_fif
from mne.datasets import sample

# 加载示例数据
data_path = sample.data_path()
raw_file = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
raw = read_raw_fif(raw_file, preload=True)

# 只保留 EEG 通道
raw.pick('eeg')

# 创建 LSL 发送端 (模拟实时流)
player = PlayerLSL(
    raw,
    chunk_size=200,       # 每次发送 200 个样本
    name='SimulatedEEG',  # 流名称
    source_id='mock-001'  # 设备 ID
)

# 开始发送
player.start()
print("✅ 模拟 EEG 流已启动")

# ========== 步骤 2: 接收数据流 (接收端) ==========
from mne_lsl.stream import StreamLSL
import time

# 创建接收端
stream = StreamLSL(bufsize=5, name='SimulatedEEG')

# 连接到流
stream.connect(acquisition_delay=0.1)
print("✅ 已连接到 EEG 流")

# 等待缓冲区填充
time.sleep(2)

# 获取最新 1 秒数据
data, times = stream.get_data(winsize=1)  # (n_channels, n_samples)
print(f"✅ 接收到数据: {data.shape}")

# 查看 Info
print(stream.info)

# 停止流
player.stop()
```

**输出**:
```
✅ 模拟 EEG 流已启动
✅ 已连接到 EEG 流
✅ 接收到数据: (60, 600)  # 60 通道, 600 样本 (1秒 @ 600Hz)
<Info | 7 non-empty values
 bads: []
 ch_names: EEG 001, EEG 002, EEG 003, ...
 chs: 60 EEG
 custom_ref_applied: False
 dig: 146 items (3 Cardinal, 4 HPI, 61 EEG, 78 Extra)
 highpass: 0.0 Hz
 lowpass: 300.0 Hz
 meas_date: 2002-12-03 19:01:10 UTC
 nchan: 60
 projs: []
 sfreq: 600.0 Hz
>
```

---

## 实战示例

### 示例 1: 实时滤波和可视化

```python
from mne_lsl.stream import StreamLSL
from mne_lsl.player import PlayerLSL
import matplotlib.pyplot as plt
import numpy as np

# 1. 启动模拟流
raw = ...  # 你的 Raw 数据
player = PlayerLSL(raw, name='RealTimeEEG')
player.start()

# 2. 连接并配置
stream = StreamLSL(bufsize=10, name='RealTimeEEG')
stream.connect()

# 3. 应用滤波器 (在线滤波)
stream.filter(l_freq=1, h_freq=40, picks='eeg')

# 4. 实时循环
plt.ion()  # 交互模式
fig, ax = plt.subplots()

for i in range(100):  # 循环 100 次
    # 获取最新 2 秒数据
    data, times = stream.get_data(winsize=2)
    
    # 选择 1 个通道绘制
    channel_data = data[0, :]  # 第一个通道
    
    # 更新图形
    ax.clear()
    ax.plot(times, channel_data)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (V)')
    ax.set_title(f'Channel {stream.info["ch_names"][0]} - Iteration {i}')
    plt.pause(0.1)  # 暂停 100ms

plt.ioff()
player.stop()
```

---

### 示例 2: 实时 Epochs 提取

```python
from mne_lsl.stream import EpochsLSL
from mne_lsl.player import PlayerLSL

# 1. 准备带事件的数据
raw = ...  # 包含 Stim 通道的 Raw
player = PlayerLSL(raw, name='EpochStream')
player.start()

# 2. 连接 Stream
from mne_lsl.stream import StreamLSL
stream = StreamLSL(bufsize=10, name='EpochStream')
stream.connect()

# 3. 创建实时 Epochs
epochs_stream = EpochsLSL(
    stream,
    bufsize=20,              # 缓冲 20 个 epochs
    event_channels='STI 014', # 事件通道
    event_id={'visual': 3},   # 事件 ID
    tmin=-0.2,               # Epoch 起始 (相对事件)
    tmax=0.5,                # Epoch 结束
    baseline=(None, 0)       # 基线校正
)

# 4. 获取实时 Epochs
import time
time.sleep(5)  # 等待积累 epochs

# 获取最新的 epochs
epochs_data = epochs_stream.get_data()  # (n_epochs, n_channels, n_times)
print(f"Collected {epochs_data.shape[0]} epochs")

# 计算平均 Evoked
evoked = epochs_data.mean(axis=0)  # (n_channels, n_times)

# 绘制
import matplotlib.pyplot as plt
plt.plot(epochs_stream.times, evoked[10, :])  # 第 10 个通道
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Real-time Evoked Response')
plt.show()

player.stop()
```

---

### 示例 3: 实时功率谱监控

```python
from mne_lsl.stream import StreamLSL
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# 连接流
stream = StreamLSL(bufsize=10, name='MyEEG')
stream.connect()

# 配置
sfreq = stream.info['sfreq']
channel_idx = 0  # 监控第一个通道

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

for i in range(100):
    # 获取 4 秒数据
    data, times = stream.get_data(winsize=4)
    channel_data = data[channel_idx, :]
    
    # 计算功率谱 (Welch 方法)
    freqs, psd = signal.welch(
        channel_data,
        fs=sfreq,
        nperseg=int(sfreq * 2)  # 2 秒窗口
    )
    
    # 计算频段功率
    def band_power(freqs, psd, low, high):
        idx = np.logical_and(freqs >= low, freqs <= high)
        return np.trapz(psd[idx], freqs[idx])
    
    delta = band_power(freqs, psd, 1, 4)
    theta = band_power(freqs, psd, 4, 8)
    alpha = band_power(freqs, psd, 8, 13)
    beta = band_power(freqs, psd, 13, 30)
    
    # 绘制时域
    ax1.clear()
    ax1.plot(times, channel_data)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Channel {stream.info["ch_names"][channel_idx]}')
    
    # 绘制频域
    ax2.clear()
    ax2.semilogy(freqs, psd)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_xlim([0, 50])
    
    # 添加频段标注
    ax2.text(2.5, max(psd), f'δ: {delta:.2e}', ha='center')
    ax2.text(6, max(psd), f'θ: {theta:.2e}', ha='center')
    ax2.text(10, max(psd), f'α: {alpha:.2e}', ha='center')
    ax2.text(20, max(psd), f'β: {beta:.2e}', ha='center')
    
    plt.tight_layout()
    plt.pause(0.5)

plt.ioff()
```

---

### 示例 4: 多流同步

```python
from mne_lsl.lsl import resolve_streams
from mne_lsl.stream import StreamLSL
import time

# 查找所有流
streams = resolve_streams(timeout=5)
print(f"发现 {len(streams)} 个流:")
for s in streams:
    print(f"  - {s.name()} ({s.type()})")

# 连接到 EEG 和 Eye Tracker
eeg_stream = StreamLSL(bufsize=5, name='EEG')
eye_stream = StreamLSL(bufsize=5, name='EyeTracker')

eeg_stream.connect()
eye_stream.connect()

time.sleep(2)

# 获取同步数据 (LSL 自动同步时间戳)
eeg_data, eeg_times = eeg_stream.get_data(winsize=1)
eye_data, eye_times = eye_stream.get_data(winsize=1)

print(f"EEG: {eeg_data.shape}, times: {eeg_times[0]:.3f} - {eeg_times[-1]:.3f}")
print(f"Eye: {eye_data.shape}, times: {eye_times[0]:.3f} - {eye_times[-1]:.3f}")

# 时间差 < 1ms (LSL 保证)
time_diff = abs(eeg_times[0] - eye_times[0])
print(f"时间同步误差: {time_diff * 1000:.2f} ms")
```

---

## 与 MNE-Python 的关系

### 对比表

| 特性 | MNE-Python | MNE-LSL |
|------|-----------|---------|
| **数据来源** | 文件 (FIF, EDF, ...) | 实时网络流 |
| **数据长度** | 固定 | 无限 (实时) |
| **访问方式** | 随机访问 | 只能访问最新数据 |
| **时间** | 离线分析 | 实时分析 |
| **缓冲** | 全部加载 | 环形缓冲区 |
| **API** | `mne.io.Raw` | `mne_lsl.stream.StreamLSL` |

---

### 无缝集成

```python
# MNE-LSL 对象可以直接用于 MNE 函数

from mne_lsl.stream import StreamLSL
import mne

# 创建 StreamLSL
stream = StreamLSL(bufsize=10)
stream.connect()

# 1. 滤波 (MNE API)
stream.filter(l_freq=1, h_freq=40)

# 2. 设置参考 (MNE API)
stream.set_eeg_reference('average')

# 3. 应用 ICA (需要先转换为 Raw)
data, times = stream.get_data(winsize=60)  # 60 秒
info = stream.info.copy()
raw = mne.io.RawArray(data, info)

# 现在可以用所有 MNE 功能
ica = mne.preprocessing.ICA(n_components=20)
ica.fit(raw)
ica.apply(raw)
```

---

## 官方资源

### Lab Streaming Layer (LSL)

- **官网**: https://labstreaminglayer.org/
- **GitHub**: https://github.com/sccn/labstreaminglayer
- **文档**: https://labstreaminglayer.readthedocs.io/
- **论文**: Kothe et al. (2025). *Imaging Neuroscience*
  - DOI: https://doi.org/10.1162/IMAG.a.136

---

### MNE-LSL

- **官网**: https://mne.tools/mne-lsl/
- **GitHub**: https://github.com/mne-tools/mne-lsl
- **文档**: https://mne.tools/mne-lsl/stable/
- **安装**: https://mne.tools/mne-lsl/stable/resources/install.html
- **论文**: https://doi.org/10.21105/joss.08088

---

### 社区支持

- **Slack**: https://labstreaminglayer.slack.com (加入 #users 频道)
- **论坛**: https://forum.labstreaminglayer.org/
- **Issues**: 
  - LSL: https://github.com/sccn/labstreaminglayer/issues
  - MNE-LSL: https://github.com/mne-tools/mne-lsl/issues

---

## 总结

### LSL 核心价值

1. **统一接口**: 所有设备使用相同协议
2. **时间同步**: 亚毫秒级精度
3. **即插即用**: 自动设备发现
4. **跨平台**: Windows/Linux/macOS/移动端
5. **开源免费**: BSD 许可证

---

### MNE-LSL 核心价值

1. **MNE 集成**: 使用熟悉的 MNE API
2. **实时分析**: NumPy/SciPy/scikit-learn
3. **易于开发**: PlayerLSL 模拟流
4. **环形缓冲**: 自动管理内存
5. **Python 优先**: 简洁的 Python 接口

---

### 适用人群

✅ **强烈推荐 LSL 如果你**:
- 做实时脑机接口 (BCI)
- 需要多设备同步采集
- 做神经反馈训练
- 需要实时信号质量监控
- 使用多台电脑协同工作

❌ **不需要 LSL 如果你**:
- 只做离线数据分析
- 单设备、单电脑
- 数据已经采集完成
- 不需要实时反馈

---

**返回**: [依赖分析总览](dependency-analysis-overview.md)  
**相关**: [EEG/EOG/EMG 核心依赖](eeg-eog-emg-core-dependencies.md)
