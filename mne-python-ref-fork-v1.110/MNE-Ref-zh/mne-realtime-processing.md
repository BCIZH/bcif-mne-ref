# MNE 实时处理完整指南 (MNE-LSL)

> **用途**: 理解 MNE-LSL 实时数据流处理  
> **创建日期**: 2026-01-31  
> **核心**: LSL 流 → 环形缓冲 → 实时分析 → 即时反馈

---

## 目录

1. [什么是实时处理](#什么是实时处理)
2. [核心类: StreamLSL](#核心类-streamlsl)
3. [环形缓冲区](#环形缓冲区)
4. [完整工作流](#完整工作流)
5. [代码示例](#代码示例)
6. [性能要求](#性能要求)

---

## 什么是实时处理

### 定义

**实时处理** = 在数据采集的同时进行分析和反馈

```
时间线:
  数据采集 (进行中) = 实时分析 (同步) = 实时反馈
  
数据流:
  硬件设备 → LSL Outlet → 网络 → LSL Inlet → MNE-LSL → 分析 → 反馈
                                              ↓
                                          实时显示/控制
```

### 核心特点

| 特性 | 说明 |
|------|------|
| **数据来源** | 网络数据流 (LSL) |
| **数据状态** | 动态、持续增长、无限长度 |
| **访问方式** | 只能访问最新数据 (环形缓冲) |
| **处理时机** | 实验进行中,有严格时间限制 |
| **延迟要求** | < 100ms (BCI), < 1s (神经反馈) |
| **典型用途** | BCI、神经反馈、实时监控 |

---

## 核心类: StreamLSL

### 类定义

```python
# 位置: mne_lsl/stream/stream_lsl.py

from mne_lsl.stream import StreamLSL

class StreamLSL(BaseStream):
    """
    实时 LSL 数据流对象
    
    类似 mne.io.Raw,但用于实时数据:
    - 无限长度 (持续采集)
    - 环形缓冲区 (固定内存)
    - 只能访问最新数据
    """
    
    def __init__(self, bufsize, *, name=None, stype=None, source_id=None):
        """
        Parameters
        ----------
        bufsize : float
            缓冲区大小 (秒),例如 5.0 = 保留最新 5 秒数据
        name : str
            LSL 流名称
        stype : str  
            LSL 流类型 (例: 'EEG')
        source_id : str
            LSL 流源 ID (唯一标识)
        """
```

### 与 Raw 的对比

| 方法/属性 | mne.io.Raw (离线) | StreamLSL (实时) |
|----------|------------------|-----------------|
| **数据长度** | 固定 (文件大小) | 无限 (持续流) |
| **数据访问** | `raw[:, start:stop]` | `stream.get_data(winsize)` |
| **内存占用** | 全部或部分 | 固定缓冲区 |
| **时间访问** | 双向 (可回溯) | 单向 (只能最新) |
| **滤波** | `filter()` 零相位 | `filter()` 因果相位 |
| **Info** | `raw.info` | `stream.info` |

---

## 环形缓冲区

### 原理图

```
┌─────────────────────────────────────────────────────┐
│           环形缓冲区 (Ring Buffer)                   │
│           bufsize = 5 秒 @ 500 Hz                   │
└─────────────────────────────────────────────────────┘

时刻 t=0:
  [─────空白缓冲区─────]
   ↑ head (读)
   ↑ tail (写)

时刻 t=2s: (填充中)
  [███已有2秒数据███───空白───]
   ↑                  ↑
  head              tail

时刻 t=5s: (填满)
  [█████5秒数据█████]
   ↑                ↑
  head            tail

时刻 t=7s: (循环覆盖)
  [新2s█████旧3s数据█]
        ↑           ↑
       tail        head
  • 最旧的 2 秒被覆盖
  • 只保留最新 5 秒

数据形状:
  • 数组: (n_channels, bufsize * sfreq)
  • 例: (64 通道, 5秒 × 500Hz) = (64, 2500)
  • 内存: 64 × 2500 × 4 字节 = 640 KB
```

### 指针管理

```python
# 内部实现 (简化版)
class RingBuffer:
    def __init__(self, n_channels, bufsize_samples):
        self.buffer = np.zeros((n_channels, bufsize_samples))
        self.head = 0  # 读指针 (最旧数据)
        self.tail = 0  # 写指针 (写入位置)
    
    def push(self, new_data):
        """写入新数据"""
        n_samples = new_data.shape[1]
        
        # 写入数据 (可能跨越数组边界)
        if self.tail + n_samples <= self.bufsize:
            self.buffer[:, self.tail:self.tail+n_samples] = new_data
        else:
            # 分两段写入 (循环)
            n_end = self.bufsize - self.tail
            self.buffer[:, self.tail:] = new_data[:, :n_end]
            self.buffer[:, :n_samples-n_end] = new_data[:, n_end:]
        
        # 更新指针
        self.tail = (self.tail + n_samples) % self.bufsize
        if self.tail <= self.head:
            self.head = self.tail  # 数据被覆盖
    
    def get_latest(self, n_samples):
        """获取最新 n_samples"""
        start = (self.tail - n_samples) % self.bufsize
        # ... 返回数据 ...
```

---

## 完整工作流

### 实时系统架构

```
┌──────────────────────────────────────────────────────┐
│  第 1 层: 硬件 + LSL Outlet                          │
│                                                      │
│  EEG 设备 (Brain Products LiveAmp)                  │
│     ↓                                                │
│  LSL Outlet (由设备驱动创建)                         │
│  • 每 200ms 发送 100 samples (500 Hz)               │
│  • Stream name: "LiveAmpSN-12345"                   │
│  • Stream type: "EEG"                               │
└────────────────┬─────────────────────────────────────┘
                 │ 网络传输 (TCP)
                 ▼
┌──────────────────────────────────────────────────────┐
│  第 2 层: MNE-LSL StreamLSL                          │
│                                                      │
│  stream = StreamLSL(bufsize=5, name="LiveAmpSN...")  │
│  stream.connect()                                    │
│  stream.filter(1, 40, phase='minimum')  # 因果滤波   │
│                                                      │
│  环形缓冲区: 保留最新 5 秒                            │
└────────────────┬─────────────────────────────────────┘
                 │ 数据获取
                 ▼
┌──────────────────────────────────────────────────────┐
│  第 3 层: 实时处理                                    │
│                                                      │
│  while True:                                         │
│      data = stream.get_data(winsize=1)  # 最新1秒   │
│      features = extract_features(data)  # < 50ms    │
│      prediction = classifier.predict(features)       │
│      send_feedback(prediction)          # 反馈      │
└──────────────────────────────────────────────────────┘
```

### 标准流程

```
┌─────────────────────────────────────────────┐
│  步骤 1: 启动 LSL 流                         │
│  • 启动硬件设备                              │
│  • 或创建 PlayerLSL (模拟)                   │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  步骤 2: 连接 StreamLSL                      │
│  stream = StreamLSL(bufsize=5)               │
│  stream.connect()                            │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  步骤 3: 配置处理参数                        │
│  stream.filter(1, 40, phase='minimum')       │
│  stream.set_eeg_reference('average')         │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  步骤 4: 等待缓冲区填充                      │
│  time.sleep(2)  # 等待 2 秒                 │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  步骤 5: 实时处理循环                        │
│  while condition:                            │
│      data = stream.get_data(winsize=1)       │
│      result = process(data)                  │
│      display_or_control(result)              │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  步骤 6: 断开连接                            │
│  stream.disconnect()                         │
└─────────────────────────────────────────────┘
```

---

## 代码示例

### 示例 1: 实时功率谱监控

```python
from mne_lsl.stream import StreamLSL
from mne_lsl.player import PlayerLSL
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import time

# ========== 1. 启动模拟流 (实际使用时替换为真实设备) ==========
from mne.io import read_raw_fif
from mne.datasets import sample

raw_file = sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
raw = read_raw_fif(raw_file, preload=True).pick('eeg')

player = PlayerLSL(raw, chunk_size=200, name='PowerMonitor', source_id='demo-001')
player.start()
print("✅ 模拟 EEG 流已启动")

# ========== 2. 连接 StreamLSL ==========
stream = StreamLSL(bufsize=5, name='PowerMonitor', source_id='demo-001')
stream.connect(acquisition_delay=0.1)
print(f"✅ 已连接,通道: {len(stream.ch_names)}, 采样率: {stream.info['sfreq']} Hz")

# ========== 3. 预处理 ==========
stream.filter(l_freq=0.5, h_freq=40, phase='minimum')  # 因果滤波

# ========== 4. 等待缓冲区填充 ==========
time.sleep(2)
print("✅ 缓冲区已填充")

# ========== 5. 实时监控循环 ==========
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 选择监控通道 (例: Oz,后脑,视觉皮层)
monitor_ch = stream.ch_names[0]  # 第一个通道
ch_idx = stream.ch_names.index(monitor_ch)

for iteration in range(50):  # 运行 50 次 (约 50 秒)
    # 获取最新 2 秒数据
    data, times = stream.get_data(winsize=2)
    
    # 提取监控通道
    ch_data = data[ch_idx, :]
    
    # 计算功率谱密度
    sfreq = stream.info['sfreq']
    freqs, psd = welch(ch_data, fs=sfreq, nperseg=int(sfreq))
    
    # 计算频段功率
    def band_power(freqs, psd, fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return np.trapz(psd[idx], freqs[idx])
    
    delta = band_power(freqs, psd, 1, 4)
    theta = band_power(freqs, psd, 4, 8)
    alpha = band_power(freqs, psd, 8, 13)
    beta = band_power(freqs, psd, 13, 30)
    
    # ========== 绘图 ==========
    # 图 1: 时域波形
    ax1.clear()
    ax1.plot(times, ch_data * 1e6)  # 转换为 μV
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('幅值 (μV)')
    ax1.set_title(f'实时波形 - {monitor_ch} (迭代 {iteration+1}/50)')
    ax1.grid(True, alpha=0.3)
    
    # 图 2: 频域功率谱
    ax2.clear()
    ax2.semilogy(freqs, psd)
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('功率谱密度')
    ax2.set_xlim([0, 40])
    ax2.grid(True, alpha=0.3)
    
    # 标注频段功率
    ax2.text(2, max(psd)*0.8, f'δ: {delta:.2e}', fontsize=10)
    ax2.text(6, max(psd)*0.6, f'θ: {theta:.2e}', fontsize=10)
    ax2.text(10, max(psd)*0.4, f'α: {alpha:.2e}', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax2.text(20, max(psd)*0.2, f'β: {beta:.2e}', fontsize=10)
    
    plt.tight_layout()
    plt.pause(1)  # 更新间隔 1 秒
    
    print(f"迭代 {iteration+1}: Alpha = {alpha:.2e}")

plt.ioff()
stream.disconnect()
player.stop()
print("✅ 监控结束")
```

---

### 示例 2: 实时 Epochs 和 Evoked

```python
from mne_lsl.stream import StreamLSL, EpochsStream
from mne_lsl.player import PlayerLSL
from mne import combine_evoked, EvokedArray
import numpy as np
import matplotlib.pyplot as plt

# ========== 1. 准备数据流 ==========
from mne.io import read_raw_fif
from mne.datasets import sample

raw = read_raw_fif(
    sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif',
    preload=True
).pick(('meg', 'stim'))

player = PlayerLSL(raw, chunk_size=200, name='EvokedDemo', source_id='demo-002')
player.start()

# ========== 2. 连接 StreamLSL ==========
stream = StreamLSL(bufsize=4, name='EvokedDemo', source_id='demo-002')
stream.connect(acquisition_delay=0.1, processing_flags='all')

# 预处理
stream.filter(None, 40, picks='grad')
stream.info['bads'] = ['MEG 2443']  # 标记坏通道

# ========== 3. 创建实时 Epochs ==========
epochs = EpochsStream(
    stream,
    bufsize=20,              # 保留最新 20 个 epochs
    event_id=1,              # 事件 ID
    event_channels='STI 014', # 事件通道
    tmin=-0.2,               # epoch 起始
    tmax=0.5,                # epoch 结束
    baseline=(None, 0),      # 基线校正
    picks='grad'             # 只保留 MEG 梯度计
)

epochs.connect(acquisition_delay=0.1)
print("✅ 实时 Epochs 已启动")

# ========== 4. 实时累积 Evoked ==========
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))

n_epochs_total = 0
evoked = None

target_epochs = 20  # 目标: 累积 20 个 epochs

while n_epochs_total < target_epochs:
    # 检查是否有新 epochs
    if epochs.n_new_epochs == 0:
        time.sleep(0.1)
        continue
    
    print(f"收到 {epochs.n_new_epochs} 个新 epochs (总计: {n_epochs_total}→{n_epochs_total + epochs.n_new_epochs})")
    
    # 获取新 epochs
    data = epochs.get_data(n_epochs=epochs.n_new_epochs)
    
    # 创建新的 Evoked
    new_evoked = EvokedArray(
        np.average(data, axis=0),  # 平均
        epochs.info,
        nave=data.shape[0],
        tmin=epochs.tmin
    )
    
    # 合并到累积 Evoked
    if evoked is None:
        evoked = new_evoked
    else:
        evoked = combine_evoked([evoked, new_evoked], weights='nave')
    
    n_epochs_total += epochs.n_new_epochs
    evoked.nave = n_epochs_total  # 修正总数
    
    # 实时绘图
    ax.clear()
    evoked.plot(axes=ax, time_unit='s', show=False)
    ax.set_title(f'实时 Evoked (N={n_epochs_total} epochs)')
    fig.canvas.draw()
    fig.canvas.flush_events()

plt.ioff()
print(f"✅ 完成! 累积了 {n_epochs_total} 个 epochs")

# 保存结果
evoked.save('realtime-evoked-ave.fif', overwrite=True)

# 清理
epochs.disconnect()
stream.disconnect()
player.stop()
```

---

### 示例 3: 简化的 BCI 分类器

```python
from mne_lsl.stream import StreamLSL
from mne_lsl.player import PlayerLSL
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import welch
import time

# ========== 离线训练部分 (使用历史数据) ==========
print("阶段 1: 离线训练分类器...")

from mne.io import read_raw_fif
from mne.datasets import sample

# 读取训练数据
raw_train = read_raw_fif(
    sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif',
    preload=True
).pick('eeg')

raw_train.filter(8, 30)  # Mu/Beta 频段

# 提取训练特征 (简化示例)
# 假设: C3/C4 通道的 mu 波功率区分左/右手想象
def extract_features(data, sfreq, ch_names):
    """提取 C3, C4 的 mu 波功率"""
    # 找到类似 C3, C4 的通道
    c3_idx = 0  # 简化: 使用前两个通道
    c4_idx = 1
    
    # 计算功率谱
    freqs, psd_c3 = welch(data[c3_idx, :], fs=sfreq)
    _, psd_c4 = welch(data[c4_idx, :], fs=sfreq)
    
    # Mu 频段 (8-12 Hz)
    mu_idx = np.logical_and(freqs >= 8, freqs <= 12)
    mu_c3 = np.mean(psd_c3[mu_idx])
    mu_c4 = np.mean(psd_c4[mu_idx])
    
    return np.array([mu_c3, mu_c4])

# 模拟训练数据 (实际应用中应从标记的 epochs 提取)
X_train = np.random.randn(100, 2)  # 100 个样本, 2 个特征
y_train = np.random.randint(0, 2, 100)  # 0=左手, 1=右手

# 训练分类器
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
print(f"✅ 分类器训练完成")

# ========== 实时 BCI 部分 ==========
print("\n阶段 2: 实时 BCI 控制...")

# 启动实时流
player = PlayerLSL(raw_train.copy(), chunk_size=200, 
                   name='BCIDemo', source_id='demo-003')
player.start()

stream = StreamLSL(bufsize=5, name='BCIDemo', source_id='demo-003')
stream.connect(acquisition_delay=0.1)
stream.filter(8, 30, phase='minimum')  # 因果滤波

time.sleep(2)  # 等待缓冲

print("\n🧠 BCI 开始! (想象左/右手运动)")
print("=" * 50)

for i in range(20):  # 运行 20 次
    # 获取最新 2 秒数据
    data, _ = stream.get_data(winsize=2)
    
    # 提取特征
    features = extract_features(data, stream.info['sfreq'], stream.ch_names)
    
    # 分类
    prediction = clf.predict(features.reshape(1, -1))[0]
    confidence = clf.predict_proba(features.reshape(1, -1))[0]
    
    # 输出控制信号
    direction = "◀◀◀ 左手" if prediction == 0 else "右手 ▶▶▶"
    conf_percent = confidence[prediction] * 100
    
    print(f"迭代 {i+1:2d}: {direction} (置信度: {conf_percent:.1f}%)")
    
    # 模拟控制外部设备
    # if prediction == 0:
    #     send_command_to_device('move_left')
    # else:
    #     send_command_to_device('move_right')
    
    time.sleep(0.5)  # 控制频率: 2 Hz

print("=" * 50)
print("✅ BCI 演示结束")

stream.disconnect()
player.stop()
```

---

## 性能要求

### 延迟要求

| 应用 | 最大延迟 | 说明 |
|------|---------|------|
| **BCI 控制** | < 100 ms | 用户感觉实时 |
| **神经反馈** | < 500 ms | 可接受反馈延迟 |
| **质量监控** | < 2 s | 及时发现问题 |

### 计算性能

```python
# ✅ 快速算法 (适合实时)
# 每次迭代 < 50ms

def good_realtime_processing(data):
    # 简单特征提取
    mean = np.mean(data, axis=1)           # < 1ms
    std = np.std(data, axis=1)             # < 1ms
    
    # Welch 功率谱
    freqs, psd = welch(data, fs=500)       # < 10ms
    
    # 带通功率
    alpha_power = bandpower(psd, 8, 12)    # < 1ms
    
    # 分类
    prediction = clf.predict([features])   # < 1ms
    
    return prediction  # 总计 < 15ms ✅

# ❌ 慢速算法 (不适合实时)
def bad_realtime_processing(data):
    # ICA 需要多次迭代
    ica = ICA(n_components=20)
    ica.fit(raw)  # 数分钟 ❌
    
    # 源定位需要复杂计算
    stc = apply_inverse(evoked, inv)  # 数秒 ❌
    
    return result
```

### 内存管理

```python
# 环形缓冲区大小选择

# 太小: 缓冲区溢出,数据丢失
stream = StreamLSL(bufsize=0.5)  # ⚠️ 只有 0.5 秒!
# 如果处理时间 > 0.5s,数据会丢失

# 合适: 足够处理时间
stream = StreamLSL(bufsize=5)  # ✅ 5 秒缓冲
# 内存: 64 ch × 5s × 500Hz × 4 bytes = 640 KB

# 太大: 浪费内存,延迟增加
stream = StreamLSL(bufsize=60)  # ⚠️ 60 秒 = 7.7 MB
# 不必要,实时应用只需要最新数据
```

---

## 总结

### 实时处理适用场景

✅ **强烈推荐用于**:
- 脑机接口 (BCI) 控制
- 神经反馈训练
- 实时信号质量监控
- 在线自适应实验
- 多模态同步采集

❌ **不适用于**:
- 复杂离线分析 (ICA, 源定位)
- 需要反复优化参数
- 批量处理多个被试
- 发表论文的精细分析

### 关键要点

1. **环形缓冲区**: 只保留最新数据,旧数据自动覆盖
2. **因果滤波**: 必须使用 `phase='minimum'`,不能用零相位
3. **时间约束**: 处理必须在新数据到达前完成
4. **简单算法**: 避免 ICA、源定位等耗时操作
5. **先离线后实时**: 先用离线数据验证算法,再部署到实时

---

**相关文档**:
- [MNE 离线处理指南](mne-offline-processing.md)
- [LSL 和 MNE-LSL 指南](lsl-mne-lsl-guide.md)
- [离线 vs 实时对比](mne-offline-vs-realtime.md)
