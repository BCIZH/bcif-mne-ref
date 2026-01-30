# MNE-Python vs MNE-LSL 对比指南

> **用途**: 理解 MNE 离线和 MNE-LSL 实时的关系与区别  
> **创建日期**: 2026-01-31  
> **核心**: 同样的 API 风格,不同的应用场景

---

## 目录

1. [核心关系](#核心关系)
2. [API 对比](#api-对比)
3. [数据模型对比](#数据模型对比)
4. [使用场景](#使用场景)
5. [混合使用策略](#混合使用策略)
6. [迁移指南](#迁移指南)

---

## 核心关系

### 项目关系图

```
┌────────────────────────────────────────────────────┐
│                 MNE 生态系统                        │
└────────────────────────────────────────────────────┘

┌─────────────────────┐         ┌──────────────────┐
│   MNE-Python        │         │    MNE-LSL       │
│                     │         │                  │
│  • 核心包           │◀────────│  • 扩展包        │
│  • 离线分析         │ 依赖    │  • 实时处理      │
│  • mne.io.Raw       │         │  • StreamLSL     │
│  • mne.Epochs       │         │  • EpochsStream  │
│  • mne.Evoked       │         │                  │
│                     │         │  基于 MNE API    │
│  Python >= 3.10     │         │  设计           │
│  NumPy, SciPy       │         │                  │
└─────────────────────┘         └──────────────────┘
         ▲                              ▲
         │                              │
         │                              │
         └──────────┬───────────────────┘
                    │
                    ▼
         ┌────────────────────┐
         │  Lab Streaming     │
         │  Layer (LSL)       │
         │                    │
         │  • liblsl (C++)    │
         │  • 网络数据流       │
         │  • 时间同步         │
         └────────────────────┘
```

### 设计理念

**MNE-LSL 的设计目标**: 让 MNE-Python 用户能用熟悉的 API 处理实时数据

```python
# MNE-Python (离线)
from mne.io import read_raw_fif
raw = read_raw_fif('data.fif')
raw.filter(1, 40)
data = raw.get_data()

# MNE-LSL (实时) - 几乎相同的 API!
from mne_lsl.stream import StreamLSL
stream = StreamLSL(bufsize=5, name='MyEEG')
stream.connect()
stream.filter(1, 40, phase='minimum')  # 唯一区别: 因果滤波
data = stream.get_data()
```

---

## API 对比

### 核心对象对比

| 功能 | MNE-Python | MNE-LSL | 区别 |
|------|-----------|---------|------|
| **原始数据** | `mne.io.Raw` | `StreamLSL` | 文件 vs 网络流 |
| **分段数据** | `mne.Epochs` | `EpochsStream` | 固定 vs 动态累积 |
| **平均数据** | `mne.Evoked` | `mne.Evoked` | 相同 (可转换) |
| **Info** | `mne.Info` | `mne.Info` | 完全相同 |

---

### 详细 API 对比表

| 操作 | MNE-Python | MNE-LSL | 说明 |
|------|-----------|---------|------|
| **创建对象** | `read_raw_fif('file.fif')` | `StreamLSL(bufsize=5)` | 文件 vs 流 |
| **连接/加载** | `preload=True` | `stream.connect()` | 加载 vs 连接 |
| **获取数据** | `raw[:, start:stop]` | `stream.get_data(winsize)` | 随机访问 vs 最新 |
| **数据形状** | `(n_channels, n_samples)` | `(n_channels, n_samples)` | 相同 |
| **滤波** | `raw.filter(1, 40)` | `stream.filter(1, 40, phase='minimum')` | 零相位 vs 因果 |
| **重参考** | `raw.set_eeg_reference('average')` | `stream.set_eeg_reference('average')` | 完全相同 |
| **通道操作** | `raw.pick_types(meg=True)` | `stream.pick_types(meg=True)` | 完全相同 |
| **Info 访问** | `raw.info` | `stream.info` | 完全相同 |
| **坏通道** | `raw.info['bads'] = [...]` | `stream.info['bads'] = [...]` | 完全相同 |
| **时间范围** | `raw.times` | `stream.times` | 固定 vs 动态 |
| **事件检测** | `find_events(raw)` | 自动 (EpochsStream) | 离线 vs 在线 |

---

### 代码对比: 基础操作

```python
# ========================================
# MNE-Python (离线)
# ========================================
import mne

# 1. 读取数据
raw = mne.io.read_raw_fif('sample.fif', preload=True)

# 2. 查看信息
print(raw.info)
print(f"采样率: {raw.info['sfreq']} Hz")
print(f"通道数: {len(raw.ch_names)}")

# 3. 滤波
raw.filter(l_freq=1.0, h_freq=40.0, phase='zero')  # 零相位

# 4. 重参考
raw.set_eeg_reference('average')

# 5. 获取数据
data = raw[:, 0:1000]  # 前 2 秒 @ 500 Hz

# 6. 选择通道
raw.pick_types(meg=False, eeg=True)

# ========================================
# MNE-LSL (实时)
# ========================================
from mne_lsl.stream import StreamLSL
import time

# 1. 连接流
stream = StreamLSL(bufsize=5, name='MyEEG')
stream.connect()

# 2. 查看信息 (完全相同!)
print(stream.info)
print(f"采样率: {stream.info['sfreq']} Hz")
print(f"通道数: {len(stream.ch_names)}")

# 3. 滤波 (必须因果!)
stream.filter(l_freq=1.0, h_freq=40.0, phase='minimum')  # 因果相位

# 4. 重参考 (完全相同!)
stream.set_eeg_reference('average')

# 5. 获取数据 (只能最新!)
time.sleep(2)  # 等待缓冲区填充
data = stream.get_data(winsize=2)  # 最新 2 秒

# 6. 选择通道 (完全相同!)
stream.pick_types(meg=False, eeg=True)
```

---

### 代码对比: Epochs

```python
# ========================================
# MNE-Python (离线 Epochs)
# ========================================
import mne

# 读取数据
raw = mne.io.read_raw_fif('sample.fif', preload=True)

# 查找事件
events = mne.find_events(raw, stim_channel='STI 014')

# 创建 Epochs
epochs = mne.Epochs(
    raw,
    events,
    event_id={'visual': 3},
    tmin=-0.2,
    tmax=0.5,
    baseline=(None, 0),
    preload=True
)

print(f"总共 {len(epochs)} 个 epochs")

# 获取所有数据
all_data = epochs.get_data()  # (n_epochs, n_channels, n_times)

# 计算平均
evoked = epochs.average()

# ========================================
# MNE-LSL (实时 Epochs)
# ========================================
from mne_lsl.stream import StreamLSL, EpochsStream
from mne import EvokedArray, combine_evoked
import numpy as np

# 连接流
stream = StreamLSL(bufsize=5, name='MyEEG')
stream.connect()

# 创建实时 Epochs (自动事件检测!)
epochs = EpochsStream(
    stream,
    bufsize=20,  # 保留最新 20 个 epochs
    event_id=3,
    event_channels='STI 014',
    tmin=-0.2,
    tmax=0.5,
    baseline=(None, 0)
)

epochs.connect()

# 实时累积 (不同于离线!)
evoked = None
n_epochs = 0

while n_epochs < 20:
    if epochs.n_new_epochs > 0:
        # 获取新 epochs
        new_data = epochs.get_data(n_epochs=epochs.n_new_epochs)
        
        # 创建新 evoked
        new_evoked = EvokedArray(
            np.average(new_data, axis=0),
            epochs.info,
            tmin=epochs.tmin
        )
        
        # 累积平均
        if evoked is None:
            evoked = new_evoked
        else:
            evoked = combine_evoked([evoked, new_evoked], weights='nave')
        
        n_epochs += epochs.n_new_epochs
        print(f"累积了 {n_epochs} 个 epochs")
```

---

## 数据模型对比

### Raw vs StreamLSL

```python
# ========================================
# mne.io.Raw 数据模型
# ========================================
class Raw:
    """
    离线数据对象
    
    数据来源: 文件
    数据长度: 固定 (n_samples)
    内存占用: 全部或按需
    """
    _data: np.ndarray  # (n_channels, n_samples) 完整数据
    _times: np.ndarray  # (n_samples,) 所有时间点
    _first_samps: np.ndarray  # 起始样本
    _last_samps: np.ndarray   # 结束样本
    
    def __getitem__(self, key):
        """随机访问任意时间段"""
        channels, samples = key
        return self._data[channels, samples]
    
    def filter(self, l_freq, h_freq, phase='zero'):
        """零相位滤波 (需要未来数据)"""
        # 双向滤波,无因果约束

# ========================================
# StreamLSL 数据模型
# ========================================
class StreamLSL:
    """
    实时数据对象
    
    数据来源: 网络流 (LSL)
    数据长度: 无限 (持续增长)
    内存占用: 固定缓冲区
    """
    _buffer: np.ndarray  # (n_channels, bufsize*sfreq) 环形缓冲
    _inlet: StreamInlet  # LSL 接收端
    _timestamps: np.ndarray  # 时间戳
    
    def get_data(self, winsize):
        """只能获取最新数据"""
        n_samples = int(winsize * self.info['sfreq'])
        # 从环形缓冲区提取最新 n_samples
        return self._extract_latest(n_samples)
    
    def filter(self, l_freq, h_freq, phase='minimum'):
        """因果滤波 (只用过去数据)"""
        # 单向滤波,实时约束
```

---

### 内存布局对比

```
┌─────────────────────────────────────────────────┐
│  MNE-Python Raw                                 │
│                                                 │
│  文件: sample.fif (600 MB)                      │
│  ↓                                              │
│  内存: (376 channels, 166800 samples)           │
│  [════════════完整数据数组════════════]          │
│   ↑ 可随机访问任意位置                           │
│                                                 │
│  特点:                                          │
│  • 固定大小                                     │
│  • 完全加载或按需加载                            │
│  • 可反复读取                                    │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  MNE-LSL StreamLSL                              │
│                                                 │
│  网络: LSL Stream (持续传输)                     │
│  ↓                                              │
│  内存: (64 channels, 2500 samples) 环形缓冲      │
│  [新数据→███████已有5秒███→覆盖旧数据]           │
│            ↑              ↑                     │
│           tail           head                   │
│                                                 │
│  特点:                                          │
│  • 固定缓冲 (bufsize=5秒)                       │
│  • 旧数据自动覆盖                                │
│  • 只能访问最新                                  │
└─────────────────────────────────────────────────┘
```

---

## 使用场景

### 决策树

```
开始
 │
 ├─ 需要实时反馈? ───YES─→ MNE-LSL
 │                         (BCI, 神经反馈, 监控)
 │
 └─ NO
     │
     ├─ 需要复杂算法? ───YES─→ MNE-Python
     │  (ICA, 源定位)           (离线深度分析)
     │
     └─ NO
         │
         ├─ 需要反复优化? ───YES─→ MNE-Python
         │                           (参数调整, 批量处理)
         │
         └─ NO
             │
             └─ 发表论文? ───YES─→ MNE-Python
                                   (标准科学分析)
```

---

### 场景对比表

| 需求 | MNE-Python | MNE-LSL | 推荐 |
|------|-----------|---------|------|
| **ERP 研究** | ✅ 完美 | ❌ | MNE-Python |
| **BCI 控制** | ❌ | ✅ 必须 | MNE-LSL |
| **神经反馈** | ❌ | ✅ 必须 | MNE-LSL |
| **时频分析** | ✅ 完美 | ⚠️ 受限 | MNE-Python |
| **源定位** | ✅ 完美 | ❌ 无法 | MNE-Python |
| **ICA 去伪迹** | ✅ 完美 | ❌ 无法 | MNE-Python |
| **实时监控** | ❌ | ✅ 完美 | MNE-LSL |
| **批量处理** | ✅ 并行 | ❌ | MNE-Python |
| **参数优化** | ✅ 可反复 | ❌ 单次 | MNE-Python |
| **发表论文** | ✅ 标准 | ❌ | MNE-Python |

---

## 混合使用策略

### 完整项目流程

```
┌────────────────────────────────────────────────────┐
│  阶段 1: 离线开发 (MNE-Python)                      │
│                                                    │
│  目标: 开发和验证算法                               │
│                                                    │
│  1. 收集训练数据                                    │
│     raw = mne.io.read_raw_fif('training_data.fif') │
│                                                    │
│  2. 探索和优化                                      │
│     # 尝试不同预处理参数                            │
│     raw.filter(1, 40)  # vs filter(0.5, 30)       │
│                                                    │
│  3. 特征工程                                        │
│     # 测试不同特征                                  │
│     features = extract_bandpower(raw)              │
│                                                    │
│  4. 训练模型                                        │
│     clf = train_classifier(features, labels)       │
│                                                    │
│  5. 离线验证                                        │
│     accuracy = cross_val_score(clf, X, y)         │
│     print(f"离线准确率: {accuracy.mean()}")         │
└──────────────────┬─────────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────────┐
│  阶段 2: 实时部署 (MNE-LSL)                         │
│                                                    │
│  目标: 在线实时应用                                 │
│                                                    │
│  1. 将算法移植到实时                                │
│     from mne_lsl.stream import StreamLSL           │
│     stream = StreamLSL(bufsize=5)                  │
│     stream.connect()                               │
│                                                    │
│  2. 应用相同预处理                                  │
│     stream.filter(1, 40, phase='minimum')          │
│                                                    │
│  3. 实时特征提取                                    │
│     data = stream.get_data(winsize=2)              │
│     features = extract_bandpower(data)  # 相同函数  │
│                                                    │
│  4. 实时分类                                        │
│     prediction = clf.predict([features])  # 相同模型│
│                                                    │
│  5. 实时反馈                                        │
│     send_feedback(prediction)                      │
└──────────────────┬─────────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────────┐
│  阶段 3: 离线评估 (MNE-Python)                      │
│                                                    │
│  目标: 分析实验数据,改进算法                        │
│                                                    │
│  1. 加载录制数据                                    │
│     # 实时实验时同步录制                            │
│     raw = mne.io.read_raw_fif('online_session.fif')│
│                                                    │
│  2. 深度分析                                        │
│     # ICA 去伪迹                                   │
│     # 源定位                                       │
│     # 统计检验                                     │
│                                                    │
│  3. 算法改进                                        │
│     # 返回阶段 1,优化算法                           │
└────────────────────────────────────────────────────┘
```

---

### 示例: BCI 项目完整代码

```python
# ====================================================
# 阶段 1: 离线训练 (MNE-Python)
# ====================================================
import mne
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

print("="*50)
print("阶段 1: 离线训练")
print("="*50)

# 1. 读取训练数据
raw = mne.io.read_raw_fif('motor_imagery_training.fif', preload=True)

# 2. 预处理
raw.filter(8, 30)  # Mu/Beta 频段
raw.set_eeg_reference('average')

# 3. 分段
events = mne.find_events(raw)
epochs = mne.Epochs(
    raw, events,
    event_id={'left_hand': 1, 'right_hand': 2},
    tmin=0, tmax=3,
    baseline=None
)

# 4. 特征提取函数
def extract_features_offline(epochs_data):
    """从 Epochs 提取特征"""
    from scipy.signal import welch
    
    features = []
    for epoch in epochs_data:
        # C3, C4 通道
        c3_idx = epochs.ch_names.index('C3')
        c4_idx = epochs.ch_names.index('C4')
        
        # 功率谱
        freqs, psd_c3 = welch(epoch[c3_idx, :], fs=epochs.info['sfreq'])
        _, psd_c4 = welch(epoch[c4_idx, :], fs=epochs.info['sfreq'])
        
        # Mu 频段功率
        mu_idx = np.logical_and(freqs >= 8, freqs <= 12)
        mu_c3 = np.mean(psd_c3[mu_idx])
        mu_c4 = np.mean(psd_c4[mu_idx])
        
        features.append([mu_c3, mu_c4])
    
    return np.array(features)

# 5. 提取特征
X = extract_features_offline(epochs.get_data())
y = epochs.events[:, 2]

# 6. 训练分类器
clf = LinearDiscriminantAnalysis()
scores = cross_val_score(clf, X, y, cv=5)

print(f"离线交叉验证准确率: {scores.mean():.2%} ± {scores.std():.2%}")

# 7. 训练最终模型
clf.fit(X, y)
print("✅ 模型训练完成\n")

# ====================================================
# 阶段 2: 实时 BCI (MNE-LSL)
# ====================================================
from mne_lsl.stream import StreamLSL
from scipy.signal import welch
import time

print("="*50)
print("阶段 2: 实时 BCI")
print("="*50)

# 1. 连接实时流
stream = StreamLSL(bufsize=5, name='MotorImageryEEG')
stream.connect()

# 2. 应用相同预处理
stream.filter(8, 30, phase='minimum')  # 因果滤波
stream.set_eeg_reference('average')

time.sleep(2)  # 等待缓冲

# 3. 特征提取函数 (与离线相同逻辑!)
def extract_features_online(data, sfreq, ch_names):
    """从实时数据提取特征"""
    # C3, C4 通道
    c3_idx = ch_names.index('C3')
    c4_idx = ch_names.index('C4')
    
    # 功率谱
    freqs, psd_c3 = welch(data[c3_idx, :], fs=sfreq)
    _, psd_c4 = welch(data[c4_idx, :], fs=sfreq)
    
    # Mu 频段功率
    mu_idx = np.logical_and(freqs >= 8, freqs <= 12)
    mu_c3 = np.mean(psd_c3[mu_idx])
    mu_c4 = np.mean(psd_c4[mu_idx])
    
    return np.array([mu_c3, mu_c4])

# 4. 实时 BCI 循环
print("\n🧠 BCI 开始运行...")
print("请想象左手或右手运动\n")

for trial in range(10):
    # 获取最新 3 秒数据
    data, _ = stream.get_data(winsize=3)
    
    # 提取特征 (相同函数!)
    features = extract_features_online(
        data, 
        stream.info['sfreq'],
        stream.ch_names
    )
    
    # 分类 (相同模型!)
    prediction = clf.predict([features])[0]
    prob = clf.predict_proba([features])[0]
    
    # 输出
    if prediction == 1:
        direction = "◀◀◀ 左手"
        conf = prob[0]
    else:
        direction = "右手 ▶▶▶"
        conf = prob[1]
    
    print(f"Trial {trial+1:2d}: {direction} (置信度: {conf*100:.1f}%)")
    
    # 发送控制信号到外部设备
    # control_device(prediction)
    
    time.sleep(1)

stream.disconnect()
print("\n✅ BCI 会话结束")
```

---

## 迁移指南

### 从离线到实时

**需要修改的部分**:

```python
# ========== 数据源 ==========
# 离线
raw = mne.io.read_raw_fif('data.fif')

# 实时
stream = StreamLSL(bufsize=5, name='MyEEG')
stream.connect()

# ========== 滤波 ==========
# 离线 (零相位)
raw.filter(1, 40, phase='zero')

# 实时 (因果)
stream.filter(1, 40, phase='minimum')  # ⚠️ 必须修改!

# ========== 数据获取 ==========
# 离线 (随机访问)
data = raw[:, 0:1000]

# 实时 (最新数据)
data, times = stream.get_data(winsize=2)  # ⚠️ 修改!

# ========== ICA (不支持) ==========
# 离线
ica = ICA(n_components=20)
ica.fit(raw)  # ✅ 可以

# 实时
# ❌ 无法在实时流上运行 ICA
# 解决方案: 离线预处理,或用简单滤波替代
```

**不需要修改的部分**:

```python
# ✅ 这些完全相同,无需修改

# Info 对象
raw.info == stream.info  # 结构完全相同

# 重参考
raw.set_eeg_reference('average')
stream.set_eeg_reference('average')  # 相同

# 通道选择
raw.pick_types(meg=False, eeg=True)
stream.pick_types(meg=False, eeg=True)  # 相同

# 坏通道
raw.info['bads'] = ['EEG 053']
stream.info['bads'] = ['EEG 053']  # 相同
```

---

## 总结

### 核心要点

1. **MNE-LSL 基于 MNE-Python**
   - 依赖 MNE-Python
   - 使用相似的 API
   - 共享 Info 对象

2. **主要区别**
   - 数据来源: 文件 vs 网络流
   - 数据访问: 随机 vs 顺序
   - 滤波方式: 零相位 vs 因果
   - 算法支持: 无限制 vs 受限

3. **选择原则**
   - 需要实时反馈 → MNE-LSL
   - 复杂离线分析 → MNE-Python
   - 最佳实践: 混合使用

4. **混合策略**
   - 离线开发算法 (MNE-Python)
   - 实时部署应用 (MNE-LSL)
   - 离线评估改进 (MNE-Python)

---

**相关文档**:
- [MNE 离线处理指南](mne-offline-processing.md)
- [MNE 实时处理指南](mne-realtime-processing.md)
- [LSL 和 MNE-LSL 指南](lsl-mne-lsl-guide.md)
