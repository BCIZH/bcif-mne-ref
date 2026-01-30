# MNE-ICALabel 深度学习后端架构详解

> **用途**: 详细拆解 MNE-ICALabel 的推理流程和后端架构  
> **创建日期**: 2026-01-31  
> **核心**: 从特征提取到模型推理的完整数据流

---

## 目录

1. [数据格式兼容性](#数据格式兼容性)
2. [完整数据流](#完整数据流)
3. [特征提取详解](#特征提取详解)
4. [模型输入格式](#模型输入格式)
5. [神经网络架构](#神经网络架构)
6. [现有后端实现](#现有后端实现)
7. [理论上可添加的后端](#理论上可添加的后端)
8. [Rust 后端可行性分析](#rust-后端可行性分析)

---

## 数据格式兼容性

### .fif 格式可以完全替换！

**重要**: MNE-ICALabel **不依赖** `.fif` 格式，只需要 `mne.io.Raw` 或 `mne.Epochs` 对象。

```python
# ✅ 这些格式都可以!
from mne_icalabel import label_components

# 方法 1: XDF (LSL 录制格式)
raw = mne.io.read_raw_xdf('recording.xdf')
ic_labels = label_components(raw, ica, method='iclabel')

# 方法 2: EDF (欧洲标准格式)
raw = mne.io.read_raw_edf('data.edf')
ic_labels = label_components(raw, ica, method='iclabel')

# 方法 3: BrainVision
raw = mne.io.read_raw_brainvision('data.vhdr')
ic_labels = label_components(raw, ica, method='iclabel')

# 方法 4: LSL 实时流 (MNE-LSL)
from mne_lsl.stream import StreamLSL
stream = StreamLSL(bufsize=10, name='YourStream')
stream.connect()
ic_labels = label_components(stream, ica, method='iclabel')  # ✅ 直接支持!
```

---

### XDF 格式 (LSL 录制) - 您的场景

#### 什么是 XDF？

```
XDF (eXtensible Data Format)
├── Lab Streaming Layer 的录制格式
├── 存储多流数据 + 时间戳同步
└── 文件扩展名: .xdf
```

#### 完整示例: XDF → ICALabel

```python
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

# ========================================
# 1. 读取 LSL 录制的 XDF 文件
# ========================================
# 安装: pip install pyxdf
raw = mne.io.read_raw_xdf(
    'my_lsl_recording.xdf',
    stream_ids=[0],  # 选择第一个流 (如果有多个)
    preload=True
)

print(f"✅ 读取成功: {len(raw.ch_names)} 通道, {raw.times[-1]:.1f} 秒")

# ========================================
# 2. 预处理 (ICLabel 要求)
# ========================================
# 滤波: 1-100 Hz
raw.filter(l_freq=1.0, h_freq=100.0)

# 设置参考 (如果是 EEG)
if 'eeg' in raw.get_channel_types():
    raw.set_eeg_reference('average')

# ========================================
# 3. ICA 分解
# ========================================
ica = ICA(
    n_components=15,
    method='infomax',
    fit_params=dict(extended=True),
    random_state=42
)
ica.fit(raw)

# ========================================
# 4. 自动分类 (与 .fif 完全相同!)
# ========================================
ic_labels = label_components(raw, ica, method='iclabel')

print("\n成分分类结果:")
for i, (label, prob) in enumerate(zip(ic_labels['labels'], 
                                       ic_labels['y_pred_proba'])):
    print(f"  ICA{i:02d}: {label:20s} ({prob*100:.1f}%)")

# 排除伪迹
exclude_idx = [i for i, label in enumerate(ic_labels['labels'])
               if label not in ['brain', 'other']]
ica.apply(raw, exclude=exclude_idx)

print(f"\n✅ 排除 {len(exclude_idx)} 个伪迹成分")
```

#### XDF 多流处理

```python
import pyxdf

# 查看 XDF 文件包含哪些流
streams, header = pyxdf.load_xdf('recording.xdf')

print("XDF 文件包含的流:")
for i, stream in enumerate(streams):
    print(f"  Stream {i}: {stream['info']['name'][0]}")
    print(f"    类型: {stream['info']['type'][0]}")
    print(f"    通道数: {stream['info']['channel_count'][0]}")

# 读取指定流
raw = mne.io.read_raw_xdf(
    'recording.xdf',
    stream_ids=[0],  # EEG 流
    preload=True
)
```

---

### EDF 格式 - 医疗标准

#### 完整示例: EDF → ICALabel

```python
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

# ========================================
# 1. 读取 EDF 文件
# ========================================
raw = mne.io.read_raw_edf(
    'patient_data.edf',
    preload=True,
    stim_channel='auto'  # 自动检测刺激通道
)

# EDF 文件信息
print(f"采样率: {raw.info['sfreq']} Hz")
print(f"通道: {raw.ch_names}")

# ========================================
# 2. 选择 EEG 通道 (EDF 可能包含其他类型)
# ========================================
raw.pick_types(eeg=True, exclude=[])

# ========================================
# 3. 预处理
# ========================================
raw.filter(1.0, 100.0)
raw.set_eeg_reference('average')

# ========================================
# 4. ICA + 自动分类
# ========================================
ica = ICA(n_components=15, method='infomax',
          fit_params=dict(extended=True))
ica.fit(raw)

ic_labels = label_components(raw, ica, method='iclabel')

# 排除伪迹
exclude_idx = [i for i, label in enumerate(ic_labels['labels'])
               if label not in ['brain', 'other']]
ica.apply(raw, exclude=exclude_idx)

# ========================================
# 5. 保存清洗后的数据 (可选)
# ========================================
# 保存为 EDF
mne.export.export_raw('cleaned_data.edf', raw, fmt='edf', overwrite=True)

# 或保存为 FIFF (MNE 原生，更快)
raw.save('cleaned_data.fif', overwrite=True)
```

#### EDF vs BDF (重要区别)

```python
# ❌ EDF: 16-bit (标准 EEG,精度有限)
raw = mne.io.read_raw_edf('data.edf')
# 精度: ±327.68 mV (16-bit 整数)
# 采样率: 通常 ≤ 256 Hz
# 适用: 临床 EEG,睡眠研究

# ✅ BDF: 24-bit (BioSemi 格式,高精度)
raw = mne.io.read_raw_bdf('data.bdf')
# 精度: ±8388.608 mV (24-bit 整数,256x 更精确)
# 采样率: 可达 16 kHz+
# 适用: 研究级 EEG/ERP

# 使用方式完全相同
ic_labels = label_components(raw, ica, method='iclabel')
```

**关键区别**:

| 特性 | EDF | BDF |
|------|-----|-----|
| **位深度** | 16-bit | **24-bit** ✅ |
| **精度** | ±32768 levels | ±8388608 levels (256x) |
| **动态范围** | ~96 dB | ~144 dB |
| **文件大小** | 较小 | 大 50% |
| **标准** | 欧洲医疗标准 (1992) | BioSemi 扩展 (2003) |
| **推荐用于** | 临床,长时程录制 | 研究,高精度分析 |

**开源实现** ✅:

| 库/工具 | BDF 支持 | EDF 支持 | 许可证 |
|---------|----------|----------|--------|
| **MNE-Python** | ✅ `mne.io.read_raw_bdf()` | ✅ `mne.io.read_raw_edf()` | BSD-3 |
| **pyEDFlib** | ✅ 完整支持 | ✅ 完整支持 | BSD-2 |
| **EDFbrowser** | ✅ 可视化 | ✅ 可视化 | GPL-3 |
| **BioSig** | ✅ C/C++/Octave | ✅ 完整支持 | GPL-3 |
| **EEGLAB (MATLAB)** | ✅ biosig 插件 | ✅ 原生支持 | GPL-2 |

**兼容性** 🔄:

```python
# ========================================
# BDF → EDF 转换 (向下兼容,损失精度)
# ========================================
import mne

# 读取 BDF (24-bit)
raw_bdf = mne.io.read_raw_bdf('high_precision.bdf', preload=True)

# 导出为 EDF (自动降采样到 16-bit)
mne.export.export_raw('converted.edf', raw_bdf, fmt='edf')
# ⚠️ 警告: 24-bit → 16-bit 会损失精度!

# ========================================
# EDF → BDF 转换 (向上兼容,无损)
# ========================================
raw_edf = mne.io.read_raw_edf('standard.edf', preload=True)

# EDF 数据可以在 BDF 系统中使用 (无需转换)
# BDF 读取器通常也能读 EDF 文件

# 如需保存为 BDF 格式:
import pyedflib
n_channels = len(raw_edf.ch_names)
signals = raw_edf.get_data() * 1e6  # 转换为 µV

with pyedflib.EdfWriter('converted.bdf', n_channels, file_type=pyedflib.FILETYPE_BDFPLUS) as f:
    channel_info = []
    for ch_name in raw_edf.ch_names:
        ch_dict = {
            'label': ch_name,
            'dimension': 'uV',
            'sample_rate': raw_edf.info['sfreq'],
            'physical_max': signals.max(),
            'physical_min': signals.min(),
            'digital_max': 8388607,   # 24-bit max
            'digital_min': -8388608,  # 24-bit min
        }
        channel_info.append(ch_dict)
    
    f.setSignalHeaders(channel_info)
    f.writeSamples(signals)
```

**格式兼容性总结**:

| 场景 | 兼容性 | 说明 |
|------|--------|------|
| **BDF 读取器读 EDF** | ✅ 完全兼容 | BDF 是 EDF 的超集 |
| **EDF 读取器读 BDF** | ⚠️ 部分兼容 | 旧版 EDF 软件可能不支持 24-bit |
| **BDF → EDF 转换** | ⚠️ 损失精度 | 24-bit → 16-bit 截断 |
| **EDF → BDF 转换** | ✅ 无损 | 16-bit 数据在 24-bit 容器中 |
| **MNE-Python** | ✅ 完全兼容 | 两种格式使用相同 API |

**您的场景推荐**:
- 如果 LSL 采集设备支持高精度 → 保存为 **BDF** ✅
- 如果需要兼容性/文件小 → 保存为 **EDF** ⚠️

---

### LSL 实时流 → ICALabel

#### 场景 1: 离线分析 LSL 录制数据

```python
# 您的场景: LSL 录制 → XDF → 离线 ICA
from mne_icalabel import label_components

# 1. 读取 LSL 录制的 XDF
raw = mne.io.read_raw_xdf('lsl_recording.xdf', preload=True)

# 2. 离线 ICA 分析
raw.filter(1, 100).set_eeg_reference('average')
ica = ICA(n_components=15, method='infomax', 
          fit_params=dict(extended=True))
ica.fit(raw)

# 3. 自动分类
ic_labels = label_components(raw, ica, method='iclabel')

# 4. 应用清洗
exclude = [i for i, l in enumerate(ic_labels['labels']) 
           if l not in ['brain', 'other']]
ica.apply(raw, exclude=exclude)
```

#### 场景 2: 准实时 ICA (1秒延迟可接受)

**您的场景**: 1秒延迟可接受 → **完全可行**!

**方案 A: 预训练 ICA + 实时应用** (推荐)

```python
# --- 步骤 1: 离线训练 ICA (一次性,5分钟) ---
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

# 收集校准数据 (可以是 XDF 录制)
raw = mne.io.read_raw_xdf('calibration_5min.xdf', preload=True)
raw.filter(1, 100).set_eeg_reference('average')

ica = ICA(n_components=15, method='infomax',
          fit_params=dict(extended=True))
ica.fit(raw)

ic_labels = label_components(raw, ica, method='iclabel')
exclude = [i for i, l in enumerate(ic_labels['labels'])
           if l not in ['brain', 'other']]
ica.exclude = exclude

# 保存模型
ica.save('trained_ica.fif')
print(f"✅ 训练完成,排除成分: {exclude}")

# --- 步骤 2: 实时应用 (低延迟 ~100ms) ---
from mne_lsl.stream import StreamLSL
from mne.preprocessing import read_ica
import numpy as np
import time

# 加载模型
ica = read_ica('trained_ica.fif')

# 连接实时流
stream = StreamLSL(bufsize=2, name='MyEEG')
stream.connect()
stream.filter(1, 100, phase='minimum')
stream.set_eeg_reference('average')

print("🚀 实时 ICA 清洗启动 (延迟 ~100ms)")

while True:
    # 获取最新 0.5 秒数据
    data, times = stream.get_data(winsize=0.5)
    
    # 快速 ICA 应用 (~10ms)
    sources = np.dot(ica.unmixing_matrix_, data)
    sources[ica.exclude, :] = 0  # 移除伪迹
    data_clean = np.dot(ica.mixing_matrix_, sources)
    
    # ✅ 总延迟: ~100ms (远低于您的 1秒要求)
    
    # 您的实时处理
    # process_realtime(data_clean)
    
    time.sleep(0.1)  # 100ms 更新
from mne.io import RawArray
info = stream.info
raw_segment = RawArray(data, info)

# 5. ICA 分析
raw_segment.filter(1, 100)
ica = ICA(n_components=15, method='infomax',
          fit_params=dict(extended=True))
ica.fit(raw_segment)

# 6. 自动分类
ic_labels = label_components(raw_segment, ica, method='iclabel')

# 7. 应用到实时流
# ⚠️ MNE-LSL 的 StreamLSL 不直接支持 ICA.apply()
# 需要手动处理或切换到离线模式
```

**实时 ICA 的局限**:
- ❌ ICA 拟合需要大量数据 (建议 >30 秒)
- ❌ 无法在"真正实时"中重新训练 ICA
- ✅ 可行方案: 离线训练 ICA → 保存 → 实时应用

#### 推荐工作流: 离线 ICA + 实时应用

```python
# ========================================
# 阶段 1: 离线训练 ICA (使用录制数据)
# ========================================
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

# 1. 读取 XDF 录制
raw = mne.io.read_raw_xdf('calibration.xdf', preload=True)
raw.filter(1, 100).set_eeg_reference('average')

# 2. 训练 ICA
ica = ICA(n_components=15, method='infomax',
          fit_params=dict(extended=True))
ica.fit(raw)

# 3. 自动分类
ic_labels = label_components(raw, ica, method='iclabel')
exclude = [i for i, l in enumerate(ic_labels['labels']) 
           if l not in ['brain', 'other']]

# 4. 保存 ICA (含排除列表)
ica.exclude = exclude
ica.save('trained_ica.fif')

print(f"✅ ICA 训练完成，排除成分: {exclude}")

# ========================================
# 阶段 2: 实时应用 ICA (使用 MNE-LSL)
# ========================================
from mne_lsl.stream import StreamLSL
from mne.preprocessing import read_ica

# 1. 加载训练好的 ICA
ica = read_ica('trained_ica.fif')

# 2. 连接实时流
stream = StreamLSL(bufsize=5, name='MyEEG')
stream.connect()
stream.filter(1, 100, phase='minimum')
stream.set_eeg_reference('average')

# 3. 实时循环
while True:
    # 获取最新数据
    data, times = stream.get_data(winsize=2)
    
    # 应用 ICA 清洗
    data_clean = ica.apply(data, exclude=ica.exclude)
    
    # 使用清洗后的数据做分析
    # analyze(data_clean)
```

---

### 格式对比 (针对您的场景)

| 格式 | 优势 | 劣势 | 推荐场景 |
|------|------|------|------|
| **XDF** | ✅ LSL 原生<br>✅ 多流同步<br>✅ 精确时间戳<br>⚠️ **离线格式** | ⚠️ 需要 pyxdf<br>⚠️ 文件可能很大<br>❌ 不支持实时 | **LSL 录制数据** (离线分析) |
| **EDF** | ✅ 医疗标准<br>✅ 广泛兼容<br>✅ 长时程录制 | ❌ 16-bit 精度<br>⚠️ 元数据有限 | 临床数据,睡眠研究 |
| **BDF** | ✅ **24-bit 精度** 🎯<br>✅ 高采样率<br>✅ BioSemi 标准 | ⚠️ 文件较大<br>⚠️ 兼容性略低 | **研究级 EEG** (您的高精度场景) |
| **FIFF** | ✅ MNE 最快<br>✅ 元数据完整 | ❌ 非通用格式 | MNE 内部处理 |
| **LSL Stream** | ✅ **真实时** 🚀<br>✅ 低延迟 (<100ms)<br>✅ 多设备同步 | ❌ 不能保存<br>⚠️ 需要 MNE-LSL | **实时 BCI/神经反馈** (您的实时场景) |

---

### 完整工作流: XDF → ICALabel → 清洗数据

```python
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

# ========================================
# Step 1: 读取 LSL 录制的 XDF
# ========================================
print("📂 读取 XDF 文件...")
raw = mne.io.read_raw_xdf('my_lsl_recording.xdf', preload=True)

print(f"✅ 数据信息:")
print(f"   通道数: {len(raw.ch_names)}")
print(f"   采样率: {raw.info['sfreq']} Hz")
print(f"   时长: {raw.times[-1]:.1f} 秒")

# ========================================
# Step 2: 预处理
# ========================================
print("\n🔧 预处理...")

# 选择 EEG 通道 (如果有其他类型)
raw.pick_types(eeg=True, exclude=[])

# 滤波: 1-100 Hz (ICLabel 要求)
raw.filter(l_freq=1.0, h_freq=100.0)

# 平均参考 (ICLabel 要求)
raw.set_eeg_reference('average')

# ========================================
# Step 3: ICA 分解
# ========================================
print("\n🧠 运行 ICA...")
ica = ICA(
    n_components=15,
    method='infomax',
    fit_params=dict(extended=True),
    random_state=42,
    max_iter='auto'
)
ica.fit(raw)

print(f"✅ ICA 完成: {ica.n_components_} 个成分")

# ========================================
# Step 4: 自动分类 (ICLabel)
# ========================================
print("\n🤖 自动分类成分...")
ic_labels = label_components(raw, ica, method='iclabel')

print("\n成分分类结果:")
print("="*70)
for i, (label, prob) in enumerate(zip(ic_labels['labels'], 
                                       ic_labels['y_pred_proba'])):
    icon = "✅" if label == "brain" else "❌"
    print(f"{icon} ICA{i:02d}: {label:20s} (置信度: {prob*100:5.1f}%)")

# ========================================
# Step 5: 排除伪迹
# ========================================
exclude_idx = [i for i, label in enumerate(ic_labels['labels'])
               if label not in ['brain', 'other']]

print(f"\n🗑️  排除成分: {exclude_idx}")
print(f"   标签: {[ic_labels['labels'][i] for i in exclude_idx]}")

# ========================================
# Step 6: 应用清洗
# ========================================
raw_clean = raw.copy()
ica.apply(raw_clean, exclude=exclude_idx)

print("\n✅ ICA 清洗完成!")

# ========================================
# Step 7: 保存结果
# ========================================
# 保存为 XDF (如果需要保持格式)
# ⚠️ MNE 不支持导出 XDF，需要其他工具

# 保存为 FIFF (推荐，MNE 原生)
raw_clean.save('cleaned_data.fif', overwrite=True)
print("💾 保存为: cleaned_data.fif")

# 保存为 EDF (如果需要医疗标准格式)
mne.export.export_raw('cleaned_data.edf', raw_clean, fmt='edf', overwrite=True)
print("💾 保存为: cleaned_data.edf")

# 保存 ICA 模型 (用于后续应用)
ica.exclude = exclude_idx
ica.save('trained_ica.fif', overwrite=True)
print("💾 保存 ICA: trained_ica.fif")
```

---

### 常见问题

#### Q1: XDF 读取报错 "No module named 'pyxdf'"？

```bash
# 安装 pyxdf
pip install pyxdf

# 或使用 conda
conda install -c conda-forge pyxdf
```

#### Q2: EDF 文件通道名不规范？

```python
# 读取 EDF
raw = mne.io.read_raw_edf('data.edf', preload=True)

# 重命名通道
raw.rename_channels({
    'FP1': 'Fp1',  # 标准化命名
    'FP2': 'Fp2',
    # ...
})

# 设置通道类型
raw.set_channel_types({
    'EOG1': 'eog',
    'ECG': 'ecg'
})
```

#### Q3: LSL 流没有电极位置？

```python
# 手动设置标准 10-20 位置
raw = mne.io.read_raw_xdf('recording.xdf')

# 使用标准 montage
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# 或自定义位置
# raw.set_montage(my_custom_montage)
```

#### Q4: 能否直接在 LSL 流上运行 ICALabel？

```python
# ✅ 可以! MNE-LSL 的 StreamLSL 兼容
from mne_lsl.stream import StreamLSL

stream = StreamLSL(bufsize=60, name='MyEEG')
stream.connect()

# ... 等待数据 ...

# 直接使用 (StreamLSL 继承自 BaseRaw)
ic_labels = label_components(stream, ica, method='iclabel')
```

---

### 总结

| 您的问题 | 答案 |
|---------|------|
| **XDF 可以用吗？** | ✅ 完全可以! `mne.io.read_raw_xdf()` |
| **EDF 可以用吗？** | ✅ 完全可以! `mne.io.read_raw_edf()` |
| **LSL 流可以吗？** | ✅ 可以! MNE-LSL `StreamLSL` 兼容 |
| **必须转 .fif 吗？** | ❌ 不需要，直接用原格式 |
| **性能有差异吗？** | ⚠️ XDF/EDF 读取稍慢，但 ICA 速度一样 |

**推荐工作流** (针对您的 LSL 场景):
1. 📼 LSL 录制 → XDF 文件
2. 📂 `mne.io.read_raw_xdf()` 读取
3. 🔧 预处理 (1-100 Hz, 平均参考)
4. 🧠 ICA 分解
5. 🤖 ICALabel 自动分类
6. 🗑️ 排除伪迹
7. 💾 保存结果 (FIFF 或 EDF)

---

## 完整数据流

### 端到端流程图

```
┌──────────────────────────────────────────────────────────────┐
│  Step 1: 原始数据                                             │
│                                                               │
│  raw = mne.io.read_raw_fif('data.fif')                       │
│  ica = ICA(n_components=15, method='infomax')                │
│  ica.fit(raw)                                                │
│                                                               │
│  输入对象:                                                    │
│  • Raw/Epochs: (n_channels, n_samples) 电压数据              │
│  • ICA: 分解矩阵 (icawinv, weights, sphere)                 │
└──────────────────────────────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 2: 特征提取 (features.py)                              │
│                                                               │
│  topo, psd, autocorr = get_iclabel_features(raw, ica)       │
│                                                               │
│  提取 3 种特征:                                               │
│  ✅ topo:    (32, 32, 1, n_components)  拓扑图               │
│  ✅ psd:     (1, 100, 1, n_components)  功率谱密度           │
│  ✅ autocorr:(1, 100, 1, n_components)  自相关               │
│                                                               │
│  特征工程细节 ▼▼▼                                             │
└──────────────────────────────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 3: 特征格式化 (network/utils.py)                       │
│                                                               │
│  _format_input(topo, psd, autocorr)                         │
│                                                               │
│  数据增强 (翻转和取反):                                       │
│  • topo: [原始, -原始, 水平翻转, -水平翻转] → x4             │
│  • psd:  复制 4 倍 → x4                                      │
│  • autocorr: 复制 4 倍 → x4                                  │
│                                                               │
│  输出形状:                                                    │
│  • topo:    (32, 32, 1, n_components * 4)                   │
│  • psd:     (1, 100, 1, n_components * 4)                   │
│  • autocorr:(1, 100, 1, n_components * 4)                   │
└──────────────────────────────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 4: 后端特定格式化                                       │
│                                                               │
│  PyTorch:  transpose(3,2,0,1) → to_tensor()                 │
│  ONNX:     transpose(3,2,0,1) → astype(float32)             │
│                                                               │
│  最终输入形状 (batch-first):                                  │
│  • topo:    (n_comp*4, 1, 32, 32)   [NCHW]                  │
│  • psd:     (n_comp*4, 1, 1, 100)   [NCHW]                  │
│  • autocorr:(n_comp*4, 1, 1, 100)   [NCHW]                  │
└──────────────────────────────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 5: 神经网络推理 (network/torch.py or onnx.py)         │
│                                                               │
│  ICLabelNet(topo, psd, autocorr)                            │
│                                                               │
│  网络结构:                                                    │
│  • 拓扑分支:  Conv2D (1→128→256→512) → 512×4×4             │
│  • PSD 分支:  Conv2D (1→128→256→1) → 1×1×100 → 重塑        │
│  • 自相关分支: Conv2D (1→128→256→1) → 1×1×100 → 重塑       │
│  • 合并: Concat → Conv2D (712→7) → Softmax                  │
│                                                               │
│  输出形状: (n_components*4, 7) 概率分布                      │
└──────────────────────────────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 6: 后处理                                               │
│                                                               │
│  平均 4 个增强样本的预测:                                     │
│  labels = reshape(labels, [n_components, 4, 7])             │
│  labels = mean(labels, axis=1)  # (n_components, 7)         │
│                                                               │
│  最终输出:                                                    │
│  array([[0.85, 0.02, 0.05, 0.01, 0.01, 0.03, 0.03],  # IC0  │
│         [0.12, 0.78, 0.03, 0.02, 0.01, 0.02, 0.02],  # IC1  │
│         ...])                                                │
│                                                               │
│  7 列对应: [brain, muscle, eye, heart, line, ch_noise, other]│
└──────────────────────────────────────────────────────────────┘
```

---

## 特征提取详解

### 1. 拓扑图特征 (Topoplot)

**目的**: 捕捉 ICA 成分的空间分布模式

```python
def _eeg_topoplot(inst, icawinv, picks):
    """
    将 ICA 分解矩阵转换为 32×32 像素的拓扑图
    
    输入:
    - icawinv: (n_channels, n_components) ICA 逆矩阵
    - inst: Raw/Epochs 对象，用于获取电极位置
    
    输出:
    - topo: (32, 32, 1, n_components) float32
    """
    
    # 步骤 1: 获取电极位置 (极坐标)
    rd, th = _mne_to_eeglab_locs(inst, picks)
    # rd: 径向距离 [0, 1]
    # th: 角度 (度)
    
    # 步骤 2: 转换为笛卡尔坐标
    th_rad = th * np.pi / 180
    x, y = pol2cart(th_rad, rd)
    
    # 步骤 3: 对每个成分进行插值
    for i in range(n_components):
        values = icawinv[:, i]  # 该成分在各电极的权重
        
        # 使用 griddata (v4) 插值到 32×32 网格
        topo[:, :, 0, i] = _gdatav4(x, y, values, 32, 32)
        
        # 归一化: 除以最大绝对值
        topo[:, :, 0, i] /= np.max(np.abs(topo[:, :, 0, i]))
    
    # 步骤 4: 遮蔽头外区域 (设为 NaN)
    mask = np.sqrt(x**2 + y**2) <= 0.5
    topo[~mask] = np.nan
    
    # 步骤 5: NaN → 0
    topo = np.nan_to_num(topo)
    
    return topo.astype(np.float32)
```

**示例可视化**:

```
眼电成分拓扑图 (32×32):        心电成分拓扑图 (32×32):
  前                              前
  ↑                               ↑
█████                           ·····
███○███  ← 前额强信号             ··○··  ← 中央强信号
·····                           ·····
  后                              后
```

---

### 2. 功率谱密度特征 (PSD)

**目的**: 捕捉 ICA 成分的频率特性

```python
def _eeg_rpsd(inst, ica, icaact):
    """
    计算 ICA 成分的功率谱密度
    
    输入:
    - icaact: (n_components, n_samples) ICA 激活时间序列
    - inst: Raw/Epochs 对象
    
    输出:
    - psd: (1, 100, 1, n_components) float32
    """
    
    # 常量
    sfreq = inst.info['sfreq']
    nyquist = int(sfreq / 2)
    nfreqs = min(nyquist, 100)  # 频率点数
    
    # 步骤 1: 分窗 (Hamming window)
    n_points = min(icaact.shape[1], int(sfreq))  # 窗长 = 1 秒
    window = np.hamming(n_points)
    
    # 步骤 2: 计算重叠窗口的索引
    hop_size = n_points // 2  # 50% 重叠
    n_segments = (icaact.shape[1] - n_points) // hop_size + 1
    
    # 步骤 3: 对每个窗口计算 FFT
    psd_all_segments = []
    for seg_idx in range(n_segments):
        start = seg_idx * hop_size
        end = start + n_points
        segment = icaact[:, start:end] * window
        
        # FFT
        fft_result = np.fft.fft(segment, axis=1)
        psd_segment = np.abs(fft_result[:, :nfreqs])**2
        psd_all_segments.append(psd_segment)
    
    # 步骤 4: 中位数 PSD (鲁棒估计)
    psd = np.median(psd_all_segments, axis=0)
    
    # 步骤 5: 归一化
    # 计算总功率
    total_power = np.sum(psd, axis=1, keepdims=True)
    psd = psd / total_power
    
    # 步骤 6: 对数变换
    psd = 10 * np.log10(psd + 1e-10)
    
    # 步骤 7: Resample 到恰好 100 个频率点
    if psd.shape[1] != 100:
        psd = resample_poly(psd, up=100, down=psd.shape[1], axis=1)
    
    # 步骤 8: 重塑为 (1, 100, 1, n_components)
    psd = psd.T.reshape(1, 100, 1, -1)
    
    return psd.astype(np.float32)
```

**频率特征示例**:

```
眼电 PSD (低频为主):        工频噪声 PSD (50Hz 尖峰):
Power                       Power
  ▲                           ▲
  │████▌                       │    ·
  │███▌··                      │    ·
  │██▌···                      │    █  ← 50 Hz
  │█▌····                      │    ·
  └────────→ Freq (Hz)         └────────→ Freq (Hz)
  0      100                   0      100
```

---

### 3. 自相关特征 (Autocorrelation)

**目的**: 捕捉 ICA 成分的时间规律性

```python
def _eeg_autocorr_welch(inst, ica, icaact):
    """
    计算 ICA 成分的自相关函数
    
    输入:
    - icaact: (n_components, n_samples) ICA 激活
    
    输出:
    - autocorr: (1, 100, 1, n_components) float32
    """
    
    # 步骤 1: 对每个成分计算自相关
    n_components = icaact.shape[0]
    n_samples = icaact.shape[1]
    autocorr_list = []
    
    for i in range(n_components):
        signal = icaact[i, :]
        
        # 去均值
        signal = signal - np.mean(signal)
        
        # 方法 1: FFT 快速自相关 (Wiener-Khinchin theorem)
        fft_signal = np.fft.fft(signal, n=2*n_samples)
        power_spectrum = np.abs(fft_signal)**2
        autocorr_full = np.fft.ifft(power_spectrum).real
        
        # 取前 100 个滞后
        autocorr_100 = autocorr_full[:100]
        
        # 归一化: 除以零滞后值
        autocorr_100 = autocorr_100 / autocorr_100[0]
        
        autocorr_list.append(autocorr_100)
    
    # 步骤 2: 堆叠为矩阵
    autocorr = np.array(autocorr_list).T  # (100, n_components)
    
    # 步骤 3: 重塑为 (1, 100, 1, n_components)
    autocorr = autocorr.reshape(1, 100, 1, -1)
    
    return autocorr.astype(np.float32)
```

**自相关模式示例**:

```
心电自相关 (周期性):        随机噪声自相关:
   1.0 ▲                       1.0 ▲
       │▲ ▲ ▲                       │▲
   0.5 │ ▼ ▼ ▼                   0.5 │ ·· random ··
       │  周期性                     │
   0.0 └────────→ Lag           0.0 └────────→ Lag
       0      100                    0      100
```

---

## 模型输入格式

### 完整输入规范

```python
# ========================================
# 特征提取后的原始形状
# ========================================
topo_raw    = (32, 32, 1, n_components)  # 拓扑图
psd_raw     = (1, 100, 1, n_components)  # PSD
autocorr_raw = (1, 100, 1, n_components)  # 自相关

# ========================================
# 数据增强 (_format_input)
# ========================================
# 拓扑图: 4 种变换
topo_aug = np.concatenate([
    topo_raw,                    # 原始
    -1 * topo_raw,               # 取反
    np.flip(topo_raw, axis=1),   # 水平翻转
    -1 * np.flip(topo_raw, axis=1)  # 翻转+取反
], axis=3)
# 形状: (32, 32, 1, n_components * 4)

# PSD/Autocorr: 简单复制 4 倍
psd_aug = np.tile(psd_raw, (1, 1, 1, 4))
autocorr_aug = np.tile(autocorr_raw, (1, 1, 1, 4))
# 形状: (1, 100, 1, n_components * 4)

# ========================================
# PyTorch 特定格式化
# ========================================
# Transpose: (H, W, C, N) → (N, C, H, W)
topo_torch = np.transpose(topo_aug, (3, 2, 0, 1))
psd_torch = np.transpose(psd_aug, (3, 2, 0, 1))
autocorr_torch = np.transpose(autocorr_aug, (3, 2, 0, 1))

# 转为 Tensor
topo_tensor = torch.from_numpy(topo_torch).float()
psd_tensor = torch.from_numpy(psd_torch).float()
autocorr_tensor = torch.from_numpy(autocorr_torch).float()

# 最终形状
print(topo_tensor.shape)     # (n_comp*4, 1, 32, 32)
print(psd_tensor.shape)      # (n_comp*4, 1, 1, 100)
print(autocorr_tensor.shape) # (n_comp*4, 1, 1, 100)

# ========================================
# ONNX 特定格式化 (几乎相同)
# ========================================
topo_onnx = np.transpose(topo_aug, (3, 2, 0, 1)).astype(np.float32)
psd_onnx = np.transpose(psd_aug, (3, 2, 0, 1)).astype(np.float32)
autocorr_onnx = np.transpose(autocorr_aug, (3, 2, 0, 1)).astype(np.float32)
```

### 示例: 15 个 IC 的输入形状

```python
n_components = 15

# 原始特征
topo:    (32, 32, 1, 15)
psd:     (1, 100, 1, 15)
autocorr:(1, 100, 1, 15)

# 数据增强后
topo:    (32, 32, 1, 60)  # 15 * 4
psd:     (1, 100, 1, 60)
autocorr:(1, 100, 1, 60)

# Batch-first (PyTorch/ONNX)
topo:    (60, 1, 32, 32)   # [Batch, Channel, Height, Width]
psd:     (60, 1, 1, 100)   # [Batch, Channel, Height, Width]
autocorr:(60, 1, 1, 100)

# 推理输出
labels:  (60, 7)  # 每个增强样本的 7 类概率

# 后处理: 平均 4 个增强样本
labels_reshaped = labels.reshape(15, 4, 7)
final_labels = labels_reshaped.mean(axis=1)  # (15, 7)
```

---

## 神经网络架构

### ICLabelNet 完整结构

```python
class ICLabelNet(nn.Module):
    """
    ICLabel 卷积神经网络
    
    输入:
    - topo:    (batch, 1, 32, 32)
    - psd:     (batch, 1, 1, 100)
    - autocorr:(batch, 1, 1, 100)
    
    输出:
    - labels: (batch, 7) Softmax 概率
    """
    
    def __init__(self):
        super().__init__()
        
        # ========================================
        # 分支 1: 拓扑图 (Image) 分支
        # ========================================
        self.img_conv = nn.Sequential(
            # Conv1: 1 → 128 channels
            nn.Conv2d(1, 128, kernel_size=4, stride=2, padding=1),
            # 输出: (batch, 128, 16, 16)
            nn.LeakyReLU(0.2),
            
            # Conv2: 128 → 256 channels
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            # 输出: (batch, 256, 8, 8)
            nn.LeakyReLU(0.2),
            
            # Conv3: 256 → 512 channels
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            # 输出: (batch, 512, 4, 4)
            nn.LeakyReLU(0.2)
        )
        
        # ========================================
        # 分支 2: PSD 分支
        # ========================================
        self.psds_conv = nn.Sequential(
            # Conv1: 1 → 128 channels (1D Conv on freq axis)
            nn.Conv2d(1, 128, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            # 输出: (batch, 128, 1, 100)
            nn.LeakyReLU(0.2),
            
            # Conv2: 128 → 256 channels
            nn.Conv2d(128, 256, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            # 输出: (batch, 256, 1, 100)
            nn.LeakyReLU(0.2),
            
            # Conv3: 256 → 1 channel (降维)
            nn.Conv2d(256, 1, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            # 输出: (batch, 1, 1, 100)
            nn.LeakyReLU(0.2)
        )
        
        # ========================================
        # 分支 3: Autocorr 分支 (与 PSD 相同结构)
        # ========================================
        self.autocorr_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LeakyReLU(0.2)
        )
        # 输出: (batch, 1, 1, 100)
        
        # ========================================
        # 融合层
        # ========================================
        # 合并后通道数: 512 + 100 + 100 = 712
        self.final_conv = nn.Conv2d(712, 7, kernel_size=4, stride=1, padding=0)
        # 输出: (batch, 7, 1, 1)
        
        self.softmax = nn.Softmax(dim=1)
    
    def reshape_psd_autocorr(self, x):
        """
        PSD/Autocorr 重塑和扩展
        
        输入: (batch, 1, 1, 100)
        输出: (batch, 100, 4, 4)  # 匹配拓扑分支
        """
        # 重塑: (batch, 1, 1, 100) → (batch, 100, 1, 1)
        x = x.permute(0, 3, 1, 2)
        
        # 复制扩展: (batch, 100, 1, 1) → (batch, 100, 4, 4)
        x = x.repeat(1, 1, 4, 4)
        
        return x
    
    def forward(self, topo, psd, autocorr):
        # 三个分支并行
        img_features = self.img_conv(topo)         # (batch, 512, 4, 4)
        psd_features = self.psds_conv(psd)         # (batch, 1, 1, 100)
        autocorr_features = self.autocorr_conv(autocorr)  # (batch, 1, 1, 100)
        
        # PSD/Autocorr 重塑匹配拓扑尺寸
        psd_reshaped = self.reshape_psd_autocorr(psd_features)
        # (batch, 100, 4, 4)
        autocorr_reshaped = self.reshape_psd_autocorr(autocorr_features)
        # (batch, 100, 4, 4)
        
        # 通道拼接
        concat = torch.cat([img_features, psd_reshaped, autocorr_reshaped], dim=1)
        # (batch, 512+100+100=712, 4, 4)
        
        # 最终分类
        out = self.final_conv(concat)  # (batch, 7, 1, 1)
        out = out.squeeze(-1).squeeze(-1)  # (batch, 7)
        out = self.softmax(out)  # Softmax 归一化
        
        return out
```

### 参数量统计

```python
# 拓扑分支
Conv1: 1×128×4×4   = 2,048 params
Conv2: 128×256×4×4 = 524,288 params
Conv3: 256×512×4×4 = 2,097,152 params

# PSD 分支
Conv1: 1×128×1×3   = 384 params
Conv2: 128×256×1×3 = 98,304 params
Conv3: 256×1×1×3   = 768 params

# Autocorr 分支 (同 PSD)
                   = 99,456 params

# 融合层
Conv: 712×7×4×4    = 79,744 params

# 总计
Total: ~2.9M parameters
```

---

## 现有后端实现

### PyTorch 后端

**文件**: `mne_icalabel/iclabel/network/torch.py`

```python
def _run_iclabel(images, psds, autocorr):
    """
    PyTorch 推理流程
    """
    # 1. 加载模型权重
    network_file = 'ICLabelNet.pt'
    model = ICLabelNet()
    model.load_state_dict(torch.load(network_file, weights_only=True))
    model.eval()  # 评估模式
    
    # 2. 格式化输入
    topo, psd, autocorr = _format_input(images, psds, autocorr)
    topo = torch.from_numpy(np.transpose(topo, (3,2,0,1))).float()
    psd = torch.from_numpy(np.transpose(psd, (3,2,0,1))).float()
    autocorr = torch.from_numpy(np.transpose(autocorr, (3,2,0,1))).float()
    
    # 3. 推理 (无梯度)
    with torch.no_grad():
        labels = model(topo, psd, autocorr)
    
    # 4. 转回 NumPy
    return labels.numpy()
```

**特点**:
- ✅ 支持 GPU 加速 (自动检测 CUDA)
- ✅ 原生 PyTorch 模型，速度快
- ❌ 依赖大 (~1GB with CUDA)

---

### ONNX 后端

**文件**: `mne_icalabel/iclabel/network/onnx.py`

```python
import onnxruntime as ort

def _run_iclabel(images, psds, autocorr):
    """
    ONNX Runtime 推理流程
    """
    # 1. 创建推理会话
    network_file = 'ICLabelNet.onnx'
    session = ort.InferenceSession(network_file)
    
    # 2. 格式化输入
    topo, psd, autocorr = _format_input(images, psds, autocorr)
    topo = np.transpose(topo, (3,2,0,1)).astype(np.float32)
    psd = np.transpose(psd, (3,2,0,1)).astype(np.float32)
    autocorr = np.transpose(autocorr, (3,2,0,1)).astype(np.float32)
    
    # 3. 推理
    labels = session.run(
        None,  # 输出名称 (None = 所有输出)
        {
            'topo': topo,
            'psds': psd,
            'autocorr': autocorr
        }
    )
    
    # 4. 返回第一个输出
    return labels[0]
```

**特点**:
- ✅ 轻量级 (~50MB)
- ✅ CPU 优化良好
- ⚠️ 稍慢于 PyTorch (~20%)

---

## 理论上可添加的后端

### 对比表

| 后端 | 可行性 | 模型转换 | 工作量 | 优势 | 劣势 |
|------|-------|---------|-------|------|------|
| **TensorFlow** | ✅ 高 | ONNX→TF | 中等 | 生态成熟，Keras API 友好 | 依赖大 |
| **TensorRT** | ✅ 高 | ONNX→TRT | 较高 | NVIDIA GPU 极致性能 | 仅 GPU，复杂 |
| **OpenVINO** | ✅ 高 | ONNX→IR | 中等 | Intel CPU/GPU 优化 | 平台限制 |
| **TensorFlow Lite** | ✅ 高 | TF→TFLite | 中等 | 移动/嵌入式 | 功能受限 |
| **Core ML** | ⚠️ 中 | ONNX→mlmodel | 较高 | iOS/macOS 原生 | 仅苹果平台 |
| **Burn (Rust)** | ⚠️ 中 | 手动实现 | 高 | 无 Python 依赖，快 | 生态不成熟 |
| **Candle (Rust)** | ✅ 高 | PyTorch→Candle | 中等 | Rust 原生，GPU 支持 | 新项目，稳定性 |
| **MLC-LLM** | ⚠️ 低 | 不适用 | - | - | 针对 LLM |

---

### 添加 TensorFlow 后端示例

#### 步骤 1: 模型转换

```python
# 方法 1: ONNX → TensorFlow
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load('ICLabelNet.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('ICLabelNet_tf')

# 方法 2: PyTorch → ONNX → TensorFlow (更可靠)
import torch
model = ICLabelNet()
model.load_state_dict(torch.load('ICLabelNet.pt'))

dummy_topo = torch.randn(1, 1, 32, 32)
dummy_psd = torch.randn(1, 1, 1, 100)
dummy_autocorr = torch.randn(1, 1, 1, 100)

torch.onnx.export(
    model,
    (dummy_topo, dummy_psd, dummy_autocorr),
    'ICLabelNet.onnx',
    input_names=['topo', 'psds', 'autocorr'],
    output_names=['labels'],
    dynamic_axes={
        'topo': {0: 'batch'},
        'psds': {0: 'batch'},
        'autocorr': {0: 'batch'}
    }
)
```

#### 步骤 2: 推理实现

```python
# mne_icalabel/iclabel/network/tensorflow.py
import tensorflow as tf
import numpy as np

def _run_iclabel_tf(images, psds, autocorr):
    """TensorFlow 推理"""
    # 加载模型
    model = tf.saved_model.load('ICLabelNet_tf')
    infer = model.signatures['serving_default']
    
    # 格式化
    topo, psd, autocorr = _format_input(images, psds, autocorr)
    topo = tf.constant(np.transpose(topo, (3,2,0,1)), dtype=tf.float32)
    psd = tf.constant(np.transpose(psd, (3,2,0,1)), dtype=tf.float32)
    autocorr = tf.constant(np.transpose(autocorr, (3,2,0,1)), dtype=tf.float32)
    
    # 推理
    outputs = infer(topo=topo, psds=psd, autocorr=autocorr)
    
    return outputs['labels'].numpy()
```

---

### 添加 OpenVINO 后端示例

```bash
# 转换模型
mo --input_model ICLabelNet.onnx \
   --output_dir openvino_model \
   --data_type FP32
```

```python
# mne_icalabel/iclabel/network/openvino.py
from openvino.runtime import Core

def _run_iclabel_openvino(images, psds, autocorr):
    """OpenVINO 推理 (Intel 优化)"""
    # 初始化
    ie = Core()
    model = ie.read_model('ICLabelNet.xml')
    compiled = ie.compile_model(model, 'CPU')
    
    # 格式化
    topo, psd, autocorr = _format_input(images, psds, autocorr)
    topo = np.transpose(topo, (3,2,0,1)).astype(np.float32)
    psd = np.transpose(psd, (3,2,0,1)).astype(np.float32)
    autocorr = np.transpose(autocorr, (3,2,0,1)).astype(np.float32)
    
    # 推理
    results = compiled([topo, psd, autocorr])
    
    return results[0]
```

---

## Rust 后端可行性分析

### Burn vs Candle 对比

| 特性 | Burn | Candle |
|------|------|--------|
| **开发者** | tracel-ai | Hugging Face |
| **成熟度** | ⚠️ 早期 (v0.13) | ⚠️ 新项目 (2023) |
| **GPU 支持** | ✅ CUDA, Metal, WebGPU | ✅ CUDA, Metal |
| **模型导入** | ❌ 需手动实现 | ✅ PyTorch 权重 |
| **Python 绑定** | ⚠️ 有限 (PyO3) | ✅ 官方支持 |
| **生态** | 小 | 成长中 |

---

### Candle 后端实现 (推荐)

**为什么选 Candle**:
1. ✅ 可以直接加载 PyTorch 权重 (`.pt` 文件)
2. ✅ 有 Python 绑定 (`candle-pyo3`)
3. ✅ GPU 支持完善
4. ✅ Hugging Face 维护，更新活跃

#### 步骤 1: 安装 Candle

```bash
# Rust 侧
cargo add candle-core candle-nn

# Python 绑定 (如果有)
pip install candle-pyo3
```

#### 步骤 2: 实现 Rust 模型

```rust
// src/iclabel_net.rs
use candle_core::{Tensor, Device, DType};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder, ops::softmax};

struct ICLabelNet {
    // 拓扑分支
    img_conv1: Conv2d,
    img_conv2: Conv2d,
    img_conv3: Conv2d,
    
    // PSD 分支
    psd_conv1: Conv2d,
    psd_conv2: Conv2d,
    psd_conv3: Conv2d,
    
    // Autocorr 分支
    autocorr_conv1: Conv2d,
    autocorr_conv2: Conv2d,
    autocorr_conv3: Conv2d,
    
    // 融合层
    final_conv: Conv2d,
}

impl ICLabelNet {
    fn new(vb: VarBuilder) -> Result<Self> {
        // 拓扑分支
        let img_conv1 = candle_nn::conv2d(
            1, 128, 4,
            Conv2dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vb.pp("img_conv.0")
        )?;
        
        // ... (类似定义其他层)
        
        Ok(Self {
            img_conv1,
            // ...
        })
    }
    
    fn forward(&self, topo: &Tensor, psd: &Tensor, autocorr: &Tensor) 
        -> Result<Tensor> 
    {
        // 拓扑分支
        let img = self.img_conv1.forward(topo)?;
        let img = img.relu()?;  // LeakyReLU 简化
        let img = self.img_conv2.forward(&img)?;
        let img = img.relu()?;
        let img = self.img_conv3.forward(&img)?;
        let img = img.relu()?;
        
        // PSD 分支
        let psd_out = self.psd_conv1.forward(psd)?;
        let psd_out = psd_out.relu()?;
        let psd_out = self.psd_conv2.forward(&psd_out)?;
        let psd_out = psd_out.relu()?;
        let psd_out = self.psd_conv3.forward(&psd_out)?;
        
        // Autocorr 分支 (同理)
        // ...
        
        // 重塑和拼接
        let psd_reshaped = psd_out.permute((0, 3, 1, 2))?.repeat((1, 1, 4, 4))?;
        let autocorr_reshaped = autocorr_out.permute((0, 3, 1, 2))?.repeat((1, 1, 4, 4))?;
        
        let concat = Tensor::cat(&[img, psd_reshaped, autocorr_reshaped], 1)?;
        
        // 最终层
        let out = self.final_conv.forward(&concat)?;
        let out = out.squeeze(2)?.squeeze(2)?;
        let out = softmax(&out, 1)?;
        
        Ok(out)
    }
}

// 加载 PyTorch 权重
fn load_model(device: &Device) -> Result<ICLabelNet> {
    let weights = candle_core::safetensors::load(
        "ICLabelNet.safetensors",  // 需从 .pt 转换
        device
    )?;
    let vb = VarBuilder::from_tensors(weights, DType::F32, device);
    ICLabelNet::new(vb)
}
```

#### 步骤 3: Python 绑定

```python
# mne_icalabel/iclabel/network/candle.py
import numpy as np
from candle_pyo3 import ICLabelNet  # Rust 编译的 Python 扩展

def _run_iclabel_candle(images, psds, autocorr):
    """Candle (Rust) 推理"""
    # 加载模型
    model = ICLabelNet.from_pretrained('ICLabelNet.safetensors')
    
    # 格式化
    topo, psd, autocorr = _format_input(images, psds, autocorr)
    topo = np.transpose(topo, (3,2,0,1)).astype(np.float32)
    psd = np.transpose(psd, (3,2,0,1)).astype(np.float32)
    autocorr = np.transpose(autocorr, (3,2,0,1)).astype(np.float32)
    
    # 推理 (调用 Rust)
    labels = model.forward(topo, psd, autocorr)
    
    return labels
```

#### 步骤 4: 权重转换

```python
# PyTorch .pt → Safetensors (Candle 支持格式)
import torch
from safetensors.torch import save_file

# 加载 PyTorch 权重
state_dict = torch.load('ICLabelNet.pt', map_location='cpu')

# 保存为 Safetensors
save_file(state_dict, 'ICLabelNet.safetensors')
```

---

### Burn 后端实现 (不推荐)

**为什么不推荐 Burn**:
- ❌ 不能直接加载 PyTorch 权重
- ❌ 需要手动实现所有层
- ❌ Python 绑定不完善
- ⚠️ 生态太新，文档少

**如果真要用 Burn**:

```rust
// 需要手动定义并训练模型
use burn::{
    nn::{conv::Conv2d, Linear},
    tensor::{Tensor, backend::Backend},
};

// 完全从头实现，不能复用已有权重
// 工作量极大，不切实际
```

---

## 后端选择决策树

```
开始
 │
 ├─ 需要 GPU 极致性能? ──YES─→ TensorRT (NVIDIA)
 │                              或 OpenVINO (Intel)
 │
 ├─ 需要嵌入式/移动? ──YES─→ TensorFlow Lite
 │                           或 Core ML (iOS)
 │
 ├─ 需要 Rust 原生? ──YES─→ Candle ✅
 │  (无 Python 依赖)          (不推荐 Burn)
 │
 ├─ 需要跨平台 CPU? ──YES─→ ONNX Runtime ✅ (已有)
 │
 └─ 默认 ──────────────────→ PyTorch ✅ (已有)
```

---

## 添加新后端的通用流程

### 完整 Checklist

#### 1. 模型转换 ✅

```bash
# 示例: PyTorch → ONNX → 目标格式
python convert_to_onnx.py
onnx-tool optimize ICLabelNet.onnx -o ICLabelNet_opt.onnx
# 然后转为目标格式 (TF, TRT, OpenVINO, etc.)
```

#### 2. 实现推理函数 ✅

```python
# mne_icalabel/iclabel/network/your_backend.py

from .utils import _format_input
import numpy as np

def _run_iclabel(images, psds, autocorr):
    """
    你的后端推理实现
    
    必须:
    1. 调用 _format_input() 进行数据增强
    2. Transpose 为 batch-first (NCHW)
    3. 推理返回 (n_components*4, 7)
    4. MNE-ICALabel 会自动处理后续平均
    """
    # 1. 数据增强
    topo, psd, autocorr = _format_input(images, psds, autocorr)
    
    # 2. Transpose
    topo = np.transpose(topo, (3, 2, 0, 1))
    psd = np.transpose(psd, (3, 2, 0, 1))
    autocorr = np.transpose(autocorr, (3, 2, 0, 1))
    
    # 3. 转为你的后端格式 (TF Tensor, TRT Input, etc.)
    topo_backend = convert_to_backend_format(topo)
    psd_backend = convert_to_backend_format(psd)
    autocorr_backend = convert_to_backend_format(autocorr)
    
    # 4. 推理
    labels = model.predict({
        'topo': topo_backend,
        'psds': psd_backend,
        'autocorr': autocorr_backend
    })
    
    # 5. 返回 NumPy
    return labels
```

#### 3. 修改后端选择逻辑 ✅

```python
# mne_icalabel/iclabel/network/__init__.py

def run_iclabel(images, psds, autocorr, backend=None):
    _check_option("backend", backend, 
                  (None, "torch", "onnx", "your_backend"))  # ← 添加
    
    if backend == "your_backend":
        import_optional_dependency("your_framework", raise_error=True)
        from .your_backend import _run_iclabel
        return _run_iclabel(images, psds, autocorr)
    
    # ... 原有逻辑
```

#### 4. 编写单元测试 ✅

```python
# mne_icalabel/iclabel/tests/test_backends.py

@requires_module("your_framework")
def test_your_backend():
    # 加载测试数据
    raw, ica = load_test_data()
    
    # PyTorch 基准
    labels_torch = iclabel_label_components(
        raw, ica, backend='torch', inplace=False
    )
    
    # 你的后端
    labels_yours = iclabel_label_components(
        raw, ica, backend='your_backend', inplace=False
    )
    
    # 数值一致性检查
    np.testing.assert_allclose(
        labels_torch, labels_yours,
        rtol=1e-5, atol=1e-6
    )
```

#### 5. 更新文档 ✅

```python
# mne_icalabel/iclabel/label_components.py

def iclabel_label_components(inst, ica, backend=None):
    """
    Parameters
    ----------
    backend : None | 'torch' | 'onnx' | 'your_backend'
        Backend to use. If None, auto-selects.
        
        - 'torch': PyTorch (fastest on GPU)
        - 'onnx': ONNX Runtime (lightweight)
        - 'your_backend': Your new backend (describe here)
    """
```

---

## 性能基准 (预估)

| 后端 | CPU 速度 | GPU 速度 | 内存占用 | 安装大小 |
|------|---------|---------|---------|---------|
| **PyTorch** | ~150ms | ~50ms | ~500MB | ~1GB |
| **ONNX** | ~200ms | ~80ms | ~100MB | ~50MB |
| **TensorFlow** | ~180ms | ~60ms | ~400MB | ~500MB |
| **TensorRT** | N/A | ~30ms ⚡ | ~300MB | ~500MB |
| **OpenVINO** | ~120ms 🚀 | ~70ms | ~150MB | ~200MB |
| **Candle (Rust)** | ~160ms | ~55ms | ~80MB 💾 | ~30MB |
| **TFLite** | ~250ms | N/A | ~50MB 💾 | ~20MB |

---

## 总结

### 核心要点

1. **模型输入 = 3 种特征**
   - 拓扑图: (32, 32, 1, n_comp) 空间分布
   - PSD: (1, 100, 1, n_comp) 频率特性
   - 自相关: (1, 100, 1, n_comp) 时间规律

2. **数据增强 = x4 样本**
   - 拓扑图翻转/取反 4 种变换
   - PSD/自相关简单复制
   - 推理后平均结果

3. **现有后端 = PyTorch + ONNX**
   - PyTorch: GPU 快，依赖大
   - ONNX: CPU 好，轻量级

4. **可添加后端**
   - ✅ **高可行性**: TensorFlow, TensorRT, OpenVINO, Candle
   - ⚠️ **中等可行性**: TFLite, Core ML
   - ❌ **不推荐**: Burn (生态不成熟)

5. **Rust 后端推荐**
   - **Candle** ✅: 可加载 PyTorch 权重，GPU 支持好
   - **Burn** ❌: 需从头实现，工作量大

---

**相关文档**:
- [MNE-ICALabel 自动分类指南](mne-icalabel-guide.md)
- [MNE 离线处理指南](mne-offline-processing.md)
- [MNE 实时处理指南](mne-realtime-processing.md)
