# MNE 离线处理完整指南

> **用途**: 理解 MNE-Python 离线数据分析的完整流程  
> **创建日期**: 2026-01-31  
> **核心**: 文件 → 预处理 → 分析 → 结果

---

## 目录

1. [什么是离线处理](#什么是离线处理)
2. [核心类: BaseRaw](#核心类-baseraw)
3. [数据结构](#数据结构)
4. [完整工作流](#完整工作流)
5. [代码示例](#代码示例)
6. [优势与限制](#优势与限制)

---

## 什么是离线处理

### 定义

**离线处理** = 在实验完成后对已保存的数据进行分析

```
时间线:
  实验采集 (第 1 天)  →  保存文件  →  离线分析 (数天/数周后)
  
数据流:
  硬件设备 → 采集软件 → 保存文件 (.fif, .edf, .bdf) → MNE-Python 分析
```

### 核心特点

| 特性 | 说明 |
|------|------|
| **数据来源** | 已保存的文件 |
| **数据状态** | 静态、完整、固定长度 |
| **访问方式** | 可随机访问任意时间点 |
| **处理时机** | 实验结束后,无时间压力 |
| **可重复性** | 100% 可重复 (数据不变) |
| **典型用途** | 科学研究、发表论文 |

---

## 核心类: BaseRaw

### 类定义

```python
# 位置: mne/io/base.py

class BaseRaw(
    ProjMixin,           # 投影操作
    ContainsMixin,       # 包含检查
    UpdateChannelsMixin, # 通道更新
    ReferenceMixin,      # 重参考
    SetChannelsMixin,    # 设置通道
    InterpolationMixin,  # 插值
    TimeMixin,           # 时间操作
    SizeMixin,           # 大小信息
    FilterMixin,         # 滤波
    SpectrumMixin,       # 频谱分析
):
    """所有 Raw 类的基类"""
```

### 继承树

```
BaseRaw (基类)
  ├── Raw (FIF 格式)
  ├── RawEDF (EDF 格式)
  ├── RawBDF (BioSemi BDF)
  ├── RawBrainVision (Brain Products)
  ├── RawCNT (Neuroscan)
  ├── RawEEGLAB (EEGLAB)
  ├── RawArray (从数组创建)
  └── ... (20+ 种格式)
```

---

## 数据结构

### 内存表示

```python
# Raw 对象的核心属性
raw = mne.io.read_raw_fif('sample.fif')

raw._data         # 数据矩阵: (n_channels, n_samples)
raw.info          # 元信息: 通道名、采样率、坏通道等
raw._times        # 时间向量: (n_samples,)
raw._first_samps  # 每个文件的起始样本
raw._last_samps   # 每个文件的结束样本
raw.preload       # 是否已加载到内存
```

### 两种加载模式

#### 模式 1: 按需加载 (preload=False)

```python
# 数据保留在硬盘,需要时才读取
raw = mne.io.read_raw_fif('data.fif', preload=False)

# 优点: 节省内存 (适合大文件)
# 缺点: 访问速度慢 (需要 I/O)

# 示例: 600 MB 文件,只占用 ~10 MB 内存
```

#### 模式 2: 全部加载 (preload=True)

```python
# 数据全部加载到内存
raw = mne.io.read_raw_fif('data.fif', preload=True)

# 优点: 访问速度快 (在内存中)
# 缺点: 占用大量内存

# 示例: 600 MB 文件,占用 ~600 MB 内存
```

### 数据访问

```python
# 随机访问任意时间段
raw = mne.io.read_raw_fif('data.fif')

# 语法: raw[channels, samples]

# 示例 1: 所有通道,前 1000 个样本
data = raw[:, 0:1000]  # 形状: (n_channels, 1000)

# 示例 2: 所有通道,中间某段
data = raw[:, 50000:51000]

# 示例 3: 指定通道
data = raw[['EEG 001', 'EEG 002'], 0:1000]

# 示例 4: 可以反复读取同一段
for i in range(10):
    data = raw[:, 0:1000]  # ✅ 完全可以
```

---

## 完整工作流

### 标准分析流程

```
┌─────────────────────────────────────────────┐
│  步骤 1: 读取数据                            │
│  raw = mne.io.read_raw_fif('data.fif')      │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  步骤 2: 数据探索                            │
│  • raw.plot()            # 可视化           │
│  • raw.info              # 查看元信息       │
│  • raw.compute_psd()     # 功率谱          │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  步骤 3: 预处理                              │
│  • raw.filter(1, 40)     # 滤波            │
│  • raw.set_eeg_reference('average')        │
│  • ica = ICA(); ica.fit(raw)  # 去伪迹     │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  步骤 4: 事件检测和分段                      │
│  • events = mne.find_events(raw)           │
│  • epochs = mne.Epochs(raw, events, ...)   │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  步骤 5: 分析                                │
│  • evoked = epochs.average()  # ERP        │
│  • power = tfr_morlet(epochs) # 时频       │
│  • stc = apply_inverse(...)   # 源定位     │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  步骤 6: 统计和可视化                        │
│  • evoked.plot()                           │
│  • cluster_stats = permutation_cluster_test│
└─────────────────────────────────────────────┘
```

---

## 代码示例

### 示例 1: 基础 ERP 分析

```python
import mne
from mne.datasets import sample

# ========== 1. 读取数据 ==========
data_path = sample.data_path()
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)

print(raw)
# <Raw | sample_audvis_raw.fif, 376 x 166800 (334.0 s)>
#        376 通道, 166800 样本 @ 500 Hz

# ========== 2. 选择通道 ==========
raw.pick_types(meg=False, eeg=True, eog=True, stim=True)
print(f"保留 {len(raw.ch_names)} 个通道")

# ========== 3. 预处理 ==========
# 3.1 滤波
raw.filter(l_freq=0.1, h_freq=40.0)

# 3.2 设置坏通道
raw.info['bads'] = ['EEG 053']
raw.interpolate_bads()

# 3.3 重参考
raw.set_eeg_reference('average', projection=True)

# ========== 4. 事件检测 ==========
events = mne.find_events(raw, stim_channel='STI 014')
print(f"找到 {len(events)} 个事件")

# 定义事件类型
event_id = {'auditory/left': 1, 'visual/left': 3}

# ========== 5. 分段 (Epoching) ==========
epochs = mne.Epochs(
    raw, 
    events, 
    event_id, 
    tmin=-0.2,      # epoch 起始 (事件前 200ms)
    tmax=0.5,       # epoch 结束 (事件后 500ms)
    baseline=(None, 0),  # 基线校正
    preload=True
)

print(epochs)
# <Epochs | 145 events (all good), -0.2 - 0.5 s, baseline off>

# ========== 6. 计算平均 ERP ==========
evoked_auditory = epochs['auditory/left'].average()
evoked_visual = epochs['visual/left'].average()

# ========== 7. 可视化 ==========
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 听觉 ERP
evoked_auditory.plot(axes=axes[0], show=False)
axes[0].set_title('Auditory ERP')

# 视觉 ERP
evoked_visual.plot(axes=axes[1], show=False)
axes[1].set_title('Visual ERP')

plt.tight_layout()
plt.show()

# ========== 8. 地形图 ==========
times = [0.1, 0.2, 0.3]  # 100, 200, 300 ms
evoked_visual.plot_topomap(times=times, time_unit='s')

# ========== 9. 保存结果 ==========
evoked_auditory.save('auditory-ave.fif')
evoked_visual.save('visual-ave.fif')
```

---

### 示例 2: ICA 去眼电

```python
import mne
from mne.preprocessing import ICA, create_eog_epochs

# 读取数据
raw = mne.io.read_raw_fif('sample.fif', preload=True)
raw.filter(1, 40)  # ICA 前推荐先滤波

# ========== 1. 创建 ICA 对象 ==========
ica = ICA(
    n_components=20,    # 提取 20 个独立成分
    random_state=42,    # 随机种子 (可重复)
    method='fastica'    # FastICA 算法
)

# ========== 2. 拟合 ICA (需要访问全部数据) ==========
ica.fit(raw, picks='eeg')
print(ica)
# <ICA | raw data decomposition, fit (fastica)>

# ========== 3. 可视化独立成分 ==========
ica.plot_sources(raw)  # 查看所有成分的时间序列
ica.plot_components()  # 查看成分的地形图

# ========== 4. 自动检测眼电成分 ==========
# 创建眼电 epochs
eog_epochs = create_eog_epochs(raw)

# 自动查找眼电相关成分
eog_indices, scores = ica.find_bads_eog(eog_epochs)
print(f"检测到眼电成分: {eog_indices}")

# 标记为坏成分
ica.exclude = eog_indices

# ========== 5. 可视化被排除的成分 ==========
ica.plot_properties(raw, picks=eog_indices)

# ========== 6. 应用 ICA (去除眼电) ==========
raw_clean = raw.copy()
ica.apply(raw_clean)

# ========== 7. 对比前后 ==========
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# 原始数据
raw.plot(duration=10, n_channels=30, 
         scalings='auto', axes=axes[0], show=False)
axes[0].set_title('Before ICA')

# 去除眼电后
raw_clean.plot(duration=10, n_channels=30,
               scalings='auto', axes=axes[1], show=False)
axes[1].set_title('After ICA (EOG removed)')

plt.tight_layout()
plt.show()

# ========== 8. 保存清洁数据 ==========
raw_clean.save('sample_clean-raw.fif', overwrite=True)
ica.save('sample-ica.fif')
```

---

### 示例 3: 时频分析

```python
import mne
from mne.time_frequency import tfr_morlet
import numpy as np

# 读取数据并分段
raw = mne.io.read_raw_fif('sample.fif', preload=True)
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id={'visual': 3},
                    tmin=-0.5, tmax=1.5, baseline=(-0.5, 0))

# ========== 1. 定义频率 ==========
freqs = np.arange(4, 40, 2)  # 4-40 Hz, 步长 2 Hz
print(f"分析频率: {freqs} Hz")

# ========== 2. 计算时频功率 ==========
# n_cycles: 每个频率的周期数 (频率越高,周期越多)
n_cycles = freqs / 2.0  # 例: 4 Hz → 2 cycles, 40 Hz → 20 cycles

power = tfr_morlet(
    epochs, 
    freqs=freqs, 
    n_cycles=n_cycles,
    return_itc=False,  # 不计算相位锁定值
    average=True,      # 跨 epochs 平均
    n_jobs=4           # 并行计算
)

print(power)
# <AverageTFR | 60 x 18 x 501, ... 4.0 - 38.0 Hz>

# ========== 3. 可视化 ==========
# 3.1 所有传感器的平均时频图
power.plot(
    baseline=(-0.5, 0),    # 基线校正
    mode='logratio',       # 对数比率
    title='Time-Frequency Power'
)

# 3.2 选择单个通道
power.plot(
    picks=['EEG 060'],     # Oz 通道 (后脑)
    baseline=(-0.5, 0),
    mode='logratio'
)

# 3.3 地形图 (特定时间-频率)
power.plot_topomap(
    tmin=0.2, tmax=0.4,    # 200-400 ms
    fmin=8, fmax=12,       # Alpha 频段
    baseline=(-0.5, 0),
    mode='logratio'
)

# ========== 4. 提取特定频段功率 ==========
# Alpha 波 (8-12 Hz) 在 0-0.5 秒的平均功率
alpha_power = power.copy().crop(
    tmin=0, tmax=0.5,
    fmin=8, fmax=12
)

# 计算每个通道的平均
alpha_mean = alpha_power.data.mean(axis=(1, 2))  # (n_channels,)

# 找到功率最强的通道
max_ch_idx = np.argmax(alpha_mean)
max_ch_name = power.ch_names[max_ch_idx]
print(f"Alpha 功率最强通道: {max_ch_name}")
```

---

### 示例 4: 批量处理多个被试

```python
import mne
from pathlib import Path
from mne.parallel import parallel_func

# ========== 定义单个被试的处理函数 ==========
def process_subject(subject_id):
    """处理单个被试的数据"""
    print(f"Processing subject {subject_id}...")
    
    # 读取数据
    fname = f'data/sub-{subject_id:02d}_raw.fif'
    raw = mne.io.read_raw_fif(fname, preload=True)
    
    # 预处理
    raw.filter(0.1, 40)
    raw.set_eeg_reference('average')
    
    # 事件检测
    events = mne.find_events(raw, stim_channel='STI 014')
    
    # 分段
    epochs = mne.Epochs(
        raw, events, 
        event_id={'target': 1, 'non-target': 2},
        tmin=-0.2, tmax=0.8,
        baseline=(None, 0),
        preload=True
    )
    
    # 计算 Evoked
    evoked_target = epochs['target'].average()
    evoked_nontarget = epochs['non-target'].average()
    
    # 保存结果
    output_dir = Path(f'results/sub-{subject_id:02d}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evoked_target.save(output_dir / 'target-ave.fif')
    evoked_nontarget.save(output_dir / 'nontarget-ave.fif')
    
    print(f"✅ Subject {subject_id} done!")
    return evoked_target, evoked_nontarget

# ========== 并行处理所有被试 ==========
subject_ids = range(1, 31)  # 30 个被试

# 方法 1: 串行处理 (逐个)
# results = [process_subject(sid) for sid in subject_ids]

# 方法 2: 并行处理 (推荐,快很多!)
parallel, run_func, n_jobs = parallel_func(process_subject, n_jobs=8)
results = parallel(run_func(sid) for sid in subject_ids)

# ========== 计算组平均 (Grand Average) ==========
evoked_targets = [r[0] for r in results]
evoked_nontargets = [r[1] for r in results]

# 组平均
grand_avg_target = mne.grand_average(evoked_targets)
grand_avg_nontarget = mne.grand_average(evoked_nontargets)

# 可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

grand_avg_target.plot(axes=axes[0], show=False)
axes[0].set_title(f'Target (N={len(evoked_targets)})')

grand_avg_nontarget.plot(axes=axes[1], show=False)
axes[1].set_title(f'Non-Target (N={len(evoked_nontargets)})')

plt.tight_layout()
plt.savefig('results/grand_average.png', dpi=300)
plt.show()

# 保存组平均
grand_avg_target.save('results/grand_avg_target-ave.fif')
grand_avg_nontarget.save('results/grand_avg_nontarget-ave.fif')

print(f"✅ 所有 {len(subject_ids)} 个被试处理完成!")
```

---

## 优势与限制

### 优势 ✅

#### 1. **完全可重复**

```python
# 可以多次运行同样的分析,结果完全一致
raw = mne.io.read_raw_fif('data.fif')

# 第 1 次分析
raw1 = raw.copy()
raw1.filter(1, 40)
result1 = raw1.get_data()

# 第 2 次分析 (数天后)
raw2 = raw.copy()
raw2.filter(1, 40)
result2 = raw2.get_data()

# 结果完全相同
assert np.array_equal(result1, result2)  # ✅ True
```

#### 2. **无时间压力**

```python
# 可以慢慢优化参数
for freq in [30, 40, 50]:
    raw_copy = raw.copy()
    raw_copy.filter(1, freq)
    # 比较不同截止频率的效果
```

#### 3. **支持复杂算法**

```python
# ICA 需要多次迭代,访问全部数据
ica = ICA(n_components=20)
ica.fit(raw)  # 可能需要数分钟,但没关系

# 源定位需要协方差矩阵
noise_cov = mne.compute_covariance(epochs)
stc = apply_inverse(evoked, inverse_operator, 
                    lambda2=1/9, method='dSPM')
```

#### 4. **随机访问**

```python
# 可以先看后面,再看前面,再看中间
data_end = raw[:, -5000:]      # 最后 10 秒
data_start = raw[:, :5000]     # 前 10 秒
data_middle = raw[:, 50000:55000]  # 中间某段
```

#### 5. **批量并行**

```python
# 可以并行处理 30 个被试
from mne.parallel import parallel_func

parallel, run_func, _ = parallel_func(process_subject, n_jobs=8)
results = parallel(run_func(sid) for sid in range(1, 31))
```

---

### 限制 ⚠️

#### 1. **无实时反馈**

```python
# ❌ 无法在实验进行中提供反馈
# 必须等实验完成,数据保存后才能分析
```

#### 2. **无法用于 BCI**

```python
# ❌ 无法用于脑机接口控制
# BCI 需要 < 100ms 的响应时间
```

#### 3. **数据质量滞后发现**

```python
# ⚠️ 数据采集时的问题(电极脱落、阻抗高等)
# 只能在事后发现,可能导致数据不可用
```

#### 4. **内存占用大**

```python
# 大文件需要大量内存
raw = mne.io.read_raw_fif('10GB_file.fif', preload=True)
# 可能需要 10+ GB RAM
```

---

## 总结

### 离线处理适用场景

✅ **强烈推荐用于**:
- 科学研究发表论文
- ERP 分析 (P300, N400, etc.)
- 时频分析 (ERD/ERS, TFR)
- 源定位 (dSPM, LCMV, MNE)
- 复杂统计分析
- 批量处理多个被试

❌ **不适用于**:
- 脑机接口 (BCI) 控制
- 神经反馈训练
- 实时信号质量监控
- 需要立即反馈的应用

---

**相关文档**:
- [MNE 实时处理指南](mne-realtime-processing.md)
- [离线 vs 实时对比](mne-offline-vs-realtime.md)
- [EEG/EOG/EMG 核心依赖](eeg-eog-emg-core-dependencies.md)
