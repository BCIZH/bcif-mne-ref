# 模块 3: Event Extraction & Epoching - 事件提取与分段

> **在数据流中的位置**: 第三步 - 从连续数据提取事件相关片段  
> **核心职责**: 事件检测、时间标注、数据分段、质量控制  
> **模块路径**: `mne/event.py`, `mne/epochs.py`, `mne/annotations.py`

---

## 目录

1. [模块概述](#模块概述)
2. [事件检测算法](#事件检测算法)
3. [Epoching 算法](#epoching-算法)
4. [质量控制算法](#质量控制算法)
5. [应用场景](#应用场景)

---

## 模块概述

### 事件驱动的分析框架

```
Continuous Data (Raw)
       ↓
[Event Detection] find_events(), annotations_from_events()
       ↓
Events [n_events × 3]  (sample, prev_id, event_id)
       ↓
[Epoching] Epochs()
       ↓
Trials [n_epochs × n_channels × n_times]
       ↓
[Quality Control] drop_bad(), equalize_event_counts()
       ↓
Clean Epochs
```

### 模块结构

```
mne/
├── event.py              # 事件处理核心 (1694行)
│   ├── find_events()     # 从刺激通道检测事件
│   ├── make_fixed_length_events()
│   ├── define_target_events()
│   └── merge_events()
│
├── annotations.py        # 时间标注 (1200+ 行)
│   ├── Annotations类    # 灵活的时间段标记
│   ├── events_from_annotations()
│   └── annotations_from_events()
│
└── epochs.py             # Epoching 实现 (5004行)
    ├── BaseEpochs        # 抽象基类
    ├── Epochs            # 从 Raw 创建
    ├── EpochsArray       # 从数组创建
    └── EpochsFIF         # 从文件读取
```

---

## 事件检测算法

### 1. 刺激通道事件检测

**算法位置**: `mne/event.py:find_events()` (行 200-500)

**原理**: 检测数字刺激通道的电平跳变

```python
def find_events(raw, stim_channel='STI 014', output='onset',
                consecutive='increasing', min_duration=0):
    """
    从刺激通道检测事件
    
    参数:
        stim_channel: 刺激通道名（或自动检测）
        output: 'onset' | 'offset' | 'step'
            - onset: 仅检测上升沿
            - offset: 仅检测下降沿  
            - step: 检测所有变化
        consecutive: 'increasing' | bool
            - True: 连续相同值合并为一个事件
            - False: 每次跳变都是事件
            - 'increasing': 仅当值增加时
        min_duration: 最小事件持续时间（样本数）
    
    算法步骤:
        1. 读取刺激通道数据
        2. 检测值变化点
        3. 应用最小持续时间过滤
        4. 格式化为事件矩阵
    """
    # 1. 获取刺激通道
    if stim_channel is None:
        stim_channel = _get_stim_channel(raw.info)
    
    stim_data = raw.get_data(picks=[stim_channel])[0]
    
    # 2. 转换为整数（刺激码）
    stim_data = stim_data.astype(np.int32)
    
    # 3. 检测变化点
    # 计算差分
    diff = np.diff(stim_data, prepend=stim_data[0])
    
    if output == 'onset':
        # 仅上升沿（0 → 非0）
        event_samples = np.where(
            (diff > 0) & (stim_data[:-1] == 0)
        )[0] + 1
    elif output == 'offset':
        # 仅下降沿（非0 → 0）
        event_samples = np.where(
            (diff < 0) & (stim_data[1:] == 0)
        )[0] + 1
    else:  # step
        # 任何变化
        event_samples = np.where(diff != 0)[0] + 1
    
    # 4. 处理连续事件
    if consecutive == 'increasing':
        # 移除连续递减的事件
        to_keep = []
        for i, sample in enumerate(event_samples):
            if i == 0:
                to_keep.append(True)
            else:
                prev_val = stim_data[event_samples[i-1]]
                curr_val = stim_data[sample]
                to_keep.append(curr_val > prev_val)
        event_samples = event_samples[to_keep]
    
    # 5. 应用最小持续时间
    if min_duration > 0:
        durations = np.diff(event_samples, append=len(stim_data))
        event_samples = event_samples[durations >= min_duration]
    
    # 6. 构建事件矩阵 [n_events × 3]
    events = np.zeros((len(event_samples), 3), dtype=np.int32)
    events[:, 0] = event_samples  # 样本位置
    events[:, 1] = 0              # 前一个事件ID（兼容性）
    events[:, 2] = stim_data[event_samples]  # 事件ID
    
    return events
```

**刺激通道编码方案**:

```python
# 示例：8位刺激编码
刺激码值 = sum(2^i for i in triggered_bits)

# 例如：
# Bit 0 (值1) + Bit 2 (值4) → 事件码 5
# Bit 1 (值2) + Bit 3 (值8) → 事件码 10

# 复杂设计：
trigger_channels = {
    'block': [0, 1],      # Bits 0-1: 4种 block 类型
    'condition': [2, 3, 4], # Bits 2-4: 8种条件
    'response': [5],      # Bit 5: 有无反应
}

# 解码
def decode_event(event_code):
    block = event_code & 0b11           # 提取 bit 0-1
    condition = (event_code >> 2) & 0b111  # 提取 bit 2-4
    response = (event_code >> 5) & 0b1     # 提取 bit 5
    return block, condition, response
```

**计算复杂度**: O(N)，N = 数据长度

---

### 2. 固定时长事件生成

**算法位置**: `mne/event.py:make_fixed_length_events()` (行 600-750)

**用途**: 静息态分析、长时程数据分段

```python
def make_fixed_length_events(raw, duration=1.0, start=0, stop=None,
                              overlap=0.0, id=1):
    """
    生成固定时长的伪事件
    
    应用:
        - 静息态 EEG/MEG 分析
        - 连续睡眠数据分段
        - 微状态分析
    """
    sfreq = raw.info['sfreq']
    start_sample = int(start * sfreq)
    
    if stop is None:
        stop_sample = raw.last_samp
    else:
        stop_sample = int(stop * sfreq)
    
    # 计算步长（考虑重叠）
    duration_samples = int(duration * sfreq)
    step_samples = int(duration_samples * (1 - overlap))
    
    # 生成事件样本位置
    event_samples = np.arange(
        start_sample,
        stop_sample - duration_samples + 1,
        step_samples
    )
    
    # 构建事件矩阵
    events = np.zeros((len(event_samples), 3), dtype=int)
    events[:, 0] = event_samples
    events[:, 2] = id
    
    return events

# 使用示例
# 生成不重叠的 2 秒段
events_rest = make_fixed_length_events(raw, duration=2.0)

# 生成 50% 重叠的 1 秒段
events_overlap = make_fixed_length_events(raw, duration=1.0, overlap=0.5)
```

---

### 3. Annotations 系统

**算法位置**: `mne/annotations.py:Annotations` (行 50-600)

**设计**: 灵活的时间段标注（取代简单事件）

```python
class Annotations:
    """
    时间标注类
    
    优势:
        - 支持时间段（不仅是时间点）
        - 字符串描述（不限于整数ID）
        - 与 Raw 对象深度集成
        - 自动处理时间同步
    """
    
    def __init__(self, onset, duration, description, orig_time=None):
        """
        参数:
            onset: 开始时间 (秒或样本)
            duration: 持续时间 (秒)
            description: 标注描述 (字符串)
            orig_time: 原始时间戳
        """
        self.onset = np.array(onset, dtype=float)
        self.duration = np.array(duration, dtype=float)
        self.description = np.array(description, dtype='<U')
        self.orig_time = orig_time
    
    def __add__(self, other):
        """合并标注"""
        return Annotations(
            onset=np.concatenate([self.onset, other.onset]),
            duration=np.concatenate([self.duration, other.duration]),
            description=np.concatenate([self.description, other.description])
        )
    
    def crop(self, tmin, tmax):
        """裁剪到时间范围"""
        mask = (self.onset + self.duration >= tmin) & (self.onset <= tmax)
        return Annotations(
            self.onset[mask],
            self.duration[mask],
            self.description[mask]
        )

# 创建标注
annot = Annotations(
    onset=[1.0, 3.5, 10.2],
    duration=[0.5, 1.0, 2.0],
    description=['bad_blink', 'bad_movement', 'bad_noise']
)

# 添加到 Raw
raw.set_annotations(annot)

# 可视化会自动显示标注
raw.plot()
```

**Annotations → Events 转换**:

```python
def events_from_annotations(raw, event_id=None, regexp=None):
    """
    从 Annotations 提取事件
    
    算法:
        1. 筛选匹配的 annotations
        2. 转换 onset 为样本位置
        3. 映射描述到事件ID
    """
    if event_id is None:
        # 自动生成 ID
        unique_desc = np.unique(raw.annotations.description)
        event_id = {desc: i+1 for i, desc in enumerate(unique_desc)}
    
    # 正则表达式过滤
    if regexp:
        mask = [re.match(regexp, desc) for desc in raw.annotations.description]
        mask = np.array([m is not None for m in mask])
    else:
        mask = np.ones(len(raw.annotations), dtype=bool)
    
    # 转换为事件
    events = []
    for onset, desc in zip(raw.annotations.onset[mask], 
                           raw.annotations.description[mask]):
        if desc in event_id:
            sample = raw.time_as_index(onset)[0]
            events.append([sample, 0, event_id[desc]])
    
    return np.array(events, dtype=int), event_id
```

---

### 4. 事件组合与操作

**算法位置**: `mne/event.py:merge_events()`, `define_target_events()`

```python
def merge_events(events, ids, new_id):
    """
    合并多个事件类型
    
    示例: 合并左右手运动为 "motor"
    """
    events_out = events.copy()
    mask = np.isin(events[:, 2], ids)
    events_out[mask, 2] = new_id
    return events_out

# 使用
events_merged = merge_events(
    events, 
    ids=[1, 2],      # 左手、右手
    new_id=10        # 合并为 "motor"
)

def define_target_events(events, reference_id, target_id, 
                         sfreq, tmin, tmax, new_id):
    """
    定义基于时间关系的新事件
    
    示例: "刺激后 200-500ms 内有按键反应"
    """
    new_events = []
    
    ref_events = events[events[:, 2] == reference_id]
    target_events = events[events[:, 2] == target_id]
    
    for ref_sample in ref_events[:, 0]:
        # 计算搜索窗口
        search_start = ref_sample + int(tmin * sfreq)
        search_end = ref_sample + int(tmax * sfreq)
        
        # 查找目标事件
        in_window = (target_events[:, 0] >= search_start) & \
                    (target_events[:, 0] <= search_end)
        
        if in_window.any():
            # 找到目标，创建新事件
            new_events.append([ref_sample, 0, new_id])
    
    return np.array(new_events, dtype=int)

# 使用
# 定义 "正确反应" 事件
correct_events = define_target_events(
    events,
    reference_id=1,    # 刺激
    target_id=2,       # 按键
    sfreq=raw.info['sfreq'],
    tmin=0.2, tmax=0.8,  # 200-800ms 窗口
    new_id=100         # 新事件ID
)
```

---

## Epoching 算法

### 1. Epochs 创建

**算法位置**: `mne/epochs.py:Epochs.__init__()` (行 3420-3700)

**核心算法**:

```python
class Epochs(BaseEpochs):
    """
    从 Raw 对象创建 Epochs
    
    关键参数:
        events: 事件矩阵 [n_events × 3]
        event_id: 事件ID映射 {'condition': event_code}
        tmin, tmax: Epoch 时间窗口
        baseline: 基线校正窗口 (tmin, tmax) 或 None
        picks: 选择的通道
        preload: 是否预加载数据
        reject: 自动拒绝阈值 {'eeg': 100e-6, 'mag': 4e-12}
        flat: 平坦信号阈值
        reject_tmin, reject_tmax: 拒绝评估窗口
    """
    
    def __init__(self, raw, events, event_id, tmin, tmax, 
                 baseline=(None, 0), picks=None, preload=False,
                 reject=None, flat=None, ...):
        
        # 1. 参数验证
        self._validate_params(events, tmin, tmax, ...)
        
        # 2. 选择事件
        selection = self._select_events(events, event_id)
        
        # 3. 构建时间向量
        n_times = int((tmax - tmin) * raw.info['sfreq']) + 1
        self.times = np.linspace(tmin, tmax, n_times)
        
        # 4. 提取数据
        if preload:
            self._data = self._get_data_from_raw(raw, events[selection])
        else:
            # 延迟加载
            self._raw = raw
            self._events = events[selection]
        
        # 5. 基线校正
        if baseline is not None:
            self.apply_baseline(baseline)
        
        # 6. 自动拒绝
        if reject is not None or flat is not None:
            self.drop_bad(reject=reject, flat=flat)
    
    def _get_data_from_raw(self, raw, events):
        """
        从 Raw 提取 epochs 数据
        
        算法:
            1. 为每个事件计算起止样本
            2. 从 Raw 读取数据段
            3. 组装为 3D 数组
        """
        n_events = len(events)
        n_channels = len(self.ch_names)
        n_times = len(self.times)
        
        data = np.zeros((n_events, n_channels, n_times))
        
        for idx, event_sample in enumerate(events[:, 0]):
            # 计算起止
            start = event_sample + int(self.tmin * raw.info['sfreq'])
            stop = event_sample + int(self.tmax * raw.info['sfreq']) + 1
            
            # 读取数据
            segment, times = raw[:, start:stop]
            
            # 存储
            data[idx] = segment
        
        return data
```

**内存管理**:
```python
# 预加载（快速访问，占内存）
epochs = Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5, preload=True)

# 延迟加载（节省内存）
epochs = Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5, preload=False)

# 内存映射（大数据集）
epochs.save('epochs.fif', split_size='2GB')
epochs = mne.read_epochs('epochs.fif', preload='memmap.dat')
```

---

### 2. 基线校正

**算法位置**: `mne/baseline.py:rescale()` (行 100-300)

**方法**:

```python
def apply_baseline(epochs, baseline=(None, 0), mode='mean'):
    """
    基线校正
    
    模式:
        - 'mean': 减去基线均值（默认）
        - 'ratio': 除以基线均值
        - 'logratio': log(data / baseline)
        - 'percent': (data - baseline) / baseline * 100
        - 'zscore': (data - μ) / σ
        - 'zlogratio': log((data - μ) / σ)
    """
    # 确定基线时间窗口
    if baseline is None:
        return epochs  # 不做校正
    
    bmin, bmax = baseline
    if bmin is None:
        bmin = epochs.tmin
    if bmax is None:
        bmax = epochs.tmax
    
    # 找到基线样本索引
    baseline_mask = (epochs.times >= bmin) & (epochs.times <= bmax)
    
    # 计算基线统计量
    data = epochs._data
    
    if mode == 'mean':
        baseline_mean = data[:, :, baseline_mask].mean(axis=2, keepdims=True)
        data -= baseline_mean
    
    elif mode == 'ratio':
        baseline_mean = data[:, :, baseline_mask].mean(axis=2, keepdims=True)
        data /= baseline_mean
    
    elif mode == 'zscore':
        baseline_mean = data[:, :, baseline_mask].mean(axis=2, keepdims=True)
        baseline_std = data[:, :, baseline_mask].std(axis=2, keepdims=True)
        data = (data - baseline_mean) / baseline_std
    
    elif mode == 'logratio':
        baseline_mean = data[:, :, baseline_mask].mean(axis=2, keepdims=True)
        data = np.log(data / baseline_mean)
    
    epochs._data = data
    return epochs
```

**选择指南**:
- `mean`: ERP分析（默认）
- `ratio` / `logratio`: 功率分析（时频）
- `zscore`: 去除幅度差异
- `percent`: 易解释的变化率

---

## 质量控制算法

### 1. 自动拒绝 (Artifact Rejection)

**算法位置**: `mne/epochs.py:drop_bad()` (行 2100-2400)

**峰-峰值阈值法**:

```python
def drop_bad(epochs, reject=None, flat=None):
    """
    基于阈值的 epoch 自动拒绝
    
    reject: 最大峰-峰值幅度
        {'eeg': 100e-6,  # 100 μV
         'mag': 4e-12,   # 4000 fT
         'grad': 4000e-13}
    
    flat: 最小峰-峰值（检测坏通道/伪迹）
        {'eeg': 1e-6}    # 1 μV
    
    算法:
        1. 计算每个 epoch 的峰-峰值
        2. 与阈值比较
        3. 标记超过阈值的 epochs
    """
    n_epochs = len(epochs)
    bad_epochs = np.zeros(n_epochs, dtype=bool)
    
    # 分通道类型检查
    for ch_type in ['eeg', 'mag', 'grad']:
        if ch_type not in reject:
            continue
        
        picks = pick_types(epochs.info, meg=(ch_type in ['mag','grad']),
                          eeg=(ch_type == 'eeg'))
        
        if len(picks) == 0:
            continue
        
        # 计算峰-峰值
        data = epochs.get_data(picks=picks)
        ptp = data.max(axis=2) - data.min(axis=2)  # [n_epochs × n_channels]
        
        # 拒绝检查
        if reject and ch_type in reject:
            bad = (ptp > reject[ch_type]).any(axis=1)
            bad_epochs |= bad
        
        # 平坦检查
        if flat and ch_type in flat:
            bad = (ptp < flat[ch_type]).any(axis=1)
            bad_epochs |= bad
    
    # 记录拒绝原因
    epochs.drop_log = []
    for idx, is_bad in enumerate(bad_epochs):
        if is_bad:
            epochs.drop_log.append(['THRESHOLD'])
        else:
            epochs.drop_log.append([])
    
    # 丢弃坏 epochs
    epochs._data = epochs._data[~bad_epochs]
    epochs.events = epochs.events[~bad_epochs]
    epochs.selection = epochs.selection[~bad_epochs]
    
    print(f"丢弃 {bad_epochs.sum()} / {n_epochs} epochs "
          f"({100*bad_epochs.sum()/n_epochs:.1f}%)")
    
    return epochs
```

**自适应阈值**:

```python
def autoreject_epochs(epochs, n_interpolate=[1, 4, 8], 
                      consensus=[0.1, 0.3, 0.5]):
    """
    基于 autoreject 算法的自适应拒绝
    
    原理:
        1. 交叉验证确定最优阈值
        2. 尝试插值修复坏通道
        3. 仅丢弃无法修复的 epochs
    
    参考: Jas et al. (2017) NeuroImage
    """
    from autoreject import AutoReject
    
    ar = AutoReject(
        n_interpolate=n_interpolate,
        consensus=consensus,
        thresh_method='bayesian_optimization',
        cv=10
    )
    
    epochs_clean = ar.fit_transform(epochs)
    
    # 查看拒绝日志
    print(f"修复: {ar.bad_segments_.sum()} 段")
    print(f"丢弃: {(ar.reject_log_.bad_epochs).sum()} epochs")
    
    return epochs_clean
```

---

### 2. 条件均衡

**算法位置**: `mne/epochs.py:equalize_event_counts()` (行 1900-2050)

**用途**: 平衡不同条件的 trial 数（避免统计偏差）

```python
def equalize_event_counts(epochs_list, method='mintime'):
    """
    均衡多个条件的 epoch 数量
    
    方法:
        - 'mintime': 保留最早的 epochs
        - 'truncate': 简单截断
    
    算法:
        1. 找到最少的 epoch 数
        2. 从其他条件随机/按时间选择相同数量
    """
    # 找最小数量
    min_count = min(len(epochs) for epochs in epochs_list)
    
    print(f"均衡到每个条件 {min_count} epochs")
    
    # 均衡每个条件
    balanced = []
    for epochs in epochs_list:
        if len(epochs) == min_count:
            balanced.append(epochs)
        else:
            if method == 'mintime':
                # 保留最早的
                indices = np.arange(min_count)
            else:  # random
                # 随机选择
                indices = np.random.choice(
                    len(epochs), min_count, replace=False
                )
                indices.sort()
            
            balanced.append(epochs[indices])
    
    return balanced

# 使用
epochs_aud = epochs['auditory']
epochs_vis = epochs['visual']

epochs_aud, epochs_vis = equalize_event_counts([epochs_aud, epochs_vis])
```

---

### 3. 元数据增强

**算法位置**: `mne/epochs.py:metadata` 属性

**功能**: 为每个 trial 附加实验信息

```python
import pandas as pd

# 创建元数据
metadata = pd.DataFrame({
    'subject_id': ['sub-01'] * len(events),
    'block': [1, 1, 1, 2, 2, 2, ...],
    'condition': ['A', 'B', 'A', 'B', ...],
    'rt': [0.523, 0.612, 0.445, ...],  # 反应时
    'accuracy': [1, 1, 0, 1, ...],      # 正确性
    'difficulty': ['easy', 'hard', ...],
})

# 附加到 epochs
epochs.metadata = metadata

# 基于元数据选择
fast_trials = epochs['rt < 0.5']
correct_hard = epochs['(accuracy == 1) & (difficulty == "hard")']

# 复杂查询
from mne.utils import query_dataframe
difficult_errors = query_dataframe(
    epochs,
    "(difficulty == 'hard') & (accuracy == 0)"
)

# 与统计分析集成
from mne.stats import linear_regression

# 回归分析：RT 对 ERP 的影响
design_matrix = metadata[['rt', 'difficulty']]
lm = linear_regression(epochs, design_matrix, names=['rt', 'difficulty'])
```

---

## 应用场景

### 场景 1: 标准 ERP 实验

```python
import mne

# 1. 加载预处理后的数据
raw = mne.io.read_raw_fif('cleaned_raw.fif', preload=True)

# 2. 检测事件
events = mne.find_events(raw, stim_channel='STI 014', min_duration=0.002)

# 查看事件
print(f"找到 {len(events)} 个事件")
print(f"事件类型: {np.unique(events[:, 2])}")

# 3. 定义事件映射
event_id = {
    'auditory/left': 1,
    'auditory/right': 2,
    'visual/left': 3,
    'visual/right': 4
}

# 4. 创建 epochs
epochs = mne.Epochs(
    raw, events, event_id,
    tmin=-0.2, tmax=0.5,        # -200 到 500 ms
    baseline=(None, 0),          # -200 到 0 ms 基线
    picks='eeg',
    preload=True,
    reject=dict(eeg=100e-6),     # 100 μV
    flat=dict(eeg=1e-6),         # 1 μV
    detrend=1,                   # 去线性趋势
    verbose=True
)

# 5. 查看拒绝日志
print(f"保留: {len(epochs)} / {len(events)} epochs")
epochs.plot_drop_log()

# 6. 均衡条件
epochs.equalize_event_counts(['auditory', 'visual'])

# 7. 保存
epochs.save('epochs-epo.fif', overwrite=True)
```

---

### 场景 2: 行为数据集成

```python
import pandas as pd
import mne

# 1. 加载行为数据
behavior = pd.read_csv('behavior_log.csv')
# 列: trial_num, onset_time, condition, response, rt, accuracy

# 2. 从行为数据创建事件
events = []
for _, row in behavior.iterrows():
    sample = raw.time_as_index(row['onset_time'])[0]
    event_code = {'A': 1, 'B': 2}[row['condition']]
    events.append([sample, 0, event_code])
events = np.array(events, dtype=int)

# 3. 创建元数据（与行为数据对齐）
metadata = behavior.copy()

# 4. 创建 epochs
epochs = mne.Epochs(
    raw, events,
    event_id={'A': 1, 'B': 2},
    tmin=-0.2, tmax=0.8,
    metadata=metadata,
    preload=True
)

# 5. 基于行为的条件选择
# 仅分析正确且快速的 trials
fast_correct = epochs['(accuracy == 1) & (rt < 0.5)']

# 按 RT 中位数分组
rt_median = metadata['rt'].median()
fast_epochs = epochs[f'rt < {rt_median}']
slow_epochs = epochs[f'rt >= {rt_median}']

# 6. 分析 RT 与 ERP 关系
evoked_fast = fast_epochs.average()
evoked_slow = slow_epochs.average()

# 对比
mne.viz.plot_compare_evokeds(
    {'Fast': evoked_fast, 'Slow': evoked_slow},
    picks='Cz'
)
```

---

### 场景 3: 复杂事件定义

```python
import mne

raw = mne.io.read_raw_fif('data.fif')
events = mne.find_events(raw)

# 场景：检测 "Go" 试验中的快速正确反应

# 1. 分离刺激和反应事件
stim_events = events[np.isin(events[:, 2], [1, 2])]  # 刺激
resp_events = events[events[:, 2] == 10]              # 反应

# 2. 定义 "Go + 快速正确反应" 事件
fast_correct = mne.event.define_target_events(
    events,
    reference_id=1,    # Go 刺激
    target_id=10,      # 反应
    sfreq=raw.info['sfreq'],
    tmin=0.2,          # 最早 200 ms
    tmax=0.6,          # 最晚 600 ms
    new_id=100         # 新事件ID
)

# 3. 合并事件
all_events = np.vstack([events, fast_correct])
all_events = all_events[all_events[:, 0].argsort()]  # 按时间排序

# 4. 创建 epochs（仅针对成功 trials）
epochs_success = mne.Epochs(
    raw, fast_correct,
    event_id={'go_fast_correct': 100},
    tmin=-0.5, tmax=1.0,
    preload=True
)
```

---

### 场景 4: 静息态分析

```python
import mne

raw = mne.io.read_raw_fif('resting_state.fif', preload=True)

# 1. 生成固定长度事件（2秒段，50%重叠）
events = mne.make_fixed_length_events(
    raw, 
    duration=2.0,
    overlap=0.5,
    id=1
)

print(f"生成 {len(events)} 个段")

# 2. 创建 epochs
epochs = mne.Epochs(
    raw, events,
    event_id={'rest': 1},
    tmin=0, tmax=2.0,
    baseline=None,      # 静息态不需要基线
    preload=True
)

# 3. 拒绝有伪迹的段
epochs.drop_bad(reject=dict(eeg=100e-6))

# 4. 计算功率谱（每段）
psds, freqs = epochs.compute_psd(
    fmin=1, fmax=40,
    method='welch',
    n_fft=2048
).get_data(return_freqs=True)

# 5. 平均功率谱
psd_mean = psds.mean(axis=0)

# 6. 分析频段功率
from mne.time_frequency import psd_array_welch

band_power = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
}

for band, (fmin, fmax) in band_power.items():
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    power = psd_mean[:, freq_mask].mean(axis=1)
    print(f"{band}: {power.mean():.2e}")
```

---

## 总结

### 核心算法汇总

| 算法 | 位置 | 复杂度 | 作用 | 场景 |
|------|------|--------|------|------|
| **事件检测** |
| `find_events()` | `event.py:200-500` | O(N) | 刺激通道跳变检测 | 任务态实验 |
| `make_fixed_length_events()` | `event.py:600-750` | O(N/L) | 固定段生成 | 静息态 |
| `events_from_annotations()` | `annotations.py` | O(M) | 标注转事件 | 灵活标记 |
| **Epoching** |
| `Epochs.__init__()` | `epochs.py:3420+` | O(E×T) | 数据分段 | 核心功能 |
| `apply_baseline()` | `baseline.py` | O(E×C×T) | 基线校正 | ERP/时频 |
| **质量控制** |
| `drop_bad()` | `epochs.py:2100+` | O(E×C×T) | 自动拒绝 | 清理数据 |
| `equalize_event_counts()` | `epochs.py:1900+` | O(E) | 条件均衡 | 统计分析 |

*注: N=数据长度, E=epochs数, C=通道数, T=时间点, M=标注数, L=段长*

### 设计亮点

1. **灵活的事件系统**: 支持简单事件、时间段、复杂查询
2. **延迟加载**: 处理大量 epochs 而不耗尽内存
3. **元数据集成**: 与行为数据无缝结合
4. **自动化 QC**: 多种拒绝策略

### 下一步

Epochs 创建后，通常进行：
1. **平均化** → **模块4: Evoked**
2. **时频分析** (跳过平均)
3. **解码分析** (单 trial 分析)
