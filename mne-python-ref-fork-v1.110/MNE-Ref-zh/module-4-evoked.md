# 模块 4: Evoked - 诱发响应平均化

> **在数据流中的位置**: 第四步 - 跨试验平均提取诱发响应  
> **核心职责**: Epochs 平均、ERP/ERF 提取、对比分析  
> **模块路径**: `mne/evoked.py`

---

## 目录

1. [模块概述](#模块概述)
2. [平均化算法](#平均化算法)
3. [对比分析算法](#对比分析算法)
4. [峰值检测算法](#峰值检测算法)
5. [应用场景](#应用场景)

---

## 模块概述

### Evoked 数据模型

```
Epochs [n_trials × n_channels × n_times]
         ↓
    [average()]
         ↓
Evoked [n_channels × n_times]
    ├── ERP (Event-Related Potential)   - EEG
    ├── ERF (Event-Related Field)       - MEG
    └── Evoked Response                 - 通用
```

### 核心类结构

**文件位置**: `mne/evoked.py` (2169 行)

```python
class Evoked:
    """
    诱发响应类
    
    属性:
        data: ndarray [n_channels × n_times]
        info: Info 对象
        times: ndarray [n_times]
        nave: int (平均的 trial 数)
        kind: str ('average' | 'standard_error')
        comment: str (条件描述)
        first: int (第一个样本索引)
        last: int (最后一个样本索引)
    
    方法:
        plot(): 波形图
        plot_topomap(): 地形图
        plot_joint(): 联合视图
        get_peak(): 峰值检测
        crop(), resample(), filter(): 数据操作
        apply_baseline(): 基线校正
    """
```

**继承关系**:
```python
Evoked(
    ProjMixin,           # SSP 投影
    ContainsMixin,       # 通道类型
    UpdateChannelsMixin, # 更新通道
    ReferenceMixin,      # 重参考
    SetChannelsMixin,    # 设置通道
    InterpolationMixin,  # 插值
    FilterMixin,         # 滤波
    ExtendedTimeMixin,   # 时间操作
    SizeMixin,           # 大小计算
    SpectrumMixin,       # 频谱分析
)
```

---

## 平均化算法

### 1. 简单平均 (均值)

**算法位置**: `mne/epochs.py:BaseEpochs.average()` (行 1600-1750)

**实现**:
```python
def average(epochs, picks=None, method='mean'):
    """
    计算诱发响应
    
    参数:
        picks: 选择通道
        method: 'mean' | 'median'
    
    算法:
        1. 获取 epochs 数据
        2. 沿 trial 维度计算统计量
        3. 创建 Evoked 对象
    """
    # 1. 获取数据
    if picks is None:
        data = epochs.get_data()  # [n_epochs × n_channels × n_times]
    else:
        data = epochs.get_data(picks=picks)
    
    # 2. 计算平均
    if method == 'mean':
        evoked_data = data.mean(axis=0)  # [n_channels × n_times]
        kind = 'average'
    elif method == 'median':
        evoked_data = np.median(data, axis=0)
        kind = 'average'  # 仍标记为 average
    
    # 3. 创建 Evoked 对象
    evoked = EvokedArray(
        data=evoked_data,
        info=epochs.info.copy(),
        tmin=epochs.tmin,
        comment=epochs.comment,
        nave=len(epochs),  # 记录平均了多少 trial
        kind=kind
    )
    
    return evoked

# 使用
evoked = epochs.average()
print(f"平均了 {evoked.nave} 个 trials")
```

**信噪比提升**:
$$
\text{SNR}_{\text{evoked}} = \sqrt{N} \times \text{SNR}_{\text{single}}
$$

其中 $N$ 是平均的 trial 数

---

### 2. 标准误差

**算法位置**: `mne/evoked.py:_make_evoked_from_data()`

```python
def compute_standard_error(epochs):
    """
    计算标准误差
    
    公式:
        SEM = σ / √N
    
    其中 σ 是跨 trial 标准差，N 是 trial 数
    """
    data = epochs.get_data()
    
    # 标准误差
    sem_data = data.std(axis=0, ddof=1) / np.sqrt(len(data))
    
    evoked_sem = EvokedArray(
        data=sem_data,
        info=epochs.info.copy(),
        tmin=epochs.tmin,
        comment=f'{epochs.comment} - SEM',
        nave=len(epochs),
        kind='standard_error'  # 标记为标准误
    )
    
    return evoked_sem

# 可视化带误差带的 ERP
evoked_mean = epochs.average()
evoked_sem = compute_standard_error(epochs)

fig, ax = plt.subplots()
times = evoked_mean.times
data_mean = evoked_mean.data[0]  # 第一个通道
data_sem = evoked_sem.data[0]

ax.plot(times, data_mean, label='Mean')
ax.fill_between(
    times,
    data_mean - data_sem,
    data_mean + data_sem,
    alpha=0.3,
    label='±SEM'
)
ax.legend()
```

---

### 3. 条件对比 (差异波)

**算法位置**: `mne/evoked.py:combine_evoked()` (行 1900-2100)

**差异波计算**:
```python
def compute_difference_wave(evoked_a, evoked_b, weights='equal'):
    """
    计算差异波
    
    应用:
        - Oddball: target - standard
        - N400: incongruent - congruent
        - MMN: deviant - standard
    
    算法:
        diff = (weights_a × evoked_a - weights_b × evoked_b) / sum(weights)
    """
    if weights == 'equal':
        weights_a = 1.0
        weights_b = 1.0
    elif weights == 'nave':
        # 根据 trial 数加权
        weights_a = evoked_a.nave
        weights_b = evoked_b.nave
    
    # 计算差异
    diff_data = evoked_a.data - evoked_b.data
    
    # 新的 nave（有效 trial 数）
    if weights == 'nave':
        nave_diff = (evoked_a.nave * evoked_b.nave) / \
                    (evoked_a.nave + evoked_b.nave)
    else:
        nave_diff = min(evoked_a.nave, evoked_b.nave)
    
    # 创建差异波对象
    evoked_diff = EvokedArray(
        data=diff_data,
        info=evoked_a.info.copy(),
        tmin=evoked_a.tmin,
        comment=f'{evoked_a.comment} - {evoked_b.comment}',
        nave=int(nave_diff)
    )
    
    return evoked_diff

# 使用
evoked_target = epochs['target'].average()
evoked_standard = epochs['standard'].average()

# 计算 P300 差异波
evoked_p300 = compute_difference_wave(evoked_target, evoked_standard)

evoked_p300.plot_joint(
    times=[0.3, 0.4, 0.5],  # P300 潜伏期
    title='P300 Difference Wave'
)
```

**MNE 内置方法**:
```python
# 使用 combine_evoked
from mne import combine_evoked

evoked_diff = combine_evoked(
    [evoked_target, evoked_standard],
    weights=[1, -1]  # target - standard
)

# 多条件组合
# 例如: (A + B) / 2 - (C + D) / 2
evoked_combined = combine_evoked(
    [evoked_a, evoked_b, evoked_c, evoked_d],
    weights=[0.5, 0.5, -0.5, -0.5]
)
```

---

### 4. Grand Average (组平均)

**算法位置**: `mne/evoked.py:grand_average()` (行 2000-2100)

**跨被试平均**:
```python
def grand_average(all_evokeds, interpolate_bads=True, drop_bads=True):
    """
    组平均 (跨被试)
    
    参数:
        all_evokeds: list of Evoked (每个被试一个)
        interpolate_bads: 是否插值坏通道
        drop_bads: 是否丢弃在任何被试中都是坏的通道
    
    算法:
        1. 对齐所有被试的通道
        2. 插值坏通道（可选）
        3. 计算平均
    """
    # 1. 确保通道一致
    from mne.channels import equalize_channels
    
    if drop_bads:
        all_evokeds = [evoked.copy() for evoked in all_evokeds]
        equalize_channels(all_evokeds)
    
    # 2. 插值坏通道
    if interpolate_bads:
        for evoked in all_evokeds:
            if evoked.info['bads']:
                evoked.interpolate_bads()
    
    # 3. 堆叠数据
    data_stack = np.array([evoked.data for evoked in all_evokeds])
    
    # 4. 计算组平均
    grand_avg_data = data_stack.mean(axis=0)
    
    # 5. 计算有效 trial 数（跨被试总和）
    total_nave = sum(evoked.nave for evoked in all_evokeds)
    
    # 6. 创建 Evoked 对象
    grand_avg = EvokedArray(
        data=grand_avg_data,
        info=all_evokeds[0].info.copy(),
        tmin=all_evokeds[0].tmin,
        comment='Grand Average',
        nave=total_nave
    )
    
    return grand_avg

# 使用
evokeds_all_subjects = []
for subject_id in ['01', '02', '03', ..., '20']:
    evoked = mne.read_evokeds(f'sub-{subject_id}-ave.fif')[0]
    evokeds_all_subjects.append(evoked)

# 组平均
grand_avg = mne.grand_average(evokeds_all_subjects)

# 可视化
grand_avg.plot_joint(
    times=[0.1, 0.2, 0.3],
    title=f'Grand Average (N={len(evokeds_all_subjects)})'
)
```

---

## 对比分析算法

### 1. 条件对比可视化

**算法位置**: `mne/viz/evoked.py:plot_compare_evokeds()`

```python
from mne.viz import plot_compare_evokeds

def compare_conditions(epochs, conditions, picks='eeg'):
    """
    对比多个条件
    
    参数:
        conditions: dict {'label': 'query_string'}
    """
    # 1. 计算每个条件的 evoked
    evokeds = {}
    for label, query in conditions.items():
        evokeds[label] = epochs[query].average()
    
    # 2. 可视化对比
    plot_compare_evokeds(
        evokeds,
        picks=picks,
        combine='mean',  # 或 'gfp' (Global Field Power)
        ci=0.95,         # 置信区间
        show_sensors='upper right',
        legend='upper left',
        title='Condition Comparison'
    )
    
    return evokeds

# 使用
conditions = {
    'Congruent': 'congruency == "congruent"',
    'Incongruent': 'congruency == "incongruent"'
}

evokeds = compare_conditions(epochs, conditions)
```

**全局场功率 (GFP)**:
```python
def compute_gfp(evoked):
    """
    计算全局场功率
    
    公式:
        GFP(t) = √(Σ_i [V_i(t) - V_mean(t)]² / N)
    
    用途:
        - 汇总多通道活动
        - 独立于参考电极
        - 峰值检测
    """
    data = evoked.data
    
    # 减去均值（去参考）
    data_centered = data - data.mean(axis=0, keepdims=True)
    
    # 计算 RMS
    gfp = np.sqrt((data_centered ** 2).mean(axis=0))
    
    return gfp

# 绘制 GFP
import matplotlib.pyplot as plt

gfp_cong = compute_gfp(evoked_congruent)
gfp_incong = compute_gfp(evoked_incongruent)

fig, ax = plt.subplots()
ax.plot(evoked_congruent.times, gfp_cong, label='Congruent')
ax.plot(evoked_incongruent.times, gfp_incong, label='Incongruent')
ax.set_xlabel('Time (s)')
ax.set_ylabel('GFP (V)')
ax.legend()
ax.axhline(0, color='k', linestyle='--')
ax.axvline(0, color='k', linestyle='--')
```

---

### 2. 地形图序列

**算法位置**: `mne/viz/topomap.py:plot_evoked_topomap()`

```python
def plot_topomaps_sequence(evoked, times, **kwargs):
    """
    绘制地形图时间序列
    
    应用:
        - 展示时空演化
        - 识别成分分布
    """
    evoked.plot_topomap(
        times=times,
        ch_type='eeg',
        average=0.05,      # 时间窗口平均（±50ms）
        colorbar=True,
        size=3,
        title='ERP Topography'
    )

# N400 示例
times_n400 = np.arange(0.3, 0.5, 0.05)  # 300-500ms，每50ms
plot_topomaps_sequence(evoked_n400, times_n400)

# 动画
from mne.viz import plot_evoked_topomap

fig = evoked.plot_topomap(
    times='auto',  # 自动选择峰值时间
    ch_type='eeg',
    time_unit='ms',
    colorbar=True
)
```

---

### 3. 联合视图

**算法位置**: `mne/viz/evoked.py:plot_evoked_joint()`

```python
def plot_joint_view(evoked, times_of_interest):
    """
    联合视图: 波形 + 地形图
    
    优势:
        - 同时查看时间演化和空间分布
        - 突出关键时间点
    """
    evoked.plot_joint(
        times=times_of_interest,
        title=evoked.comment,
        ts_args=dict(gfp=True, spatial_colors=True),
        topomap_args=dict(contours=6, sensors=True)
    )

# P300 示例
evoked_p300.plot_joint(
    times=[0.3, 0.4, 0.5],  # P300 峰值周围
    title='P300 Component'
)

# 自动检测峰值
peaks_idx, _ = evoked.get_peak(
    ch_type='eeg',
    tmin=0.3, tmax=0.5,
    mode='pos',  # 正峰
    return_amplitude=True
)

# 在峰值处绘制
peak_times = evoked.times[peaks_idx][:3]  # 前3个峰
evoked.plot_joint(times=peak_times)
```

---

## 峰值检测算法

### 1. 单通道峰值

**算法位置**: `mne/evoked.py:get_peak()` (行 800-1000)

```python
def get_peak(evoked, ch_type='eeg', tmin=None, tmax=None, 
             mode='abs', return_amplitude=False):
    """
    检测峰值
    
    参数:
        ch_type: 通道类型
        tmin, tmax: 搜索时间窗口
        mode: 'pos' | 'neg' | 'abs'
            - pos: 正峰值
            - neg: 负峰值
            - abs: 绝对值最大
        return_amplitude: 是否返回幅值
    
    返回:
        (channel_idx, time_idx) 或 (channel_idx, time_idx, amplitude)
    """
    # 1. 选择通道
    picks = pick_types(evoked.info, meg=False, eeg=(ch_type=='eeg'))
    data = evoked.data[picks]
    
    # 2. 时间窗口
    if tmin is None:
        tmin = evoked.times[0]
    if tmax is None:
        tmax = evoked.times[-1]
    
    time_mask = (evoked.times >= tmin) & (evoked.times <= tmax)
    data_window = data[:, time_mask]
    times_window = evoked.times[time_mask]
    
    # 3. 峰值检测
    if mode == 'pos':
        # 最大正值
        ch_idx, time_idx = np.unravel_index(
            np.argmax(data_window), data_window.shape
        )
    elif mode == 'neg':
        # 最大负值（最小值）
        ch_idx, time_idx = np.unravel_index(
            np.argmin(data_window), data_window.shape
        )
    else:  # abs
        # 绝对值最大
        ch_idx, time_idx = np.unravel_index(
            np.argmax(np.abs(data_window)), data_window.shape
        )
    
    # 4. 转换为全局索引
    global_time_idx = np.where(time_mask)[0][time_idx]
    global_ch_idx = picks[ch_idx]
    
    # 5. 返回
    if return_amplitude:
        amplitude = evoked.data[global_ch_idx, global_time_idx]
        return global_ch_idx, global_time_idx, amplitude
    else:
        return global_ch_idx, global_time_idx

# 使用
# 检测 P300
ch_idx, time_idx, amplitude = evoked.get_peak(
    ch_type='eeg',
    tmin=0.25, tmax=0.6,
    mode='pos',
    return_amplitude=True
)

peak_time = evoked.times[time_idx]
peak_channel = evoked.ch_names[ch_idx]

print(f"P300 峰值:")
print(f"  通道: {peak_channel}")
print(f"  潜伏期: {peak_time*1000:.1f} ms")
print(f"  幅度: {amplitude*1e6:.2f} μV")
```

---

### 2. 自动成分识别

**算法位置**: 自定义实现

```python
def detect_erp_components(evoked, components_config):
    """
    自动检测多个 ERP 成分
    
    参数:
        components_config: dict {
            'P1': {'tmin': 0.08, 'tmax': 0.12, 'polarity': 'pos'},
            'N1': {'tmin': 0.12, 'tmax': 0.18, 'polarity': 'neg'},
            'P300': {'tmin': 0.25, 'tmax': 0.6, 'polarity': 'pos'},
        }
    
    返回:
        DataFrame with columns: component, channel, latency, amplitude
    """
    import pandas as pd
    
    results = []
    
    for component, config in components_config.items():
        # 检测峰值
        ch_idx, time_idx, amp = evoked.get_peak(
            ch_type='eeg',
            tmin=config['tmin'],
            tmax=config['tmax'],
            mode=config['polarity'],
            return_amplitude=True
        )
        
        results.append({
            'component': component,
            'channel': evoked.ch_names[ch_idx],
            'latency_ms': evoked.times[time_idx] * 1000,
            'amplitude_uV': amp * 1e6
        })
    
    return pd.DataFrame(results)

# 使用
components = {
    'P1': {'tmin': 0.08, 'tmax': 0.12, 'polarity': 'pos'},
    'N1': {'tmin': 0.12, 'tmax': 0.18, 'polarity': 'neg'},
    'P2': {'tmin': 0.18, 'tmax': 0.25, 'polarity': 'pos'},
    'N2': {'tmin': 0.20, 'tmax': 0.35, 'polarity': 'neg'},
    'P300': {'tmin': 0.30, 'tmax': 0.60, 'polarity': 'pos'},
}

erp_peaks = detect_erp_components(evoked, components)
print(erp_peaks)

#    component channel  latency_ms  amplitude_uV
# 0         P1     POz       104.0          3.24
# 1         N1      Cz       156.0         -5.67
# 2         P2      Pz       228.0          8.92
# 3         N2      Fz       284.0         -3.45
# 4       P300      Pz       376.0         12.34
```

---

## 应用场景

### 场景 1: 经典 P300 分析

```python
import mne
import matplotlib.pyplot as plt

# 1. 加载 epochs
epochs = mne.read_epochs('epochs-epo.fif')

# 2. 分条件平均
evoked_target = epochs['target'].average()
evoked_standard = epochs['standard'].average()

# 3. 计算差异波
evoked_diff = mne.combine_evoked(
    [evoked_target, evoked_standard],
    weights=[1, -1]
)

# 4. 可视化对比
from mne.viz import plot_compare_evokeds

plot_compare_evokeds(
    {'Target': evoked_target, 
     'Standard': evoked_standard,
     'Difference': evoked_diff},
    picks='Pz',  # P300 典型电极
    title='P300 Oddball Paradigm'
)

# 5. 峰值分析
ch_idx, time_idx, amp = evoked_diff.get_peak(
    ch_type='eeg',
    tmin=0.3, tmax=0.6,
    mode='pos',
    return_amplitude=True
)

print(f"P300 差异波峰值: {amp*1e6:.2f} μV at {evoked_diff.times[time_idx]*1000:.0f} ms")

# 6. 地形图
evoked_diff.plot_topomap(
    times=[0.35, 0.40, 0.45],
    ch_type='eeg',
    colorbar=True,
    title='P300 Topography'
)

# 7. 保存
evoked_target.save('target-ave.fif')
evoked_standard.save('standard-ave.fif')
evoked_diff.save('difference-ave.fif')
```

---

### 场景 2: N400 语义违例

```python
import mne

epochs = mne.read_epochs('language-epo.fif')

# 分条件
evoked_congruent = epochs['congruency == "congruent"'].average()
evoked_incongruent = epochs['congruency == "incongruent"'].average()

# N400 差异
evoked_n400 = mne.combine_evoked(
    [evoked_incongruent, evoked_congruent],
    weights=[1, -1]
)

# 联合视图
evoked_n400.plot_joint(
    times=[0.35, 0.40, 0.45, 0.50],
    title='N400 Effect (Incongruent - Congruent)',
    ts_args=dict(gfp=True, spatial_colors=True)
)

# 统计检验（见模块6）
from mne.stats import permutation_cluster_test

X = [epochs['congruency == "congruent"'].get_data(),
     epochs['congruency == "incongruent"'].get_data()]

T_obs, clusters, p_values, H0 = permutation_cluster_test(
    X, threshold=dict(start=0, step=0.2), n_permutations=1000
)

# 可视化显著性
evoked_n400.plot_image(picks='eeg', mask=clusters[0])
```

---

### 场景 3: 组平均分析

```python
import mne
from pathlib import Path

# 1. 收集所有被试的 evoked
data_dir = Path('derivatives')
subjects = [f'sub-{i:02d}' for i in range(1, 21)]  # 20 被试

evokeds_all = []
for subject in subjects:
    evoked_file = data_dir / subject / f'{subject}_task-auditory_ave.fif'
    if evoked_file.exists():
        evoked = mne.read_evokeds(evoked_file)[0]  # 第一个条件
        evokeds_all.append(evoked)

print(f"加载了 {len(evokeds_all)} 个被试")

# 2. 组平均
grand_avg = mne.grand_average(evokeds_all)

# 3. 计算组水平标准误
all_data = np.array([evoked.data for evoked in evokeds_all])
sem_data = all_data.std(axis=0) / np.sqrt(len(all_data))

evoked_sem = mne.EvokedArray(
    sem_data,
    grand_avg.info.copy(),
    tmin=grand_avg.tmin,
    kind='standard_error'
)

# 4. 可视化带误差带
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# 选择代表性通道
for ch_name, ax in zip(['Fz', 'Pz'], axes):
    ch_idx = grand_avg.ch_names.index(ch_name)
    
    mean = grand_avg.data[ch_idx]
    sem = evoked_sem.data[ch_idx]
    times = grand_avg.times
    
    ax.plot(times, mean * 1e6, label='Grand Average')
    ax.fill_between(
        times,
        (mean - sem) * 1e6,
        (mean + sem) * 1e6,
        alpha=0.3,
        label='±SEM'
    )
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (μV)')
    ax.set_title(f'{ch_name} (N={len(evokeds_all)})')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. 保存组平均
grand_avg.save('grand_average-ave.fif')
```

---

### 场景 4: 自动批量分析

```python
import mne
import pandas as pd
from pathlib import Path

def analyze_subject_erp(subject_id, data_dir, output_dir):
    """单被试 ERP 分析流程"""
    
    # 1. 加载 epochs
    epochs_file = data_dir / f'sub-{subject_id}' / 'epochs-epo.fif'
    epochs = mne.read_epochs(epochs_file)
    
    # 2. 分条件平均
    conditions = {
        'target': epochs['event_id == 1'].average(),
        'standard': epochs['event_id == 2'].average(),
    }
    
    # 3. 差异波
    conditions['difference'] = mne.combine_evoked(
        [conditions['target'], conditions['standard']],
        weights=[1, -1]
    )
    
    # 4. 峰值检测
    components = {
        'N1': {'tmin': 0.08, 'tmax': 0.15, 'polarity': 'neg'},
        'P2': {'tmin': 0.15, 'tmax': 0.25, 'polarity': 'pos'},
        'N2': {'tmin': 0.20, 'tmax': 0.35, 'polarity': 'neg'},
        'P300': {'tmin': 0.30, 'tmax': 0.60, 'polarity': 'pos'},
    }
    
    results = []
    for condition_name, evoked in conditions.items():
        for comp_name, config in components.items():
            ch_idx, time_idx, amp = evoked.get_peak(
                ch_type='eeg',
                tmin=config['tmin'],
                tmax=config['tmax'],
                mode=config['polarity'],
                return_amplitude=True
            )
            
            results.append({
                'subject': subject_id,
                'condition': condition_name,
                'component': comp_name,
                'channel': evoked.ch_names[ch_idx],
                'latency_ms': evoked.times[time_idx] * 1000,
                'amplitude_uV': amp * 1e6
            })
        
        # 保存 evoked
        output_file = output_dir / f'sub-{subject_id}_{condition_name}-ave.fif'
        evoked.save(output_file, overwrite=True)
    
    return pd.DataFrame(results)

# 批处理
all_results = []
for subject_id in range(1, 21):
    try:
        df = analyze_subject_erp(
            f'{subject_id:02d}',
            Path('data'),
            Path('results')
        )
        all_results.append(df)
        print(f"✓ Subject {subject_id:02d}")
    except Exception as e:
        print(f"✗ Subject {subject_id:02d}: {e}")

# 合并结果
results_df = pd.concat(all_results, ignore_index=True)
results_df.to_csv('erp_peaks_all_subjects.csv', index=False)

# 汇总统计
summary = results_df.groupby(['condition', 'component']).agg({
    'latency_ms': ['mean', 'std'],
    'amplitude_uV': ['mean', 'std']
}).round(2)

print("\n=== ERP 成分汇总 ===")
print(summary)
```

---

## 总结

### 核心算法汇总

| 算法 | 位置 | 复杂度 | 作用 | 场景 |
|------|------|--------|------|------|
| **平均化** |
| `average()` | `epochs.py:1600+` | O(E×C×T) | 跨trial平均 | 提取诱发响应 |
| `grand_average()` | `evoked.py:2000+` | O(S×C×T) | 跨被试平均 | 组分析 |
| `combine_evoked()` | `evoked.py:1900+` | O(C×T) | 条件对比 | 差异波 |
| **峰值检测** |
| `get_peak()` | `evoked.py:800+` | O(C×T) | 峰值定位 | ERP成分识别 |
| **可视化** |
| `plot_joint()` | `viz/evoked.py` | - | 联合视图 | 时空展示 |
| `plot_topomap()` | `viz/topomap.py` | - | 地形图 | 空间分布 |
| `plot_compare_evokeds()` | `viz/evoked.py` | - | 条件对比 | 差异展示 |

*注: E=epochs数, S=被试数, C=通道数, T=时间点*

### 设计要点

1. **平均提升SNR**: $\sqrt{N}$ 倍信噪比提升
2. **差异波**: 消除共同成分，突出效应
3. **组平均**: 泛化到群体水平
4. **自动峰值检测**: 客观量化ERP成分

### 下一步

Evoked 数据通常用于：
1. **源定位** → **模块5: Source Estimation**
2. **统计分析** → **模块6: Statistics**
3. **时频分析** (返回 Epochs)
