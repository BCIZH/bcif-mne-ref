# 模块 1: I/O - 数据输入输出

> **在数据流中的位置**: 第一步 - 原始数据读取  
> **核心职责**: 多格式神经生理数据的统一读写接口  
> **模块路径**: `mne/io/`

---

## 目录

1. [模块概述](#模块概述)
2. [核心数据结构](#核心数据结构)
3. [支持的数据格式](#支持的数据格式)
4. [关键算法与实现](#关键算法与实现)
5. [应用场景](#应用场景)
6. [代码示例](#代码示例)

---

## 模块概述

### 设计哲学

```
统一抽象层 (BaseRaw)
        ↓
格式特化实现 (RawFIF, RawEDF, ...)
        ↓
透明的内部表示 (NumPy数组 + Info字典)
```

**核心优势**:
- **格式无关**: 用户代码与数据格式解耦
- **延迟加载**: 处理TB级数据集而不耗尽内存
- **元数据丰富**: 保留所有测量信息（通道位置、坐标系等）

### 模块结构

```
mne/io/
├── base.py                # BaseRaw 基类（3252行）
├── _read_raw.py           # 自动格式检测
├── constants.py           # FIFF 常量定义
├── _fiff_wrap.py          # FIFF 元数据包装器
│
├── 格式特化子模块 (30+)
│   ├── fiff/              # Neuromag/MEGIN (MEG标准格式)
│   ├── brainvision/       # Brain Products
│   ├── edf/               # European Data Format (EEG)
│   ├── eeglab/            # EEGLAB (MATLAB)
│   ├── ctf/               # CTF MEG
│   ├── bti/               # BTI/4D MEG
│   ├── kit/               # Yokogawa/KIT MEG
│   ├── snirf/             # fNIRS 标准
│   ├── eyelink/           # 眼动追踪
│   └── ...                # 20+ 其他格式
│
└── array/                 # 从 NumPy 数组创建
    └── array.py           # RawArray
```

---

## 核心数据结构

### 1. BaseRaw（抽象基类）

**文件位置**: `mne/io/base.py:106-3252`

**核心属性**:
```python
class BaseRaw:
    # 元数据
    info: Info                    # 测量信息（通道、采样率、坐标等）
    
    # 数据存储
    _data: ndarray | None         # [n_channels × n_times]，preload=True时有效
    _filenames: tuple             # 数据文件路径（支持分段文件）
    _raw_extras: list[dict]       # 格式特定的读取信息
    
    # 时间索引
    first_samps: ndarray          # 每段文件的起始样本
    last_samps: ndarray           # 每段文件的结束样本
    _first_time: float            # 第一个样本的时间戳
    _last_time: float             # 最后一个样本的时间戳
    
    # 标注
    annotations: Annotations      # 时间标注（伪迹、事件等）
    
    # 数据类型
    orig_format: str              # 原始数据格式 ('double', 'float', 'int16'等)
```

**继承的 Mixin**:
```python
BaseRaw(
    ProjMixin,           # SSP投影: add_proj(), apply_proj()
    ContainsMixin,       # 通道类型检查: _contains_ch_type()
    UpdateChannelsMixin, # 更新通道: rename_channels(), set_channel_types()
    ReferenceMixin,      # EEG重参考: set_eeg_reference()
    SetChannelsMixin,    # 设置属性: set_montage()
    InterpolationMixin,  # 插值: interpolate_bads()
    TimeMixin,           # 时间操作: crop(), time_as_index()
    SizeMixin,           # 大小计算: __sizeof__()
    FilterMixin,         # 滤波: filter(), notch_filter(), resample()
    SpectrumMixin,       # 频谱: compute_psd(), plot_psd()
)
```

### 2. Info（元数据容器）

**文件位置**: `mne/_fiff/meas_info.py`

**结构**:
```python
Info = dict with keys:
    # 通道信息
    'chs': list[dict]              # 每个通道的详细信息
        ├── 'ch_name': str         # 通道名
        ├── 'kind': int            # 类型 (FIFF.FIFFV_MEG_CH, FIFF.FIFFV_EEG_CH等)
        ├── 'coil_type': int       # 线圈/电极类型
        ├── 'loc': ndarray(12,)    # 位置+方向信息
        ├── 'unit': int            # 单位 (V, T, T/m等)
        ├── 'cal': float           # 校准因子
        └── 'range': float         # 量程
    
    # 采集信息
    'sfreq': float                 # 采样率 (Hz)
    'highpass': float              # 高通滤波截止频率
    'lowpass': float               # 低通滤波截止频率
    'line_freq': float | None      # 电源线频率 (50/60 Hz)
    
    # 坐标变换
    'dev_head_t': Transform        # 设备→头部坐标变换 (4×4矩阵)
    'ctf_head_t': Transform        # CTF头部坐标变换
    
    # 数字化点
    'dig': list[dict]              # 数字化的3D点
        ├── 'kind': int            # 点类型（基准点、HPI、EEG、Extra）
        ├── 'r': ndarray(3,)       # 3D坐标 [x, y, z]
        └── 'ident': int           # 标识符
    
    # SSP投影
    'projs': list[Projection]      # 信号空间投影向量
    
    # 补偿
    'comps': list[dict]            # MEG补偿矩阵
    
    # 时间信息
    'meas_date': datetime | None   # 测量时间
    'meas_id': dict                # 测量ID
    
    # 质量控制
    'bads': list[str]              # 坏通道列表
    
    # 被试与设备
    'subject_info': dict           # 被试信息（年龄、性别等）
    'device_info': dict            # 设备信息（制造商、型号等）
    'helium_info': dict            # MEG液氦信息
    
    # 实验信息
    'experimenter': str
    'description': str
    'proj_name': str
```

---

## 支持的数据格式

### MEG 系统（7种）

| 格式 | 函数 | 文件位置 | 厂商 | 文件扩展名 |
|------|------|---------|------|-----------|
| **FIF** | `read_raw_fif()` | `mne/io/fiff/fiff.py` | Neuromag/MEGIN | `.fif` |
| **CTF** | `read_raw_ctf()` | `mne/io/ctf/ctf.py` | CTF Systems | `.ds/` (目录) |
| **BTI/4D** | `read_raw_bti()` | `mne/io/bti/bti.py` | BTi/4D Neuroimaging | `c,rfDC` |
| **KIT** | `read_raw_kit()` | `mne/io/kit/kit.py` | Yokogawa/KIT | `.sqd`, `.con` |
| **Artemis123** | `read_raw_artemis123()` | `mne/io/artemis123/` | Tristan Technologies | `.bin` |
| **FIL** | `read_raw_fil()` | `mne/io/fil/fil.py` | Elekta/UCL OPM | `.bin` |
| **Fieldtrip** | `read_raw_fieldtrip()` | `mne/io/fieldtrip/` | FieldTrip (MATLAB) | `.mat` |

### EEG 系统（15+种）

| 格式 | 函数 | 文件位置 | 特点 | 扩展名 |
|------|------|---------|------|--------|
| **BrainVision** | `read_raw_brainvision()` | `mne/io/brainvision/` | Brain Products | `.vhdr`, `.eeg`, `.vmrk` |
| **EDF/EDF+** | `read_raw_edf()` | `mne/io/edf/edf.py` | 欧洲标准 | `.edf` |
| **BDF** | `read_raw_bdf()` | `mne/io/edf/edf.py` | BioSemi (24bit) | `.bdf` |
| **GDF** | `read_raw_gdf()` | `mne/io/edf/gdf.py` | EDF扩展 | `.gdf` |
| **EEGLAB** | `read_raw_eeglab()` | `mne/io/eeglab/` | MATLAB工具箱 | `.set`, `.fdt` |
| **EGI/MFF** | `read_raw_egi()` | `mne/io/egi/` | Electrical Geodesics | `.mff/`, `.raw` |
| **Neuroscan** | `read_raw_cnt()` | `mne/io/cnt/cnt.py` | Compumedics | `.cnt` |
| **Curry** | `read_raw_curry()` | `mne/io/curry/` | Compumedics | `.cdt`, `.dap` |
| **BESA** | `read_evoked_besa()` | `mne/io/besa/` | 仅 evoked | `.avr` |
| **Nicolet** | `read_raw_nicolet()` | `mne/io/nicolet/` | 临床EEG | `.data` |
| **Persyst** | `read_raw_persyst()` | `mne/io/persyst/` | 癫痫监测 | `.lay`, `.dat` |
| **Nihon Kohden** | `read_raw_nihon()` | `mne/io/nihon/` | 日本光电 | `.eeg`, `.21e` |
| **ANT Neuro** | `read_raw_ant()` | `mne/io/ant/` | ANT Neuro | `.cnt` |
| **EXIMIA** | `read_raw_eximia()` | `mne/io/eximia/` | Nexstim | `.nxe` |
| **Boxy** | `read_raw_boxy()` | `mne/io/boxy/` | 自定义采集 | `.boxy` |

### fNIRS 系统（2种）

| 格式 | 函数 | 文件位置 | 说明 |
|------|------|---------|------|
| **SNIRF** | `read_raw_snirf()` | `mne/io/snirf/` | fNIRS标准格式 (.snirf) |
| **NIRX** | `read_raw_nirx()` | `mne/io/nirx/` | NIRx系统 (.hdr) |
| **Hitachi** | `read_raw_hitachi()` | `mne/io/hitachi/` | 日立系统 (.csv) |

### 其他模态（5种）

| 格式 | 函数 | 文件位置 | 模态 |
|------|------|---------|------|
| **EyeLink** | `read_raw_eyelink()` | `mne/io/eyelink/` | 眼动追踪 (.asc) |
| **Neuralynx** | `read_raw_neuralynx()` | `mne/io/neuralynx/` | 动物电生理 (.ncs) |
| **Blackrock** | `read_raw_nsx()` | `mne/io/nsx/` | 颅内记录 (.ns1-6) |
| **NEDF** | `read_raw_nedf()` | `mne/io/nedf/` | Natus (.edf) |
| **Array** | `RawArray()` | `mne/io/array/` | NumPy数组 |

---

## 关键算法与实现

### 1. 延迟加载机制

**算法位置**: `mne/io/base.py:_read_segment()`  
**文件行数**: 约 470-650 行

**原理**:
```python
# 当 preload=False 时，数据不立即加载
class BaseRaw:
    def __getitem__(self, item):
        # 索引数据时才实际读取
        sel, time_slice, _ = self._parse_get_set_params(item)
        return self._read_segment(...)
    
    def _read_segment(self, start, stop, sel, ...):
        """按需读取数据片段"""
        # 1. 确定读取哪些文件段
        start_file, start_pos = self._get_file_pos(start)
        stop_file, stop_pos = self._get_file_pos(stop)
        
        # 2. 逐段读取并拼接
        data = []
        for file_idx in range(start_file, stop_file + 1):
            segment = self._read_segment_file(
                file_idx, start_pos, stop_pos, sel
            )
            data.append(segment)
        
        # 3. 应用校准和投影
        data = np.concatenate(data, axis=1)
        data = self._apply_cal(data, sel)
        if self._projector is not None:
            data = self._projector @ data
        
        return data
```

**优势**:
- 内存占用与数据集大小解耦
- 支持流式处理（如实时滤波）
- 可处理分段存储的大文件

**应用场景**:
- 长时程睡眠研究（数小时数据）
- 临床监测（连续数天）
- 大规模数据集预览

---

### 2. 格式自动检测

**算法位置**: `mne/io/_read_raw.py:read_raw()`  
**文件行数**: 约 50-150 行

**实现**:
```python
def read_raw(fname, preload=False, verbose=None, **kwargs):
    """智能识别文件格式并调用相应读取器"""
    
    # 格式识别规则
    fname_str = str(fname)
    
    # 1. 扩展名匹配
    readers = {
        '.fif': read_raw_fif,
        '.vhdr': read_raw_brainvision,
        '.edf': read_raw_edf,
        '.bdf': read_raw_bdf,
        '.set': read_raw_eeglab,
        '.sqd': read_raw_kit,
        '.con': read_raw_kit,
        '.mff': read_raw_egi,
        # ... 30+ 种格式
    }
    
    # 2. 目录结构识别（如 CTF .ds/）
    if os.path.isdir(fname) and fname.endswith('.ds'):
        return read_raw_ctf(fname, preload=preload, **kwargs)
    
    # 3. 文件头魔数识别
    ext = Path(fname).suffix.lower()
    if ext in readers:
        return readers[ext](fname, preload=preload, **kwargs)
    
    # 4. 启发式检测
    with open(fname, 'rb') as fid:
        magic = fid.read(8)
        if magic == b'\x00\x00\x00\x00BIOSEMI':
            return read_raw_bdf(fname, preload=preload, **kwargs)
    
    raise ValueError(f"不支持的文件格式: {fname}")
```

**支持的识别方式**:
1. 文件扩展名
2. 目录结构
3. 文件头魔数
4. 内容启发式分析

---

### 3. 多文件拼接

**算法位置**: `mne/io/base.py:concatenate_raws()`  
**文件行数**: 约 3100-3250 行

**算法**:
```python
def concatenate_raws(raws, preload=None, events_list=None):
    """拼接多个 Raw 对象"""
    
    # 1. 验证兼容性
    first = raws[0]
    for raw in raws[1:]:
        # 检查采样率
        if raw.info['sfreq'] != first.info['sfreq']:
            raise ValueError("采样率不一致")
        
        # 检查通道匹配
        if raw.ch_names != first.ch_names:
            raise ValueError("通道不匹配")
    
    # 2. 合并时间索引
    first_samps = []
    last_samps = []
    cumulative_samples = 0
    
    for raw in raws:
        first_samps.append(cumulative_samples)
        cumulative_samples += raw.n_times
        last_samps.append(cumulative_samples - 1)
    
    # 3. 合并标注（调整时间戳）
    annotations = Annotations([], [], [])
    onset_offset = 0
    for raw in raws:
        if raw.annotations:
            ann = raw.annotations.copy()
            ann.onset += onset_offset
            annotations += ann
        onset_offset += raw.times[-1] - raw.times[0]
    
    # 4. 创建新 Raw 对象
    combined = RawFIF(
        info=first.info.copy(),
        preload=False,
        first_samps=first_samps,
        last_samps=last_samps,
        filenames=[r._filenames for r in raws],
        raw_extras=[r._raw_extras for r in raws]
    )
    combined.annotations = annotations
    
    return combined
```

**应用场景**:
- 合并多次记录的session
- 处理自动分段的长数据
- 批处理流程

---

### 4. 坐标系变换

**算法位置**: `mne/_fiff/meas_info.py`, `mne/transforms.py`

**核心变换矩阵**:
```python
# 关键坐标系
坐标系层次:
    设备坐标系 (Device)
         ↓ dev_head_t
    头部坐标系 (Head)
         ↓ head_mri_t (从 trans 文件读取)
    MRI坐标系 (MRI/Surface RAS)
         ↓ mri_voxel_t
    体素坐标系 (Voxel)

# 变换矩阵格式 (4×4 齐次坐标)
Transform = {
    'from': int,      # FIFF.FIFFV_COORD_DEVICE
    'to': int,        # FIFF.FIFFV_COORD_HEAD
    'trans': ndarray  # 4×4 变换矩阵
}
```

**应用**:
```python
# 获取电极的头部坐标
def get_electrode_positions(info):
    locs = []
    for ch in info['chs']:
        if ch['kind'] == FIFF.FIFFV_EEG_CH:
            # ch['loc'][:3] 是设备坐标
            # 应用 dev_head_t 转换
            pos_head = apply_trans(info['dev_head_t'], ch['loc'][:3])
            locs.append(pos_head)
    return np.array(locs)
```

---

### 5. 数据类型转换与校准

**算法位置**: `mne/io/base.py:_apply_cal()`

**原理**:
```python
def _apply_cal(self, data, sel):
    """应用校准因子和单位转换"""
    for idx, ch_idx in enumerate(sel):
        ch = self.info['chs'][ch_idx]
        
        # 1. 应用硬件校准
        data[idx] *= ch['cal']
        
        # 2. 单位转换（如 V → μV）
        if ch['unit'] == FIFF.FIFF_UNIT_V:
            data[idx] *= 1e6  # 转为微伏
        elif ch['unit'] == FIFF.FIFF_UNIT_T:
            data[idx] *= 1e15  # 转为飞特斯拉
        
        # 3. 应用量程缩放
        data[idx] /= ch['range']
    
    return data
```

**支持的原始格式**:
- `int16`: 16位整数（常见于EDF）
- `int32`: 32位整数
- `float32`: 单精度浮点
- `float64`: 双精度浮点
- `dau_pack16`: DAU压缩格式

---

## 应用场景

### 场景 1: 基础数据加载

```python
import mne

# 自动识别格式
raw = mne.io.read_raw('data.fif', preload=False)

# 或指定格式
raw = mne.io.read_raw_brainvision('data.vhdr', preload=True)

# 查看基本信息
print(f"采样率: {raw.info['sfreq']} Hz")
print(f"通道数: {len(raw.ch_names)}")
print(f"时长: {raw.times[-1]:.2f} 秒")
print(f"通道类型: {raw.get_channel_types()}")
```

**适用于**:
- 快速数据预览
- 格式兼容性检查
- 元数据提取

---

### 场景 2: 大文件处理

```python
# 延迟加载 100GB 数据
raw = mne.io.read_raw_fif('huge_data.fif', preload=False)

# 仅处理需要的时间段
raw_segment = raw.copy().crop(tmin=0, tmax=60)  # 只加载前60秒
raw_segment.load_data()  # 现在加载到内存

# 或使用内存映射
raw_mmap = mne.io.read_raw_fif('huge_data.fif', preload='temp.dat')
```

**适用于**:
- 睡眠研究（8小时+数据）
- 临床长时程监测
- 计算资源受限环境

---

### 场景 3: 批量格式转换

```python
from pathlib import Path

def convert_to_fif(input_dir, output_dir):
    """批量转换为 FIF 格式"""
    for vhdr_file in Path(input_dir).glob('*.vhdr'):
        # 读取 BrainVision
        raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
        
        # 保存为 FIF
        output_file = Path(output_dir) / f"{vhdr_file.stem}_raw.fif"
        raw.save(output_file, overwrite=True)
        print(f"转换完成: {vhdr_file.name} → {output_file.name}")

convert_to_fif('raw_data/', 'fif_data/')
```

**适用于**:
- 标准化数据存储
- 实验室间数据共享
- 长期归档

---

### 场景 4: 多模态数据整合

```python
# 同时加载 MEG 和 EEG
raw_meg = mne.io.read_raw_fif('meg_data.fif', preload=True)
raw_eeg = mne.io.read_raw_brainvision('eeg_data.vhdr', preload=True)

# 重采样到相同采样率
raw_eeg.resample(raw_meg.info['sfreq'])

# 添加 EEG 通道到 MEG
raw_combined = raw_meg.copy()
raw_combined.add_channels([raw_eeg], force_update_info=True)

print(f"合并后通道数: {len(raw_combined.ch_names)}")
```

**适用于**:
- MEG-EEG 联合记录
- 多设备同步采集
- 事后数据融合

---

### 场景 5: 从数组创建 Raw

```python
import numpy as np

# 模拟数据
sfreq = 1000  # Hz
n_channels = 64
duration = 10  # 秒
data = np.random.randn(n_channels, int(sfreq * duration))

# 创建 Info
info = mne.create_info(
    ch_names=[f'EEG{i:03d}' for i in range(n_channels)],
    sfreq=sfreq,
    ch_types='eeg'
)

# 创建 Raw 对象
raw = mne.io.RawArray(data, info)

# 设置 montage
montage = mne.channels.make_standard_montage('biosemi64')
raw.set_montage(montage)
```

**适用于**:
- 仿真研究
- 自定义采集系统
- 数据生成

---

## 代码示例

### 示例 1: 完整的数据加载与检查流程

```python
import mne
import matplotlib.pyplot as plt

# 1. 加载数据
raw = mne.io.read_raw_fif(
    'sample_audvis_raw.fif',
    preload=False,
    verbose=True
)

# 2. 检查基本信息
print(raw.info)
print(f"\n通道类型分布:")
print(raw.get_channel_types(unique=True))

# 3. 查看数据质量
raw.plot(duration=10, n_channels=30, scalings='auto')

# 4. 检查坏通道
raw.info['bads'] = ['MEG 2443', 'EEG 053']

# 5. 查看功率谱（检测工频噪声）
raw.compute_psd(fmax=100).plot()

# 6. 预加载用于后续处理
raw.load_data()

# 7. 保存预处理后的数据
raw.save('cleaned_raw.fif', overwrite=True)
```

---

### 示例 2: 处理多段数据

```python
import mne

# 读取多个文件
raw1 = mne.io.read_raw_fif('run1_raw.fif', preload=False)
raw2 = mne.io.read_raw_fif('run2_raw.fif', preload=False)
raw3 = mne.io.read_raw_fif('run3_raw.fif', preload=False)

# 拼接
raw_combined = mne.concatenate_raws([raw1, raw2, raw3])

print(f"总时长: {raw_combined.times[-1] / 60:.2f} 分钟")
print(f"总样本数: {raw_combined.n_times}")

# 保存拼接结果
raw_combined.save('combined_raw.fif', split_size='2GB')
```

---

### 示例 3: 自定义读取器

```python
import numpy as np
import mne
from mne.io import BaseRaw

class RawCustom(BaseRaw):
    """自定义格式读取器示例"""
    
    def __init__(self, fname, preload=False, verbose=None):
        # 读取自定义格式文件
        data, sfreq, ch_names = self._read_custom_format(fname)
        
        # 创建 Info
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types='eeg'
        )
        
        # 调用父类初始化
        super().__init__(
            info,
            preload=data if preload else False,
            filenames=[fname],
            last_samps=[data.shape[1] - 1] if preload else None,
            orig_format='int16',
            verbose=verbose
        )
        
        if not preload:
            self._data = data
    
    def _read_custom_format(self, fname):
        """实现自定义格式解析"""
        # 这里实现具体的文件读取逻辑
        data = np.loadtxt(fname, delimiter=',').T
        sfreq = 500  # 从文件头读取
        ch_names = [f'CH{i}' for i in range(data.shape[0])]
        return data, sfreq, ch_names
    
    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """实现延迟加载的数据段读取"""
        if self._data is not None:
            return self._data[:, start:stop]
        else:
            # 从文件读取指定段
            pass

# 使用
raw_custom = RawCustom('my_data.txt', preload=True)
raw_custom.plot()
```

---

## 总结

### 核心算法列表

| 算法 | 位置 | 作用 | 场景 |
|------|------|------|------|
| 延迟加载 | `base.py:_read_segment()` | 按需读取数据 | 大文件处理 |
| 格式检测 | `_read_raw.py:read_raw()` | 自动识别格式 | 便捷使用 |
| 数据拼接 | `base.py:concatenate_raws()` | 合并多段数据 | 长时程记录 |
| 坐标变换 | `transforms.py:apply_trans()` | 坐标系转换 | 源定位 |
| 校准转换 | `base.py:_apply_cal()` | 物理单位转换 | 数据标准化 |

### 设计亮点

1. **抽象与复用**: BaseRaw 提供统一接口，30+格式共享代码
2. **内存效率**: 延迟加载 + 内存映射处理TB级数据
3. **元数据完整**: Info 保留所有测量信息用于下游分析
4. **可扩展性**: 易于添加新格式支持

### 下一步

数据加载后，通常进入 **模块2: 预处理（Preprocessing）**，进行滤波和伪迹去除。
