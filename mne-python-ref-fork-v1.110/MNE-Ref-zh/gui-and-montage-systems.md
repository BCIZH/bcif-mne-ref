# MNE-Python GUI 与电极系统

> **核心功能**: 图形界面工具 + 电极位置标准 + 可视化  
> **模块路径**: `mne/gui/`, `mne/channels/`, `mne/icons/`, `mne/viz/`

---

## 目录

1. [GUI 概述](#gui-概述)
2. [Coregistration GUI](#coregistration-gui)
3. [电极系统标准](#电极系统标准)
4. [内置电极 Montage](#内置电极-montage)
5. [电极可视化](#电极可视化)
6. [图标资源](#图标资源)
7. [应用场景](#应用场景)

---

## GUI 概述

### MNE-Python 的 GUI 组件

**文件位置**: `mne/gui/`

```
mne/gui/
├── __init__.py          # 公开 API
├── __init__.pyi         # 类型提示
├── _gui.py              # GUI 辅助函数
├── _coreg.py            # 配准界面实现
└── tests/               # 测试
```

**公开的 GUI 工具**:
```python
from mne.gui import coregistration

# 启动配准界面
frame = coregistration(
    inst='raw.fif',
    subject='fsaverage',
    subjects_dir='~/mne_data/MNE-sample-data/subjects'
)
```

---

## Coregistration GUI

### 1. 功能概述

**算法位置**: `mne/gui/_gui.py:coregistration()` (行 8-180)

**主要功能**:
- **MRI-头部配准**: 将解剖 MRI 与被试的头部形状对齐
- **数字化点对齐**: 匹配数字化的头部形状点
- **传感器位置校验**: 验证传感器位置正确性
- **交互式调整**: 实时调整平移、旋转、缩放

**启动方式**:
```bash
# 命令行
$ mne coreg

# Python
from mne.gui import coregistration

frame = coregistration(
    inst='raw.fif',           # 包含数字化数据的文件
    subject='sample',         # MRI 被试名
    subjects_dir=subjects_dir,
    head_opacity=0.8,        # 头部透明度
    head_high_res=True,      # 高分辨率头部
    trans='sample-trans.fif' # 已有的变换（可选）
)
```

---

### 2. 配准参数详解

**算法位置**: `mne/gui/_gui.py:coregistration()`

```python
def coregistration(
    *,
    width=None,              # 窗口宽度（默认800像素）
    height=None,             # 窗口高度（默认600像素）
    inst=None,               # Raw/Epochs/Evoked 文件路径
    subject=None,            # 被试名称
    subjects_dir=None,       # Freesurfer subjects 目录
    head_opacity=None,       # 头部透明度 [0, 1]（默认0.8）
    head_high_res=None,      # 使用高分辨率头部（默认True）
    trans=None,              # 头部-MRI 变换矩阵
    orient_to_surface=None,  # 朝向头皮表面（默认True）
    scale_by_distance=None,  # 按距离缩放点（默认True）
    mark_inside=None,        # 标记头内点（默认True）
    interaction=None,        # 交互模式: 'terrain' | 'trackball'
    fullscreen=None,         # 全屏模式（默认False）
    show=True,               # 显示GUI
    block=False,             # 阻塞程序执行
    verbose=None
):
    """
    启动 MRI-头部配准 GUI
    
    配置读取:
        - 优先级: 函数参数 > MNE配置文件
        - 配置文件位置: ~/.mne/mne.json
        - 可用配置项:
            * MNE_COREG_HEAD_OPACITY: 0.8
            * MNE_COREG_HEAD_HIGH_RES: true
            * MNE_COREG_WINDOW_WIDTH: 800
            * MNE_COREG_WINDOW_HEIGHT: 600
            * MNE_COREG_ORIENT_TO_SURFACE: true
            * MNE_COREG_SCALE_BY_DISTANCE: true
            * MNE_COREG_INTERACTION: terrain
            * MNE_COREG_MARK_INSIDE: true
    
    返回:
        CoregistrationUI 实例
    """
```

---

### 3. 配准工作流

**步骤指南**:

```python
import mne
from mne.gui import coregistration

# 1. 准备数据
subjects_dir = 'path/to/freesurfer/subjects'
subject = 'sample'

# 2. 读取包含数字化点的数据
raw = mne.io.read_raw_fif('sample_auditory_raw.fif')

# 或者单独读取数字化文件
# digitization = mne.io.read_fiducials('sample-fiducials.fif')

# 3. 启动配准 GUI
coreg = coregistration(
    inst=raw,
    subject=subject,
    subjects_dir=subjects_dir,
    show=True
)

# 4. 在 GUI 中:
#    a) 加载头部表面（自动）
#    b) 加载数字化点（从 inst）
#    c) 初步对齐基准点 (Nasion, LPA, RPA)
#    d) 使用 ICP 算法微调
#    e) 检查拟合质量
#    f) 保存变换矩阵

# 5. 保存变换（在GUI中完成）
# File -> Save -> 保存为 'sample-trans.fif'

# 6. 在后续分析中使用
fwd = mne.make_forward_solution(
    raw.info,
    trans='sample-trans.fif',  # ← 使用配准结果
    src=src,
    bem=bem
)
```

---

## 电极系统标准

### 1. 国际 10-20 系统

**标准位置文件**: `mne/channels/data/montages/standard_1020.elc`

**描述**: 国际标准 10-20 系统（94+3 位置）

**电极命名规则**:
```
位置编码:
    F  = Frontal (额)
    C  = Central (中央)
    P  = Parietal (顶)
    T  = Temporal (颞)
    O  = Occipital (枕)

半球标记:
    奇数 (1, 3, 5, ...) = 左半球
    偶数 (2, 4, 6, ...) = 右半球
    z                    = 中线

示例:
    Fz  : 额正中
    F3  : 左额
    F4  : 右额
    Cz  : 中央正中
    Pz  : 顶正中
    T7  : 左颞 (旧称 T3)
    T8  : 右颞 (旧称 T4)
```

**使用**:
```python
import mne

# 加载标准 10-20 系统
montage_1020 = mne.channels.make_standard_montage('standard_1020')

print(f"电极数: {len(montage_1020.ch_names)}")
print(f"电极名: {montage_1020.ch_names[:10]}")
# ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1']

# 应用到 Raw 对象
raw.set_montage(montage_1020)

# 可视化
montage_1020.plot(kind='topomap', show_names=True)
```

---

### 2. 国际 10-05 系统

**标准位置文件**: `mne/channels/data/montages/standard_1005.elc`

**描述**: 高密度扩展系统（343+3 位置）

**与 10-20 的关系**:
- **10-20**: 基础系统，电极间隔约 10% 或 20% 头围距离
- **10-10**: 中等密度扩展（约 100 个电极）
- **10-05**: 高密度扩展（343 个电极）

**命名模式**:
```
10-05 在 10-20 基础上增加:
    - 更多中间位置
    - 更精细的位置编码
    
示例电极名称:
    AF3, AF4    : Anterior Frontal
    FC1, FC2    : Frontal-Central
    CP1, CP2    : Central-Parietal
    PO3, PO4    : Parietal-Occipital
    FT7, FT8    : Frontal-Temporal
    TP7, TP8    : Temporal-Parietal
```

**使用**:
```python
import mne

# 加载 10-05 系统
montage_1005 = mne.channels.make_standard_montage('standard_1005')

print(f"电极数: {len(montage_1005.ch_names)}")
# 电极数: 343

# 对比密度
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 10-20 (低密度)
montage_1020 = mne.channels.make_standard_montage('standard_1020')
montage_1020.plot(kind='topomap', show_names=False, axes=axes[0], show=False)
axes[0].set_title(f'10-20 System ({len(montage_1020.ch_names)} electrodes)')

# 10-05 (高密度)
montage_1005.plot(kind='topomap', show_names=False, axes=axes[1], show=False)
axes[1].set_title(f'10-05 System ({len(montage_1005.ch_names)} electrodes)')

plt.tight_layout()
plt.show()
```

---

### 3. 10-20 区域选择

**算法位置**: `mne/channels/channels.py:make_1020_channel_selections()` (行 1948+)

**自动分区功能**:
```python
from mne.channels import make_1020_channel_selections

def make_1020_channel_selections(info, midline="z", *, return_ch_names=False):
    """
    根据 10-20 系统自动划分脑区
    
    参数:
        info: Info 对象
        midline: 中线标记 ('z' 或 'Z')
        return_ch_names: 返回通道名而非索引
    
    返回:
        selections: dict
            键: 'Left-frontal', 'Right-temporal', 'Vertex', ...
            值: 通道索引或通道名列表
    
    脑区划分:
        - Vertex (顶点)
        - Left-temporal, Right-temporal (颞)
        - Left-parietal, Right-parietal (顶)
        - Left-occipital, Right-occipital (枕)
        - Left-frontal, Right-frontal (额)
    """

# 使用示例
import mne

raw = mne.io.read_raw_fif('sample_auditory_raw.fif')
raw.set_montage('standard_1020')

selections = make_1020_channel_selections(
    raw.info, 
    return_ch_names=True
)

for region, channels in selections.items():
    print(f"{region}: {channels}")

# Vertex: ['Cz']
# Left-temporal: ['T7', 'T3', ...]
# Right-temporal: ['T8', 'T4', ...]
# Left-parietal: ['P3', 'P7', ...]
# Right-parietal: ['P4', 'P8', ...]
# Left-occipital: ['O1']
# Right-occipital: ['O2']
# Left-frontal: ['F3', 'F7', 'Fp1', ...]
# Right-frontal: ['F4', 'F8', 'Fp2', ...]
```

**应用于数据分析**:
```python
import numpy as np

# 分区域计算平均 ERP
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5)
selections = make_1020_channel_selections(epochs.info, return_ch_names=True)

# 左额叶 vs 右额叶
evoked_left_frontal = epochs.copy().pick(selections['Left-frontal']).average()
evoked_right_frontal = epochs.copy().pick(selections['Right-frontal']).average()

# 可视化对比
from mne.viz import plot_compare_evokeds

plot_compare_evokeds(
    {'Left Frontal': evoked_left_frontal,
     'Right Frontal': evoked_right_frontal},
    combine='mean'
)
```

---

## 内置电极 Montage

### 1. 所有内置 Montage

**算法位置**: `mne/channels/montage.py` (行 70-195)

**标准系统**:
```python
_BUILTIN_STANDARD_MONTAGES = [
    # 国际标准
    'standard_1005',         # 10-05 系统 (343+3 位置)
    'standard_1020',         # 10-20 系统 (94+3 位置)
    'standard_alphabetic',   # 字母数字命名 (A1, B2, ...) (65+3)
    'standard_postfixed',    # 后缀命名 (100+3)
    'standard_prefixed',     # 前缀命名 (74+3)
    'standard_primed',       # 撇号命名 (', '') (100+3)
    
    # BioSemi 系列
    'biosemi16',             # BioSemi 16 通道 (16+3)
    'biosemi32',             # BioSemi 32 通道 (32+3)
    'biosemi64',             # BioSemi 64 通道 (64+3)
    'biosemi128',            # BioSemi 128 通道 (128+3)
    'biosemi160',            # BioSemi 160 通道 (160+3)
    'biosemi256',            # BioSemi 256 通道 (256+3)
    
    # EasyCap 系列
    'easycap-M1',            # EasyCap M1 (74 位置)
    'easycap-M10',           # EasyCap M10 (61 位置)
    'easycap-M43',           # EasyCap M43 (64 位置)
    
    # EGI (Electrical Geodesics)
    'EGI_256',               # Geodesic Sensor Net (256 位置)
    'GSN-HydroCel-32',       # HydroCel 32 (33+3)
    'GSN-HydroCel-64_1.0',   # HydroCel 64 (64+3)
    'GSN-HydroCel-65_1.0',   # HydroCel 65 with Cz (65+3)
    'GSN-HydroCel-128',      # HydroCel 128 (128+3)
    'GSN-HydroCel-129',      # HydroCel 129 with Cz (129+3)
    'GSN-HydroCel-256',      # HydroCel 256 (256+3)
    'GSN-HydroCel-257',      # HydroCel 257 with Cz (257+3)
    
    # MGH (Massachusetts General Hospital)
    'mgh60',                 # MGH 旧版 60 通道 (60+3)
    'mgh70',                 # MGH 新版 70 通道 (70+3)
    
    # fNIRS
    'artinis-octamon',       # Artinis OctaMon (8 源 + 2 探测器)
    'artinis-brite23',       # Artinis Brite23 (11 源 + 7 探测器)
    
    # Brain Products
    'brainproducts-RNP-BA-128',  # Brain Products 128 通道
]
```

**查看所有内置 Montage**:
```python
import mne

# 列出所有
montages = mne.channels.get_builtin_montages()
print(f"共 {len(montages)} 个内置 montage:")
for m in montages:
    print(f"  - {m}")

# 共 28 个内置 montage:
#   - EGI_256
#   - GSN-HydroCel-128
#   - GSN-HydroCel-129
#   - ...
```

---

### 2. 加载和使用 Montage

**基础用法**:
```python
import mne

# 1. 创建标准 montage
montage = mne.channels.make_standard_montage('biosemi64')

# 2. 查看信息
print(f"电极数: {len(montage.ch_names)}")
print(f"基准点: {montage.dig[:3]}")  # Nasion, LPA, RPA

# 3. 获取电极位置
ch_pos = montage.get_positions()
print(f"通道位置键: {ch_pos.keys()}")
# dict_keys(['ch_pos', 'nasion', 'lpa', 'rpa', 'hsp', 'hpi'])

# 4. 应用到数据
raw = mne.io.read_raw_fif('raw.fif', preload=True)

# 检查通道名匹配
print(f"Raw 通道: {raw.ch_names[:5]}")
print(f"Montage 通道: {montage.ch_names[:5]}")

# 应用 montage
raw.set_montage(montage, on_missing='warn')

# 5. 验证位置已设置
print(raw.info['dig'])  # 数字化点
```

---

### 3. 自定义 Montage

**从坐标创建**:
```python
import mne
import numpy as np

# 定义电极位置 (笛卡尔坐标, 单位: 米)
ch_pos = {
    'Fz':  [0.000,  0.085, 0.000],
    'Cz':  [0.000,  0.000, 0.085],
    'Pz':  [0.000, -0.085, 0.000],
    'F3':  [-0.060,  0.060, 0.050],
    'F4':  [ 0.060,  0.060, 0.050],
    'C3':  [-0.085,  0.000, 0.000],
    'C4':  [ 0.085,  0.000, 0.000],
    'P3':  [-0.060, -0.060, 0.050],
    'P4':  [ 0.060, -0.060, 0.050],
}

# 定义基准点 (可选)
nasion = [0.000,  0.095,  0.000]
lpa    = [-0.080,  0.000, -0.010]  # Left PreAuricular
rpa    = [ 0.080,  0.000, -0.010]  # Right PreAuricular

# 创建 montage
montage_custom = mne.channels.make_dig_montage(
    ch_pos=ch_pos,
    nasion=nasion,
    lpa=lpa,
    rpa=rpa,
    coord_frame='head'  # 或 'unknown'
)

# 可视化
montage_custom.plot(kind='3d', show_names=True)

# 应用
info = mne.create_info(
    ch_names=list(ch_pos.keys()),
    sfreq=250,
    ch_types='eeg'
)
info.set_montage(montage_custom)
```

**从文件读取**:
```python
import mne

# 支持的格式:
# - .elc (BESA/Cartool)
# - .sfp (BESA/EGI)
# - .csd (CSD montage)
# - .elp (BESA spherical)
# - .txt (其他格式)
# - .loc (EEGLAB)
# - .locs (EEGLAB)
# - .fif (MNE)

# 示例: 读取 BESA 文件
montage = mne.channels.read_custom_montage('my_montage.elc')

# 或者自动检测格式
montage = mne.channels.read_dig_montage(
    fif='path/to/dig.fif',  # 或
    # egi='path/to/montage.xml',
    # bvct='path/to/electrodes.txt',
)
```

---

## 电极可视化

### 1. plot_montage() - Montage 可视化

**算法位置**: `mne/viz/montage.py:plot_montage()` (行 19-130)

```python
def plot_montage(
    montage,
    *,
    scale=1.0,           # 缩放电极点和标签
    show_names=True,     # 显示通道名
    kind="topomap",      # 'topomap' | '3d'
    show=True,
    sphere=None,
    axes=None,
    verbose=None,
):
    """
    绘制 montage
    
    参数:
        scale: 缩放因子 (< 1 缩小, > 1 放大)
        show_names: 显示通道名 (True | False | list)
        kind: 绘图类型
            - 'topomap': 2D 拓扑图
            - '3d': 3D 视图
    
    返回:
        fig: matplotlib Figure 对象
    """

# 使用示例
import mne
import matplotlib.pyplot as plt

montage = mne.channels.make_standard_montage('standard_1020')

# 2D 拓扑图
fig = montage.plot(kind='topomap', show_names=True)

# 3D 视图
fig = montage.plot(kind='3d', show_names=True)

# 仅显示部分通道名
show_these = ['Fz', 'Cz', 'Pz', 'Oz', 'F3', 'F4', 'C3', 'C4']
fig = montage.plot(kind='topomap', show_names=show_these)

# 调整缩放
fig = montage.plot(kind='topomap', scale=0.8)  # 缩小 20%
```

---

### 2. plot_sensors() - 传感器布局

**算法位置**: `mne/viz/utils.py:plot_sensors()` (行 930+)

```python
from mne.viz import plot_sensors

def plot_sensors(
    info,
    kind="topomap",      # 'topomap' | '3d' | 'select'
    ch_type=None,        # 通道类型过滤
    title=None,
    show_names=False,
    ch_groups=None,      # 通道分组
    to_sphere=True,
    axes=None,
    block=False,
    show=True,
    sphere=None,
    verbose=None,
):
    """
    绘制传感器位置
    
    kind 选项:
        - 'topomap': 2D 拓扑投影
        - '3d': 3D 交互视图
        - 'select': 可选择通道的交互式视图
    
    ch_type: 'mag' | 'grad' | 'eeg' | None
    
    ch_groups: 
        - 'position': 按位置分组（自动检测）
        - dict: 自定义分组 {'Group1': [ch1, ch2], ...}
    """

# 使用示例
import mne

raw = mne.io.read_raw_fif('sample_auditory_raw.fif')
raw.set_montage('standard_1020')

# 基础 2D 视图
fig = mne.viz.plot_sensors(raw.info, kind='topomap', show_names=True)

# 3D 视图
fig = mne.viz.plot_sensors(raw.info, kind='3d', ch_type='eeg')

# 仅显示 EEG
fig = mne.viz.plot_sensors(
    raw.info,
    kind='topomap',
    ch_type='eeg',
    title='EEG Electrode Layout'
)

# 自定义分组
ch_groups = {
    'Frontal': ['Fp1', 'Fp2', 'F3', 'Fz', 'F4'],
    'Central': ['C3', 'Cz', 'C4'],
    'Parietal': ['P3', 'Pz', 'P4'],
    'Occipital': ['O1', 'Oz', 'O2']
}

fig = mne.viz.plot_sensors(
    raw.info,
    kind='topomap',
    ch_groups=ch_groups,
    show_names=True
)
```

---

### 3. 电极邻接关系可视化

**算法位置**: `mne/channels/channels.py:find_ch_adjacency()`

```python
from mne.channels import find_ch_adjacency
import mne

raw = mne.io.read_raw_fif('sample_auditory_raw.fif')
raw.set_montage('standard_1020')

# 计算邻接矩阵
adjacency, ch_names = find_ch_adjacency(raw.info, ch_type='eeg')

print(f"邻接矩阵形状: {adjacency.shape}")
print(f"连接数: {adjacency.nnz}")

# 可视化邻接关系
from mne.viz import plot_ch_adjacency

fig = mne.viz.plot_ch_adjacency(raw.info, adjacency, ch_names)
```

---

## 图标资源

### 1. 图标文件结构

**文件位置**: `mne/icons/`

```
mne/icons/
├── README.rst                      # 说明文档
├── mne_icon.png                    # 主图标
├── mne_default_icon.png            # 默认图标
├── mne_bigsur_icon.png             # macOS Big Sur 图标
├── mne_icon-cropped.png            # 裁剪版图标
├── mne_splash.png                  # 启动画面
├── toolbar_*.png                   # 工具栏图标
├── dark/                           # 暗色主题图标
│   ├── index.theme
│   └── actions/
│       ├── clear.svg
│       ├── folder.svg
│       ├── help.svg
│       ├── movie.svg
│       ├── pause.svg
│       ├── play.svg
│       ├── reset.svg
│       ├── restore.svg
│       ├── scale.svg
│       ├── screenshot.svg
│       ├── visibility_off.svg
│       └── visibility_on.svg
└── light/                          # 亮色主题图标
    ├── index.theme
    └── actions/
        └── (同上)
```

---

### 2. 图标用途

**README 内容**: `mne/icons/README.rst`

```
Documentation
=============

The icons are used in ``mne/viz/_brain/_brain.py`` for the toolbar.
These Material design icons are provided by Google under the Apache 2.0 license.
```

**主要用于**:
1. **3D Brain 可视化工具栏** (`mne/viz/_brain/_brain.py`)
2. **Coregistration GUI**
3. **交互式绘图工具**

**图标功能映射**:
```python
图标文件                 → 功能
clear.svg              → 清除/重置
folder.svg             → 打开文件
help.svg               → 帮助文档
movie.svg              → 录制动画
pause.svg              → 暂停
play.svg               → 播放/开始
reset.svg              → 重置视图
restore.svg            → 恢复默认
scale.svg              → 缩放
screenshot.svg         → 截图
visibility_off.svg     → 隐藏
visibility_on.svg      → 显示
```

---

### 3. 在代码中使用图标

**示例**: Brain 可视化工具栏

```python
# 位置: mne/viz/_brain/_brain.py

from pathlib import Path

# 获取图标路径
icon_dir = Path(__file__).parent.parent.parent / 'icons'

# 根据主题选择
theme = 'dark'  # or 'light'
icon_path = icon_dir / theme / 'actions'

# 加载图标
play_icon = str(icon_path / 'play.svg')
pause_icon = str(icon_path / 'pause.svg')
screenshot_icon = str(icon_path / 'screenshot.svg')

# 在 PyQt/PySide 工具栏中使用
from qtpy import QtGui

play_action = toolbar.addAction(
    QtGui.QIcon(play_icon),
    'Play Animation'
)
```

---

## 应用场景

### 场景 1: 标准 EEG 实验设置

```python
import mne
import numpy as np

# 1. 创建实验数据
n_channels = 64
sfreq = 500
times = np.arange(0, 10, 1/sfreq)
data = np.random.randn(n_channels, len(times))

# 2. 创建 Info
ch_names = [f'EEG{i:03d}' for i in range(1, n_channels + 1)]
info = mne.create_info(ch_names, sfreq, ch_types='eeg')

# 3. 设置标准 montage
montage = mne.channels.make_standard_montage('biosemi64')
info.set_montage(montage)

# 4. 创建 Raw 对象
raw = mne.io.RawArray(data, info)

# 5. 可视化电极布局
fig = mne.viz.plot_sensors(
    raw.info,
    kind='topomap',
    show_names=True,
    title='BioSemi 64-Channel Layout'
)

# 6. 按脑区分析
from mne.channels import make_1020_channel_selections

selections = make_1020_channel_selections(raw.info, return_ch_names=True)

for region, channels in selections.items():
    print(f"{region}: {len(channels)} channels")
    # 区域特定分析...
```

---

### 场景 2: 自定义电极阵列

```python
import mne
import numpy as np

# 1. 定义自定义电极网格 (e.g., 8x8 网格)
n_rows, n_cols = 8, 8
spacing = 0.02  # 2cm 间距

ch_pos = {}
ch_names = []

for i in range(n_rows):
    for j in range(n_cols):
        ch_name = f'Grid{i}{j}'
        ch_names.append(ch_name)
        
        # 计算位置 (头顶中心为原点)
        x = (j - n_cols/2) * spacing
        y = (i - n_rows/2) * spacing
        z = 0.08  # 固定高度
        
        ch_pos[ch_name] = [x, y, z]

# 2. 创建 montage
montage_grid = mne.channels.make_dig_montage(
    ch_pos=ch_pos,
    coord_frame='head'
)

# 3. 可视化
fig = montage_grid.plot(
    kind='3d',
    show_names=False,
    sphere=(0, 0, 0, 0.095)
)

# 4. 应用到数据
info = mne.create_info(ch_names, sfreq=1000, ch_types='ecog')
info.set_montage(montage_grid)
```

---

### 场景 3: MRI-头部配准工作流

```python
import mne
from mne.gui import coregistration

# === 步骤 1: 准备 Freesurfer 解剖数据 ===
subjects_dir = 'path/to/subjects'
subject = 'sub-01'

# Freesurfer 重建应已完成:
# $ recon-all -s sub-01 -i T1.mgz -all

# === 步骤 2: 准备数字化数据 ===
raw = mne.io.read_raw_fif('sub-01_task-rest_raw.fif')

# 或单独读取数字化点
# dig = mne.io.read_fiducials('sub-01-fiducials.fif')

# === 步骤 3: 启动配准 GUI ===
coreg_ui = coregistration(
    inst=raw,
    subject=subject,
    subjects_dir=subjects_dir,
    head_opacity=0.8,
    head_high_res=True,
    show=True
)

# === 步骤 4: 在 GUI 中操作 ===
# a) 调整基准点 (Nasion, LPA, RPA)
# b) 使用 ICP (Iterative Closest Point) 拟合
# c) 检查拟合质量（距离分布）
# d) 保存变换: 'sub-01-trans.fif'

# === 步骤 5: 验证配准质量 ===
trans = mne.read_trans('sub-01-trans.fif')

# 可视化
fig = mne.viz.plot_alignment(
    raw.info,
    trans=trans,
    subject=subject,
    subjects_dir=subjects_dir,
    surfaces='head-dense',
    dig=True,
    eeg=True,
    meg=False
)

# === 步骤 6: 在源重建中使用 ===
src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir)
bem = mne.make_bem_solution(
    mne.make_bem_model(subject, subjects_dir=subjects_dir)
)

fwd = mne.make_forward_solution(
    raw.info,
    trans=trans,  # ← 使用配准结果
    src=src,
    bem=bem
)
```

---

### 场景 4: 高密度 EEG 地形图

```python
import mne
import numpy as np

# 1. 使用高密度 montage
montage = mne.channels.make_standard_montage('standard_1005')  # 343 通道

# 2. 创建模拟 ERP 数据
n_channels = len(montage.ch_names)
sfreq = 500
times = np.arange(-0.2, 0.6, 1/sfreq)

# 模拟 N170 成分 (枕颞区)
data = np.random.randn(n_channels, len(times)) * 1e-6

# 在 P7, P8 周围增加 N170
target_channels = ['P7', 'P8', 'PO7', 'PO8', 'TP7', 'TP8']
for ch in target_channels:
    if ch in montage.ch_names:
        idx = montage.ch_names.index(ch)
        # 170ms 处的负波
        peak_idx = np.argmin(np.abs(times - 0.17))
        gaussian = np.exp(-((times - 0.17)**2) / (2 * 0.02**2))
        data[idx] += -8e-6 * gaussian

# 3. 创建 Evoked 对象
info = mne.create_info(montage.ch_names, sfreq, ch_types='eeg')
info.set_montage(montage)
evoked = mne.EvokedArray(data, info, tmin=-0.2)

# 4. 绘制地形图序列
times_of_interest = [0.10, 0.14, 0.17, 0.20, 0.24]

fig = evoked.plot_topomap(
    times=times_of_interest,
    ch_type='eeg',
    time_unit='ms',
    ncols=5,
    nrows=1,
    cmap='RdBu_r',
    sensors=True,
    colorbar=True,
    size=3,
    title='N170 Component (High-Density EEG)'
)

# 5. 联合视图
fig = evoked.plot_joint(
    times=[0.17],
    title='N170 at 170ms',
    ts_args=dict(gfp=True, spatial_colors=True)
)
```

---

## 总结

### GUI 功能汇总

| 工具 | 位置 | 功能 | 场景 |
|------|------|------|------|
| **coregistration** | `gui/_gui.py` | MRI-头部配准 | 源定位前必需 |
| **CoregistrationUI** | `gui/_coreg.py` | 配准界面实现 | 交互式调整 |
| **_GUIScraper** | `gui/_gui.py` | 文档生成 | 自动截图 |

---

### 电极系统汇总

| 系统 | 电极数 | 文件 | 用途 | 典型场景 |
|------|--------|------|------|----------|
| **10-20** | 94+3 | `standard_1020.elc` | 基础系统 | 临床EEG |
| **10-10** | ~100 | - | 中等密度 | 认知研究 |
| **10-05** | 343+3 | `standard_1005.elc` | 高密度 | 高分辨率ERP |
| **BioSemi64** | 64+3 | `biosemi64.txt` | 商用系统 | 实验室标准 |
| **BioSemi128** | 128+3 | `biosemi128.txt` | 高密度商用 | 高级研究 |
| **EGI256** | 256 | `EGI_256.csd` | 超高密度 | 婴幼儿/高分辨率 |

*注: +3 指基准点 (Nasion, LPA, RPA)*

---

### 可视化工具汇总

| 函数 | 位置 | 输入 | 输出 | 主要用途 |
|------|------|------|------|----------|
| `plot_montage()` | `viz/montage.py` | DigMontage | Figure | Montage 可视化 |
| `plot_sensors()` | `viz/utils.py` | Info | Figure | 传感器布局 |
| `plot_ch_adjacency()` | `viz/utils.py` | adjacency | Figure | 邻接关系 |
| `plot_topomap()` | `viz/topomap.py` | data + Info | Figure | 地形图 |
| `plot_alignment()` | `viz/_3d.py` | trans + Info | Figure | 配准验证 |

---

### 核心设计模式

1. **分离数据与位置**:
   - Info: 通道信息（名称、类型）
   - Montage: 位置信息（3D 坐标）
   - 通过 `set_montage()` 关联

2. **标准化坐标系统**:
   - Head coordinate frame (头部坐标系)
   - Device coordinate frame (设备坐标系)
   - MRI coordinate frame (MRI 坐标系)

3. **可扩展架构**:
   - 支持自定义 montage
   - 支持多种文件格式
   - 插件式邻接矩阵

4. **交互式工作流**:
   - GUI 与脚本结合
   - 实时预览反馈
   - 配置持久化

---

### 最佳实践

1. **选择合适的电极系统**:
   - 临床应用 → 10-20
   - 标准 ERP 研究 → 64/128 通道系统
   - 高空间分辨率 → 10-05 或 256 通道

2. **配准质量检查**:
   - 基准点距离 < 5mm
   - 头部点平均距离 < 10mm
   - 目视检查对齐

3. **Montage 管理**:
   - 优先使用标准 montage
   - 自定义时记录详细元数据
   - 保存 montage 到项目目录

4. **可视化验证**:
   - 每次 `set_montage()` 后检查
   - 使用 3D 视图验证空间关系
   - 检查通道名匹配

---

**MNE-Python GUI 与电极系统文档完成！**

涵盖内容:
- ✅ Coregistration GUI 完整指南
- ✅ 10-20/10-05 电极系统详解
- ✅ 28 种内置 Montage
- ✅ 电极可视化工具
- ✅ 图标资源说明
- ✅ 4 个完整应用场景
