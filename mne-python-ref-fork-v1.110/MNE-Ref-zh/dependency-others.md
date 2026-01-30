# MNE-Python 其他重要依赖分析

> **可选依赖总览**: 25+ 包  
> **安装方式**: `pip install mne[full]`  
> **角色**: 可视化、3D 渲染、数据格式、并行计算

---

## 目录

1. [可视化依赖](#可视化依赖)
2. [数据格式与 I/O](#数据格式与-io)
3. [并行与性能](#并行与性能)
4. [神经影像支持](#神经影像支持)
5. [交互与 GUI](#交互与-gui)
6. [其他工具库](#其他工具库)

---

## 可视化依赖

### 1. Matplotlib - 2D 绘图 (必需)

**依赖声明**: `matplotlib >= 3.8, < 4`

**使用频率**: 🔥🔥🔥🔥🔥 (100% 可视化代码)

**核心位置**: `mne/viz/`

```python
import matplotlib.pyplot as plt
from matplotlib import patches, gridspec

# MNE 使用 matplotlib 的模块
- matplotlib.pyplot        # 核心绘图
- matplotlib.patches       # 图形对象 (Circle, Rectangle, Polygon)
- matplotlib.gridspec      # 复杂布局
- matplotlib.colors        # 颜色映射
- matplotlib.animation     # 动画
- matplotlib.backends      # 后端切换
```

**典型应用**:
```python
# 1. 时域波形图
evoked.plot()  # 内部使用 plt.plot()

# 2. 地形图
mne.viz.plot_topomap(data, info)
# -> matplotlib patches.Circle + imshow

# 3. 频谱图
epochs.plot_psd()  # plt.semilogy()

# 4. 联合图 (Joint plot)
evoked.plot_joint()  # gridspec 复杂布局
```

---

### 2. PyVista - 3D 可视化 (可选)

**依赖声明**: `pyvista >= 0.43, < 1`

**使用频率**: 🔥🔥🔥🔥 (3D 大脑、源空间)

**核心位置**: `mne/viz/_brain/`, `mne/viz/_3d.py`

```python
import pyvista as pv

# MNE 3D 渲染流程
class Brain:
    def __init__(self, ...):
        # 初始化 PyVista Plotter
        self._renderer = pv.Plotter(
            window_size=(800, 600),
            notebook=False
        )
    
    def add_data(self, array, ...):
        # 添加大脑网格
        mesh = pv.read('lh.pial')  # 读取 FreeSurfer 表面
        mesh['data'] = array       # 绑定数据
        
        # 渲染
        self._renderer.add_mesh(
            mesh, 
            scalars='data',
            cmap='hot',
            opacity=1.0
        )
```

**应用场景**:
- 大脑皮层激活图: `stc.plot()`
- 传感器位置: `mne.viz.plot_sensors_3d()`
- 源空间: `mne.viz.plot_source_estimates()`

**替代方案**:
- **mayavi** (老版本 MNE 使用，已弃用)
- **plotly** (基于 WebGL，MNE 部分支持)

---

## 数据格式与 I/O

### 1. H5py - HDF5 文件 (可选)

**依赖声明**: `h5py`

**使用频率**: 🔥🔥 (MEG/EEG 数据存储)

**核心位置**: `mne/io/_read_raw.py`

```python
import h5py

# 读取 HDF5 格式 MEG 数据
with h5py.File('data.h5', 'r') as f:
    data = f['dataset'][:]
    attrs = dict(f.attrs)
```

**应用**:
- CTF MEG 数据
- FieldTrip 格式
- MNE 自定义 HDF5 导出

---

### 2. Pandas - 数据表格 (可选)

**依赖声明**: `pandas >= 2.1`

**使用频率**: 🔥🔥🔥 (Epochs, Metadata)

**核心位置**: `mne/epochs.py`, `mne/io/kit/`

```python
import pandas as pd

# 1. Epochs Metadata
epochs = mne.Epochs(raw, events, metadata=df)
# df: pandas.DataFrame with columns ['subject', 'condition', ...]

epochs.metadata.query("condition == 'face'")

# 2. to_data_frame() 导出
df = epochs.to_data_frame()
# 返回 pandas DataFrame (长格式或宽格式)

# 3. 事件统计
events_df = pd.DataFrame(events, columns=['sample', 'prev', 'event_id'])
counts = events_df['event_id'].value_counts()
```

---

### 3. Nibabel - 神经影像格式 (可选)

**依赖声明**: `nibabel >= 5.2.0`

**使用频率**: 🔥🔥🔥 (MRI, NIfTI)

**核心位置**: `mne/source_space/`, `mne/transforms.py`

```python
import nibabel as nib

# 1. 读取 NIfTI MRI
mri = nib.load('T1.nii.gz')
data = mri.get_fdata()  # numpy array
affine = mri.affine     # 仿射变换矩阵

# 2. FreeSurfer 表面
from nibabel.freesurfer import read_geometry
coords, faces = read_geometry('lh.white')

# 3. 体积源空间
from mne.source_space import setup_volume_source_space
src = setup_volume_source_space(
    'subject',
    mri='T1.mgz',  # Nibabel 读取
    pos=5.0
)
```

---

### 4. PyMatReader - MATLAB 文件 (可选)

**依赖声明**: `pymatreader`

**使用频率**: 🔥🔥 (FieldTrip, EEGLAB)

**核心位置**: `mne/io/fieldtrip/`, `mne/io/eeglab/`

```python
from pymatreader import read_mat

# 读取 MATLAB .mat 文件
data = read_mat('eeg_data.mat')

# FieldTrip 结构体
ft_data = data['data']
# 包含 'trial', 'time', 'label', 'fsample' 等字段
```

---

## 并行与性能

### 1. Joblib - 并行计算 (可选)

**依赖声明**: `joblib >= 1.2.0`

**使用频率**: 🔥🔥🔥🔥 (并行循环)

**核心位置**: `mne/parallel.py`

```python
from joblib import Parallel, delayed

# MNE 并行函数
from mne.parallel import parallel_func

def process_epoch(epoch):
    # 处理单个 epoch
    return epoch.mean(axis=0)

# 并行处理
parallel, p_func, n_jobs = parallel_func(
    process_epoch, 
    n_jobs=4
)

results = parallel(
    p_func(epochs[i]) 
    for i in range(len(epochs))
)
```

**应用场景**:
- Epochs 并行处理
- 源重建 (逐时间点)
- 交叉验证

**原理**:
- 使用进程池 (multiprocessing)
- 共享内存优化 (memmap)

---

### 2. Numba - JIT 编译 (可选)

**依赖声明**: `numba >= 0.58.0`

**使用频率**: 🔥🔥 (加速关键循环)

**核心位置**: `mne/utils/_numba.py`

```python
from numba import jit

@jit(nopython=True, cache=True)
def fast_cross_3d(x, y):
    """加速 3D 叉积"""
    z = np.empty(3)
    z[0] = x[1] * y[2] - x[2] * y[1]
    z[1] = x[2] * y[0] - x[0] * y[2]
    z[2] = x[0] * y[1] - x[1] * y[0]
    return z

# 应用在几何计算中
# mne/surface.py: 表面法向量计算
```

**性能提升**:
- 几何计算: 10-50x
- 矩阵操作: 2-5x
- 依赖 LLVM

---

## 神经影像支持

### 1. Nilearn - fMRI 分析 (可选)

**依赖声明**: `nilearn`

**使用频率**: 🔥🔥 (fMRI-EEG 融合)

**核心位置**: `examples/`

```python
from nilearn import datasets, plotting

# 加载大脑图谱
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

# 绘制源空间结果到体积
plotting.plot_stat_map(
    stat_img,
    bg_img=atlas.maps,
    threshold=3.0,
    display_mode='z',
    cut_coords=5
)
```

---

### 2. Dipy - 扩散成像 (可选)

**依赖声明**: `dipy`

**使用频率**: 🔥 (DTI, 纤维追踪)

**核心位置**: `examples/`

```python
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel

# 用于 MEG-DTI 联合分析
# 白质纤维约束源空间
```

---

## 交互与 GUI

### 1. Qt - 图形界面 (可选)

**依赖声明**: `qtpy`, `PyQt6` / `PySide6`

**使用频率**: 🔥🔥🔥 (图形化工具)

**核心位置**: `mne/gui/`

```python
from qtpy import QtWidgets, QtCore

# MNE GUI 工具
mne.gui.coregistration()  # 配准界面
mne.gui.locate_ieeg()     # iEEG 定位

# 自动选择后端
# PyQt6 > PySide6 > PyQt5 > PySide2
```

**GUI 工具**:
- `mne coreg`: 头部-MRI 配准
- `mne browse_raw`: 原始数据浏览器
- `mne kit2fiff`: KIT MEG 转换器

---

### 2. IPython / Jupyter (可选)

**依赖声明**: `ipython`, `ipywidgets`, `ipympl`

**使用频率**: 🔥🔥🔥 (交互式绘图)

**核心位置**: `mne/viz/`

```python
# 1. Jupyter 自动检测
import mne
mne.viz.set_browser_backend('matplotlib')  # Jupyter 交互

# 2. IPython 小部件
epochs.plot(block=False)  # 非阻塞绘图

# 3. ipympl (交互式 matplotlib)
%matplotlib widget
evoked.plot()  # 可缩放、平移
```

---

## 其他工具库

### 1. Pooch - 数据下载 (必需)

**依赖声明**: `pooch >= 1.5`

**使用频率**: 🔥🔥🔥🔥 (示例数据)

**核心位置**: `mne/datasets/`

```python
import pooch

# MNE 数据集管理
sample_data = mne.datasets.sample.data_path()
# -> pooch 自动下载、验证 SHA256、解压

# 自定义数据集
GOODBOY = pooch.create(
    path=pooch.os_cache("mne"),
    base_url="https://osf.io/...",
    registry={
        "sample_audvis_raw.fif": "sha256:abcd1234...",
    }
)
```

---

### 2. Tqdm - 进度条 (必需)

**依赖声明**: `tqdm`

**使用频率**: 🔥🔥🔥 (长时间操作)

**核心位置**: `mne/utils/progressbar.py`

```python
from tqdm.auto import tqdm

# MNE 进度条包装
for i in tqdm(range(n_epochs), desc='Processing'):
    # 处理 epoch
    ...

# 自动选择:
# - Jupyter: ipywidgets 进度条
# - 终端: ASCII 进度条
```

---

### 3. Jinja2 - 模板引擎 (必需)

**依赖声明**: `jinja2`

**使用频率**: 🔥🔥🔥 (HTML 报告)

**核心位置**: `mne/report/`, `mne/html_templates/`

```python
from jinja2 import Environment, FileSystemLoader

# 生成 HTML 报告
report = mne.Report()
report.add_evokeds(evokeds)
report.save('report.html')

# 内部使用 Jinja2 模板
# mne/html_templates/report.html.jinja
```

---

### 4. Lazy_loader - 延迟导入 (必需)

**依赖声明**: `lazy_loader >= 0.3`

**使用频率**: 🔥🔥🔥 (加速启动)

**核心位置**: `mne/__init__.py`

```python
import lazy_loader as lazy

# 延迟导入子模块
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

# 好处:
# - import mne 快速 (不加载所有子模块)
# - 使用时才导入 (mne.viz, mne.preprocessing)
```

---

### 5. Decorator - 装饰器工具 (必需)

**依赖声明**: `decorator`

**使用频率**: 🔥🔥 (函数装饰器)

**核心位置**: `mne/utils/`

```python
from decorator import decorator

@decorator
def verbose(func, *args, **kwargs):
    """保留函数签名的装饰器"""
    # ... 日志逻辑
    return func(*args, **kwargs)

# 保留原函数的 __doc__, __name__, __signature__
```

---

## 依赖关系图

```
MNE-Python 依赖层次

┌─────────────────────────────────────────────────┐
│                   MNE-Python                    │
└─────────────────────────────────────────────────┘
                     ▲
                     │
┌────────────────────┴────────────────────────────┐
│                                                  │
▼                                                  ▼
┌──────────────────┐                   ┌──────────────────┐
│   必需依赖 (9)   │                   │  可选依赖 (25+)  │
│                  │                   │                  │
│ • NumPy          │                   │ • scikit-learn   │
│ • SciPy          │                   │ • PyVista        │
│ • Matplotlib     │                   │ • Pandas         │
│ • pooch          │                   │ • Nibabel        │
│ • tqdm           │                   │ • Nilearn        │
│ • jinja2         │                   │ • Joblib         │
│ • lazy_loader    │                   │ • Numba          │
│ • packaging      │                   │ • H5py           │
│ • decorator      │                   │ • Qt (PyQt6)     │
└──────────────────┘                   │ • IPython        │
                                        │ • ...            │
                                        └──────────────────┘
```

---

## 安装建议

### 1. 最小安装 (仅必需依赖)

```bash
pip install mne
```

**功能**:
- ✅ I/O (FIF, EDF, BrainVision, ...)
- ✅ 预处理 (滤波、ICA、重参考)
- ✅ Epochs, Evoked
- ✅ 2D 可视化 (Matplotlib)
- ❌ 3D 可视化
- ❌ 解码分析
- ❌ MRI 支持

---

### 2. 完整安装 (所有可选依赖)

```bash
pip install mne[full]
```

**功能**:
- ✅ 所有最小安装功能
- ✅ 3D 可视化 (PyVista)
- ✅ 解码分析 (scikit-learn)
- ✅ MRI 支持 (Nibabel)
- ✅ 并行计算 (Joblib)
- ✅ GUI 工具 (Qt)

---

### 3. 按需安装

```bash
# 3D 可视化
pip install mne pyvista

# 解码分析
pip install mne scikit-learn

# MRI 处理
pip install mne nibabel nilearn

# 性能优化
pip install mne numba joblib
```

---

## 总结

| 依赖类别 | 包数量 | 必需性 | 主要用途 |
|---------|--------|--------|---------|
| **核心计算** | 2 | ✅ 必需 | NumPy, SciPy |
| **可视化** | 2 | ✅/⚠️ | Matplotlib (必需), PyVista (可选) |
| **机器学习** | 1 | ⚠️ 可选 | scikit-learn |
| **数据格式** | 4 | ⚠️ 可选 | H5py, Pandas, Nibabel, PyMatReader |
| **并行性能** | 2 | ⚠️ 可选 | Joblib, Numba |
| **神经影像** | 2 | ⚠️ 可选 | Nilearn, Dipy |
| **GUI** | 1 | ⚠️ 可选 | Qt (PyQt6/PySide6) |
| **工具** | 5 | ✅ 必需 | pooch, tqdm, jinja2, lazy_loader, decorator |

**总计**: 9 必需 + 25+ 可选

---

**返回**: [依赖分析总览](dependency-analysis-overview.md)  
**相关**: [NumPy 分析](dependency-numpy.md) | [SciPy 分析](dependency-scipy.md) | [scikit-learn 分析](dependency-sklearn.md)
