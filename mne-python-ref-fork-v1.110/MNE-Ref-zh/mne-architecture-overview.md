# MNE-Python 架构与设计全景解析

> **版本**: 基于 MNE-Python 最新开发版  
> **文档日期**: 2026年1月30日  
> **许可**: BSD-3-Clause  

---

## 目录

1. [项目概述](#项目概述)
2. [核心数据模型](#核心数据模型)
3. [模块架构体系](#模块架构体系)
4. [算法生态系统](#算法生态系统)
5. [模块间协作机制](#模块间协作机制)
6. [技术栈与依赖](#技术栈与依赖)
7. [设计模式与最佳实践](#设计模式与最佳实践)
8. [扩展性与生态](#扩展性与生态)

---

## 项目概述

### 使命与定位

MNE-Python 是一个开源的 Python 包，用于探索、可视化和分析人类神经生理数据，包括：
- **MEG** (脑磁图)
- **EEG** (脑电图)
- **sEEG** (立体定向脑电图)
- **ECoG** (皮层脑电图)
- **fNIRS** (功能性近红外光谱)
- **Eye-tracking** (眼动追踪)

### 核心能力

```
数据流处理管道:
Raw Data → Preprocessing → Epoching → Evoked → Source Estimation → Statistics
   ↓           ↓              ↓          ↓            ↓                ↓
  I/O      伪迹去除        事件提取    平均化    正/逆向建模      假设检验
```

### 关键特性

- **全流程覆盖**: 从原始数据读取到源空间统计分析
- **多模态支持**: 30+ 种数据格式
- **算法丰富**: 涵盖信号处理、机器学习、统计推断
- **可视化强大**: 交互式数据探索
- **可扩展性**: 模块化设计，易于集成新算法

---

## 核心数据模型

### 数据对象层次结构

```
数据容器基类
├── TimeMixin              # 时间轴管理
├── SizeMixin              # 数据大小计算
├── ContainsMixin          # 通道类型检查
├── ProjMixin              # 信号空间投影
├── FilterMixin            # 滤波操作
└── SpectrumMixin          # 频谱分析
```

### 1. Raw（原始连续数据）

**类定义**: `BaseRaw` → 各格式特化（如 `RawFIF`, `RawEDF`）

**核心属性**:
```python
class BaseRaw:
    info: Info               # 元数据（通道、采样率、坐标等）
    _data: ndarray          # 时间序列数据 [n_channels × n_times]
    annotations: Annotations # 时间标注（伪迹、刺激等）
    first_samp: int         # 第一个样本的时间戳
    last_samp: int          # 最后一个样本的时间戳
```

**关键方法**:
- `filter()`: 时域/频域滤波
- `resample()`: 重采样
- `crop()`: 时间裁剪
- `pick()`: 通道选择
- `apply_function()`: 自定义变换
- `compute_psd()`: 功率谱密度
- `plot()`: 时间序列可视化

**设计特点**:
- **延迟加载**: `preload=False` 时按需读取，节省内存
- **分段文件**: 支持大文件自动分割（split files）
- **链式操作**: 方法返回 self，支持 `raw.filter().resample().crop()`

---

### 2. Epochs（事件相关数据段）

**类定义**: `BaseEpochs` → `Epochs`, `EpochsArray`, `EpochsFIF`

**核心属性**:
```python
class BaseEpochs:
    info: Info
    events: ndarray         # 事件矩阵 [n_events × 3]: (sample, prev_id, event_id)
    event_id: dict          # 事件标签映射 {'condition': event_code}
    tmin, tmax: float       # Epoch 时间窗口
    baseline: tuple         # 基线校正参数
    _data: ndarray          # [n_epochs × n_channels × n_times]
    metadata: DataFrame     # 与 trials 相关的元数据
    selection: ndarray      # 保留的 epoch 索引（用于 drop）
```

**关键方法**:
- `average()`: 计算诱发响应（Evoked）
- `drop_bad()`: 基于阈值的伪迹剔除
- `equalize_event_counts()`: 平衡各条件的 trial 数
- `crop()`, `resample()`, `decimate()`: 时间操作
- `apply_baseline()`: 基线校正
- `to_data_frame()`: 转为 Pandas DataFrame

**关键特性**:
- **拒绝参数**: 自动化质量控制（`reject`, `flat`）
- **元数据集成**: 支持复杂实验设计分析
- **事件语义**: 通过 `event_id` 进行条件标记

---

### 3. Evoked（诱发/平均响应）

**类定义**: `Evoked`, `EvokedArray`

**核心属性**:
```python
class Evoked:
    info: Info
    data: ndarray           # [n_channels × n_times]
    times: ndarray          # 时间向量
    nave: int               # 平均的 epoch 数量
    kind: str               # 'average' | 'standard_error'
    comment: str            # 条件描述
```

**关键方法**:
- `plot()`, `plot_topomap()`, `plot_joint()`: 多视角可视化
- `apply_baseline()`: 基线校正
- `crop()`, `resample()`: 时间处理
- `get_peak()`: 峰值检测

**应用场景**:
- ERP/ERF 分析（P300, N400, MMN 等）
- 条件对比（差异波）
- 源空间定位的输入

---

### 4. SourceEstimate（源空间估计）

**类层次**:
```
_BaseSourceEstimate
├── _BaseSurfaceSourceEstimate
│   ├── SourceEstimate           # 标量，皮层表面
│   └── VectorSourceEstimate     # 矢量，皮层表面
├── _BaseVolSourceEstimate
│   ├── VolSourceEstimate        # 标量，体积源
│   └── VolVectorSourceEstimate  # 矢量，体积源
└── MixedSourceEstimate          # 混合（表面+体积）
```

**核心属性**:
```python
class SourceEstimate:
    data: ndarray           # [n_vertices × n_times] 或 [n_dipoles × 3 × n_times]
    vertices: list          # 每个半球的顶点索引 [lh_vertices, rh_vertices]
    tmin: float
    tstep: float
    subject: str            # FreeSurfer subject ID
```

**关键方法**:
- `morph()`: 跨被试形变
- `save()`: 保存为 .stc/.w/.h5 格式
- `plot()`, `plot_3d()`: 大脑表面可视化
- `extract_label_time_course()`: ROI 时间序列
- `in_label()`: 标签内数据提取

**格式支持**:
- `.stc`: FreeSurfer 标准格式
- `.w`: FreeSurfer w 文件
- `.h5`: HDF5 高效存储

---

### 5. Info（元数据容器）

**结构**: 字典式对象，包含所有测量信息

**关键字段**:
```python
Info:
    chs: list[dict]         # 通道信息（位置、类型、单位等）
    sfreq: float            # 采样率
    meas_date: datetime     # 测量时间
    dev_head_t: Transform   # 设备到头部坐标变换
    dig: list[dict]         # 数字化点（头部形状、电极位置）
    projs: list[Projection] # SSP 投影向量
    bads: list[str]         # 坏通道列表
    subject_info: dict      # 被试信息
    device_info: dict       # 设备信息
```

**作用**:
- 所有数据对象的元数据核心
- 坐标系转换的基础
- 前向/逆向建模的必需信息

---

### 6. Forward（前向模型）

**核心属性**:
```python
Forward:
    sol: dict               # 前向解矩阵 [n_channels × n_sources]
    source_nn: ndarray      # 源的法向量
    source_rr: ndarray      # 源的位置 [n_sources × 3]
    src: SourceSpaces       # 源空间定义
    mri_head_t: Transform   # MRI 到头部坐标变换
    info: Info              # 传感器信息
```

**用途**:
- 逆问题求解的基础（MNE, dSPM, LCMV 等）
- 将源空间活动投影到传感器空间
- 仿真数据生成

**构建流程**:
```
MRI → 头模型(BEM) → 源空间 → 前向算子
      FreeSurfer   SourceSpaces   Forward
```

---

### 7. Covariance（协方差矩阵）

**核心属性**:
```python
Covariance:
    data: ndarray           # [n_channels × n_channels]
    ch_names: list          # 通道名
    nfree: int              # 自由度
    method: str             # 估计方法
    eig: ndarray            # 特征值（对角化后）
    eigvec: ndarray         # 特征向量
```

**用途**:
- 噪声协方差估计（逆问题正则化）
- 数据协方差（空间滤波器设计）
- 白化（数据标准化）

**估计方法**:
- `empirical`: 样本协方差
- `shrunk`: 收缩估计（Ledoit-Wolf）
- `diagonal`: 对角协方差
- `factor_analysis`: 因子分析

---

## 模块架构体系

### 整体架构图

```
                    ┌─────────────────────────────────────┐
                    │          User Interface              │
                    │    (Jupyter, Scripts, CLI)          │
                    └──────────────┬──────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
   ┌────▼─────┐            ┌──────▼──────┐          ┌───────▼────────┐
   │   I/O    │            │ Preprocessing│          │  Visualization │
   │  30+格式  │            │  伪迹去除     │          │   交互式图表    │
   └────┬─────┘            └──────┬──────┘          └───────┬────────┘
        │                         │                          │
        └──────────┬──────────────┴──────────────────────────┘
                   │
        ┌──────────▼─────────────────────────────────┐
        │        Core Data Objects                    │
        │  Raw │ Epochs │ Evoked │ SourceEstimate    │
        └──────────┬─────────────────────────────────┘
                   │
        ┌──────────┴─────────────┬────────────────────┐
        │                        │                     │
   ┌────▼─────┐          ┌──────▼──────┐      ┌──────▼──────┐
   │ Forward  │          │   Inverse   │      │Time-Frequency│
   │Modeling  │          │  Solutions  │      │   Analysis   │
   └────┬─────┘          └──────┬──────┘      └──────┬──────┘
        │                       │                     │
        └───────────┬───────────┴─────────────────────┘
                    │
        ┌───────────▼─────────────────────────┐
        │      Statistical Analysis            │
        │  (Clustering, Permutation, GLM)     │
        └─────────────────────────────────────┘
```

### 模块详解

#### 1. **I/O 模块** (`mne.io`)

**职责**: 多格式数据读写的统一接口

**支持格式** (30+):
```python
# 商业系统
- FIF (Neuromag/MEGIN)
- CTF (MEG)
- BTi/4D (MEG)
- KIT (Yokogawa MEG)

# EEG 格式
- BrainVision (.vhdr/.eeg/.vmrk)
- EDF/EDF+ (European Data Format)
- BDF (BioSemi)
- EEGLAB (.set/.fdt)
- Curry (.cdt/.dap)
- CNT (Neuroscan)
- EGI (.mff/.raw)
- Persyst
- BESA

# 其他模态
- SNIRF (fNIRS)
- EyeLink (.asc, 眼动)
- Neuralynx (.ncs, 动物电生理)

# 通用格式
- FieldTrip
- Array (NumPy/Pandas)
```

**统一接口**:
```python
# 所有格式统一通过 read_raw_xxx() 函数
raw = mne.io.read_raw_fif('data.fif')
raw = mne.io.read_raw_edf('data.edf')
raw = mne.io.read_raw_brainvision('data.vhdr')

# 自动检测格式
raw = mne.io.read_raw('data.edf')  # 智能识别
```

**设计亮点**:
- **BaseRaw 抽象**: 所有格式继承统一接口
- **延迟加载**: 处理 TB 级数据集
- **格式转换**: 透明的内部表示

---

#### 2. **Preprocessing 模块** (`mne.preprocessing`)

**子系统**:
```
preprocessing/
├── 伪迹去除
│   ├── ICA (独立成分分析)
│   ├── SSP (信号空间投影)
│   └── Regression (回归方法)
├── MEG 专用
│   ├── Maxwell Filter (SSS/tSSS)
│   └── Fine Calibration
├── EEG 专用
│   ├── CSD (电流源密度)
│   └── Xdawn (空间滤波)
├── 伪迹检测
│   ├── ECG/EOG 检测
│   ├── 肌电检测
│   └── 运动检测
└── 模态专用
    ├── nirs/ (fNIRS 预处理)
    ├── eyetracking/ (眼动预处理)
    └── ieeg/ (颅内脑电)
```

**核心算法** (详见 `preprocessing-module-overview.md`):
- ICA: FastICA, Infomax, Picard, JADE
- Maxwell: 球面谐波展开 (SSS)
- CSD: 球面样条拉普拉斯算子
- Xdawn: 最小二乘 ERP 增强

---

#### 3. **Forward 模块** (`mne.forward`)

**职责**: 从源空间到传感器空间的正向建模

**核心组件**:
```python
# 1. 源空间构建
src = mne.setup_source_space(subject, spacing='oct6')  # 皮层表面
src_vol = mne.setup_volume_source_space(subject)      # 体积源

# 2. 头模型 (BEM)
model = mne.make_bem_model(subject)
bem = mne.make_bem_solution(model)

# 3. 前向算子计算
fwd = mne.make_forward_solution(
    info, trans, src, bem, 
    meg=True, eeg=True, mindist=5.0
)
```

**算法**:
- **MEG**: 磁偶极子模型（Sarvas 公式）
- **EEG**: 边界元法 (BEM) / 有限元法 (FEM)
- **源方向**: 固定（法向）/ 自由（3D）

**输出**: `Forward` 对象 → 增益矩阵 $\mathbf{G}$:
$$
\mathbf{m} = \mathbf{G}\mathbf{s} + \mathbf{n}
$$
其中 $\mathbf{m}$ 是传感器测量，$\mathbf{s}$ 是源活动，$\mathbf{n}$ 是噪声。

---

#### 4. **Inverse 模块** (`mne.minimum_norm`, `mne.beamformer`, `mne.inverse_sparse`)

##### 4.1 最小范数估计 (`minimum_norm/`)

**方法族**:
- **MNE**: 最小 L2 范数
  $$\hat{\mathbf{s}} = \mathbf{G}^T(\mathbf{G}\mathbf{G}^T + \lambda\mathbf{C})^{-1}\mathbf{m}$$
  
- **dSPM**: 动态统计参数映射（归一化 MNE）
- **sLORETA**: 标准化低分辨率电磁层析成像
- **eLORETA**: 精确 LORETA

**核心函数**:
```python
# 创建逆算子
inverse_operator = mne.minimum_norm.make_inverse_operator(
    info, fwd, noise_cov, loose=0.2, depth=0.8
)

# 应用到数据
stc = mne.minimum_norm.apply_inverse(
    evoked, inverse_operator, lambda2=1/9, method='dSPM'
)
```

##### 4.2 波束形成 (`beamformer/`)

**方法**:
- **LCMV**: 线性约束最小方差
- **DICS**: 动态成像相干源（频域）
- **RAP-MUSIC**: 递归应用多信号分类
- **Scalar/Vector**: 标量/矢量波束形成

**优势**:
- 抑制相关噪声
- 时频特异性定位
- 无需强先验

**典型应用**:
```python
# 构建滤波器
filters = mne.beamformer.make_lcmv(
    evoked.info, fwd, data_cov, noise_cov, reg=0.05
)

# 应用到数据
stc = mne.beamformer.apply_lcmv(evoked, filters)
```

##### 4.3 稀疏逆解 (`inverse_sparse/`)

**方法**:
- **MCE**: 最小电流估计（L1 范数）
- **Gamma-MAP**: 伽马分布最大后验
- **MxNE**: 混合范数估计（时空稀疏）
- **TF-MxNE**: 时频混合范数

**特点**:
- 促进稀疏解（少数激活源）
- 适合局灶性活动
- 计算密集

---

#### 5. **Time-Frequency 模块** (`mne.time_frequency`)

**核心类**:
```python
# 频谱对象
Spectrum          # 单个频谱（Raw/Epochs/Evoked）
EpochsSpectrum    # 多 trial 频谱

# 时频对象
AverageTFR        # 平均时频表示
EpochsTFR         # 单 trial 时频
RawTFR            # 连续数据时频
```

**算法**:

1. **功率谱估计**:
   - Welch: 分段周期图平均
   - Multitaper: 多窗口方法（Slepian tapers）
   - FFT: 快速傅里叶变换

2. **时频分解**:
   - **Morlet 小波**: 
     $$\psi(t) = \pi^{-1/4}e^{i2\pi f_0 t}e^{-t^2/2\sigma^2}$$
   - **STFT**: 短时傅里叶变换
   - **Hilbert**: 解析信号包络
   - **Stockwell**: S变换（时频自适应）

3. **跨频耦合**:
   - **PAC**: 相位-幅度耦合
   - **PLV**: 相位锁定值
   - **Coherence**: 相干性

**示例**:
```python
# 诱发功率
power = mne.time_frequency.tfr_morlet(
    epochs, freqs=np.arange(7, 30, 3), 
    n_cycles=freqs/2, return_itc=False
)

# 诱发与总功率分解
power, itc = mne.time_frequency.tfr_morlet(
    epochs, freqs, n_cycles, return_itc=True
)
```

---

#### 6. **Stats 模块** (`mne.stats`)

**方法论**:

##### 6.1 聚类置换检验
```python
# 时空聚类
T_obs, clusters, p_values, H0 = mne.stats.spatio_temporal_cluster_test(
    X, threshold=dict(start=0, step=0.2), 
    n_permutations=1000, adjacency=adjacency
)
```

**原理**:
- 控制家族误差率 (FWER)
- 基于聚类质量统计量
- 适用于高维数据

##### 6.2 一般线性模型 (GLM)
```python
# 线性混合效应模型
from mne.stats import linear_regression

lm = linear_regression(epochs, design_matrix, names=['intercept', 'condition'])
```

##### 6.3 多重比较校正
- **FDR**: 错误发现率（Benjamini-Hochberg）
- **Bonferroni**: 保守校正
- **TFCE**: 无阈值聚类增强

---

#### 7. **Decoding 模块** (`mne.decoding`)

**核心组件**:

##### 7.1 时间解码
```python
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.linear_model import LogisticRegression

clf = SlidingEstimator(LogisticRegression(), scoring='roc_auc')
scores = cross_val_multiscore(clf, X, y, cv=5)
```

##### 7.2 时间泛化
```python
from mne.decoding import GeneralizingEstimator

gen = GeneralizingEstimator(LogisticRegression())
scores = cross_val_multiscore(gen, X, y, cv=5)  # 返回时间×时间矩阵
```

##### 7.3 空间滤波
- **CSP**: 共空间模式
- **SPoC**: 源功率协方差
- **xDAWN**: 前文已述

##### 7.4 感受野建模
```python
from mne.decoding import ReceptiveField

rf = ReceptiveField(tmin=-0.1, tmax=0.4, sfreq=1000)
rf.fit(speech_envelope, eeg_data)
rf.score(speech_test, eeg_test)
```

---

#### 8. **Visualization 模块** (`mne.viz`)

**可视化层次**:

```
交互式后端
├── matplotlib (默认)
├── PyVista (3D 大脑)
└── Qt (MNE-Qt-Browser)
```

**核心图表类型**:

1. **传感器空间**:
   - `plot_raw()`: 时间序列浏览
   - `plot_epochs()`: 单 trial 查看
   - `plot_topomap()`: 地形图
   - `plot_joint()`: 联合视图（波形+地形）
   - `plot_psd()`: 功率谱
   - `plot_tfr()`: 时频表示

2. **源空间**:
   - `plot_source_estimates()`: 大脑表面动画
   - `plot_volume_source_estimates()`: 体积源切片
   - `stc.plot_3d()`: 3D 交互式脑图

3. **统计**:
   - `plot_compare_evokeds()`: 条件对比
   - `plot_cluster_stats()`: 聚类检验结果

4. **报告**:
   - `Report`: 自动生成 HTML 报告

**设计哲学**:
- **即时反馈**: 数据对象自带 `.plot()` 方法
- **参数统一**: 一致的颜色、单位、坐标系
- **可定制**: 返回 matplotlib 对象，可进一步修改

---

#### 9. **Channels 模块** (`mne.channels`)

**功能**:
- **Montage**: 电极/传感器位置管理
- **Layout**: 2D 投影布局（用于地形图）
- **Interpolation**: 坏通道插值
- **Re-referencing**: EEG 重参考

**示例**:
```python
# 设置标准 montage
montage = mne.channels.make_standard_montage('biosemi64')
raw.set_montage(montage)

# 自定义 montage
montage_custom = mne.channels.make_dig_montage(
    ch_pos={'Fp1': [x, y, z], 'Fp2': [x2, y2, z2]},
    coord_frame='head'
)
```

---

#### 10. **Datasets 模块** (`mne.datasets`)

**内置数据集**:
- `sample`: MEG/EEG 听觉+视觉刺激
- `somato`: MEG 体感刺激
- `ssvep`: 稳态视觉诱发电位
- `multimodal`: MEG/EEG/fMRI
- `hf_sef`: 高频 SEF
- `fnirs_motor`: fNIRS 运动任务
- `eyelink`: 眼动追踪

**自动下载**:
```python
data_path = mne.datasets.sample.data_path()
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
```

---

## 算法生态系统

### 算法分类矩阵

| 领域 | 算法/方法 | 模块位置 | 论文基础 |
|------|----------|---------|---------|
| **信号分离** | ICA (FastICA, Infomax, Picard) | `preprocessing.ica` | Bell & Sejnowski 1995; Hyvärinen 1999 |
| | SSP (PCA-based) | `preprocessing.ssp` | Uusitalo & Ilmoniemi 1997 |
| | Maxwell Filter (tSSS) | `preprocessing.maxwell` | Taulu & Kajola 2005 |
| **逆问题** | MNE/dSPM/sLORETA | `minimum_norm` | Hämäläinen & Ilmoniemi 1994; Dale et al. 2000 |
| | LCMV/DICS Beamformer | `beamformer` | Van Veen et al. 1997; Gross et al. 2001 |
| | MCE/Gamma-MAP | `inverse_sparse` | Matsuura & Okabe 1995; Wipf & Nagarajan 2009 |
| **时频分析** | Morlet Wavelet | `time_frequency.tfr` | Tallon-Baudry et al. 1997 |
| | Multitaper | `time_frequency.psd` | Thomson 1982; Percival & Walden 1993 |
| | Hilbert Transform | `time_frequency` | Bruns 2004 |
| **统计推断** | Cluster Permutation | `stats.cluster_level` | Maris & Oostenveld 2007 |
| | TFCE | `stats` | Smith & Nichols 2009 |
| | GLM | `stats.regression` | Friston et al. 1994 |
| **机器学习** | CSP | `decoding.csp` | Ramoser et al. 2000 |
| | SPoC | `decoding` | Dähne et al. 2014 |
| | Sliding Estimator | `decoding` | King & Dehaene 2014 |
| **连接性** | Coherence/PLV/PLI | `mne-connectivity` (独立包) | Lachaux et al. 1999 |
| | Granger Causality | `mne-connectivity` | Ding et al. 2006 |

### 算法复杂度对比

| 算法 | 时间复杂度 | 空间复杂度 | 并行化 | 典型用时（sample数据） |
|------|-----------|-----------|-------|---------------------|
| ICA (FastICA) | O(n³p) | O(n²) | 否 | ~30秒 |
| Maxwell Filter | O(nt·L²) | O(L²) | 是 | ~5分钟 |
| MNE 逆解 | O(n²s + s²) | O(n·s) | 否 | <1秒 |
| LCMV Beamformer | O(n³) | O(n²) | 否 | ~10秒 |
| MxNE 稀疏逆解 | O(iter·n²s) | O(n·s) | 是 | ~5分钟 |
| 聚类置换检验 | O(perm·n·t·s) | O(n·t·s) | 是 | ~10分钟 |

*注: n=通道数, p=成分数, t=时间点, s=源点数, L=球面谐波阶数, perm=置换次数*

---

## 模块间协作机制

### 数据流转示例

#### 完整分析流程

```python
# ======= 1. 数据读取 (io) =======
raw = mne.io.read_raw_fif('raw.fif', preload=True)

# ======= 2. 预处理 (preprocessing, filter) =======
raw.filter(l_freq=1.0, h_freq=40.0)                    # filter
raw_sss = mne.preprocessing.maxwell_filter(raw)        # preprocessing.maxwell
ica = mne.preprocessing.ICA(n_components=20)           # preprocessing.ica
ica.fit(raw_sss)
ica.apply(raw_sss)

# ======= 3. 事件提取 (event) =======
events = mne.find_events(raw_sss, stim_channel='STI101')

# ======= 4. Epoching (epochs) =======
epochs = mne.Epochs(
    raw_sss, events, event_id={'auditory': 1, 'visual': 3},
    tmin=-0.2, tmax=0.5, baseline=(None, 0), preload=True
)

# ======= 5. 诱发响应 (evoked) =======
evoked_aud = epochs['auditory'].average()
evoked_vis = epochs['visual'].average()

# ======= 6. 源空间准备 (forward) =======
src = mne.setup_source_space(subject, spacing='oct6')  # forward
bem = mne.make_bem_solution(model)                     # bem
fwd = mne.make_forward_solution(                       # forward
    epochs.info, trans, src, bem
)

# ======= 7. 协方差估计 (cov) =======
noise_cov = mne.compute_covariance(                    # cov
    epochs, tmax=0, method='shrunk'
)
data_cov = mne.compute_covariance(
    epochs, tmin=0.04, tmax=0.15
)

# ======= 8. 逆问题求解 (minimum_norm 或 beamformer) =======
# 方案 A: MNE
inv = mne.minimum_norm.make_inverse_operator(          # minimum_norm
    epochs.info, fwd, noise_cov
)
stc_mne = mne.minimum_norm.apply_inverse(evoked_aud, inv)

# 方案 B: LCMV
filters = mne.beamformer.make_lcmv(                    # beamformer
    epochs.info, fwd, data_cov, noise_cov
)
stc_lcmv = mne.beamformer.apply_lcmv(evoked_aud, filters)

# ======= 9. 形变到模板大脑 (morph) =======
morph = mne.compute_source_morph(                      # morph
    stc_mne, subject_from=subject, subject_to='fsaverage'
)
stc_fsaverage = morph.apply(stc_mne)

# ======= 10. 统计检验 (stats) =======
X = [stc_aud, stc_vis]  # 两个条件的 stc 列表
T_obs, clusters, p_vals, H0 = mne.stats.spatio_temporal_cluster_test(
    X, threshold=3.0, n_permutations=1000
)

# ======= 11. 可视化 (viz) =======
stc_mne.plot(                                          # viz
    subjects_dir=subjects_dir, hemi='both', 
    time_viewer=True, backend='pyvistaqt'
)
```

### 模块依赖图

```
                    ┌──────────┐
                    │   io     │ (最底层，无依赖)
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │  utils   │ (工具函数)
                    └────┬─────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼─────┐    ┌────▼─────┐    ┌────▼─────┐
   │  filter  │    │ channels │    │ _fiff    │
   └────┬─────┘    └────┬─────┘    └────┬─────┘
        │               │               │
        └───────┬───────┴───────┬───────┘
                │               │
          ┌─────▼──────┐  ┌────▼─────┐
          │ preprocessing│ │  event   │
          └─────┬────────┘ └────┬─────┘
                │               │
                └───────┬───────┘
                        │
                  ┌─────▼──────┐
                  │   epochs   │
                  └─────┬──────┘
                        │
          ┌─────────────┼─────────────┐
          │             │             │
     ┌────▼─────┐  ┌───▼────┐  ┌────▼─────┐
     │  evoked  │  │time_freq│ │ decoding │
     └────┬─────┘  └────────┘  └──────────┘
          │
          ├──────┬─────────┬─────────┐
          │      │         │         │
    ┌─────▼──┐ ┌─▼──────┐┌▼────────┐│
    │ forward││ cov    ││beamformer││
    └─────┬──┘ └────┬──┘└─────────┘│
          │         │              │
          └────┬────┴──────────────┘
               │
        ┌──────▼─────────┐
        │ minimum_norm / │
        │inverse_sparse  │
        └──────┬─────────┘
               │
        ┌──────▼─────────┐
        │source_estimate │
        └──────┬─────────┘
               │
          ┌────▼─────┐
          │  stats   │
          └────┬─────┘
               │
          ┌────▼─────┐
          │   viz    │ (最顶层)
          └──────────┘
```

### Mixin 设计模式

MNE-Python 大量使用 **Mixin** 模式来实现功能复用：

```python
# 以 BaseRaw 为例
class BaseRaw(
    ProjMixin,           # SSP 投影相关方法
    ContainsMixin,       # 包含通道类型检查
    UpdateChannelsMixin, # 更新通道信息
    ReferenceMixin,      # EEG 重参考
    SetChannelsMixin,    # 设置通道属性
    InterpolationMixin,  # 插值坏通道
    TimeMixin,           # 时间相关方法
    SizeMixin,           # 大小计算
    FilterMixin,         # 滤波方法
    SpectrumMixin,       # 频谱分析
):
    # 核心 Raw 功能
    pass
```

**优势**:
- **关注点分离**: 每个 Mixin 负责一组相关功能
- **代码复用**: 同样的 Mixin 用于 Raw, Epochs, Evoked
- **可测试性**: 每个 Mixin 独立测试

**常见 Mixin**:
- `TimeMixin`: `crop()`, `time_as_index()`, `__len__()`
- `FilterMixin`: `filter()`, `notch_filter()`, `resample()`
- `ProjMixin`: `add_proj()`, `apply_proj()`, `del_proj()`

---

## 技术栈与依赖

### 核心依赖

```toml
# 必需依赖（最小安装）
Python >= 3.10
NumPy >= 1.26          # 数组运算
SciPy >= 1.11          # 科学计算（线性代数、信号处理）
Matplotlib >= 3.8      # 基础绘图
Pooch >= 1.5           # 数据集下载
tqdm                   # 进度条
Jinja2                 # HTML 模板
decorator              # 装饰器工具
lazy_loader >= 0.3     # 延迟导入
packaging              # 版本解析
```

### 可选依赖

#### 完整功能 (`mne[full]`)

```python
# GUI 与可视化
PyQt6 / PySide6        # Qt 后端
PyVista >= 0.32        # 3D 可视化
PyVistaQt              # Qt 集成
ipywidgets             # Jupyter 交互
ipympl                 # Matplotlib Jupyter 后端

# 高级分析
scikit-learn           # 机器学习（decoding）
pandas                 # 数据框架
numba                  # JIT 编译加速
dipy                   # 扩散成像（MRI 配准）
nibabel                # 神经影像格式 (NIfTI, MGH)

# 格式支持
h5py                   # HDF5 读写
pymatreader            # MATLAB 文件
edfio >= 0.4.10        # EDF/BDF
eeglabio               # EEGLAB .set
pybv                   # BrainVision
snirf                  # fNIRS

# 其他
joblib                 # 并行计算
mne-qt-browser         # 专用数据浏览器
mne-bids               # BIDS 格式支持
```

### 系统架构

```
┌─────────────────────────────────────────────┐
│           MNE-Python (Python)               │
├─────────────────────────────────────────────┤
│  NumPy/SciPy (C/Fortran BLAS/LAPACK)       │
├─────────────────────────────────────────────┤
│  Matplotlib (C/C++)                         │
├─────────────────────────────────────────────┤
│  PyVista → VTK (C++)                        │
├─────────────────────────────────────────────┤
│  Qt (C++)                                   │
└─────────────────────────────────────────────┘
```

**性能关键路径**:
- **线性代数**: 通过 NumPy 调用优化的 BLAS/LAPACK
- **信号处理**: SciPy 的 C/Fortran 实现
- **3D 渲染**: VTK 的 OpenGL 加速

---

## 设计模式与最佳实践

### 1. 链式操作 (Fluent Interface)

```python
# 方法返回 self，支持链式调用
raw.filter(1, 40).notch_filter(50).resample(250).crop(tmax=60)

# 但涉及数据修改时需 copy
raw_clean = raw.copy().filter(1, 40).apply_ica(ica)
```

### 2. 延迟计算 (Lazy Evaluation)

```python
# 延迟加载数据
raw = mne.io.read_raw_fif('huge_file.fif', preload=False)  # 不占内存
raw.filter(1, 40)  # 计算时才读取需要的片段

# 手动加载
raw.load_data()  # 现在加载到内存
```

### 3. 不可变性 (Immutability Preference)

```python
# 默认行为：修改操作返回新对象
epochs_clean = epochs.copy().drop_bad(reject=reject)

# 原地修改需显式指定
epochs.drop_bad(reject=reject)  # 修改 epochs 本身
```

### 4. 上下文管理器

```python
# 临时设置日志级别
with mne.use_log_level('ERROR'):
    # 只显示错误
    raw.filter(1, 40)

# 临时禁用进度条
with mne.utils.use_log_level(False):
    results = process_all_subjects()
```

### 5. 配置系统

```python
# 持久化配置
mne.set_config('MNE_DATA', '/path/to/data')
mne.set_config('SUBJECTS_DIR', '/path/to/freesurfer/subjects')

# 读取配置
data_path = mne.get_config('MNE_DATA')

# 查看所有配置
mne.sys_info()
```

### 6. 单元测试与 CI/CD

**测试策略**:
```python
# 每个模块都有对应的 tests/ 目录
mne/
├── preprocessing/
│   ├── ica.py
│   └── tests/
│       └── test_ica.py

# 使用 pytest 框架
@pytest.mark.parametrize('method', ['fastica', 'infomax', 'picard'])
def test_ica_methods(method):
    ica = ICA(n_components=2, method=method)
    ica.fit(raw)
    assert ica.n_components_ == 2
```

**CI 流程** (Azure Pipelines, GitHub Actions):
- 多 Python 版本测试 (3.10, 3.11, 3.12)
- 多操作系统 (Linux, macOS, Windows)
- 依赖版本矩阵（最小/最新依赖）
- 代码覆盖率 (Codecov)

### 7. 文档驱动开发

```python
# 所有公共函数都有详细文档
@verbose
def apply_inverse(evoked, inverse_operator, lambda2=1/9, method='dSPM'):
    """Apply inverse operator to evoked data.

    Parameters
    ----------
    evoked : Evoked
        The evoked data.
    inverse_operator : instance of InverseOperator
        The inverse operator.
    lambda2 : float
        The regularization parameter.
    method : 'MNE' | 'dSPM' | 'sLORETA'
        The inverse method to use.

    Returns
    -------
    stc : SourceEstimate
        The source estimate.

    References
    ----------
    .. [1] Dale et al. (2000). Dynamic Statistical Parametric Mapping.
           Neuron, 26(1), 55-67.
    """
```

**文档工具链**:
- Sphinx: 文档生成
- Sphinx-Gallery: 示例自动运行
- Napoleon: NumPy/Google docstring 支持
- Intersphinx: 跨项目引用

---

## 扩展性与生态

### MNE 生态系统

```
         ┌──────────────────────────────────┐
         │        MNE-Python (核心)          │
         └──────────────┬───────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
  ┌─────▼──────┐  ┌────▼─────┐  ┌─────▼──────┐
  │ MNE-BIDS   │  │MNE-Conn  │  │MNE-Realtime│
  │(BIDS格式)   │  │(连接性)   │  │(实时分析)   │
  └────────────┘  └──────────┘  └────────────┘
        │
  ┌─────▼──────┐  ┌────────────┐ ┌────────────┐
  │MNE-Qt-Brow │  │MNE-GUI-Add │ │MNE-NIRS    │
  │(数据浏览)   │  │(GUI工具)   │ │(fNIRS专用) │
  └────────────┘  └────────────┘ └────────────┘
```

#### 主要扩展包

1. **MNE-BIDS**:
   - 自动化 BIDS 数据集创建
   - BIDS → MNE 对象转换
   - 元数据管理

2. **MNE-Connectivity**:
   - 功能连接分析
   - 有效连接（Granger, PDC）
   - 网络分析

3. **MNE-Realtime**:
   - 实时数据流处理
   - 在线解码
   - 神经反馈

4. **MNE-Qt-Browser**:
   - 快速数据浏览
   - 批量坏段标记
   - 集成标注工具

### 与外部工具集成

#### FreeSurfer (解剖学处理)
```python
# MNE 依赖 FreeSurfer 的解剖重建
subjects_dir = '/path/to/freesurfer/subjects'
subject = 'sub-01'

# 使用 FreeSurfer 定义的表面
src = mne.setup_source_space(subject, spacing='ico5', subjects_dir=subjects_dir)
```

#### FSL/SPM (fMRI 集成)
```python
# 读取 fMRI ROI
labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
```

#### scikit-learn (机器学习)
```python
from sklearn.svm import SVC
from mne.decoding import SlidingEstimator

clf = SlidingEstimator(SVC(kernel='linear'))
clf.fit(X_train, y_train)
```

#### Pandas (数据分析)
```python
# 数据对象可转为 DataFrame
df = epochs.to_data_frame()

# Metadata 集成
epochs.metadata = pd.DataFrame({
    'subject': [...],
    'condition': [...],
    'rt': [...]
})
```

### 插件机制

**自定义读取器**:
```python
# 实现 BaseRaw 接口
class RawCustom(BaseRaw):
    def __init__(self, fname, **kwargs):
        # 读取自定义格式
        data, info = read_custom_format(fname)
        super().__init__(info, data, **kwargs)

# 自动集成到 MNE
raw = RawCustom('data.custom')
raw.filter(1, 40)  # 所有 BaseRaw 方法可用
```

**自定义逆解算法**:
```python
from mne.inverse_sparse import mixed_norm_solver

# 实现求解器接口
def my_custom_solver(M, G, alpha, **kwargs):
    # 自定义稀疏逆解算法
    X = solve(M, G, alpha)
    return X, residual

# 注册到 MNE
mne.inverse_sparse.register_solver('my_solver', my_custom_solver)
```

---

## 性能与可扩展性

### 内存管理策略

```python
# 1. 按需加载
raw = mne.io.read_raw_fif('big.fif', preload=False)  # 元数据：~KB
raw.crop(tmin=0, tmax=60)                           # 仍未加载数据
data_chunk = raw[:, 1000:2000]                      # 仅加载需要的片段

# 2. 内存映射
raw = mne.io.read_raw_fif('huge.fif', preload='mmap.dat')  # 磁盘缓存

# 3. 分块处理
for idx in range(0, len(raw.times), chunk_size):
    chunk = raw[:, idx:idx+chunk_size][0]
    process(chunk)
```

### 并行计算

```python
# 大多数计算密集型函数支持 n_jobs
# 内部使用 joblib 实现
stc = mne.beamformer.apply_lcmv_epochs(epochs, filters, n_jobs=4)

# 自定义并行
from mne.parallel import parallel_func
parallel, p_fun, n_jobs = parallel_func(my_func, n_jobs=4)
results = parallel(p_fun(data[i]) for i in range(len(data)))
```

### GPU 加速 (实验性)

```python
# CUDA 支持 (需要 cupy)
mne.cuda.init_cuda(verbose=True)

# 某些操作自动使用 GPU
epochs_tfr = mne.time_frequency.tfr_morlet(
    epochs, freqs, n_cycles, 
    use_fft=True, n_jobs='cuda'  # GPU 加速
)
```

---

## 总结与展望

### 架构优势

1. **模块化**: 清晰的功能分层，低耦合高内聚
2. **一致性**: 统一的 API 设计，学习曲线平滑
3. **可扩展**: 开放的插件系统，社区贡献友好
4. **文档化**: 完善的文档、教程、示例
5. **测试覆盖**: 高代码覆盖率，持续集成保障

### 当前发展方向

- **BIDS 标准化**: 与 BIDS 生态深度集成
- **实时处理**: 在线解码、神经反馈
- **深度学习**: 与 PyTorch/TensorFlow 集成
- **云计算**: 分布式处理支持
- **可重复性**: 自动化流程、容器化部署

### 学习路径建议

```
入门 → 基础分析 → 高级方法 → 方法开发
  ↓         ↓          ↓           ↓
教程     示例脚本    研究论文    源代码
```

**推荐资源**:
- 官方教程: https://mne.tools/stable/auto_tutorials/
- API 文档: https://mne.tools/stable/python_reference.html
- 论坛: https://mne.discourse.group/
- GitHub: https://github.com/mne-tools/mne-python

---

**文档维护者**: MNE-Python 中文社区  
**最后更新**: 2026年1月30日  
**反馈渠道**: GitHub Issues / MNE Discourse
