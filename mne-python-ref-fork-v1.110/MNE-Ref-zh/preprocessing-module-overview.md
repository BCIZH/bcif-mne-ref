# MNE-Python 预处理模块详解

## 模块概述

`mne.preprocessing` 是 MNE-Python 中用于神经电生理数据预处理的核心模块，提供了全面的伪迹检测、信号分离和数据清理功能。该模块主要针对脑电图（EEG）、脑磁图（MEG）、近红外光谱（fNIRS）和眼动追踪（Eye-tracking）等多模态神经影像数据。

## 主要功能分类

### 1. 独立成分分析（ICA）- `ica.py`

**核心算法：**
- **FastICA**: 基于非高斯性的独立成分分析
- **Infomax**: 基于最大信息传输的盲源分离算法
- **Extended Infomax**: 扩展版本，可处理超高斯和亚高斯源
- **JADE**: 联合对角化算法
- **Picard**: 预条件ICA用于实数数据

**主要功能：**
- `ICA`: ICA核心类，提供fit、apply、plot等方法
- `corrmap()`: 跨多个数据集识别相似的ICA成分
- `get_score_funcs()`: 获取自动化成分评分函数
- `ica_find_ecg_events()` / `ica_find_eog_events()`: 自动检测心电和眼电伪迹成分
- `read_ica()` / `read_ica_eeglab()`: 读取ICA模型

**应用场景：**
- 去除眼动伪迹（眨眼、眼跳）
- 去除心电伪迹
- 去除肌电伪迹
- 信号源分离

---

### 2. 信号空间投影（SSP）- `ssp.py`

**核心算法：**
- 基于主成分分析（PCA）的信号投影
- 构建正交投影向量来抑制伪迹源

**主要功能：**
- `compute_proj_ecg()`: 计算心电伪迹的投影向量
- `compute_proj_eog()`: 计算眼电伪迹的投影向量
- `compute_proj_hfc()`: 计算高频连续噪声的投影向量

**应用场景：**
- 实时数据处理（无需完整数据集）
- 与ICA互补的伪迹去除方法
- MEG/EEG环境噪声抑制

---

### 3. Maxwell滤波 - `maxwell.py`

**核心算法：**
- **信号空间分离（SSS）**: 基于球面谐波展开分解内外源
- **时间信号空间分离（tSSS）**: 结合时间信息提升性能
- **运动补偿（Movement compensation）**: 校正头部运动引起的信号变化

**主要功能：**
- `maxwell_filter()`: Maxwell滤波的主函数
- `maxwell_filter_prepare_emptyroom()`: 准备空房间数据用于噪声抑制
- `find_bad_channels_maxwell()`: 基于Maxwell滤波自动检测坏导
- `compute_maxwell_basis()`: 计算Maxwell基函数
- `compute_fine_calibration()`: 精细校准计算

**应用场景：**
- MEG数据噪声抑制
- 头部运动校正
- 坏导自动检测
- 外部磁场干扰去除

**技术细节：**
- 使用球面谐波展开（勒让德多项式）
- 区分内源（大脑信号）和外源（环境噪声）
- 支持单精度和双精度计算

---

### 4. 电流源密度（CSD）变换 - `_csd.py`

**核心算法：**
- **球面样条表面拉普拉斯算子**: 基于球形头模型
- **Perrin et al. (1987, 1989)** 的CSD算法实现

**主要功能：**
- `compute_current_source_density()`: 计算CSD变换
- `compute_bridged_electrodes()`: 检测电极桥接
- `interpolate_bridged_electrodes()`: 插值修复桥接电极

**应用场景：**
- 去除参考电极影响（无参考变换）
- 提高空间分辨率
- 减少容积传导效应
- 检测和修复EEG电极桥接问题

**数学基础：**
- 使用勒让德多项式（Legendre polynomials）
- 球面样条插值
- 正则化矩阵求逆（λ²参数控制平滑度）

---

### 5. Xdawn算法 - `xdawn.py`

**核心算法：**
- **Xdawn空间滤波**: 用于事件相关电位（ERP）增强
- 基于最小二乘估计的诱发响应提取

**主要功能：**
- `Xdawn`: Xdawn类，提供fit、transform、apply等方法
- 事件重叠校正
- 诱发响应增强

**应用场景：**
- P300检测（脑机接口）
- ERP信号增强
- 诱发电位分析
- 处理事件重叠问题

**技术特点：**
- 构建Toeplitz矩阵进行时域建模
- 最小二乘估计每个条件的独立诱发响应
- 协方差正则化处理

---

### 6. 伪迹检测与标注

#### 6.1 心电（ECG）处理 - `ecg.py`

**核心算法：**
- **QRS检测器**: Pan-Tompkins算法变体
- 自适应阈值检测

**主要功能：**
- `qrs_detector()`: QRS波检测
- `find_ecg_events()`: 查找心跳事件
- `create_ecg_epochs()`: 创建围绕心跳的epochs

**检测流程：**
1. 带通滤波（5-35 Hz）
2. 信号绝对值
3. 自适应阈值（可选"auto"）
4. 峰值检测

#### 6.2 眼电（EOG）处理 - `eog.py`

**主要功能：**
- `find_eog_events()`: 检测眼动事件
- `create_eog_epochs()`: 创建眼动相关epochs

#### 6.3 自动伪迹标注 - `artifact_detection.py`

**算法集合：**

1. **肌电伪迹检测** - `annotate_muscle_zscore()`
   - 频段：110-140 Hz（可配置）
   - 方法：包络Z分数检测
   - 阈值：默认Z>4

2. **运动伪迹检测** - `annotate_movement()`
   - 基于头部位置设备（HPI线圈）
   - 计算四元数旋转角度
   - 检测突然的头部移动

3. **中断检测** - `annotate_break()`
   - 检测数据采集中断
   - 基于时间戳跳跃

4. **幅度异常检测** - `annotate_amplitude()`
   - `_annotate_amplitude.py`: 峰值-峰值幅度检测
   - `_annotate_nan.py`: NaN值检测

5. **头部位置平均** - `compute_average_dev_head_t()`
   - 计算平均头部位置变换矩阵

---

### 7. 回归方法 - `_regress.py`

**核心算法：**
- **线性回归**: 使用参考通道预测和去除伪迹
- **EOG回归类**: `EOGRegression`

**主要功能：**
- `regress_artifact()`: 基于回归的伪迹去除
- `EOGRegression`: 专门用于EOG伪迹的回归类
- `read_eog_regression()`: 读取保存的回归模型

**应用场景：**
- 使用EOG通道去除眨眼伪迹
- 使用ECG通道去除心跳伪迹
- 自定义参考通道的伪迹去除

**方法特点：**
- 支持预计算beta系数
- 可选择是否应用SSP投影
- Gratton et al. (1983) 方法的实现

---

### 8. 近红外光谱（fNIRS）预处理 - `nirs/`

**专用算法：**

1. **光密度变换** - `optical_density()`
   - 将原始光强转换为光密度
   - 公式：OD = -log(I/I₀)

2. **Beer-Lambert定律** - `beer_lambert_law()`
   - 将光密度转换为血红蛋白浓度
   - 计算HbO（氧合血红蛋白）和HbR（脱氧血红蛋白）

3. **头皮耦合指数（SCI）** - `scalp_coupling_index()`
   - 评估光极与头皮的接触质量
   - 检测不良通道

4. **时间导数分布修复（TDDR）** - `temporal_derivative_distribution_repair()`
   - 去除运动伪迹
   - 基于信号时间导数的分布分析

**辅助功能：**
- `short_channels()`: 短通道处理
- `source_detector_distances()`: 计算源-探测器距离
- 通道频率和色团分析

---

### 9. 眼动追踪预处理 - `eyetracking/`

**主要功能：**

1. **数据类型设置** - `set_channel_types_eyetrack()`
   - 设置眼动追踪通道类型

2. **单位转换** - `convert_units()`
   - 像素、视角单位转换

3. **眨眼插值** - `interpolate_blinks()`
   - `_pupillometry.py`: 瞳孔测量数据的眨眼插值

4. **校准管理** - `calibration.py`
   - `Calibration`: 眼动校准类
   - `read_eyelink_calibration()`: 读取EyeLink校准数据

5. **辅助工具** - `utils.py`
   - `get_screen_visual_angle()`: 计算屏幕视角

---

### 10. 颅内脑电图（iEEG）预处理 - `ieeg/`

**主要功能：**

1. **投影** - `_projection.py`
   - iEEG信号的空间投影

2. **体积处理** - `_volume.py`
   - 三维体积数据处理
   - 电极定位

---

### 11. 其他高级算法

#### 11.1 局部异常因子（LOF）- `_lof.py`
- `find_bad_channels_lof()`: 基于LOF算法的坏导检测
- 无监督异常检测方法

#### 11.2 皮层信号抑制（CSS）- `_css.py`
- `cortical_signal_suppression()`: 抑制皮层信号以突出深部源

#### 11.3 过采样时间投影（OTP）- `otp.py`
- `oversampled_temporal_projection()`: 高时间分辨率投影

#### 11.4 PCA观测应用 - `_pca_obs.py`
- `apply_pca_obs()`: 应用基于观测的PCA

#### 11.5 峰值查找 - `_peak_finder.py`
- `peak_finder()`: 通用峰值检测工具

#### 11.6 数据对齐 - `realign.py`
- `realign_raw()`: 重新对齐原始数据

#### 11.7 刺激伪迹修复 - `stim.py`
- `fix_stim_artifact()`: 修复刺激引起的伪迹

#### 11.8 坏导均衡 - `bads.py`
- `equalize_bads()`: 在多个数据集间统一坏导标记

---

## 技术栈与依赖

### 核心数学库
- **NumPy**: 数组运算和线性代数
- **SciPy**: 
  - `scipy.linalg`: 线性代数（SVD、矩阵求逆）
  - `scipy.signal`: 信号处理（峰值检测、滤波）
  - `scipy.special`: 特殊函数（球面谐波、勒让德多项式）
  - `scipy.optimize`: 优化算法
  - `scipy.stats`: 统计函数

### 信号处理技术
- **滤波**: FIR/IIR滤波器设计
- **傅里叶变换**: 频域分析
- **小波变换**: 时频分析
- **球面谐波**: MEG/EEG建模

### 机器学习方法
- **PCA**: 主成分分析
- **ICA**: 独立成分分析
- **LOF**: 局部异常因子
- **回归**: 线性回归、最小二乘

---

## 典型预处理流程

### EEG/MEG标准流程
```python
import mne
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs

# 1. 加载数据
raw = mne.io.read_raw_fif('data.fif', preload=True)

# 2. 滤波
raw.filter(l_freq=1.0, h_freq=40.0)

# 3. 标记坏导
raw.info['bads'] = ['EEG 001', 'MEG 2443']

# 4. ICA去伪迹
ica = ICA(n_components=20, random_state=42)
ica.fit(raw)

# 5. 自动检测ECG/EOG成分
ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.exclude = ecg_indices + eog_indices

# 6. 应用ICA
raw_clean = ica.apply(raw.copy())
```

### MEG Maxwell滤波流程
```python
from mne.preprocessing import maxwell_filter

# Maxwell滤波（含头部运动补偿）
raw_sss = maxwell_filter(
    raw,
    origin='auto',
    st_duration=10.0,  # tSSS时间窗口
    st_correlation=0.98,
    coord_frame='head',
    destination=None
)
```

### fNIRS处理流程
```python
from mne.preprocessing.nirs import (
    optical_density,
    beer_lambert_law,
    scalp_coupling_index
)

# 1. 转换为光密度
raw_od = optical_density(raw_intensity)

# 2. 检测坏导（SCI方法）
sci = scalp_coupling_index(raw_od)
raw_od.info['bads'] = list(sci[sci < 0.5].index)

# 3. 转换为血红蛋白浓度
raw_haemo = beer_lambert_law(raw_od, ppf=0.1)
```

---

## 算法参数调优指南

### ICA参数
- **n_components**: 
  - EEG: 20-30（64通道系统）
  - MEG: 40-60
- **method**: 
  - 'fastica': 最快，适合大多数情况
  - 'infomax': 稳定性更好
  - 'picard': 最新，收敛快
- **max_iter**: 通常200-500足够

### Maxwell滤波参数
- **st_duration**: 
  - 10秒：标准选择
  - 4-6秒：快速头部运动
- **st_correlation**: 
  - 0.98: 默认
  - 0.90-0.95: 更激进的噪声抑制
- **int_order / ext_order**: 
  - (8, 3): 成人默认
  - (6, 2): 儿童（头部更小）

### CSD参数
- **lambda2**: 
  - 1e-5: 默认正则化
  - 更小: 更锐利但噪声敏感
  - 更大: 更平滑但空间分辨率降低
- **stiffness**: 
  - 4: 默认（二阶导数）
  - 3-5: 可调范围

---

## 文件组织结构特点

### 模块化设计
- **私有模块**（下划线开头）：内部实现细节
- **公共模块**：用户API
- **子包**：专业领域（nirs, eyetracking, ieeg）

### 测试覆盖
- 每个主要算法都有对应的测试文件
- 测试数据位于 `tests/data/`
- 包含EEGLAB兼容性测试

### 类型提示
- `__init__.pyi`: 提供完整的类型注解
- 支持现代IDE的自动补全和类型检查

---

## 性能优化

### 并行计算
- 支持 `n_jobs` 参数的函数可利用多核
- 大多数滤波操作支持并行

### 内存管理
- `preload=True` vs `preload=False`
- Maxwell滤波使用OLA（Overlap-Add）方法处理长数据

### 数值稳定性
- 使用安全SVD（`_safe_svd`）
- 正则化矩阵求逆
- 单精度/双精度选项

---

## 引用文献

主要算法的文献基础：

1. **ICA**: 
   - Bell & Sejnowski (1995) - Infomax
   - Hyvärinen (1999) - FastICA

2. **Maxwell滤波**: 
   - Taulu & Kajola (2005) - SSS基础
   - Taulu et al. (2004) - tSSS

3. **CSD**: 
   - Perrin et al. (1987, 1989) - 球面样条
   - Kayser & Tenke (2015) - 现代CSD综述

4. **Xdawn**: 
   - Rivet et al. (2009) - Xdawn算法

5. **肌电检测**: 
   - Muthukumaraswamy (2013) - Z分数方法

6. **EOG回归**: 
   - Gratton et al. (1983) - 经典回归方法

---

## 最佳实践建议

### 1. 预处理顺序
推荐顺序：
1. 坏导标记
2. 滤波（高通 ≥ 0.1 Hz）
3. Maxwell滤波（仅MEG）
4. ICA/SSP伪迹去除
5. 降采样（如需要）
6. Epoching

### 2. 质量控制
- 使用可视化检查每步结果
- 保存中间步骤便于回溯
- 记录所有参数以保证可重复性

### 3. 数据备份
- 使用 `copy=True` 保护原始数据
- 关键步骤后保存中间结果

### 4. 批处理
- 为多被试处理编写脚本
- 使用统一参数保证一致性
- 记录每个被试的处理日志

---

## 扩展阅读

### 官方文档
- MNE-Python官方教程
- API参考文档

### 相关工具
- **FieldTrip** (MATLAB): 类似功能
- **EEGLAB** (MATLAB): ICA处理
- **Brainstorm** (MATLAB): 图形化界面

### 学习资源
- MNE-Python workshops
- 在线示例库
- GitHub discussions

---

## 更新日志

该模块持续维护和更新：
- 新算法的添加（如Picard ICA）
- 性能优化
- Bug修复
- 与最新科学文献保持同步

---

## 许可与贡献

- **许可**: BSD-3-Clause
- **作者**: MNE-Python contributors
- **贡献**: 欢迎通过GitHub提交PR

---

**生成日期**: 2026年1月30日  
**文档版本**: 1.0  
**对应MNE-Python版本**: 最新开发版
