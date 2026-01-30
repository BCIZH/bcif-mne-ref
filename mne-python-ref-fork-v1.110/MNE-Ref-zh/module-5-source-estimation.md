# 模块 5: Source Estimation - 源空间定位

> **在数据流中的位置**: 第五步 - 从传感器空间重建大脑源活动  
> **核心职责**: 正向建模 + 逆向求解 = 源定位  
> **模块路径**: `mne/forward/`, `mne/minimum_norm/`, `mne/beamformer/`, `mne/inverse_sparse/`

---

## 目录

1. [源定位概述](#源定位概述)
2. [正向建模 (Forward Modeling)](#正向建模)
3. [逆向求解 (Inverse Solutions)](#逆向求解)
4. [波束成形 (Beamformer)](#波束成形)
5. [稀疏方法](#稀疏方法)
6. [应用场景](#应用场景)

---

## 源定位概述

### 基本原理

**正问题 (Forward Problem)**:
$$
\mathbf{M} = \mathbf{G} \mathbf{J} + \mathbf{n}
$$

- $\mathbf{M}$: 传感器测量 [n_sensors × n_times]
- $\mathbf{G}$: 导联场矩阵 (Lead Field) [n_sensors × n_sources]
- $\mathbf{J}$: 源活动 [n_sources × n_times]
- $\mathbf{n}$: 噪声

**逆问题 (Inverse Problem)**:
$$
\hat{\mathbf{J}} = \mathbf{W} \mathbf{M}
$$

- $\mathbf{W}$: 逆算子 [n_sources × n_sensors]
- $\hat{\mathbf{J}}$: 估计的源活动

**不适定性 (Ill-posed)**:
- 源数量 (~5000-20000) >> 传感器数量 (~300)
- 无穷多解
- 需要正则化约束

---

### 数据流

```
解剖结构 (MRI)
    ├── Freesurfer 重建
    ├── 源空间 (Source Space)
    └── 边界元模型 (BEM)
           ↓
    Forward Solution
    (导联场矩阵 G)
           ↓
    + Noise Covariance
    + Data Covariance
           ↓
    Inverse Operator
           ↓
    Apply to Evoked/Epochs/Raw
           ↓
    Source Time Courses (STC)
```

---

## 正向建模

### 1. 源空间构建

**算法位置**: `mne/source_space/_source_space.py` (3000+ 行)

**Surface-based 源空间**:
```python
import mne

def setup_source_space(subject, spacing='oct6', subjects_dir=None):
    """
    构建皮层表面源空间
    
    参数:
        spacing: 'oct6' | 'oct5' | 'ico4' | 'ico5'
            - oct6: ~4098 dipoles per hemisphere (spacing ~4.9mm)
            - oct5: ~1026 dipoles (spacing ~9.9mm)
            - ico4: ~2562 dipoles (spacing ~5.1mm)
    
    算法:
        1. 加载 Freesurfer 皮层表面 (白质-灰质界面)
        2. 下采样 (octahedral subdivision / icosahedron)
        3. 为每个顶点定义法向偶极子
    """
    src = mne.setup_source_space(
        subject=subject,
        spacing='oct6',
        subjects_dir=subjects_dir,
        add_dist=False  # 不计算距离矩阵（节省内存）
    )
    
    # src 包含两个 hemisphere
    print(f"左半球: {src[0]['nuse']} dipoles")
    print(f"右半球: {src[1]['nuse']} dipoles")
    
    # 可视化
    mne.viz.plot_bem(
        subject=subject,
        subjects_dir=subjects_dir,
        src=src,
        show=True
    )
    
    # 保存
    src.save(f'{subject}-oct6-src.fif', overwrite=True)
    
    return src

# 使用
src = setup_source_space('fsaverage', spacing='oct6')
```

**Volume-based 源空间** (用于深部结构):
```python
def setup_volume_source_space(subject, volume_label, subjects_dir=None):
    """
    体积源空间 (用于海马、杏仁核等皮层下结构)
    
    参数:
        volume_label: 'Left-Hippocampus' | 'Right-Amygdala' | ...
    
    算法:
        1. 加载 Freesurfer aseg.mgz 分割
        2. 提取 ROI
        3. 在体素中放置偶极子
    """
    # 皮层下核团
    volume_label = 'Left-Hippocampus'
    
    src_vol = mne.setup_volume_source_space(
        subject=subject,
        subjects_dir=subjects_dir,
        volume_label=volume_label,
        pos=5.0,  # 体素间距 (mm)
        sphere=(0.0, 0.0, 0.0, 90.0),  # 限制球形区域 (可选)
        bem=None,  # 或提供 BEM
        mindist=5.0,  # 与内颅表面最小距离
        exclude=0.0
    )
    
    print(f"{volume_label}: {src_vol[0]['nuse']} dipoles")
    
    return src_vol

# 混合源空间 (皮层 + 皮层下)
src_surface = mne.setup_source_space('fsaverage', spacing='oct6')
src_hippocampus = setup_volume_source_space('fsaverage', 'Left-Hippocampus')

# 合并
src_mixed = src_surface + src_hippocampus
```

---

### 2. 边界元模型 (BEM)

**算法位置**: `mne/bem.py` (1500+ 行)

**三层 BEM** (EEG):
```python
def create_bem_model(subject, conductivity=(0.3, 0.006, 0.3), subjects_dir=None):
    """
    创建边界元模型
    
    参数:
        conductivity: 电导率 (S/m)
            - 脑: 0.3
            - 颅骨: 0.006 (~1/50 of brain)
            - 头皮: 0.3
    
    算法:
        1. 加载 BEM 表面 (brain.surf, inner_skull.surf, outer_skull.surf, outer_skin.surf)
        2. 计算边界元矩阵
    """
    # 制作 BEM 模型
    model = mne.make_bem_model(
        subject=subject,
        subjects_dir=subjects_dir,
        ico=4,  # tessellation detail
        conductivity=(0.3, 0.006, 0.3)  # brain, skull, scalp
    )
    
    # 计算 BEM 解
    bem_sol = mne.make_bem_solution(model)
    
    # 可视化
    mne.viz.plot_bem(
        subject=subject,
        subjects_dir=subjects_dir,
        brain_surfaces='white',
        src=src,
        bem=bem_sol,
        show=True
    )
    
    # 保存
    mne.write_bem_solution(f'{subject}-5120-bem-sol.fif', bem_sol)
    
    return bem_sol

# 单层 BEM (仅 MEG)
bem_sol_meg = mne.make_bem_solution(
    mne.make_bem_model(
        subject='fsaverage',
        ico=4,
        conductivity=(0.3,)  # 只有内颅表面
    )
)
```

---

### 3. 正向解计算

**算法位置**: `mne/forward/_make_forward.py` (2000+ 行)

**导联场矩阵**:
```python
def compute_forward_solution(raw, src, bem, trans, meg=True, eeg=True):
    """
    计算正向解
    
    参数:
        raw: 包含传感器位置的数据对象
        src: 源空间
        bem: 边界元解
        trans: 头部-MRI 配准变换矩阵
        meg: 是否计算 MEG 导联场
        eeg: 是否计算 EEG 导联场
    
    算法 (MEG):
        对每个源位置和每个 MEG 传感器:
        B(r_sensor) = (μ₀ / 4π) × [3(Q·R)R / R⁵ - Q / R³]
        其中 Q 是偶极矩，R = r_sensor - r_source
    
    算法 (EEG):
        使用 BEM 计算电势:
        V = Σ_surfaces (1/4πσ) ∫ (σ(r') / |r - r'|) dS
    
    复杂度:
        - MEG: O(n_sensors × n_sources) - 解析解
        - EEG: O(n_sensors × n_sources × n_bem_vertices) - 数值积分
    """
    fwd = mne.make_forward_solution(
        raw.info,
        trans=trans,  # 'fsaverage-trans.fif' 或配准结果
        src=src,
        bem=bem,
        meg=meg,
        eeg=eeg,
        mindist=5.0,  # 最小源深度 (mm)
        n_jobs=4
    )
    
    print(fwd)
    # Forward solution:
    #   Source space          : Surface (left + right)
    #   MRI -> head transform : fsaverage-trans.fif
    #   Leadfield channels    : 306 MEG, 60 EEG
    #   Source orientations   : Free (3 components)
    #   Channels              : 366
    #   Source space points   : 8196
    #   Source space components: 24588 (3 per location)
    
    # 保存
    mne.write_forward_solution('forward-fwd.fif', fwd, overwrite=True)
    
    return fwd

# 固定方向 (法向偶极子)
fwd_fixed = mne.convert_forward_solution(
    fwd,
    surf_ori=True,   # 转换为表面法向
    force_fixed=True # 固定方向 (1 component)
)

# fwd_fixed: 8196 sources (vs 24588 for free orientation)
```

---

## 逆向求解

### 1. Minimum Norm Estimate (MNE)

**算法位置**: `mne/minimum_norm/inverse.py` (2500+ 行)

**L2 最小范数**:
```python
def make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8):
    """
    创建逆算子
    
    参数:
        noise_cov: 噪声协方差 (空白期 / 基线)
        loose: 松弛参数 (0-1)
            - 0: 完全固定法向
            - 0.2: 允许 20% 切向分量 (默认)
            - 1: 完全自由方向
        depth: 深度加权 (0-1)
            - 0.8: 典型值，补偿深部源
    
    数学形式:
        最小化: ||J||² subject to M = GJ + n
        
        解:
        Ĵ = G^T (GG^T + λC)^(-1) M
        
        其中 λ 是正则化参数，C 是噪声协方差
    """
    # 1. 计算噪声协方差
    noise_cov = mne.compute_covariance(
        epochs,
        tmin=None, tmax=0,  # 基线期
        method='shrunk'
    )
    
    # 2. 创建逆算子
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        info=raw.info,
        forward=fwd,
        noise_cov=noise_cov,
        loose=0.2,  # 松弛约束
        depth=0.8   # 深度加权
    )
    
    # 3. 查看
    print(inverse_operator)
    # Inverse operator:
    #   Method        : MNE
    #   fMRI prior    : No
    #   Channels      : 366
    #   Source space  : Surface (8196 locations)
    #   Noise cov     : 366 channels
    #   Whitener      : 366 x 366
    
    # 4. 保存
    mne.minimum_norm.write_inverse_operator(
        'inverse-inv.fif',
        inverse_operator
    )
    
    return inverse_operator

# 应用到 Evoked
stc = mne.minimum_norm.apply_inverse(
    evoked,
    inverse_operator,
    lambda2=1.0 / 9.0,  # 正则化参数 (SNR = 3)
    method='MNE',       # 'MNE' | 'dSPM' | 'sLORETA' | 'eLORETA'
    pick_ori=None       # None | 'normal' | 'vector'
)

print(stc)
# SourceEstimate
#   tmin: -0.2 s
#   tmax: 0.5 s
#   tstep: 1 ms
#   data shape: (8196, 701)
```

---

### 2. dSPM (Dynamic Statistical Parametric Mapping)

**算法位置**: `mne/minimum_norm/inverse.py:_apply_inverse_method()`

**归一化 MNE**:
```python
def apply_dspm(evoked, inverse_operator, lambda2=1.0/9.0):
    """
    dSPM: MNE 的归一化版本
    
    公式:
        dSPM(r, t) = MNE(r, t) / σ(r)
    
    其中 σ(r) 是估计的标准差 (从分辨率矩阵对角线推导)
    
    优势:
        - 单位无关 (类似 z-score)
        - 可跨被试/条件比较
        - 减少深度偏差
    """
    stc_dspm = mne.minimum_norm.apply_inverse(
        evoked,
        inverse_operator,
        lambda2=lambda2,
        method='dSPM',  # ← 归一化
        pick_ori=None
    )
    
    # dSPM 值: 类似 z-score，无单位
    # 可以设置显著性阈值 (e.g., dSPM > 5)
    
    return stc_dspm

# 可视化
brain = stc_dspm.plot(
    subject='fsaverage',
    subjects_dir=subjects_dir,
    hemi='both',
    time_viewer=True,
    clim=dict(kind='value', pos_lims=[3, 5, 7]),  # dSPM 阈值
    initial_time=0.1,
    colormap='hot'
)
```

---

### 3. sLORETA (Standardized Low Resolution Electromagnetic Tomography)

**算法位置**: `mne/minimum_norm/inverse.py`

**零定位误差**:
```python
def apply_sloreta(evoked, inverse_operator, lambda2=1.0/9.0):
    """
    sLORETA: 改进的归一化方法
    
    特性:
        - 对于点源，零定位误差 (理论保证)
        - 归一化考虑分辨率矩阵非对角元素
    
    公式:
        sLORETA(r, t) = MNE(r, t) / √(R(r,r))
    
    其中 R 是分辨率矩阵 (resolution matrix)
    """
    stc_sloreta = mne.minimum_norm.apply_inverse(
        evoked,
        inverse_operator,
        lambda2=lambda2,
        method='sLORETA',
        pick_ori=None
    )
    
    return stc_sloreta

# 对比 MNE vs dSPM vs sLORETA
methods = ['MNE', 'dSPM', 'sLORETA']
stcs = {}

for method in methods:
    stcs[method] = mne.minimum_norm.apply_inverse(
        evoked,
        inverse_operator,
        lambda2=1.0/9.0,
        method=method
    )

# 并排可视化
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (method, stc) in zip(axes, stcs.items()):
    brain = stc.plot(
        subject='fsaverage',
        hemi='lh',
        initial_time=0.1,
        time_viewer=False,
        colormap='hot',
        background='white',
        size=(400, 400)
    )
    ax.set_title(method)
```

---

## 波束成形

### 1. LCMV (Linearly Constrained Minimum Variance)

**算法位置**: `mne/beamformer/_lcmv.py` (800+ 行)

**空间滤波器**:
```python
def apply_lcmv_beamformer(epochs, fwd, noise_cov, data_cov):
    """
    LCMV 波束成形
    
    算法:
        对每个源位置 r:
        
        1. 约束: w^T G_r = 1 (单位增益)
        2. 最小化: w^T C w (输出方差)
        
        解:
        w = (G_r^T C^(-1) G_r)^(-1) G_r^T C^(-1)
    
    其中 C 是数据协方差矩阵
    
    参数:
        data_cov: 数据协方差 (包含信号的时间窗口)
        noise_cov: 噪声协方差 (基线)
    
    特性:
        - 自适应空间滤波
        - 对相关源敏感度低
        - 更好的空间分辨率
    """
    # 1. 计算数据协方差
    data_cov = mne.compute_covariance(
        epochs,
        tmin=0.0, tmax=0.5,  # 信号窗口
        method='empirical'
    )
    
    # 2. 计算噪声协方差
    noise_cov = mne.compute_covariance(
        epochs,
        tmin=None, tmax=0,  # 基线
        method='empirical'
    )
    
    # 3. 构建 LCMV filters
    filters = mne.beamformer.make_lcmv(
        epochs.info,
        forward=fwd,
        data_cov=data_cov,
        reg=0.05,            # 正则化
        noise_cov=noise_cov,
        pick_ori='max-power', # 'max-power' | 'normal' | 'vector'
        weight_norm='unit-noise-gain',  # 归一化
        rank=None
    )
    
    # 4. 应用到 Evoked
    stc_lcmv = mne.beamformer.apply_lcmv(evoked, filters)
    
    return stc_lcmv

# 使用
stc_lcmv = apply_lcmv_beamformer(epochs, fwd, noise_cov, data_cov)

# 可视化
brain = stc_lcmv.plot(
    subject='fsaverage',
    hemi='both',
    time_viewer=True,
    colormap='auto'
)
```

**Neural Activity Index (NAI)**:
```python
def compute_nai(epochs, filters):
    """
    神经活动指数: 信号与噪声功率对比
    
    NAI = √(P_signal / P_noise)
    """
    # 应用到两个时间窗口
    stc_baseline = mne.beamformer.apply_lcmv_epochs(
        epochs.copy().crop(tmin=-0.2, tmax=0),
        filters
    )
    
    stc_active = mne.beamformer.apply_lcmv_epochs(
        epochs.copy().crop(tmin=0.0, tmax=0.5),
        filters
    )
    
    # 计算功率
    power_baseline = (stc_baseline.data ** 2).mean(axis=1)
    power_active = (stc_active.data ** 2).mean(axis=1)
    
    # NAI
    nai = np.sqrt(power_active / power_baseline)
    
    # 转为 STC
    stc_nai = stc_active.copy()
    stc_nai.data = nai[:, np.newaxis]  # [n_sources × 1]
    
    return stc_nai

# 可视化 NAI
brain = stc_nai.plot(
    subject='fsaverage',
    hemi='both',
    clim=dict(kind='value', pos_lims=[1.5, 2.0, 3.0]),
    colormap='hot',
    time_label='NAI'
)
```

---

### 2. DICS (Dynamic Imaging of Coherent Sources)

**算法位置**: `mne/beamformer/_dics.py` (600+ 行)

**频域波束成形**:
```python
def apply_dics_beamformer(epochs, fwd, freq_band=(8, 12)):
    """
    DICS: 频域 LCMV
    
    应用:
        - 振荡活动定位 (alpha, beta, gamma)
        - 相干性分析
    
    算法:
        1. 计算交叉谱密度 (CSD)
        2. 在特定频段构建空间滤波器
        3. 估计源功率/相干性
    """
    from mne.time_frequency import csd_morlet
    
    # 1. 计算 CSD
    csd = csd_morlet(
        epochs,
        frequencies=np.arange(freq_band[0], freq_band[1] + 1),
        tmin=0, tmax=0.5,
        decim=10,
        n_jobs=4
    )
    
    # 平均频率
    csd = csd.mean()
    
    # 2. 构建 DICS filters
    filters = mne.beamformer.make_dics(
        epochs.info,
        forward=fwd,
        csd=csd,
        reg=0.05,
        pick_ori='max-power',
        weight_norm='unit-noise-gain'
    )
    
    # 3. 应用
    stc_dics, freqs = mne.beamformer.apply_dics_csd(csd, filters)
    
    return stc_dics

# Alpha 源定位
stc_alpha = apply_dics_beamformer(epochs, fwd, freq_band=(8, 12))

# 对比两个条件的 alpha 功率
csd_rest = csd_morlet(epochs['rest'], frequencies=np.arange(8, 13))
csd_task = csd_morlet(epochs['task'], frequencies=np.arange(8, 13))

filters = mne.beamformer.make_dics(
    epochs.info, fwd, csd_rest.mean()
)

stc_rest, _ = mne.beamformer.apply_dics_csd(csd_rest.mean(), filters)
stc_task, _ = mne.beamformer.apply_dics_csd(csd_task.mean(), filters)

# Alpha 抑制
stc_diff = stc_task / stc_rest  # 或 stc_task - stc_rest

brain = stc_diff.plot(
    subject='fsaverage',
    hemi='both',
    clim=dict(kind='percent', pos_lims=[90, 95, 99]),
    title='Alpha Suppression (Task / Rest)'
)
```

---

## 稀疏方法

### 1. Gamma-MAP

**算法位置**: `mne/inverse_sparse/mxne_inverse.py`

**稀疏先验**:
```python
from mne.inverse_sparse import gamma_map

def apply_gamma_map(evoked, fwd, noise_cov, alpha=0.2):
    """
    Gamma-MAP: 稀疏贝叶斯方法
    
    假设:
        - 大部分源不活跃 (稀疏)
        - Gamma 超先验
    
    优势:
        - 自动确定活跃源数量
        - 不需要手动设置正则化参数
    """
    stc_sparse, residual = gamma_map(
        evoked,
        fwd,
        noise_cov,
        alpha=alpha,         # 时间正则化
        loose='auto',
        depth=0.8,
        xyz_same_gamma=True, # 强制同位置3个分量共享 gamma
        return_residual=True
    )
    
    # 稀疏激活
    n_active = (stc_sparse.data.max(axis=1) > 0).sum()
    print(f"活跃源: {n_active} / {len(stc_sparse.data)}")
    
    return stc_sparse

# 使用
stc_gamma = apply_gamma_map(evoked, fwd, noise_cov)

# 对比 MNE (distributed) vs Gamma-MAP (sparse)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# MNE: 分布式
brain_mne = stc_mne.plot(hemi='both', size=(600, 400))
axes[0].set_title('MNE (Distributed)')

# Gamma-MAP: 稀疏
brain_gamma = stc_gamma.plot(hemi='both', size=(600, 400))
axes[1].set_title('Gamma-MAP (Sparse)')
```

---

### 2. Mixed Norm (MxNE)

**算法位置**: `mne/inverse_sparse/mxne_inverse.py`

**时空稀疏**:
```python
from mne.inverse_sparse import mixed_norm

def apply_mxne(evoked, fwd, noise_cov, alpha=30):
    """
    Mixed-Norm Estimate (MxNE)
    
    目标函数:
        minimize: ||M - GJ||²_F + α ||J||_{21}
    
    其中 ||J||_{21} = Σ_r ||J(r,:)||₂ (空间稀疏 + 时间平滑)
    
    参数:
        alpha: 稀疏度 (越大越稀疏)
    """
    stc_mxne = mixed_norm(
        evoked,
        fwd,
        noise_cov,
        alpha=alpha,
        loose='auto',
        depth=0.8,
        maxit=3000,
        tol=1e-4,
        active_set_size=10,  # 活跃集大小
        debias=True,         # 最后一步 least-squares
        return_residual=False
    )
    
    return stc_mxne

# 调优 alpha
alphas = [10, 30, 50, 70]
stcs = {}

for alpha in alphas:
    stcs[alpha] = apply_mxne(evoked, fwd, noise_cov, alpha=alpha)
    n_active = (stcs[alpha].data.max(axis=1) > 0).sum()
    print(f"α={alpha}: {n_active} 活跃源")

# α=10: 856 活跃源
# α=30: 234 活跃源
# α=50: 89 活跃源
# α=70: 34 活跃源
```

---

## 应用场景

### 场景 1: 听觉诱发响应源定位

```python
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse

# 1. 准备数据
epochs = mne.read_epochs('auditory-epo.fif')
evoked = epochs.average()

# 2. 加载正向解
fwd = mne.read_forward_solution('forward-fwd.fif')

# 3. 计算噪声协方差
noise_cov = mne.compute_covariance(
    epochs,
    tmin=None, tmax=0,
    method='shrunk'
)

# 4. 创建逆算子
inverse_operator = make_inverse_operator(
    evoked.info,
    fwd,
    noise_cov,
    loose=0.2,
    depth=0.8
)

# 5. 应用 dSPM
stc_dspm = apply_inverse(
    evoked,
    inverse_operator,
    lambda2=1.0/9.0,
    method='dSPM'
)

# 6. 可视化时间序列
brain = stc_dspm.plot(
    subject='fsaverage',
    subjects_dir=subjects_dir,
    hemi='both',
    time_viewer=True,
    initial_time=0.1,  # N100
    clim=dict(kind='value', pos_lims=[3, 6, 9]),
    colormap='hot',
    background='white',
    size=(800, 600)
)

# 7. 提取 ROI 时间序列
# 定义 Heschl's gyrus (初级听觉皮层)
label_lh = mne.read_label('fsaverage/label/lh.BA41-lh.label')
label_rh = mne.read_label('fsaverage/label/rh.BA41-rh.label')

# 提取平均时间序列
tc_lh = stc_dspm.in_label(label_lh).mean()
tc_rh = stc_dspm.in_label(label_rh).mean()

# 绘图
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(stc_dspm.times, tc_lh.data, label='Left Heschl')
ax.plot(stc_dspm.times, tc_rh.data, label='Right Heschl')
ax.axvline(0, color='k', linestyle='--')
ax.set_xlabel('Time (s)')
ax.set_ylabel('dSPM value')
ax.legend()
ax.set_title('Primary Auditory Cortex Activity')

# 8. 峰值定位
peak_vertex, peak_time = stc_dspm.get_peak(
    hemi='lh',
    tmin=0.08, tmax=0.15,  # N100 窗口
    vert_as_index=False
)

print(f"峰值: vertex {peak_vertex}, time {peak_time:.3f} s")

# 转换为 MNI 坐标
mni_coords = mne.vertex_to_mni(
    peak_vertex,
    hemis=0,  # 0=lh, 1=rh
    subject='fsaverage',
    subjects_dir=subjects_dir
)
print(f"MNI 坐标: {mni_coords}")
```

---

### 场景 2: Alpha 振荡源定位 (DICS)

```python
import mne
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import csd_morlet

# 1. 加载静息态 epochs
epochs = mne.read_epochs('rest-epo.fif')

# 2. 计算 alpha CSD (8-12 Hz)
freqs = np.arange(8, 13)
csd_alpha = csd_morlet(
    epochs,
    frequencies=freqs,
    tmin=0, tmax=30,  # 整个 epoch
    decim=20,
    n_jobs=4
)

# 平均频率
csd_alpha_avg = csd_alpha.mean()

# 3. 构建 DICS filters
fwd = mne.read_forward_solution('forward-fwd.fif')

filters = make_dics(
    epochs.info,
    fwd,
    csd_alpha_avg,
    reg=0.05,
    pick_ori='max-power',
    weight_norm='unit-noise-gain'
)

# 4. 应用
stc_alpha, freqs = apply_dics_csd(csd_alpha_avg, filters)

# 5. 可视化 alpha 功率分布
brain = stc_alpha.plot(
    subject='fsaverage',
    hemi='both',
    clim=dict(kind='percent', pos_lims=[90, 95, 99]),
    colormap='Reds',
    time_label='Alpha Power (8-12 Hz)',
    background='white',
    size=(800, 600)
)

# 6. 提取枕叶 alpha
label_occ = mne.read_labels_from_annot(
    'fsaverage',
    parc='aparc',
    regexp='lateraloccipital',
    subjects_dir=subjects_dir
)

for label in label_occ:
    alpha_power = stc_alpha.in_label(label).data.mean()
    print(f"{label.name}: {alpha_power:.2e}")

# 7. 闭眼 vs 睁眼对比
epochs_open = mne.read_epochs('eyes-open-epo.fif')
epochs_closed = mne.read_epochs('eyes-closed-epo.fif')

csd_open = csd_morlet(epochs_open, frequencies=freqs).mean()
csd_closed = csd_morlet(epochs_closed, frequencies=freqs).mean()

# 用闭眼数据构建 filters
filters = make_dics(epochs.info, fwd, csd_closed)

# 应用到两个条件
stc_open, _ = apply_dics_csd(csd_open, filters)
stc_closed, _ = apply_dics_csd(csd_closed, filters)

# Alpha 增强 (闭眼 / 睁眼)
stc_ratio = stc_closed.copy()
stc_ratio.data = stc_closed.data / (stc_open.data + 1e-20)  # 避免除零

brain = stc_ratio.plot(
    subject='fsaverage',
    hemi='both',
    clim=dict(kind='value', pos_lims=[1.5, 2.0, 3.0]),
    colormap='RdBu_r',
    time_label='Alpha Enhancement (Closed/Open)',
    transparent=True
)
```

---

### 场景 3: 组平均源空间分析

```python
import mne
import numpy as np
from pathlib import Path

# 1. Morph 所有被试到 fsaverage
subjects = [f'sub-{i:02d}' for i in range(1, 21)]
subjects_dir = Path('subjects')
stcs_all = []

for subject in subjects:
    # 读取单被试 STC
    stc = mne.read_source_estimate(f'results/{subject}_dspm')
    
    # Morph 到 fsaverage
    morph = mne.compute_source_morph(
        stc,
        subject_from=subject,
        subject_to='fsaverage',
        subjects_dir=subjects_dir
    )
    
    stc_fsaverage = morph.apply(stc)
    stcs_all.append(stc_fsaverage)

print(f"Morphed {len(stcs_all)} subjects to fsaverage")

# 2. 组平均
stc_avg = stcs_all[0].copy()
stc_avg.data = np.mean([stc.data for stc in stcs_all], axis=0)

# 3. 标准误
stc_sem = stcs_all[0].copy()
stc_sem.data = np.std([stc.data for stc in stcs_all], axis=0, ddof=1) / \
               np.sqrt(len(stcs_all))

# 4. 可视化组平均
brain = stc_avg.plot(
    subject='fsaverage',
    hemi='both',
    time_viewer=True,
    clim=dict(kind='value', pos_lims=[3, 6, 9]),
    title=f'Group Average (N={len(stcs_all)})'
)

# 5. 统计检验 (cluster permutation)
from mne.stats import spatio_temporal_cluster_1samp_test

# 堆叠数据
X = np.array([stc.data for stc in stcs_all])  # [n_subjects × n_sources × n_times]

# 源空间邻接矩阵
src = mne.read_source_spaces('fsaverage-oct6-src.fif')
connectivity = mne.spatial_src_adjacency(src)

# 单样本 t 检验 (vs 0)
T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
    X,
    n_permutations=1024,
    threshold=dict(start=0, step=0.2),
    adjacency=connectivity,
    n_jobs=4,
    verbose=True
)

# 显著性集群
sig_clusters = np.where(cluster_p_values < 0.05)[0]
print(f"发现 {len(sig_clusters)} 个显著集群 (p < 0.05)")

# 6. 可视化显著性
# 创建mask
stc_cluster = stc_avg.copy()
stc_cluster.data[:] = 0

for idx in sig_clusters:
    stc_cluster.data[clusters[idx][0], clusters[idx][1]] = 1

brain = stc_cluster.plot(
    subject='fsaverage',
    hemi='both',
    time_viewer=True,
    clim=dict(kind='value', lims=[0, 0.5, 1]),
    colormap='Reds',
    title='Significant Clusters (p < 0.05)'
)
```

---

## 总结

### 核心算法汇总

| 算法 | 位置 | 复杂度 | 特性 | 场景 |
|------|------|--------|------|------|
| **正向建模** |
| 源空间构建 | `source_space/` | O(N_vertices) | 皮层采样 | 所有方法前提 |
| BEM | `bem.py` | O(N_bem²) | 容积导体 | EEG必需 |
| 导联场计算 | `forward/` | O(S×C) | 物理模型 | 所有方法前提 |
| **分布式方法** |
| MNE | `minimum_norm/` | O(C³ + S×C²) | L2最小范数 | 平滑解 |
| dSPM | `minimum_norm/` | O(C³ + S×C²) | 归一化MNE | 统计推断 |
| sLORETA | `minimum_norm/` | O(C³ + S²×C) | 零定位误差 | 精确定位 |
| **波束成形** |
| LCMV | `beamformer/` | O(C³×S) | 自适应滤波 | 时域,高分辨 |
| DICS | `beamformer/` | O(F×C³×S) | 频域波束 | 振荡分析 |
| **稀疏方法** |
| Gamma-MAP | `inverse_sparse/` | O(iter×S×C²) | 贝叶斯稀疏 | 少数强源 |
| MxNE | `inverse_sparse/` | O(iter×S×C) | 混合范数 | 时空稀疏 |

*注: S=源数, C=传感器数, F=频率点, iter=迭代次数*

### 方法选择指南

| 研究问题 | 推荐方法 | 原因 |
|----------|----------|------|
| ERP源定位 | dSPM / sLORETA | 分布式，适合扩展源 |
| 深部源 (海马等) | LCMV + volume src | 体积源空间 |
| 振荡活动 | DICS | 频域分析 |
| 癫痫棘波 | Gamma-MAP / MxNE | 稀疏，局灶性 |
| 相关源网络 | MNE | 不假设稀疏 |
| 单试次分析 | LCMV | 自适应，不需平均 |

### 下一步

源空间数据用于：
1. **连接性分析**: 功能/有效连接
2. **统计分析** → **模块6: Statistics**
3. **机器学习**: 源空间解码
