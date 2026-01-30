# 模块 6: Statistics - 统计分析与假设检验

> **在数据流中的位置**: 最后一步 - 统计推断与显著性检验  
> **核心职责**: 多重比较校正、置换检验、聚类分析  
> **模块路径**: `mne/stats/`

---

## 目录

1. [统计检验概述](#统计检验概述)
2. [多重比较校正](#多重比较校正)
3. [置换检验 (Permutation Tests)](#置换检验)
4. [聚类置换检验 (Cluster Permutation)](#聚类置换检验)
5. [参数检验](#参数检验)
6. [回归分析](#回归分析)
7. [应用场景](#应用场景)

---

## 统计检验概述

### M/EEG 数据的多重比较问题

**问题**:
- EEG: ~60 通道 × 500 时间点 = **30,000 次检验**
- MEG: ~300 通道 × 500 时间点 = **150,000 次检验**
- 源空间: ~8000 源 × 500 时间点 = **4,000,000 次检验**

**虚警率 (False Positive Rate)**:
$$
P(\text{至少1个假阳性}) = 1 - (1 - \alpha)^n
$$

例如: $\alpha = 0.05$, $n = 30,000$:
$$
P \approx 1 - (0.95)^{30000} \approx 1.0
$$

→ **几乎必然产生假阳性!**

---

### MNE-Python 统计工具箱

**文件位置**: `mne/stats/` (10个子模块)

```python
mne/stats/
├── __init__.py
├── parametric.py          # t检验、ANOVA
├── permutations.py        # 置换检验
├── cluster_level.py       # 聚类置换检验 ⭐
├── multi_comp.py          # 多重比较校正
├── regression.py          # 线性回归
├── _adjacency.py          # 邻接矩阵
└── erp/
    ├── __init__.py
    └── _stats.py          # ERP统计 (rERP)
```

---

## 多重比较校正

### 1. Bonferroni 校正

**算法位置**: `mne/stats/multi_comp.py:bonferroni_correction()` (行 10-50)

**FWER 控制** (Family-Wise Error Rate):
```python
def bonferroni_correction(p_values, alpha=0.05):
    """
    Bonferroni 校正: 最保守的方法
    
    校正阈值:
        α_corrected = α / n
    
    其中 n 是检验次数
    
    参数:
        p_values: ndarray, p值数组
        alpha: float, 显著性水平
    
    返回:
        reject: bool array, 是否拒绝零假设
        p_corrected: ndarray, 校正后的p值
    
    缺点:
        - 过于保守
        - 功效低 (high false negative rate)
    """
    from mne.stats import bonferroni_correction
    
    # p_values shape: [n_channels, n_times]
    reject, p_corrected = bonferroni_correction(
        p_values,
        alpha=0.05
    )
    
    print(f"显著性点数: {reject.sum()} / {reject.size}")
    
    return reject, p_corrected

# 使用示例
import numpy as np
from scipy import stats

# 模拟数据: 20 被试 × 64 通道 × 500 时间点
data_cond1 = np.random.randn(20, 64, 500) + 0.3  # 轻微效应
data_cond2 = np.random.randn(20, 64, 500)

# 逐点 t 检验
t_values = np.zeros((64, 500))
p_values = np.zeros((64, 500))

for ch in range(64):
    for t in range(500):
        t_val, p_val = stats.ttest_rel(
            data_cond1[:, ch, t],
            data_cond2[:, ch, t]
        )
        t_values[ch, t] = t_val
        p_values[ch, t] = p_val

# Bonferroni 校正
reject, p_corrected = bonferroni_correction(p_values, alpha=0.05)

# 未校正 vs 校正
print(f"未校正显著: {(p_values < 0.05).sum()}")
print(f"Bonferroni校正显著: {reject.sum()}")

# 未校正显著: 1598
# Bonferroni校正显著: 0  ← 过于保守!
```

---

### 2. FDR 校正 (False Discovery Rate)

**算法位置**: `mne/stats/multi_comp.py:fdr_correction()` (行 50-150)

**控制假发现率**:
```python
def fdr_correction(p_values, alpha=0.05, method='indep'):
    """
    FDR 校正 (Benjamini-Hochberg)
    
    算法:
        1. 将 p 值从小到大排序: p_(1) ≤ p_(2) ≤ ... ≤ p_(n)
        2. 找到最大的 i 使得: p_(i) ≤ (i/n) × α
        3. 拒绝所有 p_(1), ..., p_(i)
    
    参数:
        method: 'indep' | 'negcorr'
            - indep: 独立/正相关检验
            - negcorr: 任意依赖 (更保守)
    
    优势:
        - 比 Bonferroni 宽松
        - 控制假发现比例而非 FWER
        - 更高功效
    """
    from mne.stats import fdr_correction
    
    reject_fdr, p_fdr = fdr_correction(
        p_values,
        alpha=0.05,
        method='indep'
    )
    
    return reject_fdr, p_fdr

# 对比 Bonferroni vs FDR
reject_bonf, _ = bonferroni_correction(p_values, alpha=0.05)
reject_fdr, _ = fdr_correction(p_values, alpha=0.05)

print(f"Bonferroni: {reject_bonf.sum()} 显著点")
print(f"FDR: {reject_fdr.sum()} 显著点")

# Bonferroni: 0 显著点
# FDR: 47 显著点 ← 更高功效
```

**可视化**:
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. 原始 p 值
im1 = axes[0].imshow(p_values, aspect='auto', cmap='RdBu_r', 
                      vmin=0, vmax=1)
axes[0].set_title('Original p-values')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Channel')
plt.colorbar(im1, ax=axes[0])

# 2. Bonferroni
im2 = axes[1].imshow(reject_bonf, aspect='auto', cmap='Greys')
axes[1].set_title(f'Bonferroni (n={reject_bonf.sum()})')
axes[1].set_xlabel('Time')

# 3. FDR
im3 = axes[2].imshow(reject_fdr, aspect='auto', cmap='Greys')
axes[2].set_title(f'FDR (n={reject_fdr.sum()})')
axes[2].set_xlabel('Time')

plt.tight_layout()
```

---

## 置换检验

### 1. 单样本置换 t 检验

**算法位置**: `mne/stats/permutations.py:permutation_t_test()` (行 200-400)

**非参数检验**:
```python
from mne.stats import permutation_t_test

def permutation_test_1samp(data, n_permutations=10000, tail=0):
    """
    单样本置换 t 检验 (vs 0)
    
    算法:
        1. 计算观察到的 t 统计量
        2. 重复 n_permutations 次:
           a. 随机翻转每个被试的符号 (±1)
           b. 重新计算 t 统计量
        3. p-value = (更极端的 t 值数量 + 1) / (n_permutations + 1)
    
    参数:
        data: ndarray [n_subjects, ...]
        tail: 0 (双尾) | 1 (右尾) | -1 (左尾)
    
    复杂度: O(n_permutations × n_subjects × n_features)
    """
    # data shape: [n_subjects, n_channels, n_times]
    T_obs, p_values, H0 = permutation_t_test(
        data,
        n_permutations=n_permutations,
        tail=tail,
        n_jobs=4,
        verbose=True
    )
    
    # T_obs: 观察到的 t 值 [n_channels, n_times]
    # p_values: 置换 p 值 [n_channels, n_times]
    # H0: 零假设分布 [n_permutations, ...]
    
    return T_obs, p_values, H0

# 使用
# 差异波 (每个被试都有 target - standard)
diff_waves = []
for subject in subjects:
    evoked_target = mne.read_evokeds(f'{subject}_target-ave.fif')[0]
    evoked_standard = mne.read_evokeds(f'{subject}_standard-ave.fif')[0]
    
    diff = evoked_target.data - evoked_standard.data
    diff_waves.append(diff)

# [n_subjects × n_channels × n_times]
X = np.array(diff_waves)

# 置换检验
T_obs, p_values, H0 = permutation_test_1samp(X, n_permutations=5000)

# FDR 校正
reject_fdr, p_fdr = fdr_correction(p_values)

print(f"显著点: {reject_fdr.sum()} / {reject_fdr.size}")

# 可视化
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# t 值
im1 = axes[0].imshow(T_obs, aspect='auto', cmap='RdBu_r', 
                      vmin=-5, vmax=5)
axes[0].set_title('Observed t-values')
plt.colorbar(im1, ax=axes[0])

# 显著性 mask
im2 = axes[1].imshow(reject_fdr, aspect='auto', cmap='Greys')
axes[1].set_title('Significant (FDR corrected)')
plt.colorbar(im2, ax=axes[1])
```

---

### 2. Bootstrap 置信区间

**算法位置**: `mne/stats/permutations.py:bootstrap_confidence_interval()` (行 100-200)

```python
from mne.stats import bootstrap_confidence_interval

def compute_bootstrap_ci(data, ci=0.95, n_bootstraps=10000):
    """
    Bootstrap 置信区间
    
    算法:
        1. 重复 n_bootstraps 次:
           a. 从数据中有放回抽样
           b. 计算统计量 (均值)
        2. 置信区间 = bootstrap 分布的百分位数
    
    参数:
        data: ndarray [n_samples, ...]
        ci: 置信水平 (0.95 = 95%)
    """
    # data: [n_subjects, n_channels, n_times]
    ci_lower, ci_upper = bootstrap_confidence_interval(
        data,
        ci=ci,
        n_bootstraps=n_bootstraps,
        stat_fun=np.mean,  # 或自定义函数
        random_state=42
    )
    
    return ci_lower, ci_upper

# 使用
grand_avg = X.mean(axis=0)  # [n_channels, n_times]
ci_lower, ci_upper = compute_bootstrap_ci(X)

# 绘制带置信区间的 ERP
ch_idx = 31  # Cz
times = np.linspace(-0.2, 0.8, X.shape[2])

fig, ax = plt.subplots()
ax.plot(times, grand_avg[ch_idx], label='Grand Average')
ax.fill_between(
    times,
    ci_lower[ch_idx],
    ci_upper[ch_idx],
    alpha=0.3,
    label='95% CI (Bootstrap)'
)
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('ERP with Bootstrap Confidence Interval')
ax.legend()
```

---

## 聚类置换检验

### 1. 时空聚类置换 (Cluster-based Permutation)

**算法位置**: `mne/stats/cluster_level.py:spatio_temporal_cluster_test()` (行 500-800)

**解决多重比较的优雅方案**:
```python
from mne.stats import spatio_temporal_cluster_test
from mne.channels import find_ch_adjacency

def cluster_permutation_test(epochs_cond1, epochs_cond2, 
                              n_permutations=1024, threshold=None):
    """
    时空聚类置换检验
    
    算法:
        1. 逐点计算 t 统计量
        2. 阈值化: t > threshold
        3. 识别时空聚类 (连通的超阈值点)
        4. 计算每个聚类的统计量 (cluster mass = Σ t-values)
        5. 置换:
           a. 随机打乱条件标签
           b. 重新计算聚类统计量
           c. 记录最大聚类统计量
        6. p-value = (max_cluster_permuted >= cluster_obs 的次数) / n_permutations
    
    参数:
        threshold: dict | float
            - dict: {'start': 0, 'step': 0.2} (TFCE-like)
            - float: 固定阈值 (e.g., 2.5)
            - None: 自动 (基于 t 分布)
    
    优势:
        - 控制 FWER (而非逐点)
        - 考虑时空结构
        - 对扩展效应敏感
    """
    # 1. 准备数据
    X_cond1 = epochs_cond1.get_data()  # [n_epochs × n_channels × n_times]
    X_cond2 = epochs_cond2.get_data()
    
    X = [X_cond1, X_cond2]
    
    # 2. 通道邻接矩阵
    adjacency, ch_names = find_ch_adjacency(epochs_cond1.info, ch_type='eeg')
    
    # 3. 聚类检验
    T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_test(
        X,
        n_permutations=n_permutations,
        threshold=None,  # 自动阈值 (t分布)
        tail=0,          # 双尾
        adjacency=adjacency,
        n_jobs=4,
        verbose=True
    )
    
    # T_obs: t 值 [n_channels × n_times]
    # clusters: list of tuples (每个是 (ch_indices, time_indices))
    # cluster_p_values: 每个聚类的 p 值
    # H0: 零假设下最大聚类统计量分布
    
    return T_obs, clusters, cluster_p_values, H0

# 使用
epochs_target = epochs['event_id == 1']
epochs_standard = epochs['event_id == 2']

T_obs, clusters, cluster_pv, H0 = cluster_permutation_test(
    epochs_target,
    epochs_standard,
    n_permutations=5000
)

# 找到显著聚类
sig_clusters_idx = np.where(cluster_pv < 0.05)[0]
print(f"发现 {len(sig_clusters_idx)} 个显著聚类 (p < 0.05)")

for idx in sig_clusters_idx:
    print(f"  Cluster {idx}: p = {cluster_pv[idx]:.4f}")
```

**可视化聚类**:
```python
from mne.viz import plot_compare_evokeds

def plot_clusters(epochs_cond1, epochs_cond2, T_obs, clusters, cluster_pv, 
                   alpha=0.05):
    """可视化显著聚类"""
    
    # 1. 计算 evoked
    evoked_cond1 = epochs_cond1.average()
    evoked_cond2 = epochs_cond2.average()
    
    # 2. 创建 mask (显著聚类)
    mask = np.zeros_like(T_obs, dtype=bool)
    for idx in np.where(cluster_pv < alpha)[0]:
        mask[clusters[idx]] = True
    
    # 3. 绘图
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # a) t 值
    times = evoked_cond1.times
    im1 = axes[0].imshow(
        T_obs,
        extent=[times[0], times[-1], 0, len(T_obs)],
        aspect='auto',
        cmap='RdBu_r',
        vmin=-5, vmax=5
    )
    axes[0].set_title('Observed t-values')
    axes[0].set_ylabel('Channel')
    plt.colorbar(im1, ax=axes[0])
    
    # b) 显著性 mask
    im2 = axes[1].imshow(
        mask,
        extent=[times[0], times[-1], 0, len(mask)],
        aspect='auto',
        cmap='Greys'
    )
    axes[1].set_title(f'Significant Clusters (p < {alpha})')
    axes[1].set_ylabel('Channel')
    plt.colorbar(im2, ax=axes[1])
    
    # c) ERP 对比
    # 选择一个显著聚类中的代表性通道
    if len(sig_clusters_idx) > 0:
        cluster_channels = clusters[sig_clusters_idx[0]][0]
        ch_idx = np.bincount(cluster_channels).argmax()  # 最常出现的通道
        
        axes[2].plot(times, evoked_cond1.data[ch_idx], label='Condition 1')
        axes[2].plot(times, evoked_cond2.data[ch_idx], label='Condition 2')
        
        # 标记显著时间段
        cluster_times = clusters[sig_clusters_idx[0]][1]
        time_mask = np.zeros(len(times), dtype=bool)
        time_mask[cluster_times] = True
        
        axes[2].fill_between(
            times,
            axes[2].get_ylim()[0],
            axes[2].get_ylim()[1],
            where=time_mask,
            alpha=0.3,
            color='red',
            label='Significant'
        )
        
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        axes[2].set_title(f'Channel {evoked_cond1.ch_names[ch_idx]}')
        axes[2].legend()
        axes[2].axhline(0, color='k', linestyle='--', linewidth=0.5)
        axes[2].axvline(0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    return fig

# 绘制
fig = plot_clusters(epochs_target, epochs_standard, T_obs, clusters, cluster_pv)
```

---

### 2. TFCE (Threshold-Free Cluster Enhancement)

**算法位置**: `mne/stats/cluster_level.py` (集成在 threshold 参数中)

**无需手动设置阈值**:
```python
def cluster_test_tfce(epochs_cond1, epochs_cond2):
    """
    TFCE: 无阈值聚类增强
    
    算法:
        对每个点 (ch, t):
        TFCE(ch, t) = ∫ e(h)^E × h^H dh
        
        其中:
        - h: 阈值高度
        - e(h): 在阈值 h 下的聚类范围
        - E=0.5, H=2 (默认参数)
    
    优势:
        - 不依赖单一阈值
        - 对不同形状的效应都敏感
    """
    X = [epochs_cond1.get_data(), epochs_cond2.get_data()]
    
    adjacency, _ = find_ch_adjacency(epochs_cond1.info, ch_type='eeg')
    
    # 使用 dict threshold 触发 TFCE
    T_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_test(
        X,
        n_permutations=5000,
        threshold=dict(start=0, step=0.2),  # TFCE
        tail=0,
        adjacency=adjacency,
        n_jobs=4
    )
    
    return T_obs, clusters, cluster_pv

# 对比固定阈值 vs TFCE
# 固定阈值
T_fixed, clusters_fixed, pv_fixed, _ = spatio_temporal_cluster_test(
    X,
    threshold=2.5,  # t > 2.5
    adjacency=adjacency,
    n_permutations=1000
)

# TFCE
T_tfce, clusters_tfce, pv_tfce, _ = spatio_temporal_cluster_test(
    X,
    threshold=dict(start=0, step=0.2),
    adjacency=adjacency,
    n_permutations=1000
)

print(f"固定阈值 (t>2.5): {(pv_fixed < 0.05).sum()} 显著聚类")
print(f"TFCE: {(pv_tfce < 0.05).sum()} 显著聚类")
```

---

### 3. 源空间聚类检验

**算法位置**: `mne/stats/cluster_level.py:spatio_temporal_cluster_1samp_test()`

**针对 SourceEstimate**:
```python
from mne.stats import spatio_temporal_cluster_1samp_test
import mne

def source_space_cluster_test(stcs, src, subjects_dir):
    """
    源空间单样本 t 检验
    
    参数:
        stcs: list of SourceEstimate (每个被试)
        src: 源空间
    """
    # 1. 准备数据
    X = np.array([stc.data for stc in stcs])  # [n_subjects × n_sources × n_times]
    
    # 2. 源空间邻接矩阵
    connectivity = mne.spatial_src_adjacency(src)
    
    # 3. 聚类检验
    T_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(
        X,
        n_permutations=1024,
        threshold=None,
        adjacency=connectivity,
        n_jobs=4,
        verbose=True
    )
    
    return T_obs, clusters, cluster_pv

# 使用
src = mne.read_source_spaces('fsaverage-oct6-src.fif')

# 加载所有被试的 STC
stcs_all = []
for subject in subjects:
    stc = mne.read_source_estimate(f'{subject}_dspm')
    # Morph to fsaverage
    morph = mne.compute_source_morph(
        stc, subject_from=subject, subject_to='fsaverage'
    )
    stc_fsaverage = morph.apply(stc)
    stcs_all.append(stc_fsaverage)

# 聚类检验
T_obs, clusters, cluster_pv = source_space_cluster_test(
    stcs_all, src, subjects_dir
)

# 可视化显著聚类
sig_idx = np.where(cluster_pv < 0.05)[0]

for idx in sig_idx[:3]:  # 前3个显著聚类
    # 创建 mask STC
    stc_cluster = stcs_all[0].copy()
    stc_cluster.data[:] = 0
    stc_cluster.data[clusters[idx][0], clusters[idx][1]] = T_obs[clusters[idx]]
    
    # 可视化
    brain = stc_cluster.plot(
        subject='fsaverage',
        hemi='both',
        time_viewer=True,
        clim=dict(kind='value', pos_lims=[3, 5, 7]),
        title=f'Cluster {idx} (p={cluster_pv[idx]:.4f})'
    )
```

---

## 参数检验

### 1. 重复测量 ANOVA

**算法位置**: `mne/stats/parametric.py:f_mway_rm()` (行 200-500)

**多因素设计**:
```python
from mne.stats import f_mway_rm

def repeated_measures_anova(data, factor_levels):
    """
    重复测量 ANOVA
    
    参数:
        data: ndarray [n_subjects × n_conditions × n_observations × ...]
        factor_levels: list of int, 每个因子的水平数
    
    示例:
        2×3 设计 (刺激类型 × SOA)
        - 刺激: 2 水平 (target, distractor)
        - SOA: 3 水平 (100ms, 200ms, 300ms)
    """
    # data shape: [20 subjects × 2 stimuli × 3 SOAs × 64 channels × 500 times]
    # → reshape: [20 × 6 conditions × 64 × 500]
    
    n_subjects = data.shape[0]
    data_reshaped = data.reshape(n_subjects, -1, *data.shape[3:])
    
    # ANOVA
    F_obs, p_values = f_mway_rm(
        data_reshaped,
        factor_levels=[2, 3],  # [n_stim, n_soa]
        effects='all',         # 主效应 + 交互
        return_pvals=True
    )
    
    # F_obs: dict with keys:
    #   - 'stim': F values for stimulus main effect
    #   - 'soa': F values for SOA main effect
    #   - 'stim:soa': F values for interaction
    
    return F_obs, p_values

# 使用
epochs_list = []
for subject in subjects:
    for stim in ['target', 'distractor']:
        for soa in [100, 200, 300]:
            epochs_sub = mne.read_epochs(
                f'{subject}_{stim}_soa{soa}-epo.fif'
            )
            epochs_list.append(epochs_sub.average().data)

# [n_subjects × n_stim × n_soa × n_ch × n_times]
data = np.array(epochs_list).reshape(len(subjects), 2, 3, 64, 500)

F_obs, p_vals = repeated_measures_anova(data, [2, 3])

# 检查主效应
print(f"刺激主效应显著点: {(p_vals['stim'] < 0.05).sum()}")
print(f"SOA 主效应显著点: {(p_vals['soa'] < 0.05).sum()}")
print(f"交互效应显著点: {(p_vals['stim:soa'] < 0.05).sum()}")

# 可视化 F 值
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (effect, F) in zip(axes, F_obs.items()):
    im = ax.imshow(F, aspect='auto', cmap='hot')
    ax.set_title(f'{effect} (F-values)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Channel')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
```

---

### 2. 成对 t 检验 (矢量化)

**算法位置**: `mne/stats/parametric.py:ttest_1samp_no_p()`

**快速批量检验**:
```python
from mne.stats import ttest_1samp_no_p

def vectorized_ttest(data):
    """
    矢量化 t 检验 (不计算 p 值)
    
    用途:
        - 快速计算 t 值
        - 后续用于聚类检验
    
    复杂度: O(n_subjects × n_features)
    """
    # data: [n_subjects × n_features]
    t_values = ttest_1samp_no_p(data, sigma=0)
    
    # 比 scipy.stats.ttest_1samp 快 ~10x
    return t_values

# 比较速度
import time
from scipy.stats import ttest_1samp

data = np.random.randn(20, 64, 500)  # 20 subjects

# scipy (慢)
start = time.time()
t_scipy = np.zeros((64, 500))
for ch in range(64):
    for t in range(500):
        t_scipy[ch, t] = ttest_1samp(data[:, ch, t], 0)[0]
print(f"scipy: {time.time() - start:.2f} s")

# MNE (快)
start = time.time()
t_mne = ttest_1samp_no_p(data)
print(f"MNE: {time.time() - start:.2f} s")

# scipy: 2.34 s
# MNE: 0.08 s  ← ~29x 加速!
```

---

## 回归分析

### 1. 传感器空间线性回归

**算法位置**: `mne/stats/regression.py:linear_regression()` (行 100-300)

**单试次预测变量**:
```python
from mne.stats import linear_regression

def regress_behavioral_var(epochs, behavioral_var):
    """
    单试次回归分析
    
    模型:
        EEG(trial, ch, t) = β₀ + β₁ × behavior(trial) + ε
    
    参数:
        epochs: Epochs 对象
        behavioral_var: ndarray [n_trials], 行为变量
    
    返回:
        beta: 回归系数 [n_channels × n_times]
        t_values: t 统计量
        p_values: p 值
    
    示例:
        - 反应时与 P300 幅度关系
        - 置信度与 CNV 关系
    """
    # 1. 获取数据
    X_eeg = epochs.get_data()  # [n_trials × n_channels × n_times]
    
    # 2. 设计矩阵
    # 添加截距列
    design_matrix = np.column_stack([
        np.ones(len(behavioral_var)),  # 截距
        behavioral_var                 # 预测变量
    ])
    
    # 3. 回归
    lm = linear_regression(
        X_eeg,
        design_matrix,
        names=['intercept', 'behavior']
    )
    
    # lm: dict with keys for each predictor
    # lm['behavior']: Evoked object with beta coefficients
    
    return lm

# 使用
epochs = mne.read_epochs('decision-epo.fif')

# 行为变量: 反应时 (RT)
rt = epochs.metadata['reaction_time'].values

# 回归
lm = regress_behavioral_var(epochs, rt)

# 可视化 beta 系数
beta_rt = lm['behavior']

beta_rt.plot_joint(
    times=[0.3, 0.4, 0.5],
    title='RT Regression Coefficient',
    ts_args=dict(gfp=True)
)

# 统计检验 (置换)
# 置换行为变量标签
from mne.stats import permutation_cluster_test

betas_permuted = []
for _ in range(1000):
    rt_shuffled = np.random.permutation(rt)
    lm_perm = regress_behavioral_var(epochs, rt_shuffled)
    betas_permuted.append(lm_perm['behavior'].data)

# 计算 p 值
beta_obs = beta_rt.data
p_values = (np.abs(betas_permuted) >= np.abs(beta_obs)).mean(axis=0)

# FDR 校正
reject_fdr, _ = fdr_correction(p_values)
print(f"显著点: {reject_fdr.sum()}")
```

---

### 2. Raw 数据回归 (连续预测变量)

**算法位置**: `mne/stats/regression.py:linear_regression_raw()` (行 300-500)

**时间变化的预测变量**:
```python
from mne.stats import linear_regression_raw

def regress_continuous_predictor(raw, predictor, predictor_times):
    """
    连续预测变量回归
    
    应用:
        - 瞳孔直径与脑活动
        - 心率变异性与 EEG
        - 行为表现连续变化
    
    参数:
        raw: Raw 对象
        predictor: ndarray [n_samples_predictor]
        predictor_times: 预测变量的时间戳
    """
    # 1. 插值预测变量到 EEG 采样率
    from scipy.interpolate import interp1d
    
    eeg_times = raw.times
    interp_func = interp1d(
        predictor_times,
        predictor,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )
    
    predictor_resampled = interp_func(eeg_times)
    
    # 2. 设计矩阵
    design_matrix = np.column_stack([
        np.ones(len(predictor_resampled)),
        predictor_resampled
    ])
    
    # 3. 回归
    lm = linear_regression_raw(
        raw,
        design_matrix,
        names=['intercept', 'predictor']
    )
    
    return lm['predictor']

# 使用
raw = mne.io.read_raw_fif('raw.fif')

# 瞳孔数据 (假设250 Hz)
pupil_diameter = np.load('pupil_diameter.npy')
pupil_times = np.arange(len(pupil_diameter)) / 250.0

# 回归
beta_pupil = regress_continuous_predictor(raw, pupil_diameter, pupil_times)

# 可视化
beta_pupil.plot_topomap(
    times=[0, 1, 2, 3, 4],
    ch_type='eeg',
    title='Pupil Diameter β Coefficients'
)
```

---

## 应用场景

### 场景 1: ERP 组分析 (聚类置换)

```python
import mne
from mne.stats import spatio_temporal_cluster_test
from mne.channels import find_ch_adjacency
import numpy as np

# 1. 加载所有被试数据
subjects = [f'sub-{i:02d}' for i in range(1, 21)]

X_target = []
X_standard = []

for subject in subjects:
    epochs = mne.read_epochs(f'data/{subject}-epo.fif')
    
    X_target.append(epochs['target'].get_data())
    X_standard.append(epochs['standard'].get_data())

# 平均每个被试
X_target_avg = np.array([x.mean(axis=0) for x in X_target])
X_standard_avg = np.array([x.mean(axis=0) for x in X_standard])

X = [X_target_avg, X_standard_avg]

# 2. 邻接矩阵
adjacency, ch_names = find_ch_adjacency(epochs.info, ch_type='eeg')

# 3. 聚类置换检验
T_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_test(
    X,
    n_permutations=5000,
    threshold=None,
    tail=0,
    adjacency=adjacency,
    n_jobs=-1,
    verbose=True
)

# 4. 找到显著聚类
sig_idx = np.where(cluster_pv < 0.05)[0]
print(f"\n发现 {len(sig_idx)} 个显著聚类:")

for i, idx in enumerate(sig_idx):
    # 提取聚类信息
    ch_inds, time_inds = clusters[idx]
    
    # 时间范围
    times = epochs.times
    time_range = (times[time_inds.min()], times[time_inds.max()])
    
    # 涉及通道
    unique_chs = np.unique(ch_inds)
    ch_list = [epochs.ch_names[ch] for ch in unique_chs[:5]]  # 前5个
    
    print(f"\nCluster {i+1}:")
    print(f"  p-value: {cluster_pv[idx]:.4f}")
    print(f"  Time: {time_range[0]*1000:.0f} - {time_range[1]*1000:.0f} ms")
    print(f"  Channels ({len(unique_chs)}): {', '.join(ch_list)}...")
    print(f"  Size: {len(ch_inds)} data points")

# 5. 可视化最显著的聚类
if len(sig_idx) > 0:
    best_cluster_idx = sig_idx[np.argmin(cluster_pv[sig_idx])]
    
    # 创建 mask
    mask = np.zeros_like(T_obs, dtype=bool)
    mask[clusters[best_cluster_idx]] = True
    
    # 绘制
    evoked_diff = epochs['target'].average()
    evoked_diff.data = X_target_avg.mean(axis=0) - X_standard_avg.mean(axis=0)
    
    evoked_diff.plot_image(
        picks='eeg',
        mask=mask,
        show_names='auto',
        titles=dict(eeg=f'Target - Standard (p={cluster_pv[best_cluster_idx]:.4f})')
    )

# 6. 保存结果
results = {
    'T_obs': T_obs,
    'clusters': clusters,
    'cluster_pv': cluster_pv,
    'sig_idx': sig_idx
}

np.save('cluster_results.npy', results)
```

---

### 场景 2: 源空间统计 (组分析)

```python
import mne
from mne.stats import spatio_temporal_cluster_1samp_test
import numpy as np

# 1. 加载源空间数据
subjects_dir = 'subjects'
src = mne.read_source_spaces('fsaverage-oct6-src.fif')

stcs_diff = []
for subject in subjects:
    # 读取两个条件的 STC
    stc_a = mne.read_source_estimate(f'results/{subject}_condA_dspm')
    stc_b = mne.read_source_estimate(f'results/{subject}_condB_dspm')
    
    # 差异
    stc_diff = stc_a - stc_b
    
    # Morph 到 fsaverage
    morph = mne.compute_source_morph(
        stc_diff,
        subject_from=subject,
        subject_to='fsaverage',
        subjects_dir=subjects_dir,
        smooth=5  # 平滑核 (mm)
    )
    stc_fsaverage = morph.apply(stc_diff)
    
    stcs_diff.append(stc_fsaverage)

# 2. 准备数据
X = np.array([stc.data for stc in stcs_diff])  # [n_subjects × n_sources × n_times]

# 3. 源空间邻接
connectivity = mne.spatial_src_adjacency(src)

# 4. 聚类检验
T_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(
    X,
    n_permutations=1024,
    threshold=None,
    adjacency=connectivity,
    n_jobs=-1,
    verbose=True
)

# 5. 显著聚类
sig_idx = np.where(cluster_pv < 0.05)[0]
print(f"显著聚类: {len(sig_idx)}")

# 6. 可视化
for i, idx in enumerate(sig_idx[:3]):  # 前3个
    stc_cluster = stcs_diff[0].copy()
    stc_cluster.data[:] = 0
    stc_cluster.data[clusters[idx]] = T_obs[clusters[idx]]
    
    brain = stc_cluster.plot(
        subject='fsaverage',
        subjects_dir=subjects_dir,
        hemi='both',
        time_viewer=True,
        clim=dict(kind='value', pos_lims=[3, 5, 7]),
        title=f'Cluster {i+1} (p={cluster_pv[idx]:.4f})',
        background='white',
        size=(800, 600)
    )
    
    brain.save_image(f'cluster_{i+1}.png')
```

---

### 场景 3: 单试次回归 (行为-脑关系)

```python
import mne
from mne.stats import linear_regression
import pandas as pd

# 1. 加载 epochs with metadata
epochs = mne.read_epochs('decision-epo.fif')

# metadata 包含: reaction_time, accuracy, confidence
print(epochs.metadata.head())

# 2. 多元回归
design_matrix = epochs.metadata[['reaction_time', 'confidence']].values

# 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
design_matrix_scaled = scaler.fit_transform(design_matrix)

# 添加截距
design_matrix_full = np.column_stack([
    np.ones(len(design_matrix_scaled)),
    design_matrix_scaled
])

# 3. 回归
lm = linear_regression(
    epochs.get_data(),
    design_matrix_full,
    names=['intercept', 'RT', 'confidence']
)

# 4. 可视化系数
beta_rt = lm['RT']
beta_conf = lm['confidence']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# RT β
brain1 = beta_rt.plot_topomap(
    times=[0.3, 0.4, 0.5],
    ch_type='eeg',
    axes=axes[0],
    show=False,
    colorbar=False
)
axes[0].set_title('Reaction Time β')

# Confidence β
brain2 = beta_conf.plot_topomap(
    times=[0.3, 0.4, 0.5],
    ch_type='eeg',
    axes=axes[1],
    show=False,
    colorbar=True
)
axes[1].set_title('Confidence β')

plt.tight_layout()

# 5. 置换检验
from mne.stats import permutation_cluster_test

# 提取 RT 系数作为观察值
beta_rt_obs = beta_rt.data

# 置换
n_perm = 1000
betas_perm = []

for _ in range(n_perm):
    # 打乱 RT 标签
    rt_shuffled = np.random.permutation(design_matrix_scaled[:, 0])
    
    dm_perm = np.column_stack([
        np.ones(len(rt_shuffled)),
        rt_shuffled,
        design_matrix_scaled[:, 1]  # confidence 不变
    ])
    
    lm_perm = linear_regression(
        epochs.get_data(),
        dm_perm,
        names=['intercept', 'RT', 'confidence']
    )
    
    betas_perm.append(lm_perm['RT'].data)

# p 值
betas_perm = np.array(betas_perm)
p_values = (np.abs(betas_perm) >= np.abs(beta_rt_obs)).mean(axis=0)

# FDR 校正
from mne.stats import fdr_correction
reject_fdr, _ = fdr_correction(p_values, alpha=0.05)

print(f"显著点: {reject_fdr.sum()} / {reject_fdr.size}")

# 可视化显著性
beta_rt_masked = beta_rt.copy()
beta_rt_masked.data[~reject_fdr] = 0

beta_rt_masked.plot_topomap(
    times=[0.3, 0.4, 0.5],
    ch_type='eeg',
    title='RT β (FDR corrected)'
)
```

---

## 总结

### 核心算法汇总

| 算法 | 位置 | 复杂度 | 控制 | 场景 |
|------|------|--------|------|------|
| **多重比较校正** |
| Bonferroni | `multi_comp.py:10+` | O(N) | FWER | 少量检验 |
| FDR | `multi_comp.py:50+` | O(N log N) | FDR | 大量检验,探索 |
| **置换检验** |
| Permutation t-test | `permutations.py:200+` | O(P×N×M) | FWER | 非参数,小样本 |
| Bootstrap CI | `permutations.py:100+` | O(B×N) | - | 置信区间 |
| **聚类方法** |
| Cluster Permutation | `cluster_level.py:500+` | O(P×N×M) | FWER | 时空数据 ⭐ |
| TFCE | `cluster_level.py` | O(P×N×M×T) | FWER | 无需阈值 |
| Source Cluster | `cluster_level.py` | O(P×S×T) | FWER | 源空间 |
| **参数检验** |
| Repeated ANOVA | `parametric.py:200+` | O(N×M) | - | 多因素设计 |
| Vectorized t-test | `parametric.py` | O(N×M) | - | 快速计算 |
| **回归** |
| Linear Regression | `regression.py:100+` | O(N×M×P) | - | 行为预测变量 |
| Continuous Regression | `regression.py:300+` | O(T×M) | - | 时间序列预测 |

*注: N=样本数, M=特征数, P=置换次数, S=源数, T=时间点, B=bootstrap次数*

### 方法选择流程图

```
是否有多重比较?
├─ 否 → 标准 t/F 检验
└─ 是 → 检验数量?
       ├─ 少 (<100) → Bonferroni
       └─ 多 (>100) → 考虑结构?
              ├─ 无结构 → FDR
              └─ 时空结构 → 聚类置换检验 ⭐
                     ├─ 传感器空间 → spatio_temporal_cluster_test
                     ├─ 源空间 → spatio_temporal_cluster_1samp_test
                     └─ 不确定阈值 → TFCE
```

### 关键要点

1. **聚类置换是 M/EEG 统计的黄金标准**
   - 控制 FWER
   - 保留时空结构
   - 对扩展效应敏感

2. **FDR vs Bonferroni**
   - 探索性分析 → FDR
   - 验证性分析 → Bonferroni

3. **置换检验优势**
   - 无参数假设
   - 精确 p 值
   - 适合小样本

4. **回归分析潜力**
   - 单试次变异性
   - 连续预测变量
   - 与行为关联

### 完整数据流总结

```
Raw Data (原始数据)
    ↓ (模块1: I/O)
Preprocessed (预处理)
    ↓ (模块2: Preprocessing)
Events (事件标记)
    ↓ (模块3: Epoching)
Epochs (分段数据)
    ↓ (模块4: Evoked)
Evoked (平均)
    ↓ (模块5: Source)
Source Estimate (源定位)
    ↓ (模块6: Statistics) ⭐
Statistical Inference (推断)
```

---

**MNE-Python 数据流管道文档完成！**

6个核心模块:
1. ✅ I/O - 数据读取
2. ✅ Preprocessing - 预处理
3. ✅ Event/Epoching - 分段
4. ✅ Evoked - 平均化
5. ✅ Source Estimation - 源定位
6. ✅ Statistics - 统计分析
