# MNE-Python scikit-learn 依赖详细分析

> **可选依赖**: `scikit-learn >= 1.3`  
> **使用频率**: 🔥🔥🔥 (解码分析模块)  
> **角色**: 机器学习、分类、回归、降维、交叉验证

---

## 目录

1. [scikit-learn 在 MNE 中的定位](#scikit-learn-在-mne-中的定位)
2. [mne.decoding 模块架构](#mnedecoding-模块架构)
3. [分类器集成](#分类器集成)
4. [特征提取与降维](#特征提取与降维)
5. [交叉验证策略](#交叉验证策略)
6. [Pipeline 设计模式](#pipeline-设计模式)
7. [Transformer API 实现](#transformer-api-实现)
8. [完整工作流示例](#完整工作流示例)

---

## scikit-learn 在 MNE 中的定位

### 1. 可选但核心

**依赖声明**: `scikit-learn >= 1.3` (在 `mne[full]` 中)

**使用场景**:
- ✅ **解码分析**: `mne.decoding` 模块
- ✅ **机器学习**: 分类、回归、聚类
- ✅ **特征工程**: PCA, ICA, CSP, SSD
- ✅ **模型评估**: 交叉验证、评分
- ❌ **必需功能**: 不影响 I/O、预处理、可视化

**安装检查**:
```python
import mne

# 如果未安装 sklearn
try:
    from sklearn import __version__
except ImportError:
    print("sklearn not installed - decoding module unavailable")
```

---

### 2. sklearn 模块使用统计

| sklearn 模块 | 使用位置 | 主要类/函数 | MNE 应用 |
|-------------|---------|------------|---------|
| **sklearn.base** | `mne/decoding/base.py` | `BaseEstimator`, `TransformerMixin` | 自定义 Transformer |
| **sklearn.model_selection** | `mne/decoding/` | `KFold`, `cross_val_score` | 交叉验证 |
| **sklearn.linear_model** | `mne/decoding/`, `mne/preprocessing/` | `LogisticRegression`, `Ridge` | 分类、回归 |
| **sklearn.decomposition** | `mne/preprocessing/ica.py` | `FastICA`, `PCA` | ICA, 降维 |
| **sklearn.discriminant_analysis** | `mne/decoding/` | `LinearDiscriminantAnalysis` | LDA 分类器 |
| **sklearn.svm** | `mne/decoding/` | `SVC`, `SVR` | 支持向量机 |
| **sklearn.preprocessing** | `mne/decoding/transformer.py` | `StandardScaler`, `RobustScaler` | 标准化 |
| **sklearn.pipeline** | `mne/decoding/` | `Pipeline`, `make_pipeline` | 工作流组合 |
| **sklearn.metrics** | `mne/decoding/` | `accuracy_score`, `r2_score` | 性能评估 |
| **sklearn.feature_extraction** | `mne/stats/` | `grid_to_graph` | 空间邻接 |
| **sklearn.neighbors** | `mne/surface.py` | `BallTree`, `LocalOutlierFactor` | 近邻、异常检测 |

---

## mne.decoding 模块架构

### 1. 模块结构

```
mne/decoding/
├── __init__.py              # 公开 API
├── base.py                  # 基础类 (BaseEstimator 复制)
├── csp.py                   # Common Spatial Pattern
├── ems.py                   # Event-Matched Spatial filter
├── _ged.py                  # Generalized Eigenvalue Decomposition
├── receptive_field.py       # Receptive Field 模型
├── search_light.py          # Searchlight 分析
├── ssd.py                   # Spatio-Spectral Decomposition
├── time_delaying_ridge.py   # Time-Delaying Ridge Regression
├── time_frequency.py        # 时频特征
├── transformer.py           # Scaler, Vectorizer, FilterEstimator
├── xdawn.py                 # Xdawn (也在 preprocessing)
└── tests/                   # 单元测试
```

---

### 2. 核心设计原则

**遵循 sklearn API 标准**:

```python
# sklearn 标准接口
class MyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """学习参数"""
        # ... 计算统计量
        return self
    
    def transform(self, X):
        """应用变换"""
        # ... 转换数据
        return X_transformed
    
    # 可选: fit_transform (自动实现)
    # def fit_transform(self, X, y=None):
    #     return self.fit(X, y).transform(X)
```

**MNE 扩展**:
```python
from mne.decoding import Scaler

# 添加 MNE 特定功能
class Scaler(TransformerMixin):
    def __init__(self, info=None, scalings='mean'):
        self.info = info  # MNE Info 对象
        self.scalings = scalings
    
    def fit(self, X, y=None):
        # X: (n_epochs, n_channels, n_times)
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True)
        return self
    
    def transform(self, X):
        return (X - self.mean_) / self.std_
```

---

## 分类器集成

### 1. 常用分类器

**位置**: `mne/decoding/base.py`, `mne/decoding/tests/`

```python
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

# 1. Logistic Regression (默认)
clf_lr = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

# 2. LDA (降维 + 分类)
clf_lda = LinearDiscriminantAnalysis(
    solver='lsqr',  # 'svd', 'lsqr', 'eigen'
    shrinkage='auto'
)

# 3. SVM
clf_svm = SVC(
    kernel='rbf',   # 'linear', 'poly', 'rbf'
    C=1.0,
    gamma='scale'
)
```

---

### 2. 分类流程示例

**位置**: `examples/decoding/decoding_csp_timefreq.py`

```python
from mne.decoding import CSP
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 准备数据
epochs = mne.Epochs(raw, events, tmin=0, tmax=1, baseline=None)
X = epochs.get_data()  # (n_epochs, n_channels, n_times)
y = epochs.events[:, -1]  # 类别标签

# 构建 Pipeline
csp = CSP(n_components=4, reg=None, log=True)
clf = LogisticRegression(max_iter=1000)
pipeline = make_pipeline(csp, clf)

# 交叉验证
scores = cross_val_score(
    pipeline, X, y, 
    cv=5,                          # 5-fold
    scoring='accuracy',
    n_jobs=-1                      # 并行
)

print(f"Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
```

---

### 3. 分类器包装 (SlidingEstimator)

**位置**: `mne/decoding/base.py`

```python
from mne.decoding import SlidingEstimator
from sklearn.linear_model import LogisticRegression

# 在每个时间点训练独立分类器
sliding = SlidingEstimator(
    LogisticRegression(),
    n_jobs=4,
    scoring='roc_auc'
)

# 拟合
sliding.fit(X, y)  # X: (n_epochs, n_channels, n_times)

# 预测
y_pred = sliding.predict(X_test)  # (n_epochs, n_times)

# 得分 (时间序列)
scores = sliding.score(X_test, y_test)  # (n_times,)
```

**应用**: 解码时间动态 (temporal decoding)

---

## 特征提取与降维

### 1. PCA - 主成分分析

**位置**: `mne/preprocessing/tests/test_infomax.py`

```python
from sklearn.decomposition import PCA

# 降维
pca = PCA(n_components=0.95,  # 保留 95% 方差
          whiten=True)

X_pca = pca.fit_transform(X)

# 解释方差
explained_var = pca.explained_variance_ratio_
print(f"Components: {pca.n_components_}")
print(f"Explained variance: {explained_var.sum():.2%}")
```

---

### 2. ICA - 独立成分分析

**位置**: `mne/preprocessing/ica.py`

```python
from sklearn.decomposition import FastICA

class ICA:
    def __init__(self, method='fastica', ...):
        if method == 'fastica':
            from sklearn.decomposition import FastICA
            
            self._ica = FastICA(
                n_components=n_components,
                algorithm='parallel',  # 'parallel', 'deflation'
                fun='logcosh',         # 'logcosh', 'exp', 'cube'
                max_iter=200,
                random_state=random_state
            )
    
    def fit(self, inst):
        data = inst.get_data()  # (n_channels, n_times)
        
        # sklearn FastICA
        self._ica.fit(data.T)  # 转置: (n_times, n_channels)
        
        # 提取混合矩阵和解混矩阵
        self.mixing_matrix_ = self._ica.mixing_  # (n_channels, n_components)
        self.unmixing_matrix_ = self._ica.components_  # (n_components, n_channels)
        
        return self
```

---

### 3. CSP - 共空间模式

**位置**: `mne/decoding/csp.py`

```python
from mne.decoding import CSP

# CSP 基于广义特征值分解
csp = CSP(
    n_components=4,      # 提取 4 个空间滤波器
    reg=None,            # 正则化 (None, 'shrinkage', float)
    log=True,            # 对特征取对数
    cov_est='concat',    # 协方差估计 ('concat', 'epoch')
    transform_into='average_power'  # 'average_power', 'csp_space'
)

# 拟合 (需要两类数据)
csp.fit(X, y)  # X: (n_epochs, n_channels, n_times), y: 类别标签

# 变换
X_csp = csp.transform(X)  # (n_epochs, n_components * 2)
# 前 n_components 个: 类 1 最大方差
# 后 n_components 个: 类 2 最大方差

# 空间模式
patterns = csp.patterns_  # (n_channels, n_components * 2)
```

**CSP 工作原理**:
1. 计算两类协方差矩阵: C1, C2
2. 求解广义特征值问题: C1 v = λ (C1 + C2) v
3. 选择最大和最小特征值对应的特征向量
4. 投影数据到 CSP 空间

---

### 4. SSD - 时空谱分解

**位置**: `mne/decoding/ssd.py`

```python
from mne.decoding import SSD

# 针对特定频段的空间滤波
ssd = SSD(
    info=epochs.info,
    filt_params_signal=(8, 12),   # 信号频段 (alpha)
    filt_params_noise=(6, 7, 13, 14),  # 噪声频段
    reg='oas',                     # 协方差正则化
    n_components=4
)

# 拟合
ssd.fit(X)

# 提取成分
X_ssd = ssd.transform(X)

# 可视化空间模式
ssd.plot_patterns(epochs.info)
```

---

## 交叉验证策略

### 1. KFold 交叉验证

**位置**: `mne/decoding/base.py`

```python
from sklearn.model_selection import KFold, StratifiedKFold

# K-Fold (回归任务)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 训练和评估

# Stratified K-Fold (分类任务 - 保持类别比例)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in skf.split(X, y):
    # 确保每折中类别分布一致
    ...
```

---

### 2. cross_val_score

**位置**: `mne/decoding/tests/`

```python
from sklearn.model_selection import cross_val_score

# 方便的 CV 评分
scores = cross_val_score(
    estimator=pipeline,
    X=X, y=y,
    cv=5,                    # 折数或 CV 对象
    scoring='accuracy',      # 'accuracy', 'roc_auc', 'r2', ...
    n_jobs=-1,               # 并行 (所有 CPU)
    verbose=1
)

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

### 3. GeneralizingEstimator

**位置**: `mne/decoding/base.py`

```python
from mne.decoding import GeneralizingEstimator

# 在时间点 i 训练，在时间点 j 测试
gen = GeneralizingEstimator(
    LogisticRegression(),
    n_jobs=4,
    scoring='roc_auc'
)

gen.fit(X_train, y_train)

# 泛化矩阵 (train_time x test_time)
scores = gen.score(X_test, y_test)  # (n_times, n_times)

# 可视化
import matplotlib.pyplot as plt
plt.imshow(scores, origin='lower', cmap='RdBu_r')
plt.xlabel('Testing Time')
plt.ylabel('Training Time')
plt.colorbar(label='ROC AUC')
```

---

## Pipeline 设计模式

### 1. make_pipeline 简化

**位置**: `examples/decoding/`

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from mne.decoding import Scaler, Vectorizer

# 方法 1: sklearn Pipeline
pipeline_sklearn = make_pipeline(
    StandardScaler(),              # 标准化
    LogisticRegression()           # 分类器
)

# 方法 2: MNE + sklearn 混合
pipeline_mne = make_pipeline(
    Scaler(scalings='mean'),       # MNE Scaler (保留 3D 形状)
    Vectorizer(),                  # 展平为 2D
    StandardScaler(),              # sklearn 标准化
    LogisticRegression()
)

# 训练
pipeline_mne.fit(X, y)  # X: (n_epochs, n_channels, n_times)

# 预测
y_pred = pipeline_mne.predict(X_test)
```

---

### 2. 自定义 Pipeline 步骤

```python
from sklearn.base import BaseEstimator, TransformerMixin

class EpochsVectorizer(BaseEstimator, TransformerMixin):
    """展平 MNE Epochs 数据"""
    
    def fit(self, X, y=None):
        return self  # 无参数需要学习
    
    def transform(self, X):
        # X: (n_epochs, n_channels, n_times)
        n_epochs = X.shape[0]
        return X.reshape(n_epochs, -1)  # (n_epochs, n_channels * n_times)

# 使用
pipeline = make_pipeline(
    EpochsVectorizer(),
    StandardScaler(),
    LogisticRegression()
)
```

---

### 3. Pipeline 参数调优

```python
from sklearn.model_selection import GridSearchCV

# 定义 Pipeline
pipeline = make_pipeline(
    CSP(n_components=4),
    LogisticRegression()
)

# 参数网格 (使用 step名称__参数名)
param_grid = {
    'csp__n_components': [2, 4, 6, 8],
    'csp__reg': [None, 'ledoit_wolf', 0.1],
    'logisticregression__C': [0.01, 0.1, 1, 10],
}

# 网格搜索
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X, y)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
```

---

## Transformer API 实现

### 1. Scaler - 标准化

**位置**: `mne/decoding/transformer.py`

```python
from sklearn.preprocessing import StandardScaler, RobustScaler

class Scaler(TransformerMixin):
    """保留 3D 数组形状的 Scaler"""
    
    def __init__(self, info=None, scalings='mean', with_mean=True, with_std=True):
        self.scalings = scalings
        self.with_mean = with_mean
        self.with_std = with_std
        
        if scalings == 'mean':
            self._scaler = StandardScaler(
                with_mean=with_mean, 
                with_std=with_std
            )
        elif scalings == 'median':
            self._scaler = RobustScaler(
                with_centering=with_mean,
                with_scaling=with_std
            )
    
    def fit(self, X, y=None):
        # X: (n_epochs, n_channels, n_times)
        n_epochs, n_channels, n_times = X.shape
        
        # 沿 epochs 和 times 轴标准化
        X_2d = X.transpose(1, 0, 2).reshape(n_channels, -1).T
        # X_2d: (n_epochs * n_times, n_channels)
        
        self._scaler.fit(X_2d)
        return self
    
    def transform(self, X):
        n_epochs, n_channels, n_times = X.shape
        X_2d = X.transpose(1, 0, 2).reshape(n_channels, -1).T
        
        X_scaled = self._scaler.transform(X_2d)
        
        # 恢复 3D 形状
        X_scaled = X_scaled.T.reshape(n_channels, n_epochs, n_times)
        X_scaled = X_scaled.transpose(1, 0, 2)
        
        return X_scaled
```

---

### 2. Vectorizer - 展平

**位置**: `mne/decoding/transformer.py`

```python
class Vectorizer(TransformerMixin):
    """将 3D epochs 展平为 2D"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X: (n_epochs, n_channels, n_times)
        n_epochs = X.shape[0]
        return X.reshape(n_epochs, -1)
    
    def inverse_transform(self, X):
        # 恢复原始形状 (需要记录)
        n_epochs = X.shape[0]
        return X.reshape(n_epochs, self.n_channels_, self.n_times_)
```

---

### 3. FilterEstimator - 滤波包装

**位置**: `mne/decoding/transformer.py`

```python
class FilterEstimator(TransformerMixin):
    """在 sklearn Pipeline 中应用 MNE 滤波"""
    
    def __init__(self, info, l_freq, h_freq, method='fir'):
        self.info = info
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.method = method
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        from mne.filter import filter_data
        
        # X: (n_epochs, n_channels, n_times)
        X_filt = np.empty_like(X)
        
        for i in range(X.shape[0]):
            X_filt[i] = filter_data(
                X[i], 
                sfreq=self.info['sfreq'],
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                method=self.method
            )
        
        return X_filt
```

---

## 完整工作流示例

### 示例: Motor Imagery 分类

```python
import mne
from mne.decoding import CSP, Scaler, Vectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 1. 加载数据
epochs = mne.read_epochs('motor_imagery_epochs-epo.fif')

# 2. 准备特征和标签
X = epochs.get_data()  # (n_epochs, n_channels, n_times)
y = epochs.events[:, -1]  # 左手 vs 右手

# 3. 构建 Pipeline
pipeline = make_pipeline(
    # Step 1: 标准化 (保留 3D)
    Scaler(epochs.info, scalings='mean'),
    
    # Step 2: CSP 空间滤波
    CSP(n_components=4, reg='ledoit_wolf', log=True),
    
    # Step 3: LDA 分类器
    LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
)

# 4. 交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    pipeline, X, y,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

print(f"分类准确率: {scores.mean():.2%} ± {scores.std():.2%}")

# 5. 训练最终模型
pipeline.fit(X, y)

# 6. 提取 CSP 模式
csp = pipeline.named_steps['csp']
patterns = csp.patterns_

# 7. 可视化空间模式
import matplotlib.pyplot as plt
from mne.viz import plot_topomap

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    plot_topomap(
        patterns[:, i], 
        epochs.info, 
        axes=axes[i],
        show=False
    )
    axes[i].set_title(f'CSP {i+1}')

plt.tight_layout()
plt.show()
```

---

## 总结

### scikit-learn 在 MNE 中的价值

| 维度 | 评分 | 说明 |
|------|------|------|
| **解码分析** | ⭐⭐⭐⭐⭐ | `mne.decoding` 核心依赖 |
| **机器学习** | ⭐⭐⭐⭐⭐ | 分类、回归标准接口 |
| **Pipeline** | ⭐⭐⭐⭐⭐ | 工作流组合关键 |
| **交叉验证** | ⭐⭐⭐⭐⭐ | 模型评估必备 |
| **可替代性** | ⭐⭐⭐ | 可用其他 ML 库，但需适配 |

---

### MNE 对 sklearn 的扩展

1. **保留数据维度**: Scaler, Vectorizer 处理 3D epochs
2. **时间解码**: SlidingEstimator, GeneralizingEstimator
3. **神经科学特定**: CSP, SSD, Xdawn
4. **无缝集成**: 遵循 sklearn API 标准

---

**返回**: [依赖分析总览](dependency-analysis-overview.md)  
**上一步**: [SciPy 依赖分析](dependency-scipy.md)
