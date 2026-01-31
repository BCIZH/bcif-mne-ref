# MNE-Python 与 scikit-learn 详细代码位置对比分析

## 概述

本文档详细分析 MNE-Python 如何使用 scikit-learn 的具体功能，重点关注：
- ICA (Independent Component Analysis)
- PCA (Principal Component Analysis)
- 交叉验证 (Cross-validation)
- 数据预处理 (StandardScaler, Normalizer)
- 分类器 (SVM, LogisticRegression)

---

## 1. ICA - 独立成分分析

### 1.1 FastICA 算法

#### MNE 使用位置

**文件**: `mne/preprocessing/ica.py`
**行号**: 960-1020
**功能**: 去除伪迹（眼动、心电等）

```python
# mne/preprocessing/ica.py:963-1020
from sklearn.decomposition import FastICA

class ICA:
    """独立成分分析"""
    
    def fit(self, inst, picks=None, start=None, stop=None, 
            decim=None, reject=None, tstep=2.0):
        """拟合 ICA"""
        # 预处理数据
        data = self._get_data(inst, picks, start, stop, decim)
        
        # 白化（可选）
        if self.noise_cov is not None:
            data = self._pre_whiten(data)
        
        # 应用 FastICA
        if self.method == "fastica":
            # 创建 FastICA 对象
            ica = FastICA(
                n_components=self.n_components,
                algorithm='parallel',      # 或 'deflation'
                whiten=False,              # 我们已经预白化
                fun='logcosh',             # 对比函数
                fun_args={'alpha': 1.0},   # logcosh 参数
                max_iter=200,
                tol=1e-4,
                random_state=self.random_state,
                **self.fit_params
            )
            
            # 拟合数据
            ica.fit(data[:, sel].T)  # (n_samples, n_features)
            
            # 提取解混矩阵（独立成分）
            self.unmixing_matrix_ = ica.components_
            # shape: (n_components, n_channels)
            
            # 提取混合矩阵（伪逆）
            self.mixing_matrix_ = pinv(self.unmixing_matrix_)
            
            # 迭代次数
            self.n_iter_ = ica.n_iter_
        
        return self
```

**对应 scikit-learn 位置**:
- **源码**: `sklearn/decomposition/_fastica.py`
- **类定义**: `sklearn/decomposition/_fastica.py:FastICA` (line ~280-550)
- **核心算法**: `sklearn/decomposition/_fastica.py:_ica_par()` (并行) 和 `_ica_def()` (deflation)

**FastICA 源码详细**:

```python
# sklearn/decomposition/_fastica.py:50-150
def _ica_par(X, tol, g, fun_args, max_iter, w_init):
    """并行 FastICA 算法"""
    n, p = X.shape
    
    # 初始化权重矩阵
    W = w_init
    
    # 迭代优化
    for i in range(max_iter):
        # 计算对比函数的期望
        gwtx, g_wtx = g(np.dot(W, X.T), fun_args)
        # g: 非高斯性度量（logcosh, exp, cube）
        
        # 更新 W
        W1 = np.dot(gwtx, X) / n - np.dot(g_wtx[:, None], W)
        
        # 对称正交化（防止退化）
        W1 = _sym_decorrelation(W1)
        
        # 检查收敛
        lim = np.max(np.abs(np.abs(np.diag(np.dot(W1, W.T))) - 1))
        W = W1
        
        if lim < tol:
            break
    
    return W, i + 1

def _sym_decorrelation(W):
    """对称正交化: W = (W W^T)^{-1/2} W"""
    s, u = linalg.eigh(np.dot(W, W.T))
    # W = u * diag(1/sqrt(s)) * u^T * W
    return np.dot(np.dot(u * (1.0 / np.sqrt(s)), u.T), W)
```

**对比函数实现**:
```python
# sklearn/decomposition/_fastica.py:200-250
def _logcosh(x, fun_args={'alpha': 1.0}):
    """logcosh 对比函数"""
    alpha = fun_args.get('alpha', 1.0)
    x *= alpha
    
    # g(x) = tanh(x)
    gx = np.tanh(x)
    
    # g'(x) = 1 - tanh^2(x)
    g_x = alpha * (1 - gx ** 2)
    
    return gx, g_x.mean(axis=-1)

def _exp(x, fun_args=None):
    """指数对比函数"""
    exp = np.exp(-(x ** 2) / 2)
    gx = x * exp
    g_x = (1 - x ** 2) * exp
    return gx, g_x.mean(axis=-1)

def _cube(x, fun_args=None):
    """立方对比函数"""
    return x ** 3, (3 * x ** 2).mean(axis=-1)
```

**FastICA 类接口**:
```python
# sklearn/decomposition/_fastica.py:280-550
class FastICA(TransformerMixin, BaseEstimator):
    """FastICA: 快速独立成分分析"""
    
    def __init__(self, n_components=None, algorithm='parallel',
                 whiten=True, fun='logcosh', fun_args=None,
                 max_iter=200, tol=1e-4, w_init=None,
                 random_state=None):
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.random_state = random_state
    
    def fit(self, X, y=None):
        """拟合模型"""
        X = self._validate_data(X)
        
        # 白化
        if self.whiten:
            X_white, self.whitening_, self.mean_ = _whiten(X)
        else:
            X_white = X - X.mean(axis=0)
            self.mean_ = X.mean(axis=0)
        
        # 运行 FastICA
        if self.algorithm == 'parallel':
            W, self.n_iter_ = _ica_par(
                X_white, self.tol, _get_g_fun(self.fun),
                self.fun_args, self.max_iter, self.w_init
            )
        elif self.algorithm == 'deflation':
            W, self.n_iter_ = _ica_def(
                X_white, self.tol, _get_g_fun(self.fun),
                self.fun_args, self.max_iter, self.w_init
            )
        
        # 保存解混矩阵
        self.components_ = W
        
        # 计算混合矩阵
        if self.whiten:
            self.mixing_ = np.dot(
                self.whitening_.T,
                pinv(self.components_)
            )
        else:
            self.mixing_ = pinv(self.components_)
        
        return self
    
    def transform(self, X):
        """应用 ICA 分离"""
        X = X - self.mean_
        if self.whiten:
            X = np.dot(X, self.whitening_)
        # S = W * X
        return np.dot(self.components_, X.T).T
    
    def inverse_transform(self, X):
        """从独立成分重建"""
        return np.dot(X, self.components_) + self.mean_
```

**白化函数**:
```python
# sklearn/decomposition/_fastica.py:30-45
def _whiten(X, n_components=None):
    """数据白化（PCA 预处理）"""
    # 中心化
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    
    # SVD
    U, S, Vt = linalg.svd(X_centered, full_matrices=False)
    
    # 白化矩阵: K = diag(1/s) * V^T
    K = (Vt[:n_components] / S[:n_components, None]).T
    
    # 白化数据
    X_white = np.dot(X_centered, K)
    
    return X_white, K, X_mean
```

---

### 1.2 MNE 中的 ICA 应用模式

#### MNE 使用位置：去除眼动伪迹

**文件**: `mne/preprocessing/ica.py`
**行号**: 1500-1600
**功能**: 自动检测和去除眼动伪迹

```python
# mne/preprocessing/ica.py:1550
def find_bads_eog(self, inst, ch_name=None, threshold=3.0):
    """查找与 EOG 相关的 ICA 成分"""
    # 获取 EOG 通道
    eog_data = inst.get_data(picks='eog')
    
    # 获取 ICA 源信号
    sources = self.get_sources(inst)
    
    # 计算相关性
    from scipy.stats import pearsonr
    
    correlations = []
    for src in sources.get_data():
        r, p = pearsonr(src, eog_data[0])
        correlations.append(abs(r))
    
    # 找到高相关成分
    bad_idx = np.where(np.array(correlations) > threshold)[0]
    
    # 标记为坏成分
    self.exclude = list(bad_idx)
    
    return bad_idx

def apply(self, inst, exclude=None):
    """应用 ICA（去除坏成分）"""
    if exclude is None:
        exclude = self.exclude
    
    # 获取源信号
    sources = np.dot(self.unmixing_matrix_, inst._data)
    
    # 将排除的成分置零
    sources[exclude] = 0
    
    # 重建数据
    inst._data = np.dot(self.mixing_matrix_, sources)
    
    return inst
```

---

## 2. PCA - 主成分分析

### 2.1 用于降维和白化

#### MNE 使用位置

**文件**: `mne/decoding/base.py`
**行号**: ~200-300
**功能**: 数据降维

```python
# mne/decoding/base.py:~250
from sklearn.decomposition import PCA

class Vectorizer:
    """向量化和降维"""
    
    def __init__(self, n_components=0.95):
        self.pca = PCA(n_components=n_components)
    
    def fit(self, X, y=None):
        """拟合 PCA"""
        # X: (n_samples, n_channels, n_times)
        
        # 展平到 2D
        X_2d = X.reshape(X.shape[0], -1)
        
        # 拟合 PCA
        self.pca.fit(X_2d)
        
        return self
    
    def transform(self, X):
        """降维"""
        X_2d = X.reshape(X.shape[0], -1)
        return self.pca.transform(X_2d)
```

**对应 scikit-learn 位置**:
- **源码**: `sklearn/decomposition/_pca.py`
- **类定义**: `sklearn/decomposition/_pca.py:PCA` (line ~120-800)

**PCA 源码详细**:
```python
# sklearn/decomposition/_pca.py:150-800
class PCA(BaseEstimator, TransformerMixin):
    """主成分分析"""
    
    def __init__(self, n_components=None, whiten=False, 
                 svd_solver='auto', random_state=None):
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.random_state = random_state
    
    def fit(self, X, y=None):
        """拟合 PCA 模型"""
        X = self._validate_data(X)
        
        # 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 选择 SVD 方法
        if self.svd_solver == 'auto':
            if X.shape[0] >= X.shape[1]:
                svd_solver = 'full'
            else:
                svd_solver = 'randomized'
        else:
            svd_solver = self.svd_solver
        
        # 执行 SVD
        if svd_solver == 'full':
            U, S, Vt = linalg.svd(X_centered, full_matrices=False)
        elif svd_solver == 'randomized':
            U, S, Vt = randomized_svd(
                X_centered, 
                n_components=self.n_components,
                random_state=self.random_state
            )
        
        # 主成分（特征向量）
        self.components_ = Vt[:self.n_components]
        
        # 解释方差
        explained_variance = (S ** 2) / (X.shape[0] - 1)
        self.explained_variance_ = explained_variance[:self.n_components]
        
        total_var = explained_variance.sum()
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_var
        )
        
        # 奇异值
        self.singular_values_ = S[:self.n_components]
        
        return self
    
    def transform(self, X):
        """投影到主成分空间"""
        X_centered = X - self.mean_
        X_transformed = np.dot(X_centered, self.components_.T)
        
        if self.whiten:
            # 白化：除以奇异值
            X_transformed /= np.sqrt(self.explained_variance_)
        
        return X_transformed
    
    def inverse_transform(self, X):
        """从主成分空间重建"""
        if self.whiten:
            X = X * np.sqrt(self.explained_variance_)
        
        return np.dot(X, self.components_) + self.mean_
```

**随机化 SVD** (用于大数据):
```python
# sklearn/utils/extmath.py:300-500
def randomized_svd(M, n_components, n_oversamples=10, 
                   n_iter='auto', random_state=None):
    """随机化 SVD（Halko 算法）"""
    n_random = n_components + n_oversamples
    
    # 随机投影
    Q = randomized_range_finder(
        M, size=n_random, n_iter=n_iter, 
        random_state=random_state
    )
    
    # 投影矩阵到低维
    B = Q.T @ M
    
    # 小矩阵的 SVD
    Uhat, s, Vt = linalg.svd(B, full_matrices=False)
    
    # 恢复 U
    U = Q @ Uhat
    
    return U[:, :n_components], s[:n_components], Vt[:n_components]
```

---

## 3. 交叉验证

### 3.1 K-Fold 交叉验证

#### MNE 使用位置

**文件**: `mne/decoding/time_gen.py`
**行号**: ~150-250
**功能**: 解码器性能评估

```python
# mne/decoding/time_gen.py:~200
from sklearn.model_selection import cross_val_score, StratifiedKFold

class TimeDecoding:
    """时间解码"""
    
    def cross_validate(self, X, y, cv=5):
        """交叉验证评分"""
        # X: (n_epochs, n_channels, n_times)
        # y: (n_epochs,) 标签
        
        # 创建分层 K-Fold
        skf = StratifiedKFold(n_splits=cv, shuffle=True, 
                              random_state=0)
        
        # 对每个时间点进行交叉验证
        scores = []
        for t in range(X.shape[-1]):
            X_t = X[:, :, t]  # (n_epochs, n_channels)
            
            # 交叉验证
            cv_scores = cross_val_score(
                self.clf,
                X_t,
                y,
                cv=skf,
                scoring='roc_auc'
            )
            scores.append(cv_scores.mean())
        
        return np.array(scores)
```

**对应 scikit-learn 位置**:
- **cross_val_score**: `sklearn/model_selection/_validation.py:cross_val_score()` (line ~450-550)
- **StratifiedKFold**: `sklearn/model_selection/_split.py:StratifiedKFold` (line ~800-950)

**StratifiedKFold 源码**:
```python
# sklearn/model_selection/_split.py:800-950
class StratifiedKFold(BaseCrossValidator):
    """分层 K-Fold 交叉验证"""
    
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y, groups=None):
        """生成训练/测试索引"""
        y = np.asarray(y)
        n_samples = len(y)
        
        # 统计每个类别
        unique_y, y_inversed = np.unique(y, return_inverse=True)
        class_counts = np.bincount(y_inversed)
        
        # 确保每个折叠都有所有类别
        if np.min(class_counts) < self.n_splits:
            raise ValueError("每个类别至少要有 n_splits 个样本")
        
        # 为每个类别分配折叠
        indices = np.arange(n_samples)
        for test_fold_idx in range(self.n_splits):
            test_idx = []
            
            # 从每个类别中选择样本
            for cls in unique_y:
                cls_indices = indices[y == cls]
                
                # 分层采样
                n_cls_samples = len(cls_indices)
                start = test_fold_idx * n_cls_samples // self.n_splits
                end = (test_fold_idx + 1) * n_cls_samples // self.n_splits
                
                test_idx.extend(cls_indices[start:end])
            
            # 训练集 = 所有样本 - 测试集
            test_idx = np.array(test_idx)
            train_idx = np.setdiff1d(indices, test_idx)
            
            yield train_idx, test_idx
```

**cross_val_score 实现**:
```python
# sklearn/model_selection/_validation.py:450-550
def cross_val_score(estimator, X, y=None, groups=None, 
                    scoring=None, cv=None, n_jobs=None):
    """交叉验证评分"""
    cv_results = cross_validate(
        estimator, X, y, groups=groups, 
        scoring=scoring, cv=cv, n_jobs=n_jobs,
        return_train_score=False
    )
    return cv_results['test_score']

def cross_validate(estimator, X, y=None, groups=None,
                   scoring=None, cv=None, n_jobs=None,
                   return_train_score=True):
    """完整的交叉验证"""
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    
    # 并行执行每个折叠
    parallel = Parallel(n_jobs=n_jobs)
    results = parallel(
        delayed(_fit_and_score)(
            clone(estimator), X, y,
            scorer, train_idx, test_idx,
            parameters=None
        )
        for train_idx, test_idx in cv.split(X, y, groups)
    )
    
    # 汇总结果
    test_scores = [r['test_scores'] for r in results]
    
    return {
        'test_score': np.array(test_scores),
        'fit_time': np.array([r['fit_time'] for r in results]),
        'score_time': np.array([r['score_time'] for r in results])
    }
```

---

## 4. 数据预处理

### 4.1 StandardScaler (标准化)

#### MNE 使用位置

**文件**: `mne/decoding/transformer.py`
**行号**: ~100-150
**功能**: 特征标准化

```python
# mne/decoding/transformer.py:~120
from sklearn.preprocessing import StandardScaler

class Scaler:
    """时间序列标准化"""
    
    def __init__(self, scalings='mean'):
        self.scalings = scalings
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        """拟合缩放器"""
        # X: (n_epochs, n_channels, n_times)
        
        # 展平时间维度
        X_2d = X.reshape(-1, X.shape[1])  # (n_epochs*n_times, n_channels)
        
        # 拟合 StandardScaler
        self.scaler.fit(X_2d)
        
        return self
    
    def transform(self, X):
        """应用标准化"""
        orig_shape = X.shape
        X_2d = X.reshape(-1, X.shape[1])
        
        X_scaled = self.scaler.transform(X_2d)
        
        return X_scaled.reshape(orig_shape)
```

**对应 scikit-learn 位置**:
- **源码**: `sklearn/preprocessing/_data.py:StandardScaler` (line ~600-850)

**StandardScaler 源码**:
```python
# sklearn/preprocessing/_data.py:650-850
class StandardScaler(TransformerMixin, BaseEstimator):
    """标准化特征（零均值，单位方差）"""
    
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
    
    def fit(self, X, y=None, sample_weight=None):
        """计算均值和标准差"""
        X = self._validate_data(X)
        
        if self.with_mean:
            self.mean_ = np.average(X, axis=0, weights=sample_weight)
        else:
            self.mean_ = None
        
        if self.with_std:
            if self.with_mean:
                X_centered = X - self.mean_
            else:
                X_centered = X
            
            self.scale_ = np.sqrt(
                np.average(X_centered ** 2, axis=0, weights=sample_weight)
            )
            # 避免除零
            self.scale_[self.scale_ == 0] = 1.0
        else:
            self.scale_ = None
        
        return self
    
    def transform(self, X):
        """应用标准化"""
        X = X.copy()
        
        if self.with_mean:
            X -= self.mean_
        
        if self.with_std:
            X /= self.scale_
        
        return X
    
    def inverse_transform(self, X):
        """逆变换"""
        X = X.copy()
        
        if self.with_std:
            X *= self.scale_
        
        if self.with_mean:
            X += self.mean_
        
        return X
```

---

### 4.2 LabelEncoder (标签编码)

#### MNE 使用位置

**文件**: `mne/decoding/search_light.py`
**行号**: ~80
**功能**: 分类标签编码

```python
# mne/decoding/search_light.py:~90
from sklearn.preprocessing import LabelEncoder

def prepare_labels(events):
    """准备分类标签"""
    # events: 字符串标签 ['left', 'right', 'left', ...]
    
    le = LabelEncoder()
    y = le.fit_transform(events)
    # y: [0, 1, 0, ...]  (整数编码)
    
    return y, le.classes_
```

**对应 scikit-learn 位置**:
- **源码**: `sklearn/preprocessing/_label.py:LabelEncoder` (line ~100-250)

---

## 5. 分类器

### 5.1 SVM (支持向量机)

#### MNE 使用位置

**文件**: `mne/decoding/classifier.py`
**行号**: ~50-100
**功能**: 时间解码分类

```python
# mne/decoding/classifier.py:~70
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

def create_decoder(C=1.0, kernel='linear'):
    """创建 SVM 解码器"""
    clf = make_pipeline(
        StandardScaler(),
        SVC(C=C, kernel=kernel, class_weight='balanced')
    )
    return clf

# 使用示例
clf = create_decoder()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

**对应 scikit-learn 位置**:
- **SVC**: `sklearn/svm/_classes.py:SVC` (line ~600-900)
- **核心实现**: `sklearn/svm/_libsvm.pyx` (Cython 封装 LIBSVM)

**SVC 源码**:
```python
# sklearn/svm/_classes.py:650-900
class SVC(BaseSVC):
    """C-支持向量分类"""
    
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 max_iter=-1, decision_function_shape='ovr',
                 random_state=None):
        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, nu=0., shrinking=shrinking,
            probability=probability, cache_size=cache_size,
            class_weight=class_weight, max_iter=max_iter,
            random_state=random_state
        )
        self.decision_function_shape = decision_function_shape
    
    def fit(self, X, y, sample_weight=None):
        """拟合 SVM 模型"""
        # 调用 LIBSVM
        return self._fit(X, y, sample_weight)
```

**LIBSVM Cython 封装**:
```python
# sklearn/svm/_libsvm.pyx:100-300
def fit(X, y, svm_type, kernel, C, ...):
    """LIBSVM 拟合"""
    # 准备数据
    X_csr = sp.csr_matrix(X)
    
    # 调用 C 库
    cdef svm_problem problem
    problem.l = X.shape[0]
    problem.x = <svm_node **> X_csr.data
    problem.y = <double *> y
    
    # 训练
    cdef svm_model *model = svm_train(&problem, &param)
    
    # 提取支持向量
    support_vectors = ...
    dual_coef = ...
    
    return support_vectors, dual_coef, ...
```

---

### 5.2 Logistic Regression (逻辑回归)

#### MNE 使用位置

**文件**: `mne/decoding/time_gen.py`
**行号**: ~100
**功能**: 快速线性解码

```python
# mne/decoding/time_gen.py:~110
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    multi_class='ovr'  # one-vs-rest
)
```

**对应 scikit-learn 位置**:
- **源码**: `sklearn/linear_model/_logistic.py:LogisticRegression` (line ~1200-1600)
- **求解器**: 
  - `lbfgs`: `scipy.optimize.fmin_l_bfgs_b`
  - `liblinear`: `sklearn/svm/_liblinear.pyx`
  - `saga`: `sklearn/linear_model/_sag.py`

**LogisticRegression 源码**:
```python
# sklearn/linear_model/_logistic.py:1250-1600
class LogisticRegression(LinearClassifierMixin, BaseEstimator):
    """逻辑回归分类器"""
    
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 solver='lbfgs', max_iter=100, multi_class='auto'):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
    
    def fit(self, X, y, sample_weight=None):
        """拟合逻辑回归"""
        # 选择求解器
        if self.solver == 'lbfgs':
            self.coef_, self.intercept_, n_iter_ = _fit_lbfgs(
                X, y, self.C, self.penalty, self.tol, self.max_iter
            )
        elif self.solver == 'liblinear':
            self.coef_, self.intercept_ = _fit_liblinear(
                X, y, self.C, self.penalty, self.dual
            )
        
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        decision = self.decision_function(X)
        # Sigmoid 函数
        return expit(decision)
```

**L-BFGS 求解器**:
```python
# sklearn/linear_model/_logistic.py:500-700
def _fit_lbfgs(X, y, C, penalty, tol, max_iter):
    """使用 L-BFGS-B 优化逻辑回归"""
    
    def func(w, X, y, C):
        """目标函数：负对数似然 + L2 正则"""
        z = X @ w
        # Log-loss
        loss = np.log(1 + np.exp(-y * z)).sum()
        # L2 正则
        reg = 0.5 / C * (w ** 2).sum()
        return loss + reg
    
    def grad(w, X, y, C):
        """梯度"""
        z = X @ w
        s = expit(-y * z)
        grad_loss = -(X.T @ (y * s))
        grad_reg = w / C
        return grad_loss + grad_reg
    
    # 调用 scipy.optimize.fmin_l_bfgs_b
    from scipy.optimize import fmin_l_bfgs_b
    
    w0 = np.zeros(X.shape[1])
    w_opt, f_opt, info = fmin_l_bfgs_b(
        func, w0, fprime=grad,
        args=(X, y, C),
        maxiter=max_iter,
        pgtol=tol
    )
    
    return w_opt[:-1], w_opt[-1], info['nit']
```

---

## 总结表：MNE → scikit-learn 映射

| MNE 功能 | MNE 文件 | sklearn 函数/类 | sklearn 源码位置 | 核心算法 |
|---------|---------|----------------|-----------------|---------|
| ICA 去伪迹 | preprocessing/ica.py:963 | decomposition.FastICA | decomposition/_fastica.py:280 | 并行/deflation + 对称正交化 |
| 降维 | decoding/base.py:250 | decomposition.PCA | decomposition/_pca.py:150 | SVD (full/randomized) |
| 交叉验证 | decoding/time_gen.py:200 | model_selection.cross_val_score | model_selection/_validation.py:450 | K-Fold 并行评估 |
| 分层 K-Fold | decoding/time_gen.py:200 | model_selection.StratifiedKFold | model_selection/_split.py:800 | 分层采样 |
| 标准化 | decoding/transformer.py:120 | preprocessing.StandardScaler | preprocessing/_data.py:650 | (X - μ) / σ |
| 标签编码 | decoding/search_light.py:90 | preprocessing.LabelEncoder | preprocessing/_label.py:100 | 字符串 → 整数 |
| SVM 分类 | decoding/classifier.py:70 | svm.SVC | svm/_classes.py:650 | LIBSVM (C) |
| 逻辑回归 | decoding/time_gen.py:110 | linear_model.LogisticRegression | linear_model/_logistic.py:1250 | L-BFGS-B |

---

## 性能关键路径

### 1. FastICA（最耗时）
- **并行算法**: `_ica_par()` - 迭代优化
- **对称正交化**: `_sym_decorrelation()` - 防止退化
- **复杂度**: O(n_iter * n_components² * n_samples)

### 2. PCA（高维数据）
- **随机化 SVD**: `randomized_svd()` - 适用于大矩阵
- **复杂度**: O(n_components * n_samples * n_features)

### 3. SVM（训练慢）
- **LIBSVM**: C 实现的序列最小优化（SMO）
- **复杂度**: O(n_samples² * n_features)（最坏情况）

### 4. 交叉验证（可并行）
- **joblib 并行**: `Parallel(n_jobs=-1)`
- **复杂度**: O(n_folds * 训练时间)

---

## Rust 替代建议

### 1. ICA
- **crate**: `ndarray` + 自定义实现（无现成库）
- **策略**: 移植 FastICA 算法（约 500 行代码）

### 2. PCA
- **crate**: `ndarray-linalg` (SVD)
- **示例**:
  ```rust
  use ndarray_linalg::SVD;
  let (u, s, vt) = matrix.svd(true, true)?;
  ```

### 3. 交叉验证
- **crate**: 手动实现（简单）
- **策略**: 使用 `rayon` 并行化

### 4. 分类器
- **SVM**: `smartcore::svm::SVC`
- **逻辑回归**: `linfa-logistic::LogisticRegression`

---

## 下一步

继续阅读：
- [Rust 生态详细对比](./04_Rust替代方案详细分析.md)
- [代码移植优先级](./05_代码移植优先级.md)
