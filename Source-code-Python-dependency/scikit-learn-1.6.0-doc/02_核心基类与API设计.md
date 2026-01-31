# scikit-learn 核心基类与 API 设计

## 1. 简介

scikit-learn 的 API 设计是其最大的优势之一。所有估计器都遵循统一的接口，这使得用户可以轻松地切换不同的算法而无需修改太多代码。

### 1.1 设计原则

scikit-learn 的 API 设计遵循以下原则：

1. **一致性（Consistency）**: 所有对象共享一致的接口
2. **可检查性（Inspection）**: 学习后的参数易于访问
3. **非增殖（Non-proliferation of classes）**: 算法通过参数控制，而不是派生新类
4. **组合（Composition）**: 可组合成复杂的管道
5. **合理默认值（Sensible defaults）**: 提供合理的默认参数

## 2. BaseEstimator 基类

### 2.1 核心功能

`BaseEstimator` 是所有估计器的基类，位于 `sklearn/base.py`。

**继承关系**:
```python
BaseEstimator
    ├── _HTMLDocumentationLinkMixin  # HTML 文档链接
    └── _MetadataRequester            # 元数据请求
```

**提供的默认实现**:
- 参数的获取和设置（`get_params`、`set_params`）
- 文本和 HTML 表示
- 估计器序列化
- 参数验证
- 数据验证
- 特征名称验证

### 2.2 参数管理

#### 2.2.1 get_params()

获取估计器的参数：

```python
def get_params(self, deep=True):
    """获取此估计器的参数。
    
    Parameters
    ----------
    deep : bool, default=True
        如果为 True，将返回此估计器和包含的子对象的参数。
    
    Returns
    -------
    params : dict
        参数名称映射到它们的值。
    """
```

**使用示例**:
```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=5)
params = clf.get_params()
# {'n_estimators': 100, 'max_depth': 5, 'random_state': None, ...}
```

**嵌套参数**（用于 Pipeline）:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

params = pipe.get_params(deep=True)
# {
#   'scaler': StandardScaler(),
#   'classifier': SVC(),
#   'scaler__with_mean': True,
#   'classifier__C': 1.0,
#   ...
# }
```

#### 2.2.2 set_params()

设置估计器的参数：

```python
def set_params(self, **params):
    """设置此估计器的参数。
    
    Parameters
    ----------
    **params : dict
        估计器参数。
    
    Returns
    -------
    self : 估计器实例
    """
```

**使用示例**:
```python
clf = RandomForestClassifier()
clf.set_params(n_estimators=200, max_depth=10)
```

**嵌套参数设置**:
```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

pipe.set_params(svm__C=10)  # 使用 __ 分隔符
```

### 2.3 参数约束验证

#### 2.3.1 _parameter_constraints

定义参数的类型和值约束：

```python
class MyEstimator(BaseEstimator):
    _parameter_constraints = {
        'alpha': [Interval(Real, 0, None, closed='left')],  # alpha > 0
        'penalty': [StrOptions({'l1', 'l2', 'elasticnet'})],  # 字符串选项
        'max_iter': [Interval(Integral, 1, None, closed='left')],  # 整数 >= 1
    }
    
    def __init__(self, alpha=1.0, penalty='l2', max_iter=100):
        self.alpha = alpha
        self.penalty = penalty
        self.max_iter = max_iter
```

#### 2.3.2 _validate_params()

验证参数的类型和值：

```python
def fit(self, X, y):
    self._validate_params()  # 在 fit 开始时验证
    # ... 拟合逻辑
```

### 2.4 估计器克隆

#### 2.4.1 clone() 函数

创建估计器的深拷贝：

```python
def clone(estimator, *, safe=True):
    """构造一个具有相同参数的新的未拟合估计器。
    
    Parameters
    ----------
    estimator : {list, tuple, set, frozenset} of estimator instance or a single \
            estimator instance
        要克隆的估计器。
    safe : bool, default=True
        如果为 True，如果估计器不符合 scikit-learn 约定，则克隆将失败。
    
    Returns
    -------
    estimator : object
        未拟合的估计器。
    """
```

**实现逻辑**:
```python
# 1. 支持集合类型
if type(estimator) is dict:
    return {k: clone(v, safe=safe) for k, v in estimator.items()}
elif type(estimator) in (list, tuple, set, frozenset):
    return type(estimator)([clone(e, safe=safe) for e in estimator])

# 2. 检查是否有 get_params 方法
if not hasattr(estimator, "get_params"):
    raise TypeError("Cannot clone object...")

# 3. 获取参数并递归克隆
klass = estimator.__class__
new_object_params = estimator.get_params(deep=False)
for name, param in new_object_params.items():
    new_object_params[name] = clone(param, safe=False)

# 4. 创建新实例
new_object = klass(**new_object_params)
```

**使用示例**:
```python
from sklearn.base import clone
from sklearn.svm import SVC

original = SVC(C=1.0, kernel='rbf')
cloned = clone(original)

# cloned 是一个新的未拟合的 SVC 实例
# cloned.C == 1.0
# cloned.kernel == 'rbf'
```

### 2.5 表示和序列化

#### 2.5.1 __repr__()

自定义的字符串表示：

```python
>>> from sklearn.ensemble import RandomForestClassifier
>>> clf = RandomForestClassifier(n_estimators=100, max_depth=5)
>>> print(clf)
RandomForestClassifier(max_depth=5, n_estimators=100)
```

#### 2.5.2 _repr_html_()

Jupyter Notebook 中的 HTML 表示：

```python
# 在 Jupyter 中显示为交互式图表
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

pipe  # 显示交互式图表
```

#### 2.5.3 __getstate__() 和 __setstate__()

支持 pickle 序列化：

```python
import pickle
from sklearn.svm import SVC

clf = SVC().fit(X, y)

# 保存
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# 加载
with open('model.pkl', 'rb') as f:
    clf_loaded = pickle.load(f)
```

### 2.6 标签系统（Tags）

#### 2.6.1 __sklearn_tags__()

定义估计器的元数据标签：

```python
def __sklearn_tags__(self):
    return Tags(
        estimator_type=None,  # 'classifier', 'regressor', 'clusterer', 等
        target_tags=TargetTags(required=False),  # 是否需要 y
        transformer_tags=None,  # 转换器特定标签
        regressor_tags=None,    # 回归器特定标签
        classifier_tags=None,   # 分类器特定标签
    )
```

## 3. Mixin 类

Mixin 类为估计器提供特定类型的功能。遵循多重继承时 Mixin 在左侧的约定。

### 3.1 ClassifierMixin

为分类器提供功能：

```python
class ClassifierMixin:
    """分类器的 Mixin 类。
    
    提供的功能：
    - 设置 estimator_type 为 "classifier"
    - 默认的 score 方法（准确率）
    - 强制 fit 需要 y 参数
    """
    
    _estimator_type = "classifier"
    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        tags.target_tags.required = True
        return tags
    
    def score(self, X, y, sample_weight=None):
        """返回给定测试数据和标签的平均准确率。"""
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
```

**使用示例**:
```python
from sklearn.base import BaseEstimator, ClassifierMixin

class MyClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, param=1):
        self.param = param
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        return np.zeros(len(X))
```

### 3.2 RegressorMixin

为回归器提供功能：

```python
class RegressorMixin:
    """回归器的 Mixin 类。
    
    提供的功能：
    - 设置 estimator_type 为 "regressor"
    - 默认的 score 方法（R² 分数）
    - 强制 fit 需要 y 参数
    """
    
    _estimator_type = "regressor"
    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()
        tags.target_tags.required = True
        return tags
    
    def score(self, X, y, sample_weight=None):
        """返回预测的决定系数 R²。"""
        from .metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)
```

**R² 分数公式**:
$$R^2 = 1 - \frac{u}{v}$$

其中：
- $u = \sum_{i}(y_i - \hat{y}_i)^2$ (残差平方和)
- $v = \sum_{i}(y_i - \bar{y})^2$ (总平方和)

### 3.3 ClusterMixin

为聚类器提供功能：

```python
class ClusterMixin:
    """聚类器的 Mixin 类。
    
    提供的功能：
    - 设置 estimator_type 为 "clusterer"
    - fit_predict 方法
    """
    
    _estimator_type = "clusterer"
    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "clusterer"
        return tags
    
    def fit_predict(self, X, y=None, **kwargs):
        """对 X 执行聚类并返回聚类标签。"""
        self.fit(X, **kwargs)
        return self.labels_
```

**使用示例**:
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)  # 一步完成拟合和预测
```

### 3.4 TransformerMixin

为转换器提供功能：

```python
class TransformerMixin(_SetOutputMixin):
    """转换器的 Mixin 类。
    
    提供的功能：
    - fit_transform 方法
    - set_output 方法
    """
    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.transformer_tags = TransformerTags()
        return tags
    
    def fit_transform(self, X, y=None, **fit_params):
        """拟合数据，然后转换它。"""
        # 检查是否需要路由元数据到 transform
        if _routing_enabled():
            transform_params = self.get_metadata_routing().consumes(
                method="transform", params=fit_params.keys()
            )
            if transform_params:
                warnings.warn(...)
        
        return self.fit(X, y, **fit_params).transform(X)
```

**使用示例**:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 等同于 fit().transform()
```

### 3.5 BiclusterMixin

为双聚类器提供功能：

```python
class BiclusterMixin:
    """双聚类器的 Mixin 类。
    
    提供的功能：
    - biclusters_ 属性
    - get_indices 方法
    - get_shape 方法
    - get_submatrix 方法
    """
    
    @property
    def biclusters_(self):
        """返回行和列指示器。"""
        return self.rows_, self.columns_
    
    def get_indices(self, i):
        """返回第 i 个双聚类的行和列索引。"""
        rows = self.rows_[i]
        columns = self.columns_[i]
        return np.nonzero(rows)[0], np.nonzero(columns)[0]
```

### 3.6 OutlierMixin

为异常检测器提供功能：

```python
class OutlierMixin:
    """异常检测器的 Mixin 类。
    
    提供的功能：
    - fit_predict 方法（预测异常）
    """
    
    _estimator_type = "outlier_detector"
    
    def fit_predict(self, X, y=None, **kwargs):
        """执行拟合并返回是否为异常（1 或 -1）。"""
```

**使用示例**:
```python
from sklearn.ensemble import IsolationForest

clf = IsolationForest()
labels = clf.fit_predict(X)  # 1 为正常，-1 为异常
```

## 4. 估计器 API

### 4.1 核心接口

所有估计器都应实现以下接口：

#### 4.1.1 fit(X, y=None, **fit_params)

训练模型：

```python
def fit(self, X, y=None, **fit_params):
    """拟合模型。
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        训练数据。
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), optional
        目标值。
    **fit_params : dict
        其他拟合参数。
    
    Returns
    -------
    self : object
        拟合的估计器。
    """
    # 1. 验证参数
    self._validate_params()
    
    # 2. 验证输入数据
    X, y = validate_data(self, X, y)
    
    # 3. 执行拟合逻辑
    # ...
    
    # 4. 存储学习到的参数（以 _ 结尾）
    self.coef_ = ...
    self.intercept_ = ...
    
    # 5. 返回 self
    return self
```

**关键约定**:
- 必须返回 `self`，以支持链式调用
- 学习到的参数以下划线结尾（如 `coef_`、`classes_`）
- 超参数不以下划线结尾（如 `max_depth`、`n_estimators`）

#### 4.1.2 predict(X)

预测（分类器和回归器）：

```python
def predict(self, X):
    """预测 X 的目标值。
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        输入样本。
    
    Returns
    -------
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        预测值。
    """
    # 检查是否已拟合
    check_is_fitted(self)
    
    # 验证输入
    X = validate_data(self, X, reset=False)
    
    # 预测
    return self._predict(X)
```

#### 4.1.3 predict_proba(X)

预测概率（分类器）：

```python
def predict_proba(self, X):
    """预测 X 的类概率。
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        输入样本。
    
    Returns
    -------
    proba : array-like of shape (n_samples, n_classes)
        每个类的概率。
    """
```

#### 4.1.4 transform(X)

转换数据（转换器）：

```python
def transform(self, X):
    """转换 X。
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        输入样本。
    
    Returns
    -------
    X_transformed : array-like of shape (n_samples, n_features_new)
        转换后的数据。
    """
```

#### 4.1.5 score(X, y)

评估性能：

```python
def score(self, X, y, sample_weight=None):
    """返回模型的性能分数。
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        测试样本。
    y : array-like of shape (n_samples,)
        真实标签或值。
    sample_weight : array-like of shape (n_samples,), optional
        样本权重。
    
    Returns
    -------
    score : float
        性能分数。
    """
```

**默认实现**:
- `ClassifierMixin`: 返回准确率
- `RegressorMixin`: 返回 R² 分数

### 4.2 可选接口

#### 4.2.1 decision_function(X)

决策函数（分类器）：

```python
def decision_function(self, X):
    """计算决策函数。
    
    Returns
    -------
    decision : array-like of shape (n_samples,) or (n_samples, n_classes)
        决策值。
    """
```

#### 4.2.2 fit_transform(X, y=None)

拟合并转换（转换器）：

```python
def fit_transform(self, X, y=None, **fit_params):
    """拟合数据，然后转换它。"""
    return self.fit(X, y, **fit_params).transform(X)
```

某些转换器（如 PCA）有优化的实现。

#### 4.2.3 inverse_transform(X)

逆转换（某些转换器）：

```python
def inverse_transform(self, X):
    """将转换后的数据还原回原始空间。"""
```

#### 4.2.4 partial_fit(X, y=None)

增量学习（支持在线学习的估计器）：

```python
def partial_fit(self, X, y=None, classes=None):
    """增量拟合。
    
    Parameters
    ----------
    classes : array-like, optional
        所有可能的类（首次调用时需要）。
    """
```

**使用示例**:
```python
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier()
classes = np.unique(y)

for X_batch, y_batch in batches:
    clf.partial_fit(X_batch, y_batch, classes=classes)
```

#### 4.2.5 get_feature_names_out()

获取输出特征名称（转换器）：

```python
def get_feature_names_out(self, input_features=None):
    """获取转换后的特征名称。
    
    Returns
    -------
    feature_names_out : ndarray of str objects
        输出特征名称。
    """
```

## 5. 学习到的参数

### 5.1 命名约定

所有在 `fit` 过程中学习到的参数都应以下划线结尾：

```python
class LinearRegression(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        # 学习到的参数
        self.coef_ = ...           # 系数
        self.intercept_ = ...      # 截距
        self.n_features_in_ = ...  # 输入特征数
        self.feature_names_in_ = ...  # 输入特征名称（如果提供）
        return self
```

### 5.2 通用的学习参数

大多数估计器会存储以下参数：

| 参数 | 说明 |
|-----|------|
| `n_features_in_` | 拟合时的特征数 |
| `feature_names_in_` | 拟合时的特征名称（如果是 DataFrame） |
| `n_iter_` | 实际迭代次数 |
| `classes_` | 类标签（分类器） |
| `n_classes_` | 类的数量（分类器） |

### 5.3 特定估计器的参数

**线性模型**:
- `coef_`: 系数
- `intercept_`: 截距

**树模型**:
- `tree_`: 树结构
- `feature_importances_`: 特征重要性

**集成模型**:
- `estimators_`: 基估计器列表
- `feature_importances_`: 特征重要性

**聚类**:
- `labels_`: 聚类标签
- `cluster_centers_`: 聚类中心（如果适用）

## 6. 输入验证

### 6.1 validate_data()

验证和预处理输入数据：

```python
from sklearn.utils.validation import validate_data

def fit(self, X, y=None):
    # 在 fit 中验证并存储信息
    X, y = validate_data(
        self, X, y,
        accept_sparse=False,      # 是否接受稀疏矩阵
        dtype=np.float64,         # 数据类型
        ensure_min_samples=2,     # 最小样本数
        ensure_min_features=1,    # 最小特征数
        force_all_finite=True,    # 是否允许 inf/nan
        reset=True                # 是否重置 n_features_in_
    )
    
def predict(self, X):
    # 在 predict 中只验证，不重置
    X = validate_data(self, X, reset=False)
```

### 6.2 check_is_fitted()

检查估计器是否已拟合：

```python
from sklearn.utils.validation import check_is_fitted

def predict(self, X):
    # 检查是否已拟合（检查是否有学习到的参数）
    check_is_fitted(self, ['coef_', 'intercept_'])
    # 或者
    check_is_fitted(self)  # 检查任何以 _ 结尾的属性
```

## 7. 元数据路由

### 7.1 元数据请求

从 scikit-learn 1.3 开始，支持元数据路由：

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class MyClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        """
        Parameters
        ----------
        sample_weight : array-like of shape (n_samples,), default=None
            样本权重。
        """
        # 使用 sample_weight
```

### 7.2 设置元数据请求

```python
from sklearn import set_config
from sklearn.utils.metadata_routing import MetadataRequest

# 启用元数据路由
set_config(enable_metadata_routing=True)

# 定义元数据请求
clf.set_fit_request(sample_weight=True)
```

## 8. 输出配置

### 8.1 set_output()

配置输出容器类型：

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = StandardScaler()
scaler.set_output(transform="pandas")  # 输出 DataFrame

X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
X_scaled = scaler.fit_transform(X)
# X_scaled 是 DataFrame
```

**支持的输出类型**:
- `"default"`: NumPy 数组（默认）
- `"pandas"`: pandas DataFrame
- `"polars"`: Polars DataFrame

## 9. 实现自定义估计器

### 9.1 基本模板

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, validate_data
import numpy as np

class MyClassifier(ClassifierMixin, BaseEstimator):
    """我的自定义分类器。
    
    Parameters
    ----------
    param : int, default=1
        参数说明。
    """
    
    # 定义参数约束
    _parameter_constraints = {
        'param': [Interval(Integral, 1, None, closed='left')],
    }
    
    def __init__(self, param=1):
        self.param = param
    
    def fit(self, X, y):
        """拟合分类器。
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            训练数据。
        y : array-like of shape (n_samples,)
            目标值。
        
        Returns
        -------
        self : object
            拟合的分类器。
        """
        # 1. 验证参数
        self._validate_params()
        
        # 2. 验证并存储输入数据
        X, y = validate_data(self, X, y)
        
        # 3. 存储类信息
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # 4. 拟合逻辑
        # ...
        
        # 5. 存储其他学习到的参数
        self.coef_ = ...
        
        return self
    
    def predict(self, X):
        """预测类标签。
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            输入样本。
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            预测的类标签。
        """
        # 1. 检查是否已拟合
        check_is_fitted(self)
        
        # 2. 验证输入
        X = validate_data(self, X, reset=False)
        
        # 3. 预测
        # ...
        return predictions
```

### 9.2 自定义转换器模板

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

class MyTransformer(TransformerMixin, BaseEstimator):
    """我的自定义转换器。"""
    
    def __init__(self, param=1):
        self.param = param
    
    def fit(self, X, y=None):
        """学习转换参数。"""
        # 验证输入
        X = validate_data(self, X)
        
        # 学习参数
        self.mean_ = X.mean(axis=0)
        
        return self
    
    def transform(self, X):
        """应用转换。"""
        # 检查是否已拟合
        check_is_fitted(self)
        
        # 验证输入
        X = validate_data(self, X, reset=False)
        
        # 转换
        return X - self.mean_
    
    def get_feature_names_out(self, input_features=None):
        """获取输出特征名称。"""
        check_is_fitted(self)
        
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        
        return np.array([f"{name}_transformed" for name in input_features])
```

## 10. 最佳实践

### 10.1 构造函数

```python
# 好：所有参数都是显式的关键字参数
def __init__(self, *, alpha=1.0, max_iter=100):
    self.alpha = alpha
    self.max_iter = max_iter

# 不好：使用 *args 或 **kwargs
def __init__(self, *args, **kwargs):  # 避免
    pass
```

### 10.2 fit 方法

```python
def fit(self, X, y):
    # 1. 始终在开始时验证参数
    self._validate_params()
    
    # 2. 验证输入数据
    X, y = validate_data(self, X, y)
    
    # 3. 执行拟合
    # ...
    
    # 4. 始终返回 self
    return self
```

### 10.3 predict 方法

```python
def predict(self, X):
    # 1. 检查是否已拟合
    check_is_fitted(self)
    
    # 2. 验证输入（reset=False）
    X = validate_data(self, X, reset=False)
    
    # 3. 预测
    return predictions
```

### 10.4 命名约定

- **超参数**: 不以下划线结尾（`alpha`, `max_iter`）
- **学习参数**: 以下划线结尾（`coef_`, `classes_`）
- **私有方法**: 以下划线开头（`_fit_impl`, `_validate`）

## 11. 总结

scikit-learn 的 API 设计优雅而强大：

1. **BaseEstimator** 提供统一的参数管理
2. **Mixin 类** 为不同类型的估计器提供特定功能
3. **一致的接口** 使得算法可以轻松互换
4. **克隆机制** 支持模型选择和超参数调优
5. **验证工具** 确保输入数据的正确性
6. **元数据路由** 支持灵活的参数传递
7. **输出配置** 允许自定义输出格式

---

**相关文档**:
- [01_scikit-learn架构总览.md](01_scikit-learn架构总览.md)
- [03_监督学习模块.md](03_监督学习模块.md)
- [07_Pipeline与组合器.md](07_Pipeline与组合器.md)
