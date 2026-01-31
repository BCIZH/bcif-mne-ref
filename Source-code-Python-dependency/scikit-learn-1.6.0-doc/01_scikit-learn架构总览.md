# scikit-learn 1.6.0 架构总览

## 1. 简介

scikit-learn 是一个基于 Python 的机器学习库，建立在 NumPy、SciPy 和 matplotlib 之上。它提供了简单高效的数据挖掘和数据分析工具，适用于各种机器学习任务。

### 1.1 核心特点
- **简单一致的 API**: 所有估计器遵循相同的接口（fit、predict、transform）
- **丰富的算法**: 涵盖分类、回归、聚类、降维等
- **高性能**: 核心算法使用 Cython 优化
- **良好的文档**: 完善的文档和大量示例
- **BSD 许可证**: 商业友好的开源许可

### 1.2 版本信息
- **版本**: 1.6.0
- **开始时间**: 2007年（Google Summer of Code项目）
- **官网**: https://scikit-learn.org
- **文档**: https://scikit-learn.org/stable/
- **源代码**: https://github.com/scikit-learn/scikit-learn

## 2. 整体架构

### 2.1 架构层次

```
┌─────────────────────────────────────────────────┐
│         用户 API 层（估计器接口）                 │
│  - BaseEstimator                                │
│  - ClassifierMixin, RegressorMixin, etc.        │
├─────────────────────────────────────────────────┤
│         算法实现层                               │
│  - 监督学习（分类、回归）                        │
│  - 无监督学习（聚类、降维）                      │
│  - 半监督学习                                    │
├─────────────────────────────────────────────────┤
│         工具和辅助层                             │
│  - 数据预处理                                    │
│  - 特征工程                                      │
│  - 模型选择和评估                                │
│  - 管道（Pipeline）                              │
├─────────────────────────────────────────────────┤
│         底层优化层                               │
│  - Cython 实现                                   │
│  - 稀疏矩阵支持                                  │
│  - 并行计算（joblib）                            │
├─────────────────────────────────────────────────┤
│         依赖库层                                 │
│  - NumPy, SciPy                                 │
│  - joblib, threadpoolctl                        │
└─────────────────────────────────────────────────┘
```

### 2.2 核心设计原则

1. **一致性**: 所有对象共享一致的接口
2. **可检查性**: 学习后的参数易于访问
3. **可组合性**: 可以组合成 Pipeline
4. **合理默认值**: 提供良好的默认参数
5. **限制对象层次**: 避免深层继承

## 3. 目录结构

### 3.1 顶层目录
```
scikit-learn-1.6.0/
├── sklearn/            # 主要源代码
├── doc/               # 文档源文件
├── examples/          # 示例代码
├── benchmarks/        # 性能基准测试
└── meson.build        # 构建配置
```

### 3.2 sklearn/ 核心目录结构
```
sklearn/
├── __init__.py           # 包初始化
├── base.py              # 基础类定义
├── _config.py           # 全局配置
├── exceptions.py        # 异常定义
├── utils/               # 通用工具函数
│
├── # 监督学习
├── linear_model/        # 线性模型
├── tree/                # 决策树
├── ensemble/            # 集成学习
├── svm/                 # 支持向量机
├── neural_network/      # 神经网络
├── neighbors/           # 最近邻
├── naive_bayes.py       # 朴素贝叶斯
├── discriminant_analysis.py  # 判别分析
├── gaussian_process/    # 高斯过程
│
├── # 无监督学习
├── cluster/             # 聚类
├── decomposition/       # 矩阵分解
├── manifold/            # 流形学习
├── mixture/             # 混合模型
├── covariance/          # 协方差估计
│
├── # 模型选择和评估
├── model_selection/     # 模型选择
├── metrics/             # 评估指标
├── inspection/          # 模型检查
│
├── # 数据处理
├── preprocessing/       # 数据预处理
├── feature_extraction/  # 特征提取
├── feature_selection/   # 特征选择
├── impute/              # 缺失值处理
│
├── # 其他
├── pipeline.py          # 管道
├── compose/             # 组合器
├── datasets/            # 数据集
├── calibration.py       # 概率校准
├── semi_supervised/     # 半监督学习
├── multiclass.py        # 多类分类
├── multioutput.py       # 多输出
├── dummy.py             # 虚拟估计器
├── isotonic.py          # 等渗回归
├── kernel_approximation.py  # 核近似
├── kernel_ridge.py      # 核岭回归
├── random_projection.py # 随机投影
└── cross_decomposition/ # 交叉分解
```

## 4. 模块分类

### 4.1 监督学习模块

#### 4.1.1 分类和回归

| 模块 | 说明 | 主要算法 |
|-----|------|---------|
| `linear_model` | 线性模型 | 线性回归、逻辑回归、Lasso、Ridge、ElasticNet、SGD |
| `tree` | 决策树 | DecisionTreeClassifier、DecisionTreeRegressor |
| `ensemble` | 集成方法 | RandomForest、GradientBoosting、AdaBoost、Bagging、Voting、Stacking |
| `svm` | 支持向量机 | SVC、SVR、LinearSVC、NuSVC |
| `neural_network` | 神经网络 | MLPClassifier、MLPRegressor |
| `neighbors` | 最近邻 | KNeighborsClassifier、KNeighborsRegressor、RadiusNeighbors |
| `gaussian_process` | 高斯过程 | GaussianProcessClassifier、GaussianProcessRegressor |
| `naive_bayes` | 朴素贝叶斯 | GaussianNB、MultinomialNB、BernoulliNB |
| `discriminant_analysis` | 判别分析 | LinearDiscriminantAnalysis、QuadraticDiscriminantAnalysis |

#### 4.1.2 其他监督学习

| 模块 | 说明 |
|-----|------|
| `calibration` | 概率校准 |
| `isotonic` | 等渗回归 |
| `kernel_ridge` | 核岭回归 |
| `multiclass` | 多类分类策略 |
| `multioutput` | 多输出策略 |

### 4.2 无监督学习模块

| 模块 | 说明 | 主要算法 |
|-----|------|---------|
| `cluster` | 聚类 | KMeans、DBSCAN、AgglomerativeClustering、SpectralClustering |
| `decomposition` | 降维/矩阵分解 | PCA、NMF、FastICA、FactorAnalysis、TruncatedSVD |
| `manifold` | 流形学习 | TSNE、Isomap、MDS、LocallyLinearEmbedding |
| `mixture` | 混合模型 | GaussianMixture、BayesianGaussianMixture |
| `covariance` | 协方差估计 | EmpiricalCovariance、ShrunkCovariance、GraphicalLasso |

### 4.3 模型选择和评估

| 模块 | 说明 | 主要功能 |
|-----|------|---------|
| `model_selection` | 模型选择 | 交叉验证、网格搜索、学习曲线、数据集划分 |
| `metrics` | 评估指标 | 准确率、精确率、召回率、F1、ROC-AUC、MSE、R² |
| `inspection` | 模型检查 | 部分依赖图、排列重要性 |

### 4.4 数据预处理和特征工程

| 模块 | 说明 | 主要功能 |
|-----|------|---------|
| `preprocessing` | 数据预处理 | 标准化、归一化、编码、分箱 |
| `feature_extraction` | 特征提取 | 文本向量化、图像特征 |
| `feature_selection` | 特征选择 | 方差阈值、单变量选择、递归特征消除 |
| `impute` | 缺失值处理 | SimpleImputer、IterativeImputer、KNNImputer |

### 4.5 工具和辅助模块

| 模块 | 说明 |
|-----|------|
| `pipeline` | 管道工具 |
| `compose` | 转换器组合 |
| `datasets` | 数据集加载 |
| `utils` | 通用工具函数 |
| `dummy` | 基线估计器 |
| `exceptions` | 异常类 |

### 4.6 实验性和专用模块

| 模块 | 说明 | 状态 |
|-----|------|-----|
| `semi_supervised` | 半监督学习 | 稳定 |
| `random_projection` | 随机投影 | 稳定 |
| `kernel_approximation` | 核近似 | 稳定 |
| `cross_decomposition` | 交叉分解 | 稳定 |
| `experimental` | 实验性功能 | 实验 |
| `frozen` | 冻结功能 | 已弃用 |

## 5. 核心概念

### 5.1 估计器（Estimator）

scikit-learn 的核心概念，所有算法都实现为估计器：

```python
from sklearn.linear_model import LinearRegression

# 创建估计器
model = LinearRegression()

# 拟合数据
model.fit(X_train, y_train)

# 预测
predictions = model.predict(X_test)
```

**估计器接口**:
- `fit(X, y)`: 训练模型
- `predict(X)`: 预测
- `score(X, y)`: 评估性能
- `get_params()`: 获取参数
- `set_params(**params)`: 设置参数

### 5.2 转换器（Transformer）

用于数据转换的估计器：

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**转换器接口**:
- `fit(X, y=None)`: 学习转换参数
- `transform(X)`: 应用转换
- `fit_transform(X, y=None)`: 拟合并转换
- `inverse_transform(X)`: 逆转换（某些转换器）

### 5.3 预测器（Predictor）

用于预测的估计器（分类器、回归器）：

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)
```

**预测器接口**:
- `predict(X)`: 预测标签
- `predict_proba(X)`: 预测概率（分类器）
- `decision_function(X)`: 决策函数
- `score(X, y)`: 评分

### 5.4 管道（Pipeline）

组合多个步骤：

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### 5.5 Mixin 类

提供特定功能的混入类：

- `ClassifierMixin`: 分类器的默认 `score` 方法
- `RegressorMixin`: 回归器的默认 `score` 方法
- `TransformerMixin`: 提供 `fit_transform` 方法
- `ClusterMixin`: 聚类器的 `fit_predict` 方法
- `OutlierMixin`: 异常检测的 `fit_predict` 方法

## 6. 依赖关系

### 6.1 核心依赖

```
必需:
- Python >= 3.9
- NumPy >= 1.19.5
- SciPy >= 1.6.0
- joblib >= 1.2.0
- threadpoolctl >= 3.1.0

可选:
- matplotlib >= 3.3.4 (绘图)
- pandas >= 1.1.5 (数据处理)
- scikit-image >= 0.17.2 (图像)
- plotly >= 5.14.0 (交互式可视化)
```

### 6.2 依赖关系图

```
scikit-learn
    ├── NumPy (数组操作)
    ├── SciPy (科学计算)
    │   └── 优化、稀疏矩阵、统计
    ├── joblib (并行计算)
    │   └── 内存化、持久化
    └── threadpoolctl (线程控制)
        └── BLAS/OpenMP 线程管理
```

## 7. API 设计哲学

### 7.1 一致性

所有估计器遵循相同的接口：

```python
# 所有估计器都有相同的基本方法
estimator.fit(X, y)
estimator.predict(X)
estimator.score(X, y)
```

### 7.2 检查性

训练后的参数以下划线结尾：

```python
model.fit(X, y)
# 学习到的参数
print(model.coef_)        # 系数
print(model.intercept_)   # 截距
print(model.feature_importances_)  # 特征重要性
```

### 7.3 非增殖接口

避免创建太多相似的类：

```python
# 不好: 为每种正则化创建新类
# LinearRegressionL1, LinearRegressionL2, ...

# 好: 通过参数控制
from sklearn.linear_model import ElasticNet
model = ElasticNet(l1_ratio=0.5)  # L1和L2的混合
```

### 7.4 组合优于继承

使用 Pipeline 和 FeatureUnion 组合功能：

```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

# 组合多个转换器
combined = FeatureUnion([
    ('pca', PCA(n_components=2)),
    ('select', SelectKBest(k=1))
])

# 创建管道
pipeline = Pipeline([
    ('features', combined),
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])
```

### 7.5 合理的默认值

提供在大多数情况下有效的默认参数：

```python
from sklearn.ensemble import RandomForestClassifier

# 可以直接使用默认参数
clf = RandomForestClassifier()  # n_estimators=100, max_depth=None, ...
```

## 8. 性能优化

### 8.1 Cython 优化

关键算法使用 Cython 实现以提高性能：

```
sklearn/
├── tree/_tree.pyx           # 决策树核心
├── ensemble/_gradient_boosting.pyx  # 梯度提升
├── linear_model/_cd_fast.pyx        # 坐标下降
└── svm/src/libsvm/          # LIBSVM C++ 实现
```

### 8.2 稀疏矩阵支持

许多算法支持稀疏矩阵，节省内存：

```python
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

X_sparse = csr_matrix(X)
model = LogisticRegression()
model.fit(X_sparse, y)
```

### 8.3 并行计算

通过 `joblib` 实现并行：

```python
from sklearn.ensemble import RandomForestClassifier

# n_jobs=-1 使用所有 CPU 核心
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X, y)
```

### 8.4 增量学习

某些算法支持增量学习，适合大数据集：

```python
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier()
for X_batch, y_batch in batches:
    clf.partial_fit(X_batch, y_batch, classes=np.unique(y))
```

## 9. 构建系统

### 9.1 构建工具

- **主要构建系统**: Meson (现代化)
- **配置文件**: `pyproject.toml`, `meson.build`

### 9.2 编译特性

- Cython 代码编译
- OpenMP 并行支持
- BLAS/LAPACK 集成

## 10. 测试框架

### 10.1 测试工具

- **测试框架**: pytest
- **覆盖率**: pytest-cov
- **最低版本**: pytest >= 7.1.2

### 10.2 测试组织

每个模块都有对应的 `tests/` 目录：

```
sklearn/linear_model/
├── _base.py
├── _ridge.py
└── tests/
    ├── test_base.py
    ├── test_ridge.py
    └── ...
```

## 11. 文档结构

```
doc/
├── user_guide.rst          # 用户指南
├── api_reference.py        # API 参考
├── modules/                # 详细模块文档
├── tutorial/               # 教程
├── auto_examples/          # 自动生成的示例
└── whats_new/              # 版本更新说明
```

## 12. 常用工具函数

### 12.1 utils 模块

```python
sklearn.utils/
├── validation.py           # 输入验证
├── multiclass.py           # 多类工具
├── class_weight.py         # 类权重计算
├── extmath.py              # 扩展数学函数
├── fixes.py                # 兼容性修复
├── sparsefuncs.py          # 稀疏矩阵函数
├── graph.py                # 图算法
└── random.py               # 随机数工具
```

### 12.2 常用函数

```python
from sklearn.utils import (
    check_array,            # 检查数组
    check_X_y,              # 检查X和y
    check_random_state,     # 检查随机状态
    shuffle,                # 打乱数据
    resample,               # 重采样
)
```

## 13. 版本演进

### 13.1 主要里程碑

- **0.1** (2010): 首个公开版本
- **0.20** (2018): 最后支持 Python 2.7
- **1.0** (2021): 首个稳定的 API
- **1.1** (2022): 最低 Python 3.8
- **1.6** (2024): 当前版本

### 13.2 API 稳定性

- **稳定 API**: 保证向后兼容
- **实验性**: `sklearn.experimental` 中的功能
- **弃用策略**: 至少两个版本的弃用期

## 14. 社区和贡献

### 14.1 开发流程

- GitHub 问题跟踪
- Pull Request 审查
- 持续集成（Azure Pipelines, CircleCI）
- 代码审查和测试

### 14.2 代码规范

- PEP 8 Python 代码风格
- Black 代码格式化
- NumPy 风格的文档字符串

## 15. 许可证

- **许可证类型**: BSD 3-Clause License
- **商业友好**: 可用于商业项目
- **开源**: 完全开源

## 16. 相关资源

- **官方文档**: https://scikit-learn.org/stable/
- **教程**: https://scikit-learn.org/stable/tutorial/
- **GitHub**: https://github.com/scikit-learn/scikit-learn
- **Stack Overflow**: 标签 `scikit-learn`
- **邮件列表**: scikit-learn@python.org

---

**下一步阅读**:
- [02_核心基类与API设计.md](02_核心基类与API设计.md) - 估计器API详解
- [03_监督学习模块.md](03_监督学习模块.md) - 分类和回归算法
- [04_无监督学习模块.md](04_无监督学习模块.md) - 聚类和降维
- [05_模型选择与评估.md](05_模型选择与评估.md) - 交叉验证和指标
- [06_数据预处理与工程.md](06_数据预处理与工程.md) - 特征工程和转换
