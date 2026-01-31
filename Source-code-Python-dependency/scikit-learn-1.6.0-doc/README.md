# scikit-learn 1.6.0 文档索引

欢迎查阅 scikit-learn 1.6.0 源码分析文档！本文档集旨在帮助您深入理解 scikit-learn 的架构、模块和实现细节。

## 📚 文档结构

### 核心文档

1. **[01_scikit-learn架构总览.md](01_scikit-learn架构总览.md)**
   - scikit-learn 简介和核心特点
   - 整体架构和设计原则
   - 目录结构和模块分类
   - 依赖关系和构建系统
   - API 设计哲学

2. **[02_核心基类与API设计.md](02_核心基类与API设计.md)**
   - BaseEstimator 基类详解
   - 参数管理（get_params、set_params）
   - 估计器克隆机制
   - Mixin 类（ClassifierMixin、RegressorMixin 等）
   - 估计器 API 规范
   - 实现自定义估计器

3. **[03_监督学习模块.md](03_监督学习模块.md)**
   - linear_model（线性模型、逻辑回归、SGD）
   - tree（决策树）
   - ensemble（随机森林、梯度提升、AdaBoost）
   - svm（支持向量机）
   - neural_network（多层感知器）
   - neighbors（K 近邻）
   - 其他监督学习算法
   - 算法选择指南

4. **[04_无监督学习模块.md](04_无监督学习模块.md)**
   - cluster（KMeans、DBSCAN、层次聚类）
   - decomposition（PCA、NMF、ICA）
   - manifold（t-SNE、Isomap、LLE）
   - mixture（高斯混合模型）
   - covariance（协方差估计）
   - 聚类和降维方法对比

5. **[05_模型选择与评估.md](05_模型选择与评估.md)**
   - 数据划分（train_test_split、交叉验证）
   - 超参数调优（GridSearchCV、RandomizedSearchCV）
   - 学习曲线和验证曲线
   - 评估指标（分类、回归、聚类）
   - 模型检查（部分依赖图、排列重要性）

6. **[06_数据预处理与特征工程.md](06_数据预处理与特征工程.md)**
   - 特征缩放（StandardScaler、MinMaxScaler）
   - 编码（OneHotEncoder、LabelEncoder）
   - 非线性变换
   - 特征选择（过滤、包装、嵌入）
   - 缺失值处理
   - Pipeline 和 ColumnTransformer

## 🎯 快速导航

### 按任务查找

#### 分类任务
- [逻辑回归](03_监督学习模块.md#2251-逻辑回归-logisticregression)
- [随机森林](03_监督学习模块.md#421-随机森林-randomforest)
- [梯度提升](03_监督学习模块.md#422-梯度提升-gradientboosting)
- [支持向量机](03_监督学习模块.md#51-核心算法)
- [K 近邻](03_监督学习模块.md#62-neighbors---最近邻)
- [朴素贝叶斯](03_监督学习模块.md#63-naive_bayes---朴素贝叶斯)

#### 回归任务
- [线性回归](03_监督学习模块.md#221-线性回归-linearregression)
- [岭回归](03_监督学习模块.md#222-岭回归-ridge)
- [Lasso](03_监督学习模块.md#223-lasso)
- [随机森林回归](03_监督学习模块.md#421-随机森林-randomforest)
- [梯度提升回归](03_监督学习模块.md#422-梯度提升-gradientboosting)

#### 聚类任务
- [KMeans](04_无监督学习模块.md#211-kmeans)
- [DBSCAN](04_无监督学习模块.md#212-dbscan)
- [层次聚类](04_无监督学习模块.md#213-层次聚类-agglomerativeclustering)
- [谱聚类](04_无监督学习模块.md#214-谱聚类-spectralclustering)
- [高斯混合](04_无监督学习模块.md#51-高斯混合模型-gaussianmixture)

#### 降维任务
- [PCA](04_无监督学习模块.md#311-pca-主成分分析)
- [t-SNE](04_无监督学习模块.md#411-t-sne)
- [NMF](04_无监督学习模块.md#312-nmf-非负矩阵分解)
- [ICA](04_无监督学习模块.md#313-ica-独立成分分析)

#### 数据预处理
- [特征缩放](06_数据预处理与特征工程.md#21-特征缩放)
- [类别编码](06_数据预处理与特征工程.md#22-分类特征编码)
- [缺失值处理](06_数据预处理与特征工程.md#5-impute---缺失值处理)
- [特征选择](06_数据预处理与特征工程.md#4-feature_selection---特征选择)

### 按主题查找

#### API 和基础
- [BaseEstimator 类](02_核心基类与API设计.md#2-baseestimator-基类)
- [Mixin 类](02_核心基类与API设计.md#3-mixin-类)
- [估计器接口](02_核心基类与API设计.md#4-估计器-api)
- [自定义估计器](02_核心基类与API设计.md#9-实现自定义估计器)

#### 模型评估
- [交叉验证](05_模型选择与评估.md#22-交叉验证)
- [网格搜索](05_模型选择与评估.md#231-gridsearchcv---网格搜索)
- [分类指标](05_模型选择与评估.md#31-分类指标)
- [回归指标](05_模型选择与评估.md#32-回归指标)
- [学习曲线](05_模型选择与评估.md#24-学习曲线)

#### 特征工程
- [Pipeline](06_数据预处理与特征工程.md#61-pipeline---管道)
- [ColumnTransformer](06_数据预处理与特征工程.md#62-columntransformer---列转换器)
- [文本特征提取](06_数据预处理与特征工程.md#31-文本特征提取)
- [多项式特征](06_数据预处理与特征工程.md#233-polynomialfeatures---多项式特征)

## 📊 常用代码示例

### 基本工作流程

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. 数据预处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 模型训练
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# 4. 预测和评估
y_pred = clf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

### 使用 Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', SVC())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### 超参数调优

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

## 🔍 核心概念速查

### 估计器类型

| 类型 | Mixin | score() 方法 | 主要方法 |
|-----|-------|-------------|---------|
| 分类器 | ClassifierMixin | accuracy_score | predict, predict_proba |
| 回归器 | RegressorMixin | r2_score | predict |
| 聚类器 | ClusterMixin | - | fit_predict |
| 转换器 | TransformerMixin | - | fit, transform, fit_transform |

### 常用参数

| 参数 | 说明 | 典型值 |
|-----|------|-------|
| `n_estimators` | 集成模型的基估计器数量 | 100, 200 |
| `max_depth` | 树的最大深度 | 10, 20, None |
| `learning_rate` | 学习率 | 0.01, 0.1, 1.0 |
| `C` | 正则化参数（SVM、逻辑回归） | 0.1, 1, 10 |
| `alpha` | 正则化强度（线性模型） | 0.001, 0.01, 0.1 |
| `n_neighbors` | K 近邻数量 | 3, 5, 7 |

### 评估指标选择

| 任务 | 推荐指标 | 何时使用 |
|-----|---------|---------|
| 平衡分类 | Accuracy | 类别数量相近 |
| 不平衡分类 | F1, ROC-AUC, PR-AUC | 类别不均衡 |
| 多类分类 | Macro/Weighted F1 | 关注所有类别 |
| 回归 | R², RMSE, MAE | R² 解释性好 |
| 聚类 | Silhouette, CH Index | 无真实标签 |

## 🛠️ 实用工具

### 常用导入

```python
# 数据处理
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 监督学习
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier

# 无监督学习
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.manifold import TSNE

# 评估
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score,
    confusion_matrix, classification_report
)

# 工具
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
```

### 调试技巧

```python
# 查看估计器的所有参数
clf = RandomForestClassifier()
print(clf.get_params())

# 检查是否已拟合
from sklearn.utils.validation import check_is_fitted
try:
    check_is_fitted(clf)
    print("Model is fitted")
except:
    print("Model is not fitted")

# 查看 Pipeline 中的步骤
pipeline = Pipeline([('scaler', StandardScaler()), ('clf', SVC())])
print(pipeline.named_steps)
print(pipeline['scaler'])  # 访问特定步骤

# 获取特征名称
preprocessor.get_feature_names_out()
```

## 📖 学习路径建议

### 初学者路径
1. 阅读 [架构总览](01_scikit-learn架构总览.md) 了解整体设计
2. 学习 [核心基类](02_核心基类与API设计.md) 理解 API 规范
3. 实践 [监督学习](03_监督学习模块.md) 的基本算法
4. 掌握 [模型选择与评估](05_模型选择与评估.md)
5. 学习 [数据预处理](06_数据预处理与特征工程.md) 和 Pipeline

### 进阶路径
1. 深入理解各种算法的实现细节
2. 学习 [无监督学习](04_无监督学习模块.md) 算法
3. 掌握高级特征工程技巧
4. 实现自定义估计器和转换器
5. 优化模型性能和调参策略

### 实践项目建议
1. **分类项目**: 使用 Pipeline + GridSearchCV 构建完整分类流程
2. **回归项目**: 特征工程 + 集成学习
3. **聚类项目**: 探索性数据分析 + 多种聚类算法对比
4. **文本分类**: TF-IDF + 分类器
5. **自定义估计器**: 实现符合 scikit-learn API 的自定义算法

## 🔗 相关资源

### 官方资源
- [scikit-learn 官方网站](https://scikit-learn.org)
- [官方文档](https://scikit-learn.org/stable/documentation.html)
- [GitHub 仓库](https://github.com/scikit-learn/scikit-learn)
- [用户指南](https://scikit-learn.org/stable/user_guide.html)
- [API 参考](https://scikit-learn.org/stable/modules/classes.html)

### 相关依赖
- [NumPy 文档](../numpy-2.2.0-doc/)
- [SciPy 文档](https://docs.scipy.org)
- [Pandas 文档](https://pandas.pydata.org/docs/)
- [Matplotlib 文档](https://matplotlib.org/stable/contents.html)

## ❓ 常见问题

### 何时使用哪种缩放方法？
- 数据正态分布 → StandardScaler
- 有异常值 → RobustScaler
- 需要 [0,1] 范围 → MinMaxScaler
- 树模型 → 不需要缩放

### 如何避免数据泄漏？
使用 Pipeline 确保所有预处理步骤只在训练集上拟合：
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
```

### 如何处理类别不平衡？
1. 使用 `class_weight='balanced'`
2. 重采样（SMOTE）
3. 选择合适的评估指标（F1, ROC-AUC）
4. 分层采样 `stratify=y`

### 如何选择合适的算法？
参考 [监督学习模块](03_监督学习模块.md#7-选择合适的算法) 中的决策流程图。

## 📝 更新日志

### 当前版本: scikit-learn 1.6.0
- 文档创建日期: 2024
- 覆盖模块: 核心 API、监督学习、无监督学习、模型选择、数据预处理
- 文档数量: 6 个核心文档

---

**文档维护**: 本文档基于 scikit-learn 1.6.0 源代码分析整理。如有疑问或建议，欢迎反馈。

**快速开始**: 建议从 [01_架构总览](01_scikit-learn架构总览.md) 开始阅读！
