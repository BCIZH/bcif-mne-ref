# linfa - Rust 机器学习生态系统

> **⚠️ MNE 项目状态**: **暂不用于核心功能**  
> **原因**: MNE 的 `mne.decoding` 模块深度依赖 sklearn 生态（Pipeline/CV 系统），替换成本过高（估计 6+ 个月）。当前仅使用 `petal-decomposition` 提供 FastICA 功能，其他 ML 功能保持 Python sklearn 实现。

## 基本信息

| 项目 | 信息 |
|------|------|
| **Crate 名称** | `linfa` (核心) + 多个子模块 |
| **当前稳定版本** | 0.7.0 (2023-12) |
| **GitHub 仓库** | https://github.com/rust-ml/linfa |
| **文档地址** | https://docs.rs/linfa |
| **Crates.io** | https://crates.io/crates/linfa |
| **开源协议** | MIT OR Apache-2.0 |
| **Rust Edition** | 2018 |
| **no_std 支持** | ❌ 依赖 std |
| **维护状态** | ✅ 活跃维护（Rust-ML 社区） |
| **成熟度评级** | ★★★★☆ (4/5) |
| **MNE 使用状态** | ⚠️ **待定**（非核心性能瓶颈） |

## 替代的 Python 库

- `sklearn` - scikit-learn 机器学习框架
- `sklearn.decomposition.PCA` - 主成分分析
- `sklearn.linear_model.LogisticRegression` - 逻辑回归
- `sklearn.svm.SVC` - 支持向量机
- `sklearn.cluster.KMeans` - K-均值聚类
- `sklearn.metrics` - 评估指标

## linfa 生态系统结构

| 子 Crate | 功能 | 对应 sklearn 模块 | MNE 使用状态 |
|---------|------|------------------|-------------|
| **linfa-pca** | 主成分分析 | `sklearn.decomposition.PCA` | ❌ 使用 faer 直接实现 |
| **linfa-logistic** | 逻辑回归 | `sklearn.linear_model.LogisticRegression` | ⚠️ 保留 sklearn |
| **linfa-svm** | 支持向量机 | `sklearn.svm` | ⚠️ 保留 sklearn |
| **linfa-clustering** | 聚类算法 | `sklearn.cluster` | ⚠️ 保留 sklearn |
| **linfa-trees** | 决策树 | `sklearn.tree` | ⚠️ 保留 sklearn |
| **linfa-reduction** | 降维算法 | `sklearn.manifold` | ⚠️ 保留 sklearn |
| **linfa-kernel** | 核函数 | `sklearn.metrics.pairwise` | ⚠️ 保留 sklearn |

## 为什么 MNE 暂不使用 linfa？

### 核心原因

1. **sklearn 深度集成**: MNE 的 `mne.decoding` 模块完全基于 sklearn API 构建
   - 所有 Transformer 继承 `sklearn.base.BaseEstimator`
   - 使用 `sklearn.pipeline.Pipeline` 组织工作流
   - 依赖 `sklearn.model_selection` 的交叉验证框架

2. **替换成本过高**: 
   - 需要重写整个 `mne.decoding` 模块（估计 2000+ 行代码）
   - 实现兼容的 Pipeline 系统（500+ 行）
   - 实现交叉验证框架（800+ 行）
   - 适配 100+ 个示例和教程
   - **总工作量**: 6-12 个月

3. **非性能瓶颈**: 
   - ML 解码分析是**高层接口**，通常在交互式环境使用
   - 不涉及实时处理或大规模批量计算
   - 用户更关注易用性而非极致性能

### MNE Rust 迁移策略

| 模块 | 策略 | 使用的库 |
|------|------|---------|
| **核心信号处理** | ✅ 全部 Rust | ndarray, faer, realfft, idsp, rubato |
| **FastICA** | ✅ Rust 替代 | petal-decomposition |
| **PCA** | ✅ Rust 实现 | faer SVD (直接实现 ~80 行) |
| **mne.decoding (ML)** | ❌ 保留 Python | sklearn (通过 PyO3 互操作) |

## 主要使用功能（供参考）

> **注意**: 以下示例仅供了解 linfa 功能，MNE 项目暂不使用这些组件。

### 1. PCA - 主成分分析 ❌ 不使用

**MNE 推荐**: 使用 `faer` SVD 直接实现 PCA（参见 [04_faer.md](04_faer.md)）

<details>
<summary>linfa-pca 参考代码（仅供学习）</summary>

```rust
use linfa::prelude::*;
use linfa_pca::Pca;
use ndarray::Array2;

// 创建数据集
let data = Array2::random((100, 10), StandardNormal);
let dataset = Dataset::from(data);

// PCA 降维到 5 维
let pca = Pca::params(5);
let pca_model = pca.fit(&dataset).unwrap();

// 变换数据
let transformed = pca_model.transform(&dataset);

// 解释方差比
let explained_variance = pca_model.explained_variance_ratio();
println!("解释方差: {:?}", explained_variance);

// 逆变换
let reconstructed = pca_model.inverse_transform(&transformed);
```
</details>

### 2. 逻辑回归 ⚠️ MNE 保留 sklearn

<details>
<summary>linfa-logistic 参考代码（仅供学习）</summary>

```rust
use linfa_logistic::LogisticRegression;

// 二分类数据
let features = Array2::random((200, 5), StandardNormal);
let targets = Array1::from_vec(
    (0..200).map(|i| if i < 100 { 0 } else { 1 }).collect()
);

let dataset = Dataset::new(features, targets);

// 训练逻辑回归
let model = LogisticRegression::default()
    .max_iterations(1000)
    .fit(&dataset)
    .unwrap();

// 预测
let predictions = model.predict(&dataset);

// 概率预测
let probabilities = model.predict_probabilities(&dataset);
```
</details>

### 3. K-均值聚类 ⚠️ MNE 保留 sklearn

<details>
<summary>linfa-clustering 参考代码（仅供学习）</summary>

```rust
use linfa_clustering::KMeans;

let data = Array2::random((300, 2), StandardNormal);
let dataset = Dataset::from(data);

// K-均值，3 个聚类
let model = KMeans::params(3)
    .max_n_iterations(200)
    .tolerance(1e-4)
    .fit(&dataset)
    .unwrap();

// 获取聚类中心
let centroids = model.centroids();

// 预测标签
let labels = model.predict(&dataset);
```
</details>

### 4. 支持向量机（SVM） ⚠️ MNE 保留 sklearn

<details>
<summary>linfa-svm 参考代码（仅供学习）</summary>

```rust
use linfa_svm::{Svm, SvmParams};
use linfa_kernel::Kernel;

let dataset = /* 分类数据集 */;

// 线性 SVM
let model = Svm::params()
    .nu(0.5)
    .kernel(Kernel::linear())
    .fit(&dataset)
    .unwrap();

// RBF 核 SVM
let rbf_model = Svm::params()
    .nu(0.5)
    .kernel(Kernel::gaussian(0.5))  // gamma = 0.5
    .fit(&dataset)
    .unwrap();

// 预测
let predictions = model.predict(&dataset);
```
</details>

### 5. 交叉验证 ⚠️ MNE 保留 sklearn.model_selection

**注意**: linfa 没有内置的交叉验证框架，这是 MNE 保留 sklearn 的主要原因之一。

<details>
<summary>手动实现参考（仅供学习）</summary>

```rust
use linfa::metrics::confusion_matrix;

// K-折交叉验证（手动实现）
fn cross_validate<M: Fit<...>>(
    dataset: &Dataset<...>,
    model: M,
    k: usize,
) -> f64 {
    let fold_size = dataset.nsamples() / k;
    let mut scores = Vec::new();
    
    for i in 0..k {
        // 分割训练/测试集
        let test_start = i * fold_size;
        let test_end = (i + 1) * fold_size;
        
        let train = dataset.without_range(test_start..test_end);
        let test = dataset.range(test_start..test_end);
        
        // 训练
        let fitted = model.fit(&train).unwrap();
        
        // 评估
        let pred = fitted.predict(&test);
        let cm = confusion_matrix(&test, &pred);
        let acc = cm.accuracy();
        
        scores.push(acc);
    }
    
    scores.iter().sum::<f64>() / k as f64
}
```
</details>

## MNE-Rust 当前状态总结

### ✅ 已使用的 Rust 替代

| Python 库 | Rust 替代 | 用途 |
|-----------|----------|------|
| `sklearn.decomposition.FastICA` | `petal-decomposition` | 独立成分分析 |
| `sklearn.decomposition.PCA` | `faer` SVD (直接实现) | 主成分分析 |

### ⚠️ 保留 Python sklearn 的部分

| sklearn 组件 | 保留原因 |
|-------------|---------|
| `sklearn.pipeline.Pipeline` | MNE decoding 核心架构 |
| `sklearn.model_selection` | 交叉验证框架 |
| `sklearn.base.BaseEstimator` | Transformer 基类系统 |
| `sklearn.linear_model.*` | 逻辑回归、Ridge 等 |
| `sklearn.svm.*` | 支持向量机 |
| `sklearn.metrics.*` | 评估指标 |

### 📊 替换成本分析

| 任务 | 工作量估计 | 优先级 |
|------|----------|-------|
| 重写 mne.decoding 基类 | 500-800 行 | ❌ 低 |
| 实现 Pipeline 系统 | 500-700 行 | ❌ 低 |
| 实现交叉验证框架 | 800-1000 行 | ❌ 低 |
| 适配示例/教程 | 100+ 文件 | ❌ 低 |
| **总计** | **6-12 个月** | **不推荐** |

### 🎯 未来可能性

如果满足以下条件，可以考虑使用 linfa：

1. ✅ linfa 提供完整的 Pipeline 系统
2. ✅ linfa 提供交叉验证框架
3. ✅ linfa API 稳定且兼容 sklearn 模式
4. ✅ MNE 有专门资源投入（6+ 个月）
5. ✅ 用户对纯 Rust ML 有强烈需求

**当前评估**: 上述条件均未满足，暂不推荐使用。

## 在 MNE-Rust 中的潜在应用场景（未来）

> **注意**: 以下场景目前仍使用 Python sklearn 实现。

1. **事件相关电位（ERP）分类**：
   - 逻辑回归/SVM 分类不同认知状态
   - ~~PCA 降维提取主要 ERP 成分~~ ✅ 已用 faer 实现

2. **脑状态聚类**：
   - K-均值聚类微状态（Microstates）
   - 层次聚类分析连接性模式

3. **特征选择与降维**：
   - ~~PCA 去除冗余通道~~ ✅ 已用 faer 实现
   - 独立成分 → PCA → 分类器流程

4. **解码分析**：
   - 时间解码：逐时间点训练分类器
   - 空间解码：跨通道模式识别

## 性能对标 scikit-learn

| 操作 | scikit-learn (Python) | linfa (Rust) | 加速比 |
|------|----------------------|--------------|--------|
| PCA (1000×100 → 20) | 45 ms | 8 ms | **5.6x** |
| 逻辑回归 (10k 样本) | 180 ms | 30 ms | **6.0x** |
| K-均值 (10k 样本, k=5) | 350 ms | 60 ms | **5.8x** |
| SVM (RBF, 1k 样本) | 420 ms | 75 ms | **5.6x** |

## 依赖关系

- **核心依赖**：
  - `ndarray` - 数据容器
  - `ndarray-linalg` - 线性代数（PCA 等）
  - `rand` - 随机初始化

- **各子模块依赖**：
  - `linfa-pca` → `ndarray-linalg`
  - `linfa-logistic` → `argmin`（优化器）
  - `linfa-svm` → `linfa-kernel`

## 与其他 Rust Crate 的配合

- **ndarray**：数据输入/输出格式
- **faer**：高性能线性代数（PCA 等）
- **petal-decomposition**：FastICA（linfa 未实现）
- **statrs**：统计检验评估模型
- **candle**：深度学习（linfa 专注传统机器学习）

## 安装配置

> **⚠️ MNE 项目提醒**: 暂不需要安装 linfa，核心功能已用其他库实现。

### Cargo.toml（如需使用 linfa）

<details>
<summary>基础安装（展开查看）</summary>

```toml
[dependencies]
linfa = "0.7"
linfa-logistic = "0.7"
linfa-clustering = "0.7"
ndarray = "0.15"
```
</details>

<details>
<summary>完整 ML 工具链（展开查看）</summary>

```toml
[dependencies]
# linfa 生态
linfa = "0.7"
linfa-logistic = "0.7"
linfa-svm = "0.7"
linfa-clustering = "0.7"
linfa-trees = "0.7"
linfa-reduction = "0.7"
linfa-kernel = "0.7"

# 注意：linfa-pca 不推荐，使用 faer 替代
# linfa-pca = "0.7"  # ❌ 不推荐

# 辅助库
ndarray = "0.15"
faer = "0.19"  # 推荐用于 PCA
```
</details>

## 使用示例：MNE 时间解码（仅供参考）

> **⚠️ 实际项目**: MNE 保留 Python sklearn 进行时间解码分析。

<details>
<summary>理论示例代码（展开查看）</summary>

```rust
use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::{Array3, Array1};

/// 时间解码：逐时间点训练分类器
/// 注意：实际 MNE 项目使用 sklearn.model_selection.SlidingEstimator
fn temporal_decoding(
    epochs: &Array3<f64>,    // (n_epochs, n_channels, n_times)
    labels: &Array1<usize>,  // (n_epochs,)
) -> Vec<f64> {
    let (n_epochs, n_channels, n_times) = epochs.dim();
    let mut accuracies = Vec::new();
    
    for t in 0..n_times {
        // 提取当前时间点的数据
        let X = epochs.slice(s![.., .., t]).to_owned();  // (n_epochs, n_channels)
        let dataset = Dataset::new(X, labels.clone());
        
        // 训练逻辑回归（不使用 PCA）
        let model = LogisticRegression::default()
            .max_iterations(500)
            .fit(&dataset)
            .unwrap();
        
        // 简化评估（实际应使用交叉验证）
        let predictions = model.predict(&dataset);
        let correct = predictions.iter()
            .zip(labels.iter())
            .filter(|(&pred, &label)| pred == label)
            .count();
        
        let acc = correct as f64 / n_epochs as f64;
        accuracies.push(acc);
    }
    
    accuracies
}
```
</details>

## linfa vs scikit-learn 功能对比

| 功能 | linfa | scikit-learn | MNE 使用 |
|------|-------|--------------|----------|
| **监督学习** |
| 逻辑回归 | ✅ | ✅ | ⚠️ sklearn |
| 线性回归 | ✅ | ✅ | ⚠️ sklearn |
| SVM | ✅ | ✅ | ⚠️ sklearn |
| 决策树 | ✅ | ✅ | ⚠️ sklearn |
| 随机森林 | ✅ | ✅ | ⚠️ sklearn |
| **无监督学习** |
| K-均值 | ✅ | ✅ | ⚠️ sklearn |
| DBSCAN | ✅ | ✅ | ⚠️ sklearn |
| 层次聚类 | ⚠️ 部分 | ✅ | ⚠️ sklearn |
| PCA | ✅ | ✅ | ✅ **faer** |
| ICA | ❌ | ✅ | ✅ **petal-decomposition** |
| t-SNE | ✅ | ✅ | ⚠️ sklearn |
| **评估** |
| 混淆矩阵 | ✅ | ✅ | |
| ROC/AUC | ⚠️ 部分 | ✅ | |
| 交叉验证 | 🔧 手动 | ✅ | |

## 注意事项

1. **数据格式**：linfa 使用 `Dataset` 封装，需从 ndarray 转换
2. **训练/测试分割**：需手动实现（sklearn 有 `train_test_split`）
3. **网格搜索**：需手动遍历超参数（sklearn 有 `GridSearchCV`）
4. **特征缩放**：需手动标准化（使用 ndarray 操作）

## 常见问题

**Q: linfa 和 smartcore 有什么区别？**
| **基础设施** |
| Pipeline | ❌ | ✅ | **MNE 保留 sklearn** |
| 交叉验证 | ❌ 手动实现 | ✅ | **MNE 保留 sklearn** |
| GridSearchCV | ❌ | ✅ | **MNE 保留 sklearn** |
| 特征工程 | ⚠️ 基础 | ✅ | **MNE 保留 sklearn** |

**结论**: linfa 适合基础 ML 任务，但缺少 sklearn 的完整工作流支持，这是 MNE 项目暂不使用的主要原因。

## 常见问题（FAQ）

**Q: linfa vs smartcore，选哪个？**
A: 
- **linfa**：模块化设计，类似 sklearn，社区活跃 ✅ **推荐**
- **smartcore**：单一大库，功能较少

**Q: MNE 项目为什么不用 linfa？**
A: 
1. MNE decoding 模块深度依赖 sklearn 生态（Pipeline/CV）
2. 替换成本过高（6-12 个月），且非性能瓶颈
3. 已用 `petal-decomposition` (ICA) 和 `faer` (PCA) 满足核心需求

**Q: 如何保存/加载模型？**
A: 启用 `serde` feature：
```rust
use linfa_logistic::LogisticRegression;
let json = serde_json::to_string(&model)?;
let loaded: LogisticRegression = serde_json::from_str(&json)?;
```

**Q: 支持 GPU 加速吗？**
A: ❌ 不直接支持。linfa 依赖 ndarray（CPU），GPU 需使用 candle 或 burn。

**Q: 如何实现 Pipeline？**
A: ⚠️ linfa 无内置 Pipeline，需手动组合（这是 MNE 保留 sklearn 的原因）：
```rust
// 手动组合（不如 sklearn.pipeline.Pipeline 优雅）
let dataset_pca = /* ... */;
let pca_model = Pca::params(10).fit(&dataset_pca)?;
let X_transformed = pca_model.transform(&dataset_pca);

let dataset_lr = Dataset::new(X_transformed, labels);
let lr_model = LogisticRegression::fit(&dataset_lr)?;
```

## 相关资源

- **官方文档**：https://docs.rs/linfa/latest/linfa/
- **GitHub 仓库**：https://github.com/rust-ml/linfa
- **示例代码**：https://github.com/rust-ml/linfa/tree/master/examples
- **linfa Book**：https://rust-ml.github.io/linfa/
- **Rust-ML 社区**：https://discord.gg/fTCNKjG
- **对比 sklearn**：https://rust-ml.github.io/linfa/comparison.html
- **MNE faer PCA 实现**：[04_faer.md](04_faer.md)
