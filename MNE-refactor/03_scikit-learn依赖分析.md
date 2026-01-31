# MNE-Python scikit-learn 依赖分析

## 概述

scikit-learn 是 MNE-Python 的**机器学习核心**，主要用于：
1. **ICA 分解**（独立成分分析）- 伪影去除
2. **降维**（PCA, FA）- 数据降维和特征提取
3. **分类/回归**（解码分析）- 脑信号解码
4. **交叉验证**（模型评估）
5. **协方差估计**（收缩估计）

---

## scikit-learn 使用统计

### 模块导入频率

| 模块 | 使用频率 | 关键程度 | Rust 替代难度 |
|------|---------|---------|--------------|
| `sklearn.decomposition` | ⭐⭐⭐⭐⭐ | 必需 | 中-高 |
| `sklearn.base` | ⭐⭐⭐⭐ | 必需 | 低 |
| `sklearn.model_selection` | ⭐⭐⭐⭐ | 必需 | 中 |
| `sklearn.preprocessing` | ⭐⭐⭐ | 重要 | 低 |
| `sklearn.linear_model` | ⭐⭐⭐ | 重要 | 中 |
| `sklearn.covariance` | ⭐⭐⭐ | 重要 | 中 |
| `sklearn.utils` | ⭐⭐⭐⭐ | 必需 | 低 |
| `sklearn.discriminant_analysis` | ⭐⭐ | 可选 | 中 |
| `sklearn.svm` | ⭐⭐ | 可选 | 高 |
| `sklearn.ensemble` | ⭐⭐ | 可选 | 高 |
| `sklearn.neighbors` | ⭐⭐ | 可选 | 低 |

---

## 1. sklearn.decomposition - 分解算法（最重要）

### 1.1 FastICA - 快速独立成分分析

#### Python/sklearn 模式

```python
from sklearn.decomposition import FastICA

# MNE ICA 的核心
class ICA:
    def __init__(self, n_components=None, method='fastica', max_iter=200):
        self.method = method
        self.max_iter = max_iter
        
    def fit(self, data):
        """拟合 ICA 模型"""
        if self.method == 'fastica':
            from sklearn.decomposition import FastICA
            
            ica = FastICA(
                n_components=self.n_components,
                algorithm='parallel',  # 或 'deflation'
                fun='logcosh',  # 非高斯性度量: 'logcosh', 'exp', 'cube'
                max_iter=self.max_iter,
                tol=1e-4,
                random_state=self.random_state,
                whiten='arbitrary-variance',  # 白化方式
            )
            
            # 拟合并提取成分
            self.unmixing_matrix_ = ica.fit(data.T).components_
            self.mixing_matrix_ = pinv(self.unmixing_matrix_)
            
        return self
    
    def apply(self, data, exclude=[]):
        """应用 ICA，排除指定成分"""
        # 获取源信号
        sources = self.unmixing_matrix_ @ data
        
        # 排除伪影成分
        sources[exclude] = 0
        
        # 重建数据
        reconstructed = self.mixing_matrix_ @ sources
        
        return reconstructed

# 使用示例
ica = ICA(n_components=20, method='fastica')
ica.fit(raw)
ica.exclude = [0, 1]  # 排除眨眼和心跳成分
raw_clean = ica.apply(raw)
```

#### Rust 等价

```rust
use ndarray::prelude::*;
use ndarray_linalg::*;

// ICA 算法需要自己实现
pub struct FastICA {
    n_components: usize,
    algorithm: Algorithm,
    fun: ContrastFunction,
    max_iter: usize,
    tol: f64,
    random_state: u64,
}

#[derive(Debug, Clone)]
pub enum Algorithm {
    Parallel,     // 并行提取所有成分
    Deflation,    // 逐个提取成分
}

#[derive(Debug, Clone)]
pub enum ContrastFunction {
    LogCosh,  // tanh(x)
    Exp,      // x * exp(-x^2/2)
    Cube,     // x^3
}

impl FastICA {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            algorithm: Algorithm::Parallel,
            fun: ContrastFunction::LogCosh,
            max_iter: 200,
            tol: 1e-4,
            random_state: 42,
        }
    }
    
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<&Self> {
        // 1. 中心化
        let mean = data.mean_axis(Axis(0)).unwrap();
        let centered = data - &mean.insert_axis(Axis(0));
        
        // 2. 白化（PCA 白化）
        let (whitened, whitening_matrix) = self.whiten(&centered)?;
        
        // 3. FastICA 迭代
        match self.algorithm {
            Algorithm::Parallel => {
                self.unmixing_matrix_ = self.fit_parallel(&whitened)?;
            }
            Algorithm::Deflation => {
                self.unmixing_matrix_ = self.fit_deflation(&whitened)?;
            }
        }
        
        // 4. 组合白化和解混矩阵
        self.unmixing_matrix_ = self.unmixing_matrix_.dot(&whitening_matrix);
        self.mixing_matrix_ = self.unmixing_matrix_.pinv(1e-15)?;
        
        Ok(self)
    }
    
    fn whiten(&self, data: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        // PCA 白化
        let cov = data.t().dot(data) / (data.nrows() as f64 - 1.0);
        let (eigenvalues, eigenvectors) = cov.eigh(UPLO::Lower)?;
        
        // 只保留 n_components 个成分
        let k = self.n_components;
        let eig_vals = eigenvalues.slice(s![eigenvalues.len()-k..]);
        let eig_vecs = eigenvectors.slice(s![.., eigenvectors.ncols()-k..]);
        
        // 白化矩阵
        let diag = Array2::from_diag(&eig_vals.mapv(|x| 1.0 / x.sqrt()));
        let whitening = diag.dot(&eig_vecs.t());
        
        // 白化数据
        let whitened = data.dot(&whitening.t());
        
        Ok((whitened, whitening))
    }
    
    fn fit_parallel(&self, whitened: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_features) = whitened.dim();
        
        // 随机初始化解混矩阵
        let mut rng = StdRng::seed_from_u64(self.random_state);
        let mut W: Array2<f64> = Array::random_using(
            (n_features, n_features),
            StandardNormal,
            &mut rng
        );
        
        // 正交化
        W = self.decorrelate(&W)?;
        
        // FastICA 迭代
        for iter in 0..self.max_iter {
            // 计算 W * X
            let wx = W.dot(&whitened.t());
            
            // 应用对比函数（非线性函数）
            let (g, g_prime) = self.contrast_function(&wx);
            
            // 更新 W
            let W_new = (g.dot(&whitened) / n_samples as f64) 
                - (g_prime.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1)).dot(&W));
            
            // 正交化
            let W_new = self.decorrelate(&W_new)?;
            
            // 检查收敛
            let lim = (&W_new * &W).sum_axis(Axis(1)).mapv(|x| x.abs() - 1.0).mapv(f64::abs).max().unwrap();
            
            W = W_new;
            
            if lim < self.tol {
                println!("FastICA 收敛于第 {} 次迭代", iter + 1);
                break;
            }
        }
        
        Ok(W)
    }
    
    fn fit_deflation(&self, whitened: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_features) = whitened.dim();
        let mut W = Array2::<f64>::zeros((n_features, n_features));
        
        // 逐个提取成分
        for comp_idx in 0..n_features {
            // 随机初始化
            let mut w = Array1::random_using(n_features, StandardNormal, &mut rng);
            w = w / w.norm_l2();
            
            // 正交化（相对于已提取的成分）
            for j in 0..comp_idx {
                let w_j = W.row(j);
                w = &w - &(w.dot(&w_j) * &w_j);
            }
            w = w / w.norm_l2();
            
            // 迭代优化
            for _ in 0..self.max_iter {
                let wx = w.dot(&whitened.t());
                let (g, g_prime) = self.contrast_function_1d(&wx);
                
                let w_new = (whitened.t().dot(&g) / n_samples as f64) 
                    - g_prime.mean() * &w;
                
                // 正交化
                for j in 0..comp_idx {
                    let w_j = W.row(j);
                    w_new = &w_new - &(w_new.dot(&w_j) * &w_j);
                }
                w_new = w_new / w_new.norm_l2();
                
                // 检查收敛
                if (w_new.dot(&w).abs() - 1.0).abs() < self.tol {
                    break;
                }
                
                w = w_new;
            }
            
            W.row_mut(comp_idx).assign(&w);
        }
        
        Ok(W)
    }
    
    fn decorrelate(&self, W: &Array2<f64>) -> Result<Array2<f64>> {
        // 对称正交化：W = (W * W^T)^(-1/2) * W
        let WWT = W.dot(&W.t());
        let (eigenvalues, eigenvectors) = WWT.eigh(UPLO::Lower)?;
        
        let diag = Array2::from_diag(&eigenvalues.mapv(|x| 1.0 / x.sqrt()));
        let sqrt_inv = eigenvectors.dot(&diag).dot(&eigenvectors.t());
        
        Ok(sqrt_inv.dot(W))
    }
    
    fn contrast_function(&self, wx: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
        match self.fun {
            ContrastFunction::LogCosh => {
                // g(x) = tanh(x), g'(x) = 1 - tanh^2(x)
                let tanh = wx.mapv(|x| x.tanh());
                let g_prime = tanh.mapv(|t| 1.0 - t * t).mean_axis(Axis(1)).unwrap();
                (tanh, g_prime)
            }
            ContrastFunction::Exp => {
                // g(x) = x * exp(-x^2/2), g'(x) = (1 - x^2) * exp(-x^2/2)
                let exp_term = wx.mapv(|x| (-x * x / 2.0).exp());
                let g = wx * &exp_term;
                let g_prime = exp_term.mapv(|e| e) - wx.mapv(|x| x * x) * &exp_term;
                let g_prime_mean = g_prime.mean_axis(Axis(1)).unwrap();
                (g, g_prime_mean)
            }
            ContrastFunction::Cube => {
                // g(x) = x^3, g'(x) = 3x^2
                let g = wx.mapv(|x| x.powi(3));
                let g_prime = wx.mapv(|x| 3.0 * x * x).mean_axis(Axis(1)).unwrap();
                (g, g_prime)
            }
        }
    }
}

// MNE 风格的 ICA 类
pub struct ICA {
    n_components: usize,
    method: String,
    ica: FastICA,
    pub unmixing_matrix_: Option<Array2<f64>>,
    pub mixing_matrix_: Option<Array2<f64>>,
    pub exclude: Vec<usize>,
}

impl ICA {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            method: "fastica".to_string(),
            ica: FastICA::new(n_components),
            unmixing_matrix_: None,
            mixing_matrix_: None,
            exclude: Vec::new(),
        }
    }
    
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<&Self> {
        self.ica.fit(data)?;
        self.unmixing_matrix_ = Some(self.ica.unmixing_matrix_.clone());
        self.mixing_matrix_ = Some(self.ica.mixing_matrix_.clone());
        Ok(self)
    }
    
    pub fn apply(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let unmixing = self.unmixing_matrix_.as_ref().unwrap();
        let mixing = self.mixing_matrix_.as_ref().unwrap();
        
        // 提取源信号
        let mut sources = unmixing.dot(data);
        
        // 排除伪影成分
        for &idx in &self.exclude {
            sources.row_mut(idx).fill(0.0);
        }
        
        // 重建
        Ok(mixing.dot(&sources))
    }
}
```

**关键挑战**:
1. **FastICA 算法复杂**，需要完整实现
2. 对比函数的选择和优化
3. 正交化和白化需要仔细处理
4. 收敛判据

**优势**:
- 算法本身是确定性的（给定随机种子）
- 可以优化得比 Python 更快
- Rust 的并行特性可加速

---

### 1.2 PCA - 主成分分析

#### Python/sklearn 模式

```python
from sklearn.decomposition import PCA

# 降维
pca = PCA(n_components=50, whiten=True)
pca.fit(data)
reduced = pca.transform(data)

# 解释方差
explained_var = pca.explained_variance_ratio_

# MNE 用法：数据降维
def reduce_dimensionality(epochs, n_components=50):
    """使用 PCA 降维"""
    from sklearn.decomposition import PCA
    
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape
    
    # 重塑为 2D
    data_2d = data.reshape(n_epochs, -1)
    
    # PCA
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data_2d)
    
    return reduced, pca
```

#### Rust 等价

```rust
use ndarray::prelude::*;
use ndarray_linalg::*;

pub struct PCA {
    n_components: usize,
    whiten: bool,
    pub components_: Option<Array2<f64>>,
    pub explained_variance_: Option<Array1<f64>>,
    pub explained_variance_ratio_: Option<Array1<f64>>,
    pub mean_: Option<Array1<f64>>,
}

impl PCA {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            whiten: false,
            components_: None,
            explained_variance_: None,
            explained_variance_ratio_: None,
            mean_: None,
        }
    }
    
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<&Self> {
        let (n_samples, n_features) = data.dim();
        
        // 中心化
        let mean = data.mean_axis(Axis(0)).unwrap();
        let centered = data - &mean.insert_axis(Axis(0));
        
        // 协方差矩阵
        let cov = centered.t().dot(&centered) / (n_samples as f64 - 1.0);
        
        // 特征分解
        let (mut eigenvalues, eigenvectors) = cov.eigh(UPLO::Lower)?;
        
        // 按降序排列
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());
        
        // 提取前 k 个成分
        let k = self.n_components.min(n_features);
        let selected_indices = &indices[..k];
        
        let mut components = Array2::zeros((k, n_features));
        let mut selected_eigenvalues = Array1::zeros(k);
        
        for (i, &idx) in selected_indices.iter().enumerate() {
            components.row_mut(i).assign(&eigenvectors.column(idx));
            selected_eigenvalues[i] = eigenvalues[idx];
        }
        
        // 解释方差比例
        let total_var = eigenvalues.sum();
        let variance_ratio = &selected_eigenvalues / total_var;
        
        self.components_ = Some(components);
        self.explained_variance_ = Some(selected_eigenvalues);
        self.explained_variance_ratio_ = Some(variance_ratio);
        self.mean_ = Some(mean);
        
        Ok(self)
    }
    
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mean = self.mean_.as_ref().unwrap();
        let components = self.components_.as_ref().unwrap();
        
        // 中心化
        let centered = data - &mean.insert_axis(Axis(0));
        
        // 投影
        let transformed = centered.dot(&components.t());
        
        // 如果白化
        if self.whiten {
            let std = self.explained_variance_.as_ref().unwrap().mapv(|x| x.sqrt());
            transformed / &std.insert_axis(Axis(0))
        } else {
            transformed
        }
    }
    
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(data)?;
        Ok(self.transform(data))
    }
    
    pub fn inverse_transform(&self, transformed: &Array2<f64>) -> Array2<f64> {
        let components = self.components_.as_ref().unwrap();
        let mean = self.mean_.as_ref().unwrap();
        
        // 反白化
        let unwhitened = if self.whiten {
            let std = self.explained_variance_.as_ref().unwrap().mapv(|x| x.sqrt());
            transformed * &std.insert_axis(Axis(0))
        } else {
            transformed.to_owned()
        };
        
        // 反投影
        let reconstructed = unwhitened.dot(components);
        
        // 加回均值
        reconstructed + &mean.insert_axis(Axis(0))
    }
}
```

**优势**:
- PCA 实现简单直接
- 基于 SVD/特征分解，Rust 已有支持
- 性能优于 Python

---

### 1.3 FactorAnalysis - 因子分析

#### Python/sklearn 模式

```python
from sklearn.decomposition import FactorAnalysis

# MNE 用于协方差降噪
fa = FactorAnalysis(n_components=10, max_iter=1000)
fa.fit(epochs_data)
```

#### Rust 等价

因子分析较复杂，涉及 EM 算法，建议优先级较低或使用 PCA 替代。

---

## 2. sklearn.linear_model - 线性模型

### 2.1 Ridge Regression - 岭回归

#### Python/sklearn 模式

```python
from sklearn.linear_model import Ridge, RidgeCV

# 时间延迟岭回归（mne.decoding.ReceptiveField）
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

# 交叉验证选择 alpha
ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])
ridge_cv.fit(X, y)
best_alpha = ridge_cv.alpha_
```

#### Rust 等价

```rust
use ndarray::prelude::*;
use ndarray_linalg::*;

pub struct Ridge {
    alpha: f64,
    pub coef_: Option<Array2<f64>>,
    pub intercept_: Option<Array1<f64>>,
}

impl Ridge {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            coef_: None,
            intercept_: None,
        }
    }
    
    pub fn fit(&mut self, X: &Array2<f64>, y: &Array2<f64>) -> Result<&Self> {
        let (n_samples, n_features) = X.dim();
        
        // 添加偏置项（截距）
        let X_with_bias = self.add_intercept(X);
        
        // 岭回归闭式解: (X^T X + alpha * I)^(-1) X^T y
        let XtX = X_with_bias.t().dot(&X_with_bias);
        let identity = Array2::<f64>::eye(n_features + 1);
        let regularization = &identity * self.alpha;
        
        // 正则化矩阵（不惩罚截距）
        let mut reg = regularization.clone();
        reg[[0, 0]] = 0.0;  // 不惩罚截距项
        
        let XtX_reg = XtX + reg;
        let Xty = X_with_bias.t().dot(y);
        
        // 求解
        let coef_with_bias = XtX_reg.solve_into(Xty)?;
        
        // 分离截距和系数
        self.intercept_ = Some(coef_with_bias.row(0).to_owned());
        self.coef_ = Some(coef_with_bias.slice(s![1.., ..]).to_owned());
        
        Ok(self)
    }
    
    pub fn predict(&self, X: &Array2<f64>) -> Array2<f64> {
        let coef = self.coef_.as_ref().unwrap();
        let intercept = self.intercept_.as_ref().unwrap();
        
        X.dot(coef) + &intercept.insert_axis(Axis(0))
    }
    
    fn add_intercept(&self, X: &Array2<f64>) -> Array2<f64> {
        let n_samples = X.nrows();
        let ones = Array2::ones((n_samples, 1));
        ndarray::concatenate(Axis(1), &[ones.view(), X.view()]).unwrap()
    }
}
```

---

### 2.2 Lasso / MultiTaskLasso

MNE 用于稀疏源估计（MxNE 算法）

```python
from sklearn.linear_model import MultiTaskLasso

# 稀疏源估计
mtl = MultiTaskLasso(alpha=0.1, max_iter=1000)
mtl.fit(forward_matrix, measured_data)
source_estimates = mtl.coef_
```

Rust 实现需要坐标下降算法，较复杂。

---

## 3. sklearn.model_selection - 模型选择

### 3.1 交叉验证

#### Python/sklearn 模式

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# K 折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(classifier, X, y, cv=kf)

# 分层 K 折（保持类别比例）
skf = StratifiedKFold(n_splits=5)
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

#### Rust 等价

```rust
use rand::prelude::*;

pub struct KFold {
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

impl KFold {
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: false,
            random_state: None,
        }
    }
    
    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..n_samples).collect();
        
        // 打乱
        if self.shuffle {
            if let Some(seed) = self.random_state {
                let mut rng = StdRng::seed_from_u64(seed);
                indices.shuffle(&mut rng);
            }
        }
        
        let fold_size = n_samples / self.n_splits;
        let mut splits = Vec::new();
        
        for i in 0..self.n_splits {
            let test_start = i * fold_size;
            let test_end = if i == self.n_splits - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };
            
            let test_indices: Vec<usize> = indices[test_start..test_end].to_vec();
            let train_indices: Vec<usize> = indices[..test_start]
                .iter()
                .chain(indices[test_end..].iter())
                .copied()
                .collect();
            
            splits.push((train_indices, test_indices));
        }
        
        splits
    }
}

// 交叉验证评分
pub fn cross_val_score<M, X, Y>(
    model: &M,
    X: &X,
    y: &Y,
    cv: &KFold,
    scorer: fn(&Y, &Y) -> f64,
) -> Vec<f64>
where
    M: Estimator<X, Y>,
    X: Indexable,
    Y: Indexable,
{
    let n_samples = X.len();
    let splits = cv.split(n_samples);
    
    let mut scores = Vec::new();
    
    for (train_idx, test_idx) in splits {
        let X_train = X.index(&train_idx);
        let X_test = X.index(&test_idx);
        let y_train = y.index(&train_idx);
        let y_test = y.index(&test_idx);
        
        let mut model_clone = model.clone();
        model_clone.fit(&X_train, &y_train);
        let y_pred = model_clone.predict(&X_test);
        
        let score = scorer(&y_test, &y_pred);
        scores.push(score);
    }
    
    scores
}
```

---

## 4. sklearn.preprocessing - 预处理

### 4.1 StandardScaler - 标准化

#### Python/sklearn 模式

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### Rust 等价

```rust
pub struct StandardScaler {
    pub mean_: Option<Array1<f64>>,
    pub std_: Option<Array1<f64>>,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            mean_: None,
            std_: None,
        }
    }
    
    pub fn fit(&mut self, X: &Array2<f64>) -> &Self {
        self.mean_ = Some(X.mean_axis(Axis(0)).unwrap());
        self.std_ = Some(X.std_axis(Axis(0), 1.0));
        self
    }
    
    pub fn transform(&self, X: &Array2<f64>) -> Array2<f64> {
        let mean = self.mean_.as_ref().unwrap();
        let std = self.std_.as_ref().unwrap();
        (X - &mean.insert_axis(Axis(0))) / &std.insert_axis(Axis(0))
    }
    
    pub fn fit_transform(&mut self, X: &Array2<f64>) -> Array2<f64> {
        self.fit(X);
        self.transform(X)
    }
    
    pub fn inverse_transform(&self, X_scaled: &Array2<f64>) -> Array2<f64> {
        let mean = self.mean_.as_ref().unwrap();
        let std = self.std_.as_ref().unwrap();
        X_scaled * &std.insert_axis(Axis(0)) + &mean.insert_axis(Axis(0))
    }
}
```

---

### 4.2 RobustScaler - 鲁棒缩放

使用中位数和四分位数，对异常值更鲁棒。

```rust
pub struct RobustScaler {
    pub median_: Option<Array1<f64>>,
    pub iqr_: Option<Array1<f64>>,
}

impl RobustScaler {
    pub fn fit(&mut self, X: &Array2<f64>) -> &Self {
        use ndarray_stats::QuantileExt;
        
        let median = X.median_axis(Axis(0)).unwrap();
        let q25 = X.quantile_axis(Axis(0), 0.25).unwrap();
        let q75 = X.quantile_axis(Axis(0), 0.75).unwrap();
        let iqr = &q75 - &q25;
        
        self.median_ = Some(median);
        self.iqr_ = Some(iqr);
        self
    }
    
    pub fn transform(&self, X: &Array2<f64>) -> Array2<f64> {
        let median = self.median_.as_ref().unwrap();
        let iqr = self.iqr_.as_ref().unwrap();
        (X - &median.insert_axis(Axis(0))) / &iqr.insert_axis(Axis(0))
    }
}
```

---

## 5. sklearn.covariance - 协方差估计

### 5.1 收缩协方差估计

#### Python/sklearn 模式

```python
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance

# Ledoit-Wolf 收缩
lw = LedoitWolf()
lw.fit(data)
cov_lw = lw.covariance_

# Oracle Approximating Shrinkage
oas = OAS()
oas.fit(data)
cov_oas = oas.covariance_

# MNE 用法：噪声协方差估计
def compute_noise_cov_shrinkage(epochs):
    """使用收缩估计噪声协方差"""
    from sklearn.covariance import LedoitWolf
    
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    
    # 重塑为 2D
    n_epochs, n_channels, n_times = data.shape
    data_2d = data.transpose(0, 2, 1).reshape(-1, n_channels)
    
    # Ledoit-Wolf 估计
    lw = LedoitWolf()
    lw.fit(data_2d)
    
    return lw.covariance_, lw.shrinkage_
```

#### Rust 等价

```rust
// Ledoit-Wolf 收缩估计
pub fn ledoit_wolf(X: &Array2<f64>) -> (Array2<f64>, f64) {
    let (n_samples, n_features) = X.dim();
    
    // 样本协方差矩阵
    let mean = X.mean_axis(Axis(0)).unwrap();
    let centered = X - &mean.insert_axis(Axis(0));
    let S = centered.t().dot(&centered) / (n_samples as f64);
    
    // 目标矩阵（收缩目标）：对角矩阵，对角线为 S 的迹除以维度
    let trace_S = S.diag().sum();
    let target = Array2::from_diag(&Array1::from_elem(n_features, trace_S / n_features as f64));
    
    // 计算最优收缩参数（简化版）
    let mut shrinkage = 0.0;
    
    // 计算 ||S - target||^2
    let diff = &S - &target;
    let beta = diff.mapv(|x| x * x).sum() / n_features as f64;
    
    // 计算 delta（渐近方差）
    let mut delta = 0.0;
    for i in 0..n_samples {
        let x = centered.row(i);
        let cov_i = x.insert_axis(Axis(1)).dot(&x.insert_axis(Axis(0)));
        let diff_i = &cov_i - &S;
        delta += diff_i.mapv(|x| x * x).sum();
    }
    delta /= (n_samples as f64).powi(2);
    
    // 最优 shrinkage
    shrinkage = (delta / beta).min(1.0).max(0.0);
    
    // 收缩协方差
    let cov_shrunk = (1.0 - shrinkage) * &S + shrinkage * &target;
    
    (cov_shrunk, shrinkage)
}
```

**注意**: 完整的 Ledoit-Wolf 实现更复杂，这里是简化版。

---

## 6. sklearn.discriminant_analysis - 判别分析

### 6.1 LDA - 线性判别分析

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train, y_train)
X_transformed = lda.transform(X_test)
y_pred = lda.predict(X_test)
```

Rust 实现涉及类内和类间散度矩阵的计算。

---

## 7. sklearn.svm 和 sklearn.ensemble - 高级分类器

### 7.1 SVM - 支持向量机

```python
from sklearn.svm import SVC

svc = SVC(kernel='rbf', C=1.0, gamma='scale')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
```

### 7.2 RandomForest - 随机森林

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

**Rust 替代**:
- `smartcore` - 完整的 ML 库（包括 SVM, RF）
- `linfa` - Rust ML 框架

---

## 8. sklearn.base 和 sklearn.utils - 基础设施

### 8.1 BaseEstimator - 基类

```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param=1.0):
        self.param = param
    
    def fit(self, X, y=None):
        # 拟合逻辑
        return self
    
    def transform(self, X):
        # 转换逻辑
        return X * self.param
```

#### Rust 等价

```rust
pub trait Estimator<X, Y> {
    fn fit(&mut self, X: &X, y: &Y) -> Result<&Self>;
    fn predict(&self, X: &X) -> Y;
}

pub trait Transformer<X> {
    fn fit(&mut self, X: &X) -> Result<&Self>;
    fn transform(&self, X: &X) -> X;
    
    fn fit_transform(&mut self, X: &X) -> Result<X> {
        self.fit(X)?;
        Ok(self.transform(X))
    }
}
```

---

## 9. sklearn.neighbors - 近邻算法

### BallTree / KDTree

```python
from sklearn.neighbors import BallTree

tree = BallTree(points, leaf_size=40)
distances, indices = tree.query(query_points, k=5)
```

**Rust 替代**:
- `kiddo` - KD 树
- 需要自己实现 BallTree 或用 `rstar`

---

## Rust ML 生态总览

### 推荐 Crate 组合

```toml
[dependencies]
# 核心数组
ndarray = "0.15"
ndarray-linalg = "0.16"
ndarray-stats = "0.5"

# 机器学习框架
linfa = { version = "0.7", features = ["all"] }
linfa-reduction = "0.7"  # PCA
linfa-clustering = "0.7"  # KMeans
linfa-linear = "0.7"     # 线性模型

# 或使用 smartcore
smartcore = { version = "0.3", features = ["all"] }

# 随机数
rand = "0.8"
rand_distr = "0.4"

# 近邻
kiddo = "2.0"

# 优化（用于自定义实现）
argmin = "0.8"
```

### linfa vs smartcore

| 特性 | linfa | smartcore |
|------|-------|-----------|
| 风格 | 模块化，类似 Python | 全功能库 |
| PCA | ✅ | ✅ |
| ICA | ❌ 需自实现 | ❌ 需自实现 |
| Ridge | ✅ | ✅ |
| SVM | ✅ | ✅ |
| Random Forest | ✅ | ✅ |
| LDA | ❌ | ✅ |
| 成熟度 | 中 | 中 |

---

## 需要自己实现的关键算法

### 优先级 P0（必需）

#### 1. **FastICA**
```rust
// src/decomposition/fastica.rs
pub struct FastICA {
    // 如前所示完整实现
}
```

#### 2. **PCA**（可用 linfa 但建议自己实现以集成）
```rust
// src/decomposition/pca.rs
pub struct PCA {
    // 如前所示
}
```

---

### 优先级 P1（重要）

#### 3. **Ridge Regression**
```rust
// src/linear_model/ridge.rs
```

#### 4. **Cross-Validation**
```rust
// src/model_selection/cross_validation.rs
```

#### 5. **StandardScaler**
```rust
// src/preprocessing/scaler.rs
```

---

### 优先级 P2（可选）

- LDA（线性判别分析）
- Lasso（L1 正则化）
- 高级分类器（SVM, RF）- 可用 `smartcore`

---

## MNE 中的典型使用场景

### 场景 1: ICA 伪影去除

**Python**:
```python
from mne.preprocessing import ICA

# 拟合 ICA
ica = ICA(n_components=20, method='fastica', random_state=97)
ica.fit(raw)

# 自动检测 EOG/ECG 成分
eog_indices, eog_scores = ica.find_bads_eog(raw)
ecg_indices, ecg_scores = ica.find_bads_ecg(raw)

# 排除伪影
ica.exclude = eog_indices + ecg_indices

# 应用
raw_clean = ica.apply(raw)
```

**Rust**:
```rust
// 拟合
let mut ica = ICA::new(20);
ica.fit(&raw.get_data())?;

// 自动检测（需要实现相关性分析）
let eog_indices = ica.find_bads_correlation(&eog_channel)?;
let ecg_indices = ica.find_bads_correlation(&ecg_channel)?;

// 排除并应用
ica.exclude.extend(eog_indices);
ica.exclude.extend(ecg_indices);
let raw_clean = ica.apply(&raw.get_data())?;
```

---

### 场景 2: 解码分析

**Python**:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from mne.decoding import Scaler, Vectorizer

# 管道
from sklearn.pipeline import make_pipeline
clf = make_pipeline(
    Scaler(epochs.info),
    Vectorizer(),
    LogisticRegression(solver='liblinear')
)

# 交叉验证
scores = cross_val_score(clf, X, y, cv=5)
```

**Rust**:
```rust
// 预处理
let mut scaler = StandardScaler::new();
let X_scaled = scaler.fit_transform(&X)?;

// 分类器
let mut clf = LogisticRegression::new(C=1.0);
clf.fit(&X_scaled, &y)?;

// 交叉验证
let kfold = KFold::new(5).with_shuffle(true);
let scores = cross_val_score(&clf, &X_scaled, &y, &kfold, accuracy);
```

---

## 难度评估和实现策略

| 组件 | 难度 | 优先级 | 建议 |
|------|------|--------|------|
| FastICA | ⭐⭐⭐⭐⭐ | P0 | 必须自实现 |
| PCA | ⭐⭐ | P0 | 简单，自实现 |
| Ridge | ⭐⭐ | P1 | 闭式解，易实现 |
| StandardScaler | ⭐ | P1 | 简单 |
| KFold | ⭐⭐ | P1 | 简单 |
| Ledoit-Wolf | ⭐⭐⭐ | P1 | 中等复杂度 |
| LDA | ⭐⭐⭐ | P2 | 可后期添加 |
| SVM | ⭐⭐⭐⭐ | P2 | 用 smartcore |
| Random Forest | ⭐⭐⭐⭐ | P2 | 用 smartcore/linfa |

---

## 总结

### 关键要点

1. **FastICA 是最大挑战**
   - 必须自己实现
   - 算法复杂但清晰
   - 核心预处理功能

2. **基础组件容易实现**
   - PCA, StandardScaler, KFold
   - 直接的数学公式
   - 高性能潜力

3. **线性模型可实现**
   - Ridge 有闭式解
   - Lasso 需要迭代优化

4. **高级分类器可复用**
   - 使用 `smartcore` 或 `linfa`
   - 优先级较低

5. **sklearn.base 基础设施**
   - 需要设计 Rust trait 系统
   - 统一接口

### 实现路线图

**阶段 1**（核心预处理）:
- [x] PCA
- [x] StandardScaler
- [ ] **FastICA** ⭐⭐⭐⭐⭐

**阶段 2**（模型评估）:
- [ ] KFold / StratifiedKFold
- [ ] cross_val_score

**阶段 3**（线性模型）:
- [ ] Ridge
- [ ] Ledoit-Wolf 协方差

**阶段 4**（高级功能）:
- [ ] LDA
- [ ] 集成 smartcore 分类器

继续阅读：[04_其他依赖分析.md](04_其他依赖分析.md)
