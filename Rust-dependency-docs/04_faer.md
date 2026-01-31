# faer + faer-ndarray - 纯 Rust 线性代数库

## 基本信息

| 项目 | 信息 |
|------|------|
| **Crate 名称** | `faer` + `faer-ndarray` |
| **faer 版本** | 0.23.0 (2025-09) |
| **faer-ndarray 版本** | 0.1.0 (2024-11) |
| **GitHub 仓库** | https://github.com/sarah-ek/faer-rs |
| **文档地址** | https://docs.rs/faer |
| **Crates.io** | https://crates.io/crates/faer/0.19.4 |
| **开源协议** | MIT |
| **Rust Edition** | 2021 |
| **no_std 支持** | ✅ 支持（需 alloc） |
| **维护状态** | ✅ 活跃维护 |
| **成熟度评级** | ★★★★★ (5/5) |

## 替代的 Python 库

- `scipy.linalg` - 线性代数操作（SVD、特征分解、矩阵求逆等）
- `numpy.linalg` - 基础线性代数
- **替代 Rust 库**: `ndarray-linalg`（需要 C 依赖 OpenBLAS/MKL）

## 核心优势

### ✅ 完全纯 Rust
- 无需 C 库依赖（OpenBLAS、Intel MKL、LAPACK）
- 跨平台编译简单（无需 gfortran）
- 静态链接容易（单一二进制文件）
- WebAssembly 支持（可编译到 WASM）
- 嵌入式友好（支持 no_std + alloc）

### 🚀 性能接近 BLAS
| 操作 | faer (纯 Rust) | OpenBLAS | 性能差距 |
|------|----------------|----------|---------|
| SVD (1000×500) | 185 ms | 175 ms | +6% |
| Eigh (500×500) | 70 ms | 62 ms | +13% |
| 矩阵乘法 (1000×1000) | 52 ms | 45 ms | +16% |
| Cholesky (1000×1000) | 18 ms | 15 ms | +20% |

## 主要使用功能

### 1. SVD - 奇异值分解

```rust
use faer::prelude::*;
use faer_ndarray::{IntoFaer, IntoNdarray};
use ndarray::Array2;

// ndarray → faer 转换
let A: Array2<f64> = Array2::zeros((100, 50));
let A_faer = A.view().into_faer();

// 执行 SVD
let svd = A_faer.svd();

// 获取结果
let u = svd.u();                    // 左奇异向量
let s = svd.s_diagonal();           // 奇异值（对角线）
let vt = svd.v().transpose();       // 右奇异向量转置

// faer → ndarray 转换
let u_nd = u.as_ref().into_ndarray();
let s_nd = Array1::from_iter(s.column_vector_as_slice().iter().copied());
let vt_nd = vt.as_ref().into_ndarray();

// 重建矩阵: A = U * Σ * V^T
let A_reconstructed = u_nd.dot(&Array::from_diag(&s_nd)).dot(&vt_nd);
```

### 2. 特征分解（对称矩阵）

```rust
use faer::prelude::*;
use faer_ndarray::{IntoFaer, IntoNdarray};

// 对称矩阵特征分解
let A_faer = symmetric_matrix.view().into_faer();
let eigen = A_faer.selfadjoint_eigendecomposition(faer::Side::Lower);

// 获取特征值和特征向量
let eigenvalues_faer = eigen.s_diagonal();
let eigenvectors_faer = eigen.u();

// 转回 ndarray
let eigenvalues = Array1::from_iter(
    eigenvalues_faer.column_vector_as_slice().iter().copied()
);
let eigenvectors = eigenvectors_faer.as_ref().into_ndarray();
```

### 3. 矩阵求逆

```rust
use faer::prelude::*;
use faer_ndarray::{IntoFaer, IntoNdarray};

// 直接求逆（方阵）
let A_faer = A.view().into_faer();
let A_inv_faer = A_faer.inverse();
let A_inv = A_inv_faer.as_ref().into_ndarray();
```

### 4. 伪逆（基于 SVD）

```rust
fn pinv_faer(A: &Array2<f64>, rcond: f64) -> Array2<f64> {
    let A_faer = A.view().into_faer();
    let svd = A_faer.svd();
    
    let u = svd.u();
    let s = svd.s_diagonal();
    let vt = svd.v().transpose();
    
    // 截断小奇异值
    let cutoff = rcond * s.column_vector_as_slice()[0];
    let s_inv: Vec<f64> = s.column_vector_as_slice()
        .iter()
        .map(|&si| if si > cutoff { 1.0 / si } else { 0.0 })
        .collect();
    
    // A^+ = V * diag(1/s) * U^T
    let s_inv_mat = faer::Mat::from_fn(s_inv.len(), s_inv.len(), |i, j| {
        if i == j { s_inv[i] } else { 0.0 }
    });
    
    let result = vt.transpose() * &s_inv_mat * u.transpose();
    result.as_ref().into_ndarray()
}
```

### 5. 线性方程组求解

```rust
use faer::prelude::*;
use faer_ndarray::IntoFaer;

// LU 分解求解 Ax = b
let A_faer = A.view().into_faer();
let b_faer = b.view().into_faer();

let x_faer = A_faer.partial_piv_lu().solve(&b_faer);
let x = Array1::from(x_faer.col_as_slice(0).to_vec());

// Cholesky 分解（正定矩阵，更快）
let L = A_faer.cholesky(faer::Side::Lower).unwrap();
let x_faer = L.solve(&b_faer);
let x = Array1::from(x_faer.col_as_slice(0).to_vec());
```

## MNE 应用场景

### 1. 最小范数估计（Minimum Norm Estimate）

```rust
use faer::prelude::*;
use faer_ndarray::{IntoFaer, IntoNdarray};

// 正则化最小范数：X = G^T (G G^T + λI)^{-1} M
fn minimum_norm_estimate(
    G: &Array2<f64>,      // 导联场矩阵 (n_sensors, n_sources)
    M: &Array2<f64>,      // 测量数据 (n_sensors, n_times)
    lambda: f64           // 正则化参数
) -> Array2<f64> {
    let G_faer = G.view().into_faer();
    let M_faer = M.view().into_faer();
    
    // 计算 G * G^T
    let GGT = &G_faer * G_faer.transpose();
    
    // 添加正则化: G G^T + λI
    let n = GGT.nrows();
    let I = faer::Mat::from_fn(n, n, |i, j| if i == j { lambda } else { 0.0 });
    let A = GGT + &I;
    
    // 求解 A * Y = M
    let Y = A.partial_piv_lu().solve(&M_faer);
    
    // X = G^T * Y
    let X_faer = G_faer.transpose() * &Y;
    
    X_faer.as_ref().into_ndarray()
}
```

### 2. ICA 白化（Whitening）

```rust
// PCA 白化：X_white = (X - μ) * K
// 其中 K = V * Σ^{-1}，来自 SVD(X)
fn whiten_data(X: &Array2<f64>, n_components: usize) -> (Array2<f64>, Array2<f64>) {
    // 中心化
    let mean = X.mean_axis(Axis(0)).unwrap();
    let X_centered = X - &mean.insert_axis(Axis(0));
    
    // SVD
    let X_faer = X_centered.t().view().into_faer();
    let svd = X_faer.svd();
    
    let vt = svd.v().transpose();
    let s = svd.s_diagonal();
    
    // K = V * Σ^{-1}（取前 n_components 个）
    let s_inv: Vec<f64> = s.column_vector_as_slice()[..n_components]
        .iter()
        .map(|&x| 1.0 / x)
        .collect();
    
    // 构造白化矩阵
    let vt_nd = vt.as_ref().into_ndarray();
    let K = vt_nd.slice(s![..n_components, ..]).t().to_owned() 
        / &Array1::from(s_inv).insert_axis(Axis(0));
    
    // 白化变换
    let X_white = X_centered.dot(&K);
    
    (X_white, K)
}
```

### 3. 协方差矩阵正则化

```rust
// 噪声协方差正则化（Ledoit-Wolf 收缩）
fn regularize_covariance(
    C: &Array2<f64>,      // 样本协方差矩阵
    shrinkage: f64        // 收缩系数 (0-1)
) -> Array2<f64> {
    let C_faer = C.view().into_faer();
    
    // 特征分解
    let eigen = C_faer.selfadjoint_eigendecomposition(faer::Side::Lower);
    let lambda = eigen.s_diagonal();
    let V = eigen.u();
    
    // 目标矩阵：对角线 = 平均特征值
    let lambda_slice = lambda.column_vector_as_slice();
    let mu = lambda_slice.iter().sum::<f64>() / lambda_slice.len() as f64;
    
    // 收缩：λ_reg = (1-α)λ + α*μ
    let lambda_reg: Vec<f64> = lambda_slice
        .iter()
        .map(|&x| (1.0 - shrinkage) * x + shrinkage * mu)
        .collect();
    
    // 重建：C_reg = V * diag(λ_reg) * V^T
    let lambda_mat = faer::Mat::from_fn(lambda_reg.len(), lambda_reg.len(), |i, j| {
        if i == j { lambda_reg[i] } else { 0.0 }
    });
    
    let C_reg = &V * &lambda_mat * V.transpose();
    C_reg.as_ref().into_ndarray()
}
```

## 性能优化技巧

### 1. 并行计算
```rust
// faer 自动使用 Rayon 并行化（如果启用 rayon feature）
[dependencies]
faer = { version = "0.19", features = ["rayon"] }
```

### 2. 原地操作
```rust
// 避免不必要的内存分配
let mut A_faer = A.view_mut().into_faer();
A_faer.cholesky_inplace(faer::Side::Lower);
```

### 3. 稀疏矩阵交互
```rust
// faer 专注稠密矩阵，稀疏用 sprs
use sprs::CsMat;

// 稀疏 × 稠密
let result = &sparse_matrix * &dense_vector;
```

## 安装与配置

### Cargo.toml（推荐配置）

```toml
[dependencies]
ndarray = "0.16"
faer = { version = "0.19", features = ["rayon"] }
faer-ndarray = "0.1"

# 可选：ndarray-linalg（性能对比）
# ndarray-linalg = { version = "0.16", features = ["openblas-static"] }
```

### no_std 配置（嵌入式）

```toml
[dependencies]
faer = { version = "0.19", default-features = false, features = ["std"] }
faer-ndarray = { version = "0.1", default-features = false }
```

## 常见问题

### Q1: faer 比 OpenBLAS 慢多少？
**A**: 约慢 10-20%，但完全纯 Rust，无 C 依赖，部署简单。

### Q2: 什么时候选择 faer？
**A**: 
- ✅ 需要跨平台部署（Windows/Linux/macOS/WASM）
- ✅ 希望静态链接单一二进制文件
- ✅ 嵌入式或 no_std 环境
- ❌ HPC 集群极致性能 → 用 Intel MKL

### Q3: faer-ndarray 是必需的吗？
**A**: 如果你使用 ndarray 生态，是的。它提供 `IntoFaer`/`IntoNdarray` trait 实现零拷贝转换。

### Q4: faer 支持 GPU 加速吗？
**A**: 不支持。GPU 加速需要 cuBLAS 等 CUDA 库。

### Q5: 如何选择后端？

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 生产部署（简化） | faer | 无 C 依赖，易部署 |
| HPC 集群 | Intel MKL | 最快（Intel CPU） |
| 科研原型 | OpenBLAS | 平衡性能/兼容性 |
| WASM/嵌入式 | faer | 唯一纯 Rust 选择 |

## 相关资源

- **官方文档**: https://docs.rs/faer
- **性能基准**: https://github.com/sarah-ek/faer-rs/tree/main/bench
- **faer vs BLAS 对比**: https://github.com/sarah-ek/faer-bench
- **教程**: https://github.com/sarah-ek/faer-rs/tree/main/examples

## 总结

faer 是 Rust 生态中最先进的纯 Rust 线性代数库，性能接近优化的 BLAS 实现（仅慢 10-20%），但完全无需 C 库依赖。对于需要简化部署、跨平台支持或嵌入式应用的场景，是 ndarray-linalg 的最佳替代品。
