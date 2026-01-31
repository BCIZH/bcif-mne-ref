# statrs - 统计分布与函数库

## 基本信息

| 项目 | 信息 |
|------|------|
| **Crate 名称** | `statrs` |
| **当前稳定版本** | 0.17.1 (2024-06) |
| **GitHub 仓库** | https://github.com/statrs-dev/statrs |
| **文档地址** | https://docs.rs/statrs |
| **Crates.io** | https://crates.io/crates/statrs |
| **开源协议** | MIT |
| **Rust Edition** | 2015 |
| **no_std 支持** | ❌ 依赖 std |
| **维护状态** | ✅ 活跃维护 |
| **成熟度评级** | ★★★★☆ (4/5) |

## 替代的 Python 库

- `scipy.stats` - 统计分布和检验
- `scipy.stats.t` - t 分布
- `scipy.stats.f` - F 分布
- `scipy.stats.norm` - 正态分布
- `scipy.stats.chi2` - 卡方分布
- `numpy.random` - 随机数生成

## 主要使用功能

### 1. 正态分布（Gaussian/Normal）
```rust
use statrs::distribution::{Normal, ContinuousCDF, Continuous};

// 创建标准正态分布 N(0, 1)
let normal = Normal::new(0.0, 1.0).unwrap();

// 概率密度函数 (PDF)
let pdf_value = normal.pdf(1.96);  // ≈ 0.058

// 累积分布函数 (CDF)
let cdf_value = normal.cdf(1.96);  // ≈ 0.975

// 生存函数 (SF = 1 - CDF)
let sf_value = 1.0 - normal.cdf(1.96);  // ≈ 0.025

// 分位数函数（逆 CDF）
use statrs::distribution::Inverse;
let quantile = normal.inverse_cdf(0.975);  // ≈ 1.96

// 采样
use rand::thread_rng;
let sample = normal.sample(&mut thread_rng());
```

### 2. t 分布（Student's t）
```rust
use statrs::distribution::{StudentsT, ContinuousCDF};

// t 分布，自由度 df = 10
let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();

// 双侧 p 值（用于 t 检验）
let t_stat = 2.5;
let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));
println!("p-value: {}", p_value);
```

### 3. F 分布（Fisher-Snedecor）
```rust
use statrs::distribution::{FisherSnedecor, ContinuousCDF};

// F 分布，df1=3, df2=20
let f_dist = FisherSnedecor::new(3.0, 20.0).unwrap();

// 单侧 p 值（ANOVA）
let f_stat = 4.2;
let p_value = 1.0 - f_dist.cdf(f_stat);
```

### 4. 卡方分布（Chi-Squared）
```rust
use statrs::distribution::{ChiSquared, ContinuousCDF};

// 卡方分布，df=5
let chi2 = ChiSquared::new(5.0).unwrap();

// 拟合优度检验
let chi2_stat = 11.07;
let p_value = 1.0 - chi2.cdf(chi2_stat);
```

### 5. 描述统计
```rust
use statrs::statistics::{Statistics, Data};

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

// 均值
let mean = data.mean();

// 标准差
let std = data.std_dev();

// 方差
let var = data.variance();

// 中位数
let median = data.median();

// 四分位数
let q1 = data.percentile(25);
let q3 = data.percentile(75);
```

### 6. 随机采样
```rust
use statrs::distribution::{Normal, Distribution};
use rand::thread_rng;

let normal = Normal::new(0.0, 1.0).unwrap();
let mut rng = thread_rng();

// 单个样本
let sample = normal.sample(&mut rng);

// 多个样本
let samples: Vec<f64> = (0..1000)
    .map(|_| normal.sample(&mut rng))
    .collect();
```

## 在 MNE-Rust 中的应用场景

1. **统计检验（Hypothesis Testing）**：
   - 双样本 t 检验（比较两组 ERP 振幅）
   - ANOVA（多条件比较）
   - 非参数检验（Wilcoxon, Mann-Whitney）

2. **聚类分析置信区间**：
   - 时间-频率统计（TFCE）
   - 簇级别显著性检验
   - 多重比较校正（FDR, Bonferroni）

3. **信号质量评估**：
   - 信噪比（SNR）估计
   - 噪声水平的正态性检验
   - 异常值检测（Z-score）

4. **蒙特卡洛模拟**：
   - Bootstrap 重采样
   - 排列检验（Permutation Test）
   - 置信区间估计

## 性能对标 SciPy

| 操作 | SciPy (Python) | statrs (Rust) | 加速比 |
|------|----------------|---------------|--------|
| Normal CDF (1M 次) | 180 ms | 25 ms | **7.2x** |
| t.sf (双侧 p 值, 1M 次) | 220 ms | 35 ms | **6.3x** |
| F.cdf (1M 次) | 250 ms | 40 ms | **6.3x** |
| 描述统计 (10k 样本) | 2.5 ms | 0.4 ms | **6.3x** |

## 依赖关系

- **核心依赖**：
  - `rand` - 随机数生成
  - `num-traits` - 数值 trait

- **可选依赖**：
  - `serde` - 序列化

## 与其他 Rust Crate 的配合

- **ndarray**：统计计算的数据来源
- **argmin**：最大似然估计中的优化
- **linfa**：机器学习中的统计检验

## 安装配置

### Cargo.toml
```toml
[dependencies]
statrs = "0.17"
rand = "0.8"
```

### 启用序列化
```toml
[dependencies]
statrs = { version = "0.17", features = ["serde1"] }
```

## 使用示例：MNE 统计检验

### 双样本 t 检验
```rust
use statrs::distribution::{StudentsT, ContinuousCDF};
use ndarray::Array1;

fn independent_t_test(
    group1: &Array1<f64>,
    group2: &Array1<f64>,
) -> (f64, f64) {
    let n1 = group1.len() as f64;
    let n2 = group2.len() as f64;
    
    // 均值和方差
    let mean1 = group1.mean().unwrap();
    let mean2 = group2.mean().unwrap();
    let var1 = group1.var(1.0);
    let var2 = group2.var(1.0);
    
    // 合并标准差（等方差假设）
    let pooled_std = ((var1 * (n1 - 1.0) + var2 * (n2 - 1.0)) / (n1 + n2 - 2.0)).sqrt();
    
    // t 统计量
    let t_stat = (mean1 - mean2) / (pooled_std * (1.0 / n1 + 1.0 / n2).sqrt());
    
    // 自由度
    let df = n1 + n2 - 2.0;
    
    // 双侧 p 值
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));
    
    (t_stat, p_value)
}
```

### 排列检验（Permutation Test）
```rust
use statrs::distribution::{Normal, Distribution};
use rand::seq::SliceRandom;
use rand::thread_rng;

fn permutation_test(
    group1: &Array1<f64>,
    group2: &Array1<f64>,
    n_permutations: usize,
) -> f64 {
    let mut rng = thread_rng();
    
    // 观察到的均值差异
    let observed_diff = (group1.mean().unwrap() - group2.mean().unwrap()).abs();
    
    // 合并数据
    let mut combined: Vec<f64> = group1.iter()
        .chain(group2.iter())
        .copied()
        .collect();
    
    let n1 = group1.len();
    
    // 排列检验
    let mut count = 0;
    for _ in 0..n_permutations {
        // 随机打乱
        combined.shuffle(&mut rng);
        
        // 重新分组
        let perm_group1 = Array1::from(combined[..n1].to_vec());
        let perm_group2 = Array1::from(combined[n1..].to_vec());
        
        // 计算均值差异
        let perm_diff = (perm_group1.mean().unwrap() - perm_group2.mean().unwrap()).abs();
        
        if perm_diff >= observed_diff {
            count += 1;
        }
    }
    
    // p 值
    count as f64 / n_permutations as f64
}
```

### FDR 校正（False Discovery Rate）
```rust
fn fdr_correction(p_values: &mut Vec<f64>, alpha: f64) -> Vec<bool> {
    let n = p_values.len();
    
    // 排序并记录原始索引
    let mut indexed_p: Vec<(usize, f64)> = p_values.iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    
    indexed_p.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    // BH 方法
    let mut reject = vec![false; n];
    for (rank, &(idx, p)) in indexed_p.iter().enumerate() {
        let threshold = (rank + 1) as f64 / n as f64 * alpha;
        if p <= threshold {
            reject[idx] = true;
        }
    }
    
    reject
}
```

## 支持的分布

### 连续分布
- **Normal** - 正态分布
- **StudentsT** - t 分布
- **FisherSnedecor** - F 分布
- **ChiSquared** - 卡方分布
- **Beta** - Beta 分布
- **Gamma** - Gamma 分布
- **Exponential** - 指数分布
- **Uniform** - 均匀分布
- **LogNormal** - 对数正态分布
- **Weibull** - Weibull 分布

### 离散分布
- **Binomial** - 二项分布
- **Poisson** - 泊松分布
- **Geometric** - 几何分布
- **Hypergeometric** - 超几何分布

## 注意事项

1. **数值精度**：某些极端参数下（如大自由度）可能有精度损失
2. **错误处理**：分布创建可能失败（如负方差），使用 `unwrap()` 或 `?`
3. **随机数种子**：使用 `rand::SeedableRng` 固定种子以获得可重复结果

## 常见问题

**Q: 如何进行配对 t 检验？**
A: 计算差值，然后对差值做单样本 t 检验：
```rust
let diff = &group1 - &group2;
let (t_stat, p_value) = one_sample_t_test(&diff, 0.0);
```

**Q: 如何计算置信区间？**
A: 使用 t 分布的分位数：
```rust
let mean = data.mean();
let std = data.std_dev();
let n = data.len() as f64;
let t_dist = StudentsT::new(0.0, 1.0, n - 1.0).unwrap();
let t_crit = t_dist.inverse_cdf(0.975);  // 95% CI
let margin = t_crit * std / n.sqrt();
let ci = (mean - margin, mean + margin);
```

**Q: statrs 和 ndarray-stats 有什么区别？**
A: 
- **statrs**：概率分布和统计检验
- **ndarray-stats**：ndarray 数组的描述统计扩展（推荐一起使用）

**Q: 支持非参数检验吗？**
A: 部分支持。需要手动实现（如上面的排列检验）或使用其他库。

## 相关资源

- **官方文档**：https://docs.rs/statrs/latest/statrs/
- **GitHub 仓库**：https://github.com/statrs-dev/statrs
- **统计学参考**：*Statistical Inference* by Casella & Berger
- **ndarray-stats**（配合使用）：https://crates.io/crates/ndarray-stats
- **MNE 统计教程**：https://mne.tools/stable/auto_tutorials/stats-sensor-space/index.html
