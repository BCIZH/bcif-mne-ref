# SciPy 1.15.0 中文参考文档

## 📚 文档总览

本文档集提供 SciPy 1.15.0 的全面中文参考，涵盖架构、核心模块和实用示例。

**SciPy** 是 Python 科学计算的基石库，提供数学算法和函数，建立在 NumPy 之上。

---

## 📖 文档导航

### 1️⃣ [SciPy 架构总览](01_SciPy架构总览.md)
**推荐首先阅读** ⭐

- SciPy 简介和历史
- 整体架构设计
- 19 个核心子模块概览
- 依赖关系和构建系统
- 设计哲学和最佳实践
- 与 NumPy 的关系
- 性能优化技术

**适合**: 
- 初次接触 SciPy 的开发者
- 需要了解整体架构的用户
- 想理解 SciPy 设计理念的研究者

---

### 2️⃣ [线性代数和稀疏矩阵](02_线性代数和稀疏矩阵.md)

#### 核心内容：

**scipy.linalg - 线性代数**
- 基本运算（求逆、行列式、范数）
- 矩阵分解（LU、QR、SVD、Cholesky、Schur）
- 特征值问题
- 矩阵函数（指数、对数、平方根）
- 特殊矩阵
- 线性系统求解

**scipy.sparse - 稀疏矩阵**
- 稀疏矩阵格式（COO、CSR、CSC、LIL、DOK）
- 稀疏矩阵运算
- 稀疏线性代数
- 格式转换和优化
- 内存高效存储

**应用场景**:
- 大规模线性系统
- 有限元分析
- 图算法
- 机器学习（特征矩阵）
- 网络分析

---

### 3️⃣ [优化和求解器](03_优化和求解器.md)

#### 核心内容：

**scipy.optimize - 优化**
- 标量/多元函数优化（minimize, minimize_scalar）
- 优化算法（BFGS, L-BFGS-B, Nelder-Mead, CG）
- 约束优化（线性/非线性约束、边界）
- 全局优化（differential_evolution, basinhopping）
- 曲线拟合（curve_fit, least_squares）
- 方程求根（root, fsolve）
- 线性规划（linprog）

**scipy.integrate - 积分**
- 数值积分（quad, dblquad, tplquad）
- 常微分方程（solve_ivp, odeint）
- 边值问题（solve_bvp）
- ODE 求解器选择（RK45, BDF, LSODA）

**scipy.interpolate - 插值**
- 1D/2D/nD 插值
- 样条插值
- 散点插值
- 多种插值方法

**应用场景**:
- 参数估计和模型拟合
- 工程优化问题
- 物理模拟（ODE/PDE）
- 数据插值和平滑
- 资源分配（线性规划）

---

### 4️⃣ [信号和图像处理](04_信号和图像处理.md)

#### 核心内容：

**scipy.signal - 信号处理**
- 滤波器设计（Butterworth, Chebyshev, 椭圆, Bessel）
- IIR 和 FIR 滤波器
- 信号滤波（filtfilt, lfilter, sosfilt）
- 频谱分析（periodogram, welch, spectrogram）
- 时频分析（STFT, CWT）
- 卷积和相关
- 峰值检测
- LTI 系统分析
- 重采样

**scipy.ndimage - 多维图像处理**
- n 维图像滤波
- 形态学操作（膨胀、腐蚀、开闭运算）
- 几何变换（旋转、平移、缩放）
- 特征测量
- 连通分量标记
- 边缘检测

**scipy.fft - 快速傅里叶变换**
- 1D/2D/nD FFT
- 实数 FFT 优化（rfft）
- DCT/DST
- FFT 性能优化

**应用场景**:
- 音频处理和分析
- 生物医学信号（ECG, EEG）
- 图像处理和计算机视觉
- 振动分析
- 通信系统
- 频谱分析

---

### 5️⃣ [统计和概率](05_统计和概率.md)

#### 核心内容：

**scipy.stats - 统计函数**
- **100+ 概率分布**
  - 连续分布（正态、t、卡方、F、Beta、Gamma 等）
  - 离散分布（二项、泊松、几何、负二项等）
  - 多元分布
- **描述统计**
  - 均值、中位数、众数
  - 方差、标准差
  - 偏度、峰度
  - 分位数
- **统计检验**
  - t 检验（单样本、独立、配对）
  - ANOVA
  - 非参数检验（Mann-Whitney, Wilcoxon, Kruskal-Wallis）
  - 正态性检验（Shapiro-Wilk, K-S, Anderson）
  - 卡方检验
- **相关性分析**
  - Pearson 相关
  - Spearman 秩相关
  - Kendall τ
- **其他功能**
  - 线性回归
  - 核密度估计（KDE）
  - 分布拟合
  - Bootstrap 方法
  - 置信区间

**应用场景**:
- 实验数据分析
- A/B 测试
- 假设检验
- 概率建模
- 风险分析
- 质量控制
- 生物统计

---

## 🔍 快速查找

### 按功能查找

| 需求 | 推荐文档 | 关键函数 |
|------|---------|---------|
| 求解线性方程组 | [02 线性代数](02_线性代数和稀疏矩阵.md#2-基本线性代数运算) | `linalg.solve()` |
| 矩阵分解 | [02 线性代数](02_线性代数和稀疏矩阵.md#3-矩阵分解) | `linalg.svd()`, `linalg.qr()` |
| 处理大型稀疏矩阵 | [02 线性代数](02_线性代数和稀疏矩阵.md#7-scipysparse---稀疏矩阵) | `sparse.csr_matrix()` |
| 函数最小化 | [03 优化](03_优化和求解器.md#3-多元函数优化无约束) | `optimize.minimize()` |
| 拟合曲线 | [03 优化](03_优化和求解器.md#6-曲线拟合) | `optimize.curve_fit()` |
| 求解 ODE | [03 优化](03_优化和求解器.md#93-常微分方程ode) | `integrate.solve_ivp()` |
| 数值积分 | [03 优化](03_优化和求解器.md#92-数值积分) | `integrate.quad()` |
| 信号滤波 | [04 信号处理](04_信号和图像处理.md#2-滤波器设计) | `signal.butter()`, `signal.filtfilt()` |
| 频谱分析 | [04 信号处理](04_信号和图像处理.md#4-频谱分析) | `signal.welch()`, `signal.spectrogram()` |
| 图像滤波 | [04 信号处理](04_信号和图像处理.md#92-图像滤波) | `ndimage.gaussian_filter()` |
| 峰值检测 | [04 信号处理](04_信号和图像处理.md#6-峰值检测) | `signal.find_peaks()` |
| 统计检验 | [05 统计](05_统计和概率.md#4-统计检验) | `stats.ttest_ind()`, `stats.mannwhitneyu()` |
| 概率分布 | [05 统计](05_统计和概率.md#2-概率分布) | `stats.norm`, `stats.binom` |
| 相关性分析 | [05 统计](05_统计和概率.md#5-相关性分析) | `stats.pearsonr()`, `stats.spearmanr()` |

---

## 🎯 按应用领域查找

### 数据科学和机器学习
- [02 线性代数](02_线性代数和稀疏矩阵.md) - SVD、主成分分析
- [03 优化](03_优化和求解器.md) - 参数估计、模型拟合
- [05 统计](05_统计和概率.md) - 假设检验、特征相关性

### 信号处理
- [04 信号处理](04_信号和图像处理.md) - 完整的信号处理工具链
- 滤波器设计、频谱分析、时频分析

### 图像处理
- [04 信号处理](04_信号和图像处理.md#9-scipyndimage---多维图像处理) - ndimage 模块
- 滤波、形态学、几何变换

### 科学计算
- [02 线性代数](02_线性代数和稀疏矩阵.md) - 矩阵运算
- [03 优化](03_优化和求解器.md) - 数值优化、ODE 求解

### 统计分析
- [05 统计](05_统计和概率.md) - 完整的统计工具
- 描述统计、推断统计、概率建模

---

## 📊 核心模块速览

### SciPy 19 个子模块

| 模块 | 功能 | 详细文档 |
|------|------|---------|
| `scipy.cluster` | 聚类算法 | [01 架构](01_SciPy架构总览.md#611-cluster---聚类算法) |
| `scipy.constants` | 物理和数学常数 | [01 架构](01_SciPy架构总览.md#619-constants---物理常数) |
| `scipy.datasets` | 示例数据集 | [01 架构](01_SciPy架构总览.md) |
| `scipy.differentiate` | 数值微分 | [01 架构](01_SciPy架构总览.md) |
| `scipy.fft` | 快速傅里叶变换 | [04 信号处理](04_信号和图像处理.md#10-scipyfft---快速傅里叶变换) |
| `scipy.fftpack` | FFT（遗留） | [04 信号处理](04_信号和图像处理.md) |
| `scipy.integrate` | 积分和 ODE | [03 优化](03_优化和求解器.md#9-scipyintegrate---数值积分和微分方程) |
| `scipy.interpolate` | 插值 | [03 优化](03_优化和求解器.md#10-scipyinterpolate---插值) |
| `scipy.io` | 数据 I/O | [01 架构](01_SciPy架构总览.md#618-io---数据输入输出) |
| `scipy.linalg` | 线性代数 | [02 线性代数](02_线性代数和稀疏矩阵.md#1-scipylinalg---线性代数) |
| `scipy.ndimage` | n 维图像处理 | [04 信号处理](04_信号和图像处理.md#9-scipyndimage---多维图像处理) |
| `scipy.odr` | 正交距离回归 | [01 架构](01_SciPy架构总览.md) |
| `scipy.optimize` | 优化和求根 | [03 优化](03_优化和求解器.md#1-scipyoptimize---优化和求根) |
| `scipy.signal` | 信号处理 | [04 信号处理](04_信号和图像处理.md#1-scipysignal---信号处理) |
| `scipy.sparse` | 稀疏矩阵 | [02 线性代数](02_线性代数和稀疏矩阵.md#7-scipysparse---稀疏矩阵) |
| `scipy.spatial` | 空间算法 | [01 架构](01_SciPy架构总览.md#612-spatial---空间数据结构和算法) |
| `scipy.special` | 特殊函数 | [01 架构](01_SciPy架构总览.md#613-special---特殊数学函数) |
| `scipy.stats` | 统计函数 | [05 统计](05_统计和概率.md#1-scipystats---统计函数) |

---

## 🚀 入门指南

### 安装

```bash
# 使用 pip
pip install scipy

# 使用 conda
conda install scipy

# 或从源码安装（需要 Meson）
git clone https://github.com/scipy/scipy.git
cd scipy
pip install .
```

### 基本用法

```python
import numpy as np
from scipy import linalg, optimize, stats, signal

# 线性代数
A = np.array([[1, 2], [3, 4]])
eigenvalues, eigenvectors = linalg.eig(A)

# 优化
def f(x):
    return x**2 + 2*x + 1

result = optimize.minimize_scalar(f)

# 统计
data = np.random.randn(100)
statistic, p_value = stats.shapiro(data)

# 信号处理
sos = signal.butter(4, 0.1, output='sos')
filtered = signal.sosfilt(sos, data)
```

---

## 📚 推荐学习路径

### 初学者路径
1. **[01 架构总览](01_SciPy架构总览.md)** - 了解整体结构
2. **[05 统计](05_统计和概率.md)** - 最常用的模块，从描述统计开始
3. **[03 优化](03_优化和求解器.md)** - 学习基本优化和拟合
4. **[02 线性代数](02_线性代数和稀疏矩阵.md)** - 掌握矩阵运算

### 信号处理方向
1. **[01 架构总览](01_SciPy架构总览.md)** - 基础知识
2. **[04 信号处理](04_信号和图像处理.md)** - 核心内容
3. **[03 优化](03_优化和求解器.md)** - 系统分析和滤波器设计
4. **[02 线性代数](02_线性代数和稀疏矩阵.md)** - 高级数值方法

### 科学计算方向
1. **[01 架构总览](01_SciPy架构总览.md)** - 整体理解
2. **[02 线性代数](02_线性代数和稀疏矩阵.md)** - 线性代数基础
3. **[03 优化](03_优化和求解器.md)** - 优化和微分方程
4. **[05 统计](05_统计和概率.md)** - 数据分析

### 数据科学方向
1. **[05 统计](05_统计和概率.md)** - 统计分析核心
2. **[02 线性代数](02_线性代数和稀疏矩阵.md)** - SVD、PCA 等
3. **[03 优化](03_优化和求解器.md)** - 模型拟合
4. **[01 架构总览](01_SciPy架构总览.md)** - 深入理解

---

## 🔗 相关资源

### 官方资源
- [SciPy 官方文档](https://docs.scipy.org/)
- [SciPy GitHub](https://github.com/scipy/scipy)
- [SciPy 教程](https://docs.scipy.org/doc/scipy/tutorial/)

### 相关库文档
- [NumPy 参考文档](../numpy-2.2.0-doc/) - SciPy 的基础
- [scikit-learn 参考文档](../scikit-learn-1.6.0-doc/) - 机器学习
- [MNE-Python 参考文档](../../MNE-Ref-zh/) - 神经科学应用

### 学习资源
- SciPy Lecture Notes
- Python for Data Analysis (Wes McKinney)
- Numerical Python (Robert Johansson)

---

## 💡 使用技巧

### 性能优化
1. **使用向量化**代替循环
2. **选择合适的算法**（见各文档的"方法选择"部分）
3. **使用 SOS 而非 b/a**（滤波器）
4. **利用稀疏矩阵**处理大规模问题
5. **查看文档中的性能比较**章节

### 常见陷阱
1. **未检查假设**（如正态性）
2. **忽略数值稳定性**
3. **错误的参数顺序**（注意文档中的参数说明）
4. **未使用 `ddof=1`**（样本统计量）
5. **过度拟合**（优化问题）

### 调试技巧
1. **检查输入形状**和数据类型
2. **从简单例子开始**
3. **可视化中间结果**
4. **阅读错误信息**和警告
5. **参考文档中的实用案例**

---

## 📝 文档约定

### 代码示例
- 所有示例都是**可运行**的
- 使用**标准导入**惯例
- 包含**注释**和**输出说明**

### 数学公式
- 使用 **LaTeX** 格式
- 提供**直观解释**
- 配合**代码实现**

### 版本信息
- 基于 **SciPy 1.15.0**
- 注明版本特定功能
- 指出过时 API

---

## 🤝 贡献

发现错误或有改进建议？欢迎：
- 提交 Issue
- 发起 Pull Request
- 提供反馈

---

## 📅 更新日志

- **2024**: 初始版本（SciPy 1.15.0）
  - 完整的架构文档
  - 5 个核心主题文档
  - 大量实用示例
  - 中文优化

---

## 版权声明

本文档基于 SciPy 1.15.0 官方文档编写，遵循 BSD 许可证。

SciPy 版权归属 SciPy 开发者所有。

---

**开始学习**: [01_SciPy架构总览.md](01_SciPy架构总览.md) ⭐

**快速参考**: 使用上面的**快速查找表**直接跳转到您需要的内容！
