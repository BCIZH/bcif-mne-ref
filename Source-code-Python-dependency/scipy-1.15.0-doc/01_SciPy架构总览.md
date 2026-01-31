# SciPy 1.15.0 架构总览

## 1. 简介

SciPy（读作"Sigh Pie"）是一个用于数学、科学和工程的开源 Python 库。它建立在 NumPy 之上，提供了大量用户友好且高效的数值计算例程。

### 1.1 核心特点

- **科学计算工具包**: 提供优化、积分、插值、线性代数等功能
- **基于 NumPy**: 与 NumPy 数组无缝集成
- **高性能**: 核心算法使用 Fortran、C 和 Cython 实现
- **经过验证**: 被世界领先的科学家和工程师依赖
- **开源免费**: BSD 许可证

### 1.2 版本信息

- **版本**: 1.15.0
- **官网**: https://scipy.org
- **文档**: https://docs.scipy.org
- **源代码**: https://github.com/scipy/scipy
- **依赖**: NumPy >= 1.23.5, < 2.5.0

## 2. 整体架构

### 2.1 架构层次

```
┌─────────────────────────────────────────────────┐
│          用户 API 层（Python）                   │
│  - 高级函数接口                                  │
│  - NumPy 数组集成                                │
├─────────────────────────────────────────────────┤
│          核心算法层                              │
│  - 科学计算算法                                  │
│  - 优化和求解器                                  │
│  - 信号和图像处理                                │
│  - 统计分析                                      │
├─────────────────────────────────────────────────┤
│          底层实现层                              │
│  - Fortran/C/C++ 库（BLAS、LAPACK、等）         │
│  - Cython 优化代码                               │
│  - 特殊函数库                                    │
├─────────────────────────────────────────────────┤
│          依赖库层                                │
│  - NumPy（核心数组操作）                         │
│  - 外部科学计算库                                │
└─────────────────────────────────────────────────┘
```

### 2.2 设计理念

1. **模块化**: 每个子模块专注特定领域
2. **性能**: 核心循环用低级语言实现
3. **易用性**: 提供简洁的 Python API
4. **兼容性**: 与 NumPy 生态系统紧密集成
5. **可靠性**: 基于成熟的数值算法库

## 3. 目录结构

### 3.1 顶层目录

```
scipy-1.15.0/
├── scipy/              # 主要源代码
├── doc/                # 文档源文件
├── benchmarks/         # 性能基准测试
├── tools/              # 开发工具
├── requirements/       # 依赖管理
└── meson.build         # 构建配置
```

### 3.2 scipy/ 核心目录结构

```
scipy/
├── __init__.py              # 包初始化
├── _lib/                    # 内部工具库
│
├── # 优化和求解
├── optimize/                # 优化算法
├── integrate/               # 数值积分
├── interpolate/             # 插值
├── odr/                     # 正交距离回归
├── differentiate/           # 数值微分
│
├── # 线性代数
├── linalg/                  # 线性代数
├── sparse/                  # 稀疏矩阵
│
├── # 信号和图像处理
├── signal/                  # 信号处理
├── ndimage/                 # N 维图像处理
├── fft/                     # 快速傅里叶变换
├── fftpack/                 # FFT（旧版）
│
├── # 统计和数据
├── stats/                   # 统计函数
├── spatial/                 # 空间算法
├── cluster/                 # 聚类算法
│
├── # 特殊函数和其他
├── special/                 # 特殊函数
├── io/                      # 数据输入/输出
├── constants/               # 物理和数学常数
├── datasets/                # 数据集
└── misc/                    # 杂项工具
```

## 4. 核心子模块

### 4.1 优化和求解器

| 模块 | 功能 | 主要算法 |
|-----|------|---------|
| `optimize` | 优化 | 最小化、求根、曲线拟合、线性规划 |
| `integrate` | 积分 | ODE 求解器、数值积分 |
| `interpolate` | 插值 | 1D/2D 插值、样条 |
| `odr` | 回归 | 正交距离回归 |
| `differentiate` | 微分 | 有限差分 |

### 4.2 线性代数

| 模块 | 功能 |
|-----|------|
| `linalg` | 线性代数运算、矩阵分解、特征值 |
| `sparse` | 稀疏矩阵格式和算法 |

### 4.3 信号和图像处理

| 模块 | 功能 |
|-----|------|
| `signal` | 信号处理（滤波器、卷积、频谱分析） |
| `ndimage` | N 维图像处理（滤波、形态学、测量） |
| `fft` | 离散傅里叶变换（现代接口） |
| `fftpack` | 离散傅里叶变换（旧版接口） |

### 4.4 统计和数据

| 模块 | 功能 |
|-----|------|
| `stats` | 统计分布、统计检验、描述统计 |
| `spatial` | 空间数据结构（KD 树、Voronoi）、距离计算 |
| `cluster` | 向量量化、K-means |

### 4.5 其他模块

| 模块 | 功能 |
|-----|------|
| `special` | 特殊数学函数（贝塞尔、伽马、误差函数等） |
| `io` | 数据文件读写（MATLAB、NetCDF、WAV 等） |
| `constants` | 物理和数学常数 |
| `datasets` | 示例数据集 |

## 5. 模块分类

### 5.1 按应用领域分类

**数学计算**:
- `linalg`: 线性代数
- `sparse`: 稀疏矩阵
- `special`: 特殊函数
- `integrate`: 积分
- `differentiate`: 微分

**数值优化**:
- `optimize`: 优化算法
- `odr`: 正交距离回归

**信号处理**:
- `signal`: 信号处理
- `fft`: 傅里叶变换
- `ndimage`: 图像处理

**数据分析**:
- `stats`: 统计
- `spatial`: 空间分析
- `cluster`: 聚类
- `interpolate`: 插值

**工具和数据**:
- `io`: 数据读写
- `constants`: 常数
- `datasets`: 数据集

### 5.2 按使用频率分类

**核心模块**（最常用）:
- `optimize`
- `linalg`
- `stats`
- `signal`
- `integrate`
- `interpolate`

**专业模块**:
- `sparse`
- `spatial`
- `ndimage`
- `special`

**工具模块**:
- `io`
- `constants`
- `datasets`
- `fft`

## 6. 依赖关系

### 6.1 核心依赖

```
必需:
- Python >= 3.10
- NumPy >= 1.23.5, < 2.5.0

可选（编译时）:
- BLAS 库（OpenBLAS、MKL、ATLAS 等）
- LAPACK 库
- Fortran 编译器（gfortran）
- C/C++ 编译器

可选（运行时）:
- matplotlib（绘图）
- Pillow（图像处理）
```

### 6.2 依赖关系图

```
SciPy
├── NumPy (必需)
│   └── 数组操作、基本数学函数
├── BLAS/LAPACK (可选但推荐)
│   └── 线性代数加速
├── 编译工具链
│   ├── Fortran 编译器
│   ├── C/C++ 编译器
│   └── Cython
└── 外部库
    ├── ARPACK (sparse.linalg)
    ├── UMFPACK (sparse.linalg)
    └── SuperLU (sparse.linalg)
```

## 7. 构建系统

### 7.1 构建工具

- **主要构建系统**: Meson
- **配置文件**: `meson.build`, `pyproject.toml`
- **开发工具**: `dev.py` (统一开发入口)

### 7.2 编译特性

- Fortran 代码编译（核心算法）
- Cython 扩展编译
- C/C++ 扩展
- BLAS/LAPACK 链接
- OpenMP 并行支持（部分模块）

## 8. 性能优化

### 8.1 低级语言实现

SciPy 的性能关键代码使用：

- **Fortran**: 线性代数、特殊函数、积分
- **C/C++**: 优化算法、空间数据结构
- **Cython**: Python/C 接口、性能关键循环

### 8.2 外部库集成

```
scipy/linalg/
├── _blas.pyx           # BLAS 封装
├── _lapack.pyx         # LAPACK 封装
└── blas.py             # BLAS 接口

scipy/sparse/linalg/
├── _isolve/            # 迭代求解器
└── _dsolve/            # 直接求解器（SuperLU、UMFPACK）

scipy/special/
└── cython_special/     # 特殊函数（Cython）
```

### 8.3 并行计算

- 某些算法支持 OpenMP 多线程
- BLAS/LAPACK 库本身可能支持并行
- 部分函数支持向量化操作

## 9. API 设计模式

### 9.1 函数式 API

大多数 SciPy 函数采用函数式接口：

```python
from scipy import optimize

# 函数式 API
result = optimize.minimize(fun, x0)
```

### 9.2 面向对象 API

某些模块提供类接口：

```python
from scipy.interpolate import CubicSpline

# OOP API
spline = CubicSpline(x, y)
y_new = spline(x_new)
```

### 9.3 一致的返回格式

许多函数返回命名元组或结果对象：

```python
from scipy.optimize import minimize

result = minimize(fun, x0)
# result.x        # 最优解
# result.fun      # 最优值
# result.success  # 是否成功
# result.message  # 信息
```

## 10. 测试框架

### 10.1 测试工具

- **测试框架**: pytest
- **每个模块都有**: `tests/` 目录
- **运行测试**: `scipy.test()`

### 10.2 测试组织

```
scipy/optimize/
├── __init__.py
├── _minimize.py
└── tests/
    ├── test_minimize.py
    ├── test_linprog.py
    └── ...
```

## 11. 文档结构

```
doc/
├── source/
│   ├── index.rst           # 主页
│   ├── tutorial/           # 教程
│   ├── reference/          # API 参考
│   └── dev/                # 开发者指南
└── ...
```

## 12. 版本演进

### 12.1 主要里程碑

- **0.1** (2001): 首次发布
- **1.0** (2017): 首个稳定版本
- **1.5** (2020): Python 3 only
- **1.10** (2023): 性能改进
- **1.15** (2024): 当前版本

### 12.2 向后兼容性

- 遵循语义版本控制
- 弃用功能会提供警告
- 至少保持一个版本的弃用期

## 13. 与 NumPy 的关系

### 13.1 建立在 NumPy 之上

```python
import numpy as np
from scipy import linalg

# SciPy 函数接受 NumPy 数组
A = np.array([[1, 2], [3, 4]])
eigenvalues = linalg.eigvals(A)  # 返回 NumPy 数组
```

### 13.2 功能划分

| NumPy | SciPy |
|-------|-------|
| 基本线性代数 | 高级线性代数 |
| 基本 FFT | 高级 FFT |
| 基本统计 | 高级统计、分布 |
| 数组操作 | 科学算法 |

## 14. 常用工作流程

### 14.1 优化问题

```python
from scipy import optimize
import numpy as np

# 定义目标函数
def rosen(x):
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# 优化
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
result = optimize.minimize(rosen, x0, method='Nelder-Mead')
print(result.x)
```

### 14.2 插值

```python
from scipy import interpolate
import numpy as np

x = np.arange(0, 10)
y = np.exp(-x/3.0)

# 创建插值函数
f = interpolate.interp1d(x, y)

# 插值
x_new = np.arange(0, 9, 0.1)
y_new = f(x_new)
```

### 14.3 统计分析

```python
from scipy import stats

# 创建分布
dist = stats.norm(loc=0, scale=1)

# 生成随机数
samples = dist.rvs(size=1000)

# 计算概率
p = dist.cdf(1.96)  # P(X <= 1.96)

# 假设检验
t_stat, p_value = stats.ttest_ind(group1, group2)
```

## 15. 扩展和自定义

### 15.1 LowLevelCallable

允许从 Python 调用 C/Cython 函数：

```python
from scipy import LowLevelCallable

# 定义 C 回调
callback = LowLevelCallable.from_cython(module, 'func_name')
```

### 15.2 子类化

某些类可以子类化以扩展功能。

## 16. 最佳实践

### 16.1 选择合适的模块

- 优化 → `scipy.optimize`
- 线性系统 → `scipy.linalg`
- 稀疏矩阵 → `scipy.sparse`
- 信号处理 → `scipy.signal`
- 统计分析 → `scipy.stats`

### 16.2 性能建议

1. 使用向量化操作
2. 选择合适的求解器/方法
3. 利用稀疏矩阵（如果适用）
4. 避免不必要的数据复制
5. 使用 `scipy.linalg` 而非 `numpy.linalg`（更快、更稳定）

### 16.3 调试技巧

```python
# 查看 SciPy 版本
import scipy
print(scipy.__version__)

# 查看配置信息
scipy.show_config()

# 运行测试
scipy.test('optimize')  # 测试特定模块
```

## 17. 相关生态系统

### 17.1 核心科学计算栈

```
SciPy
├── NumPy (基础)
├── Matplotlib (可视化)
├── Pandas (数据分析)
└── Jupyter (交互式环境)
```

### 17.2 专业库

- **scikit-learn**: 机器学习
- **scikit-image**: 图像处理
- **statsmodels**: 统计建模
- **SymPy**: 符号计算
- **NetworkX**: 图论

## 18. 社区和贡献

### 18.1 开发流程

- GitHub 问题跟踪
- Pull Request 审查
- 邮件列表讨论

### 18.2 代码规范

- PEP 8 Python 风格
- NumPy 风格的文档字符串
- 完整的单元测试

## 19. 许可证

- **许可证类型**: BSD 3-Clause License
- **商业友好**: 可用于商业项目
- **开源**: 完全开源

## 20. 相关资源

- **官方文档**: https://docs.scipy.org
- **教程**: https://docs.scipy.org/doc/scipy/tutorial/
- **Cookbook**: https://scipy-cookbook.readthedocs.io
- **GitHub**: https://github.com/scipy/scipy
- **论坛**: https://discuss.scientific-python.org/c/contributor/scipy
- **Stack Overflow**: 标签 `scipy`

---

**下一步阅读**:
- [02_线性代数和稀疏矩阵.md](02_线性代数和稀疏矩阵.md) - linalg 和 sparse 模块详解
- [03_优化和求解器.md](03_优化和求解器.md) - optimize、integrate、interpolate
- [04_信号和图像处理.md](04_信号和图像处理.md) - signal、ndimage、fft
- [05_统计和概率.md](05_统计和概率.md) - stats 模块详解
- [06_空间算法和特殊函数.md](06_空间算法和特殊函数.md) - spatial、special 等
