# NumPy 2.2.0 架构总览

## 1. 简介

NumPy (Numerical Python) 是 Python 科学计算的基础库，提供了高性能的多维数组对象和用于处理这些数组的工具。

### 1.1 核心功能
- 强大的 N 维数组对象 (`ndarray`)
- 复杂的广播（broadcasting）功能
- 集成 C/C++ 和 Fortran 代码的工具
- 线性代数、傅里叶变换和随机数生成功能

### 1.2 版本信息
- **版本**: 2.2.0
- **发布日期**: 2026年
- **官网**: https://www.numpy.org
- **文档**: https://numpy.org/doc
- **源代码**: https://github.com/numpy/numpy

## 2. 整体架构

### 2.1 架构层次

```
┌─────────────────────────────────────────────────┐
│           NumPy 用户接口层 (Python API)          │
│  (主命名空间 + 专用子模块)                        │
├─────────────────────────────────────────────────┤
│           核心功能层 (_core 模块)                │
│  - 数组对象实现                                  │
│  - 通用函数 (ufunc)                              │
│  - 数据类型系统                                  │
├─────────────────────────────────────────────────┤
│           C/C++ 实现层                           │
│  - multiarray (多维数组核心)                     │
│  - umath (数学函数)                              │
│  - 内存管理                                      │
├─────────────────────────────────────────────────┤
│         底层数值库 (BLAS/LAPACK)                 │
│  - OpenBLAS / MKL / ATLAS                       │
└─────────────────────────────────────────────────┘
```

### 2.2 核心设计原则

1. **性能优先**: 使用 C/C++ 实现核心数组操作
2. **内存高效**: 支持视图（view）机制，避免不必要的数据复制
3. **类型安全**: 严格的数据类型系统
4. **广播机制**: 自动处理不同形状数组的操作
5. **可扩展性**: 支持用户自定义数据类型和通用函数

## 3. 目录结构

### 3.1 顶层目录
```
numpy-2.2.0/
├── numpy/              # 主要源代码目录
├── doc/               # 文档源文件
├── benchmarks/        # 性能基准测试
├── tools/             # 开发工具
├── requirements/      # 依赖需求文件
├── meson_cpu/        # CPU 特定配置
└── vendored-meson/   # 内嵌的 Meson 构建工具
```

### 3.2 numpy/ 核心目录结构
```
numpy/
├── __init__.py           # 主入口文件
├── _core/               # 核心实现（原 numpy.core）
├── lib/                 # 辅助功能库
├── linalg/              # 线性代数
├── random/              # 随机数生成
├── fft/                 # 傅里叶变换
├── polynomial/          # 多项式
├── ma/                  # 掩码数组
├── testing/             # 测试工具
├── f2py/                # Fortran 绑定
├── distutils/           # 构建工具（已弃用）
├── typing/              # 类型标注
└── [其他模块]/
```

## 4. 主要模块分类

### 4.1 核心模块
| 模块 | 说明 | 状态 |
|-----|------|-----|
| `numpy._core` | 数组对象和基础操作 | 核心 |
| `numpy.lib` | 通用工具函数 | 稳定 |
| `numpy.dtypes` | 数据类型类 | 核心 |

### 4.2 数值计算模块
| 模块 | 说明 | 状态 |
|-----|------|-----|
| `numpy.linalg` | 线性代数运算 | 推荐 |
| `numpy.fft` | 离散傅里叶变换 | 推荐 |
| `numpy.random` | 随机数生成 | 推荐 |
| `numpy.polynomial` | 多项式运算 | 推荐 |

### 4.3 辅助模块
| 模块 | 说明 | 状态 |
|-----|------|-----|
| `numpy.testing` | 测试工具 | 推荐 |
| `numpy.typing` | 类型标注 | 推荐 |
| `numpy.exceptions` | 异常定义 | 推荐 |
| `numpy.strings` | 字符串操作 | 推荐 |

### 4.4 专用模块
| 模块 | 说明 | 状态 |
|-----|------|-----|
| `numpy.ctypeslib` | ctypes 接口 | 特殊用途 |
| `numpy.emath` | 自动域数学函数 | 特殊用途 |
| `numpy.rec` | 记录数组 | 特殊用途 |
| `numpy.version` | 版本信息 | 特殊用途 |

### 4.5 遗留模块
| 模块 | 说明 | 状态 |
|-----|------|-----|
| `numpy.char` | 固定宽度字符串 | 遗留（不推荐） |
| `numpy.ma` | 掩码数组 | 遗留（需改进） |
| `numpy.matlib` | 矩阵库 | 待弃用 |
| `numpy.distutils` | 构建工具 | 已弃用 |
| `numpy.f2py` | Fortran 绑定 | 命令行工具 |

## 5. 构建系统

### 5.1 构建工具
- **主要构建系统**: Meson (现代化构建系统)
- **旧构建系统**: distutils (已弃用)
- **配置文件**: `pyproject.toml`, `meson.build`

### 5.2 编译特性
- 支持多种 BLAS/LAPACK 实现
- CPU 指令集优化（SIMD）
- 跨平台支持（Windows, macOS, Linux）

## 6. 依赖关系

### 6.1 核心依赖
```
Python >= 3.10
```

### 6.2 构建依赖
```
- meson >= 1.2.0
- ninja
- pkg-config
- C/C++ 编译器
- BLAS/LAPACK 库
```

### 6.3 可选依赖
```
- pytest (测试)
- hypothesis (测试)
- sphinx (文档)
- cython (部分扩展)
```

## 7. 关键概念

### 7.1 ndarray (多维数组)
NumPy 的核心数据结构，提供：
- 连续内存布局
- 任意维度
- 同质数据类型
- 高效的元素访问

### 7.2 dtype (数据类型)
定义数组元素的类型：
- 基础类型：int8, int16, int32, int64, float32, float64, complex128 等
- 结构化类型：记录数组
- 自定义类型：用户定义的数据类型

### 7.3 ufunc (通用函数)
对数组元素逐个操作的函数：
- 向量化操作
- 广播支持
- 类型转换
- 输出类型推断

### 7.4 广播（Broadcasting）
自动扩展不同形状的数组以进行元素级操作

### 7.5 视图（View）vs 拷贝（Copy）
- 视图：共享底层数据，不占用额外内存
- 拷贝：独立的数据副本

## 8. 性能优化

### 8.1 SIMD 优化
- 针对不同 CPU 架构的优化
- x86 (SSE, AVX, AVX2, AVX512)
- ARM (NEON)
- PowerPC (VSX)

### 8.2 并行计算
- 依赖 BLAS/LAPACK 的多线程实现
- 可通过 `threadpoolctl` 控制线程数

### 8.3 内存优化
- 内存对齐
- 缓存友好的内存布局
- 延迟求值（某些操作）

## 9. 命名空间组织

### 9.1 主命名空间 (`numpy`)
直接从 `numpy` 导入的常用函数和类：
```python
import numpy as np
np.array()      # 创建数组
np.zeros()      # 零数组
np.ones()       # 全1数组
np.arange()     # 范围数组
np.dot()        # 点积
# ... 等等
```

### 9.2 延迟导入
NumPy 2.x 使用延迟导入机制来提高启动速度：
```python
# 子模块在首次访问时才导入
import numpy as np
np.linalg  # 此时才真正导入 linalg 模块
```

## 10. API 演变

### 10.1 NumPy 2.0 重大变化
- `numpy.core` → `numpy._core` (标记为内部实现)
- 移除部分已弃用的 API
- 改进的数据类型系统
- Array API 标准兼容性

### 10.2 向后兼容性
- 遵循语义化版本控制
- 渐进式弃用策略
- 提供迁移指南

## 11. 文档结构

```
doc/
├── source/
│   ├── reference/        # API 参考
│   ├── user/            # 用户指南
│   ├── dev/             # 开发者指南
│   └── release/         # 版本发布说明
├── changelog/           # 变更日志
└── neps/               # NumPy 增强提案
```

## 12. 测试框架

- **测试工具**: pytest
- **测试发现**: `numpy.test()`
- **测试目录**: 每个模块都有对应的 `tests/` 子目录
- **性能测试**: 使用 `asv` (Airspeed Velocity)

## 13. 社区与贡献

### 13.1 开发流程
- GitHub 问题跟踪
- Pull Request 审查
- 持续集成（Azure Pipelines, CircleCI）

### 13.2 代码规范
- PEP 8 Python 代码风格
- C 代码风格指南
- 文档字符串格式（NumPy 风格）

## 14. 许可证

- **许可证类型**: BSD 3-Clause License
- **商业友好**: 可用于商业项目
- **开源**: 完全开源

## 15. 相关资源

- **官方文档**: https://numpy.org/doc/
- **教程**: https://numpy.org/learn/
- **GitHub**: https://github.com/numpy/numpy
- **Stack Overflow**: 标签 `numpy`
- **邮件列表**: numpy-discussion@python.org

---

**下一步阅读**:
- [02_核心模块详解.md](02_核心模块详解.md) - _core 模块详细分析
- [03_数值计算模块.md](03_数值计算模块.md) - linalg, fft, random 等模块
- [04_数据类型系统.md](04_数据类型系统.md) - dtype 和类型转换
- [05_通用函数ufunc.md](05_通用函数ufunc.md) - ufunc 机制详解
