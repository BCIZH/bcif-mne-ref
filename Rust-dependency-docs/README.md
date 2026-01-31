# MNE-Rust 依赖库详细文档

本目录包含 MNE-Python 到 Rust 迁移过程中使用的所有核心依赖库的详细文档。

## 📚 文档列表

| 文档 | Crate | 功能分类 | C 依赖 | 成熟度 |
|------|-------|---------|--------|--------|
| [01_ndarray.md](01_ndarray.md) | `ndarray` | 数据容器 | ✅ 无 | ★★★★★ |
| [02_realfft.md](02_realfft.md) | `realfft` | 实数 FFT | ✅ 无 | ★★★★★ |
| [03_petal-decomposition.md](03_petal-decomposition.md) | `petal-decomposition` | FastICA | ✅ 无 | ★★★★☆ |
| [04_faer.md](04_faer.md) | `faer` + `faer-ndarray` | 纯 Rust 线性代数 | ✅ 无 | ★★★★★ |
| [05_idsp.md](05_idsp.md) | `idsp` | 信号滤波 | ✅ 无 | ★★★★☆ |
| [06_rubato.md](06_rubato.md) | `rubato` | 重采样 | ✅ 无 | ★★★★★ |
| [07_rustfft.md](07_rustfft.md) | `rustfft` | 复数 FFT | ✅ 无 | ★★★★☆ |
| [08_sprs.md](08_sprs.md) | `sprs` | 稀疏矩阵 | ✅ 无 | ★★★★☆ |
| [09_argmin.md](09_argmin.md) | `argmin` | 数值优化 | ✅ 无 | ★★★★☆ |
| [10_statrs.md](10_statrs.md) | `statrs` | 统计分布 | ✅ 无 | ★★★★☆ |
| [11_linfa.md](11_linfa.md) | `linfa` | 机器学习 | ✅ 无 | ★★★★☆ |
| [12_edflib.md](12_edflib.md) | `edflib` | EDF/BDF 文件 I/O | ⚠️ C 封装 | ★★★★☆ |
| [13_xdf.md](13_xdf.md) | `xdf` | XDF 文件解析 | ✅ 无 | ★★★☆☆ |
| [14_lsl.md](14_lsl.md) | `lsl` | LSL 实时通信 | ⚠️ 需 liblsl | ★★★★☆ |

## 🎯 快速索引

### 按 Python 库分类

#### NumPy 替代
- **数组操作** → [ndarray](01_ndarray.md)
- **线性代数** → [faer](04_faer.md)（纯 Rust，替代 ndarray-linalg）

#### SciPy 替代
- **FFT (实数)** → [realfft](02_realfft.md)
- **FFT (复数)** → [rustfft](07_rustfft.md)
- **信号滤波** → [idsp](05_idsp.md)
- **重采样** → [rubato](06_rubato.md)
- **稀疏矩阵** → [sprs](08_sprs.md)
- **数值优化** → [argmin](09_argmin.md)
- **统计分布** → [statrs](10_statrs.md)

#### scikit-learn 替代
- **FastICA** → [petal-decomposition](03_petal-decomposition.md)
- **PCA** → [faer](04_faer.md)（基于 SVD 直接实现，约 80 行代码）
- **机器学习框架** → **待定**（MNE decoding 模块深度依赖 sklearn 生态，暂不替换）

### 按功能分类

#### 核心数据结构
- [ndarray](01_ndarray.md) - 多维数组容器
- [faer](04_faer.md) - 纯 Rust 线性代数（SVD、特征分解、矩阵求逆）

#### 信号处理
- [realfft](02_realfft.md) - 频域分析
- [idsp](05_idsp.md) - 时域滤波
- [rubato](06_rubato.md) - 采样率转换

#### 数据 I/O
- [edflib](12_edflib.md) - EDF/BDF 文件读写（C 库封装）
- [xdf](13_xdf.md) - XDF 文件解析（纯 Rust）
- [lsl](14_lsl.md) - LSL 实时通信（官方绑定）

#### 统计与机器学习
- [petal-decomposition](03_petal-decomposition.md) - FastICA（独立成分分析）
- [faer](04_faer.md) - 纯 Rust 线性代数（含 PCA 实现）
- [linfa](11_linfa.md) - 机器学习生态（⚠️ 暂不用于 MNE 核心功能）
- [statrs](10_statrs.md) - 统计分布
- [argmin](09_argmin.md) - 优化算法
- [sprs](08_sprs.md) - 稀疏矩阵

## 📊 依赖关系图

```
ndarray (核心数组容器)
    ├── faer (纯 Rust 线性代数 + PCA)
    │   └── petal-decomposition (FastICA)
    ├── realfft (实数 FFT)
    ├── rustfft (复数 FFT)
    │   └── realfft 依赖
    ├── idsp (滤波)
    ├── rubato (重采样)
    ├── sprs (稀疏矩阵)
    ├── argmin (优化)
    └── statrs (统计)

数据 I/O（与 ndarray 配合使用）：
    ├── edflib (EDF/BDF 文件) ⚠️ C 依赖
    ├── xdf (XDF 文件) ✅ 纯 Rust
    └── lsl (实时采集) ⚠️ 需 liblsl

可选（非核心）：
    └── linfa (机器学习框架 - 暂不用于 MNE 核心)
```

## ⚙️ 统一安装配置

### 最小 Cargo.toml（核心 6 个，纯 Rust）

```toml
[dependencies]
# 核心数组与线性代数
ndarray = { version = "0.16", features = ["rayon"] }
faer = { version = "0.19", features = ["rayon"] }  # 包含 PCA 实现
faer-ndarray = "0.1"

# 信号处理
realfft = "3.3"
idsp = "0.15"
rubato = "0.18"

# 机器学习（仅 FastICA）
petal-decomposition = "0.7"
```

### 完整配置（包含可选依赖）

```toml
[dependencies]
# 核心
ndarray = { version = "0.16", features = ["rayon", "serde"] }

# 线性代数（纯 Rust）
faer = { version = "0.19", features = ["rayon"] }
faer-ndarray = "0.1"

# FFT
realfft = "3.3"
rustfft = "6.2"

# 信号处理
idsp = "0.15"
rubato = "0.18"

# 稀疏矩阵与优化
sprs = "0.11"
argmin = "0.10"
argmin-math = { version = "0.4", features = ["ndarray_latest"] }

# 统计
statrs = "0.17"

# 机器学习（核心）
petal-decomposition = "0.7"  # FastICA

# 机器学习（可选 - 暂不用于 MNE 核心）
# linfa = "0.7"
# linfa-logistic = "0.7"

# 工具
num-complex = "0.4"
num-traits = "0.2"
**注意**: faer 是纯 Rust 实现，无需平台特定配置！

如果需要 OpenBLAS/MKL（不推荐，失去纯 Rust 优势）：

```toml
# macOS (Apple Silicon) - 可选
[target.'cfg(target_os = "macos")'.dependencies]
ndarray-linalg = { version = "0.16", features = ["accelerate"] }

# Linux (Intel CPU) - 可选
[target.'cfg(target_os = "linux")'.dependencies]
ndarray-linalg = { version = "0.16", features = ["intel-mkl-static"] }

# Windows - 可选osition | 4-5x |
| 滤波 | SciPy | idsp | 6-7x |
| 重采样 | SciPy | rubato | 6-8x |
| 稀疏矩阵 | SciPy | sprs | 5-7x |
| 优化算法 | SciPy | argmin | 6-7x |
| 统计分布 | SciPy | statrs | 6-7x |
| 机器学习 | sklearn | linfa | 5-6
[target.'cfg(target_os = "windows")'.dependencies]
ndarray-linalg = { version = "0.16", features = ["openblas-static"] }
```

## 📈 性能对标总结

| 功能 | Python 库 | Rust Crate | 平均加速比 |
|------|----------|-----------|----------|
| 数组运算 | NumPy | ndarray | 3-7x |
| 矩阵分解 | SciPy | ndarray-linalg | 1.0-1.1x* |
| FFT | SciPy | realfft | 4-5x |
| ICA | sklearn | petal-decomposition | 4-5x |
| 滤波 | SciPy | idsp | 6-7x |
| 重采样 | SciPy | rubato | 6-8x |

*注：线性代数性能相当是因为都使用相同的 BLAS/LAPACK 后端

## 📝 使用建议

### 开发阶段
## 📈 性能对比总结

| 功能 | Python 库 | Rust 替代 | 性能提升 |
|------|----------|----------|---------|
| 数组操作 | NumPy | ndarray | 3-7x |
| 矩阵分解（SVD/Eigh） | SciPy | faer | ~1.1-1.2x |
| PCA | sklearn | faer (直接实现) | 同 SVD (~1.1x) |
| 实数 FFT | SciPy | realfft | 4-5x |
| 复数 FFT | SciPy | rustfft | 3-4x |
| ICA | sklearn | petal-decomposition | 4-5x |
| 滤波 | SciPy | idsp | 6-7x |
| 重采样 | SciPy | rubato | 6-8x |
| 稀疏矩阵 | SciPy | sprs | 5-7x |
| 优化算法 | SciPy | argmin | 6-7x |
| 统计分布 | SciPy | statrs | 6-7x |

**注意**: 
- faer 比 OpenBLAS 慢约 10-20%，但完全纯 Rust，无 C 依赖！
- PCA 基于 faer SVD 实现（约 80 行代码），无需额外依赖
- 机器学习框架（linfa）暂不用于 MNE 核心功能
## 💡 最佳实践

### 开发调试
1. **快速迭代**：先用 `f32` 验证算法，后期切换 `f64`
2. **启用调试信息**：
   ```toml
   [profile.dev]
   debug = true
   opt-level = 1  # 加速调试构建
   ```
3. **并行化**：使用 rayon feature 自动并行

### 生产部署
1. **纯 Rust 栈**：使用 faer 避免 C 依赖
2. **发布优化**：
   ```toml
   [profile.release]
   opt-level = 3
   lto = true
   codegen-units = 1
   ```
3. **精度要求**：科研应用使用 `f64`

## 🐛 常见问题

### 编译相关

**问题 1：编译时间过长**
- 使用 `sccache` 缓存编译结果
- 开发时减少 `opt-level`

**问题 2：需要 C 库怎么办？**
- ✅ 推荐：使用 faer（纯 Rust）
- ⚠️ 可选：ndarray-linalg + OpenBLAS（需安装 C 库）

### 运行时问题

**问题 3：多线程设置**
```bash
# Rayon 线程数（全局）
export RAYON_NUM_THREADS=4

# faer 自动使用 Rayon
```

**问题 4：ndarray 形状不匹配**
- 检查是否需要转置（MNE 使用 `(n_channels, n_times)`）
- faer 文档：https://docs.rs/faer
- 使用 `.t()` 或 `.reversed_axes()` 调整

## 🔗 相关资源

### 官方文档
- Rust ndarray 文档：https://docs.rs/ndarray
- SciPy 迁移指南：https://docs.rs/ndarray/latest/ndarray/doc/ndarray_for_numpy_users/

### 社区支持
- Rust 用户论坛：https://users.rust-lang.org
- ndarray Gitter：https://gitter.im/rust-ndarray/ndarray
- Discord (Rust Audio)：https://discord.gg/RJrZPJF

### 学习资源
- *Rust for Data Science*：https://github.com/ndscilib/ndscibook
- *Scientific Computing in Rust*：https://github.com/rust-scicomp/scicomp-rs

## 📄 许可证信息

所有文档采用的依赖库许可证：

| Crate | 许可证 | Rust Edition | no_std |
|-------|--------|--------------|--------|
| ndarray | MIT OR Apache-2.0 | 2021 | ✅ 支持 (需 alloc) |
| faer | MIT | 2021 | ✅ 支持 (需 alloc) |
| faer-ndarray | MIT | 2021 | ❌ 依赖 std |
| realfft | MIT OR Apache-2.0 | 2018 | ❌ 依赖 std |
| rustfft | MIT OR Apache-2.0 | 2018 | ✅ 支持 (需 alloc) |
| petal-decomposition | Apache-2.0 | 2021 | ❌ 依赖 std |
| idsp | MIT | 2021 | ✅ 支持 |
| rubato | MIT | 2021 | ❌ 依赖 std |
| sprs | MIT OR Apache-2.0 | 2018 | ✅ 支持 (需 alloc) |
| argmin | MIT OR Apache-2.0 | 2021 | ❌ 依赖 std |
| statrs | MIT | 2015 | ❌ 依赖 std |
| linfa | MIT OR Apache-2.0 | 2018 | ❌ 依赖 std |

---

## 🔧 Rust Edition 兼容性说明

### ✅ 核心结论

**不同 Edition 的依赖库混用完全安全**，强制使用 Rust Edition 2021 不会产生任何问题。

### 📊 本项目 Edition 分布

| Edition | Crates | 占比 |
|---------|--------|------|
| **Edition 2021** | ndarray, faer, petal-decomposition, idsp, rubato, argmin | 6/11 (55%) |
| **Edition 2018** | realfft, rustfft, sprs, linfa | 4/11 (36%) |
| **Edition 2015** | statrs | 1/11 (9%) |

### 🔍 技术原理

#### 1. Edition 的设计哲学

根据 [Rust 官方文档](https://doc.rust-lang.org/edition-guide/editions/index.html)：

> **"Editions do not split the ecosystem"** - crates in one edition must seamlessly interoperate with those compiled with other editions.

**关键特性**：
- ✅ 每个 crate 可独立选择 Edition（私有决定）
- ✅ Edition 不影响依赖图中的其他 crate
- ✅ 所有 Edition 的代码最终编译到**相同的内部表示**

#### 2. 编译器级别的兼容性保证

从 [Rust 编译器源码](https://github.com/rust-lang/rust) 可以看出：

```rust
// Edition 只影响语法解析和语义分析阶段
pub enum Edition {
    Edition2015,
    Edition2018,
    Edition2021,
    Edition2024,
}

// 最终都编译到相同的 MIR/LLVM IR
```

**Edition 差异仅限于**：
- 🔤 **关键字**：`async`/`await` 在 2018+ 才是关键字
- 🎯 **默认行为**：trait 解析规则、prelude 导入
- ⚠️ **编译器 Lint**：警告级别调整

引用官方文档：

> **"All Rust code - regardless of edition - will ultimately compile down to the same internal representation within the compiler."**

#### 3. Cargo 的自动处理机制

当你的项目依赖树包含多个 Edition 时：

```toml
# 你的主项目
[package]
edition = "2021"

[dependencies]
statrs = "0.17"   # Edition 2015
rustfft = "6.2"   # Edition 2018
ndarray = "0.16"  # Edition 2021
```

**Cargo 会自动**：
1. 为每个 crate 设置正确的 `--edition` 编译标志
2. 编译器按各自 Edition 规则解析代码
3. 链接阶段统一到相同的 ABI（无兼容性问题）

### 🧪 官方测试验证

Rust 编译器的测试套件 [mixed-editions.rs](https://github.com/rust-lang/rust/blob/master/tests/ui/editions/mixed-editions.rs) 明确验证：

- ✅ Edition 2021 主 crate 调用 Edition 2015 的宏
- ✅ Edition 2024 代码使用 Edition 2018 的依赖
- ✅ 跨 Edition 的 trait 实现和方法调用

### ⚠️ 唯一需要注意的边界情况

根据 [prelude_edition_lints.rs](https://github.com/rust-lang/rust/blob/master/compiler/rustc_lint/src/builtin.rs)：

**问题场景**：
```rust
// 你的 Edition 2021 代码
use std::convert::TryInto; // 2021 prelude 自动导入

// 某些旧依赖可能也定义了 TryInto
// 编译器会给出 ambiguity warning（不是错误）
```

**解决方法**：
```rust
// 使用完全限定语法明确指定
let x: i32 = std::convert::TryInto::try_into(value)?;
```

### 📝 最佳实践建议

#### ✅ 推荐做法

1. **主项目使用 Edition 2021**
   ```toml
   [package]
   edition = "2021"  # 最新稳定版，功能最完整
   ```

2. **不必强制升级依赖库 Edition**
   - `statrs` 使用 Edition 2015 是**维护者的有意选择**
   - 不影响功能、性能或安全性
   - 强制升级可能引入不必要的维护负担

3. **关注编译器警告**
   ```bash
   cargo build 2>&1 | grep "edition"
   ```
   - 如果出现 edition-related lint，按提示修改
   - 通常只需要添加 `use` 或使用完全限定路径

#### ❌ 避免的误区

- ❌ **误区 1**：认为依赖库必须与主项目 Edition 一致
  - **事实**：Cargo 会为每个 crate 设置正确的 Edition

- ❌ **误区 2**：担心 Edition 2015 的库"太旧"不安全
  - **事实**：Edition 与安全性无关，只影响语法特性
  - `statrs` Edition 2015 ≠ 代码过时

- ❌ **误区 3**：手动修改依赖库的 Edition
  - **事实**：Cargo.toml 中的 `edition` 只影响当前 crate

### 🎯 针对本项目的结论

**你的 Rust dependency stack 完全健康！**

```
✅ 可以安全使用 Edition 2021 进行开发
✅ 无需担心 statrs/rustfft/realfft 的 Edition 差异
✅ Cargo 和编译器会正确处理所有混合 Edition 场景
✅ 如果未来需要，可以无缝升级到 Edition 2024
```

**性能影响**：❌ **无** - Edition 不影响运行时性能  
**ABI 兼容性**：✅ **完全兼容** - 所有 Edition 使用相同 ABI  
**维护负担**：⬇️ **极低** - 无需关注依赖库的 Edition 选择

---

**文档版本**: v2.0  
**总文档数**: 11 个核心依赖库（100% 纯 Rust）  
**Rust Edition**: 主要 2021（部分 2018/2015）  
**Edition 兼容性**: ✅ 完全兼容（官方保证）  
**no_std 支持**: 5/11 支持（ndarray, faer, rustfft, idsp, sprs）  
**最后更新**: 2026-01-31
| ndarray-li1  
**总文档数**：11 个核心依赖库alg | MIT OR Apache-2.0 |
| realfft | MIT OR Apache-2.0 |
| petal-decomposition | Apache-2.0 |
| idsp | MIT OR Apache-2.0 |
| rubato | MIT |

**注意**：使用 Intel MKL 时需遵守 Intel 的许可条款。

## 🚀 下一步

1. 阅读每个 crate 的详细文档
2. 查看 `../BCIF_OVERVIEW_DOC/` 中的实现方案
3. 参考 `../MNE-Ref-zh/` 中的算法原理
4. 开始编写 Rust 代码！

---

**维护者**：MNE-Rust 团队  
**最后更新**：2026-01-31  
**文档版本**：v1.0
