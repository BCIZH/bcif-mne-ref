# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BCIF (Brain-Computer Interface Framework) is a **documentation and reference repository** for building a Rust-first signal processing framework to replace Python scientific computing dependencies (NumPy/SciPy/scikit-learn) in brain-computer interface applications. It is based on MNE-Python v1.110.

**Important**: This is NOT a buildable Rust workspace. There is no root `Cargo.toml`. This repository contains:
- Reference MNE-Python source code (`mne-python-ref-fork-v1.110/`)
- Rust crate source code for reference (`Source-code-Rust-dependency/`)
- Python library source code for reference (`Source-code-Python-dependency/`)
- Architecture documentation and migration guides

## Repository Structure

```
bcif-mne-ref/
├── BCIF_Agent_Prompt.md          # AI development guidelines (v2.0.0)
├── BCIF_Core_Pipeline.md         # Five-layer architecture documentation
├── BCIF_OVERVIEW_DOC/            # 8 analysis documents (dependency mapping, migration priorities)
├── Rust_Guideline/               # Rust coding standards (std and embedded)
├── C++_Guideline/                # C++17 coding standards (std and embedded)
├── Rust-dependency-docs/         # Documentation for each Rust crate
├── MNE-refactor/                 # Dependency analysis documents
├── mne-python-ref-fork-v1.110/   # Reference MNE-Python v1.110 source
├── Source-code-Rust-dependency/  # Rust crate source (ndarray, faer, realfft, etc.)
└── Source-code-Python-dependency/# Python source (numpy, scipy, scikit-learn)
```

## Key Technical Decisions

### Rust Crate Selection (Pure Rust Priority)

| Functionality | Python | Rust Selection |
|---------------|--------|----------------|
| Data Container | `numpy.ndarray` | **ndarray** |
| Real FFT | `scipy.fft.rfft` | **realfft** |
| Complex FFT | `scipy.fft.fft` | **rustfft** |
| ICA | `sklearn.FastICA` | **petal-decomposition** |
| Linear Algebra | OpenBLAS/MKL | **faer + faer-ndarray** |
| IIR Filtering | `scipy.signal.butter` | **idsp** |
| Resampling | `scipy.signal.resample` | **rubato** |
| PCA | `sklearn.PCA` | **faer** (direct SVD implementation) |
| Sparse Matrix | `scipy.sparse` | **sprs** |
| Optimization | `scipy.optimize` | **argmin** |
| Statistics | `scipy.stats` | **statrs** |

### Target Deployment Profiles

- **Full Profile** (Desktop/Server): `std + alloc + rayon + BLAS + FFTW`
- **Embedded Profile** (ARM Cortex-M, RISC-V): `no_std + alloc (optional)`, fixed-size buffers

## Coding Guidelines

### Rust Development Rules

Follow guidelines in `Rust_Guideline/Rust_AI_Coding_Guideline_Std.md`:

1. **Bilingual comments** - English first, Chinese second
2. **Explicit types** - Always annotate types, avoid inference for complex types
3. **No unwrap() in production** - Use `?`, `unwrap_or()`, or `if let`
4. **No magic numbers** - Use named constants
5. **Clone when confused** - Optimize later if profiling shows bottleneck
6. **Structs own data** - Avoid lifetimes in structs unless necessary

```rust
// ✅ GOOD: Explicit types, bilingual comments
/// Calculate mean of samples.
/// 计算采样均值。
fn calculate_mean(data: &[f64]) -> f64 {
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

// ❌ BAD: Implicit types, no comments
let x = data.iter().sum::<f64>() / data.len() as f64;
```

### Forbidden Patterns

- `unwrap()` in production code
- Complex macros (AI cannot parse)
- Deep generics `<T: A + B + C>`
- Lifetimes in structs `<'a>` (unless necessary)
- `unsafe` blocks (without justification)
- `Rc`/`Arc`/`RefCell` (simplify data flow instead)

## Five-Layer Architecture

```
Layer 0: Data Acquisition (ADC → μV, LSL streams, file parsing)
    ↓
Layer 1: Core Data Structures (Raw, Info, Epochs, Evoked)
    ↓
Layer 2: Preprocessing (Filtering, Resampling, Re-referencing, ICA)
    ↓
Layer 3: Feature Extraction (PSD, Time-frequency, Connectivity)
    ↓
Layer 4: Application (BCI control, Sleep staging, Statistics)
```

## Planned Workspace Structure

When implementing BCIF, use this module structure:

```
bcif/
├── bcif-core/      # Core types, errors, feature flags
├── bcif-math/      # Basic statistics, window functions
├── bcif-dsp/       # FFT, STFT, filters, resampling
├── bcif-la/        # Matrix decomposition (EVD/SVD)
├── bcif-algo/      # PCA, ICA, CSP, CCA, LDA
├── bcif-pipeline/  # Offline/online processing pipelines
├── bcif-io/        # File format readers/writers
├── bcif-python/    # PyO3 bindings (optional)
└── bcif-cli/       # Command-line tools (optional)
```

## Key Reference Documents

- **Architecture**: `BCIF_Core_Pipeline.md`, `Preview_BCIF_Arch_TBD.md`
- **Dependency Mapping**: `BCIF_OVERVIEW_DOC/00.Table.md`
- **Rust Alternatives**: `BCIF_OVERVIEW_DOC/04_Rust替代方案详细分析.md`
- **Migration Priority**: `BCIF_OVERVIEW_DOC/05_代码移植优先级.md`
- **FFT Analysis**: `BCIF_OVERVIEW_DOC/08_MNE中FFT算法详细分析.md`
- **Coding Standards**: `Rust_Guideline/Rust_AI_Coding_Guideline_Std.md`

## Quality Checklist

When implementing algorithms:

- [ ] Bilingual documentation (English + Chinese)
- [ ] Explicit type annotations
- [ ] Complete error handling (no unwrap)
- [ ] Test cases with expected outputs
- [ ] Numerical validation against Python baseline (MNE/SciPy)
- [ ] No magic numbers (named constants)
- [ ] Memory safety (no unsafe without justification)
