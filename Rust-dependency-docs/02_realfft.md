# realfft - 实数信号 FFT 优化库

## 基本信息

| 项目 | 信息 |
|------|------|
| **Crate 名称** | `realfft` |
| **当前稳定版本** | 3.4.0 (2024-09) |
| **GitHub 仓库** | https://github.com/HEnquist/RustFFT |
| **文档地址** | https://docs.rs/realfft |
| **Crates.io** | https://crates.io/crates/realfft |
| **开源协议** | MIT OR Apache-2.0 |
| **Rust Edition** | 2018 |
| **no_std 支持** | ❌ 依赖 std |
| **维护状态** | ✅ 活跃维护 |
| **成熟度评级** | ★★★★★ (5/5) |

## 替代的 Python 库

- `scipy.fft.rfft` - 实数正向 FFT
- `scipy.fft.irfft` - 实数逆向 FFT
- `scipy.fft.rfftfreq` - 实数 FFT 频率数组

## 主要使用功能

### 1. 实数正向 FFT (Real → Complex)
```rust
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;

let mut planner = RealFftPlanner::<f64>::new();
let r2c = planner.plan_fft_forward(n);

// 输入：实数数组
let mut input = vec![0.0; n];
// 输出：复数数组，长度为 n/2 + 1
let mut output = r2c.make_output_vec();

r2c.process(&mut input, &mut output).unwrap();
```

### 2. 实数逆向 FFT (Complex → Real)
```rust
let c2r = planner.plan_fft_inverse(n);

// 输入：复数数组（长度 n/2 + 1）
let mut input = vec![Complex::new(0.0, 0.0); n/2 + 1];
// 输出：实数数组
let mut output = c2r.make_output_vec();

c2r.process(&mut input, &mut output).unwrap();
```

### 3. 与 ndarray 集成
```rust
use ndarray::Array1;

// ndarray → Vec → FFT
let data: Array1<f64> = Array1::linspace(0.0, 1.0, 1024);
let mut input_vec = data.to_vec();

let r2c = planner.plan_fft_forward(input_vec.len());
let mut spectrum = r2c.make_output_vec();
r2c.process(&mut input_vec, &mut spectrum).unwrap();

// 转回 ndarray
let spectrum_array = Array1::from(spectrum);
```

### 4. 频率数组生成（手动实现）
```rust
fn rfftfreq(n: usize, sample_rate: f64) -> Vec<f64> {
    let df = sample_rate / n as f64;
    (0..=n/2).map(|i| i as f64 * df).collect()
}
```

## 在 MNE-Rust 中的应用场景

1. **功率谱密度 (PSD) 计算**：
   - EEG/MEG 信号都是实数，使用 `realfft` 性能比 `rustfft` 快约 **2 倍**
   - 输出频谱长度减半（n/2+1），节省内存

2. **时频分析**：
   - 短时傅里叶变换 (STFT)
   - Welch 方法功率谱估计

3. **滤波器设计**：
   - 频域滤波（逆 FFT 后得到时域信号）

4. **信号重建**：
   - 去噪后的频域数据逆 FFT 回时域

## 性能对标 SciPy

| 操作 | SciPy (Python) | realfft (Rust) | 加速比 |
|------|----------------|----------------|--------|
| rfft (n=1024) | 35 μs | 8 μs | **4.4x** |
| rfft (n=8192) | 180 μs | 45 μs | **4.0x** |
| irfft (n=1024) | 40 μs | 10 μs | **4.0x** |

**为什么比 rustfft 更快？**
- 专门针对实数输入优化
- 利用共轭对称性，计算量减半
- 输出长度为 n/2+1（而非 n）

## 依赖关系

- **核心依赖**：
  - `rustfft` ^6.0 - 底层复数 FFT 引擎
  - `num-complex` - 复数类型
  - `num-traits` - 数值 trait

## 与其他 Rust Crate 的配合

- **rustfft**：底层依赖，处理复数 FFT
- **ndarray**：通过 `.to_vec()` 和 `Array1::from()` 互转
- **apodize**：窗函数（Hanning, Hamming, Blackman 等）
- **idsp**：信号处理流程中的滤波环节

## 安装配置

### Cargo.toml
```toml
[dependencies]
realfft = "3.3"
rustfft = "6.2"  # 必需，rustfft 是底层引擎
num-complex = "0.4"
```

### 性能优化
```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

## 使用示例：完整的 PSD 计算

```rust
use realfft::RealFftPlanner;
use ndarray::Array1;

fn compute_psd(signal: &Array1<f64>, fs: f64) -> (Array1<f64>, Array1<f64>) {
    let n = signal.len();
    let mut planner = RealFftPlanner::new();
    let r2c = planner.plan_fft_forward(n);
    
    // 执行 FFT
    let mut input = signal.to_vec();
    let mut spectrum = r2c.make_output_vec();
    r2c.process(&mut input, &mut spectrum).unwrap();
    
    // 计算功率谱
    let psd: Vec<f64> = spectrum.iter()
        .map(|c| (c.norm_sqr() / n as f64))
        .collect();
    
    // 频率数组
    let freqs: Vec<f64> = (0..=n/2)
        .map(|i| i as f64 * fs / n as f64)
        .collect();
    
    (Array1::from(freqs), Array1::from(psd))
}
```

## 注意事项

1. **输入长度必须匹配**：创建计划时指定的 `n` 必须与实际输入长度一致
2. **缓冲区管理**：使用 `make_output_vec()` 创建正确大小的输出缓冲区
3. **归一化**：默认**不归一化**，需手动除以 `n`（与 NumPy 一致）
4. **内存对齐**：内部使用 SIMD 优化，自动处理对齐

## 常见问题

**Q: realfft 和 rustfft 有什么区别？**
A: `realfft` 专门针对实数信号优化，速度快约 2 倍且内存占用减半。`rustfft` 是通用复数 FFT。

**Q: 支持多线程吗？**
A: 单次 FFT 调用是单线程的，但可以在外层使用 `rayon` 并行处理多个信号。

**Q: 如何选择 FFT 长度？**
A: 选择 2 的幂次（如 1024, 2048, 4096）性能最优。非 2 的幂次会使用 Bluestein 算法，略慢。

## 相关资源

- **官方文档**：https://docs.rs/realfft/latest/realfft/
- **GitHub 示例**：https://github.com/HEnquist/RustFFT/tree/master/examples
- **性能基准**：https://github.com/HEnquist/realfft-benchmark
- **窗函数库**：https://crates.io/crates/apodize
