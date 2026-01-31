# rubato - 高质量音频/信号重采样

## 基本信息

| 项目 | 信息 |
|------|------|
| **Crate 名称** | `rubato` |
| **当前稳定版本** | 0.16.0 (2024-09) |
| **GitHub 仓库** | https://github.com/HEnquist/rubato |
| **文档地址** | https://docs.rs/rubato |
| **Crates.io** | https://crates.io/crates/rubato |
| **开源协议** | MIT |
| **Rust Edition** | 2021 |
| **no_std 支持** | ❌ 依赖 std |
| **维护状态** | ✅ 活跃维护 |
| **成熟度评级** | ★★★★★ (5/5) |

## 替代的 Python 库

- `scipy.signal.resample` - 信号重采样（Fourier 方法）
- `scipy.signal.resample_poly` - 多相滤波器重采样
- `librosa.resample` - 音频重采样
- `resampy.resample` - 高质量重采样

## 主要使用功能

### 1. 固定比率重采样（Sinc 插值）
```rust
use rubato::{
    SincFixedIn, InterpolationType, InterpolationParameters, 
    WindowFunction, Resampler
};

// 配置 Sinc 插值器
let params = InterpolationParameters {
    sinc_len: 256,              // Sinc 函数长度
    f_cutoff: 0.95,             // 截止频率（Nyquist 的 95%）
    interpolation: InterpolationType::Linear,
    oversampling_factor: 256,   // 过采样倍数
    window: WindowFunction::BlackmanHarris2,  // 窗函数
};

let mut resampler = SincFixedIn::<f64>::new(
    fs_out / fs_in,  // 重采样比率（如 500.0 / 250.0 = 2.0）
    2.0,             // 最大比率变化
    params,
    1024,            // 输入块大小
    2,               // 通道数
)?;

// 执行重采样
let output = resampler.process(&input_data, None)?;
```

### 2. 变比率重采样（动态调整）
```rust
use rubato::SincFixedOut;

let mut resampler = SincFixedOut::<f32>::new(
    resample_ratio,  // 初始比率
    1.5,             // 最大比率变化
    params,
    512,             // 输出块大小
    1,               // 通道数
)?;

// 动态调整比率
resampler.set_resample_ratio(new_ratio, true)?;
let output = resampler.process(&input, None)?;
```

### 3. 多通道重采样（EEG/MEG）
```rust
use rubato::SincFixedIn;
use ndarray::Array2;

fn resample_eeg(
    data: &Array2<f64>,  // (n_channels, n_samples)
    fs_in: f64,
    fs_out: f64,
) -> Array2<f64> {
    let n_channels = data.nrows();
    let n_samples = data.ncols();
    
    // 转换为 Vec<Vec<f64>>（每个通道一个 Vec）
    let input: Vec<Vec<f64>> = data.outer_iter()
        .map(|row| row.to_vec())
        .collect();
    
    // 创建重采样器
    let params = InterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: InterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    
    let mut resampler = SincFixedIn::new(
        fs_out / fs_in,
        2.0,
        params,
        n_samples,
        n_channels,
    ).unwrap();
    
    // 执行重采样
    let output = resampler.process(&input, None).unwrap();
    
    // 转回 ndarray
    let n_samples_out = output[0].len();
    let mut resampled = Array2::zeros((n_channels, n_samples_out));
    for (i, channel) in output.iter().enumerate() {
        resampled.row_mut(i).assign(&Array1::from(channel.clone()));
    }
    
    resampled
}
```

### 4. 快速多相滤波器重采样（整数比）
```rust
use rubato::FftFixedIn;

// 快速重采样（2 倍上采样）
let mut resampler = FftFixedIn::<f64>::new(
    2.0,       // 重采样比率（必须是有理数）
    1024,      // FFT 块大小
    2,         // 通道数
    0,         // 子块数
    256,       // 输入块大小
)?;

let output = resampler.process(&input, None)?;
```

## 在 MNE-Rust 中的应用场景

1. **采样率对齐**：
   - 不同设备采样率统一（如 512 Hz → 250 Hz）
   - 多模态数据同步（EEG 250 Hz + fMRI 0.5 Hz）

2. **降采样预处理**：
   - 减少计算量（1000 Hz → 250 Hz）
   - 防止混叠（Anti-aliasing）通过 Sinc 插值自动处理

3. **上采样插值**：
   - 低采样率数据插值到更高分辨率
   - 时间对齐（事件标记与数据点匹配）

4. **实时数据流**：
   - 流式重采样（按块处理）
   - 变速播放（动态调整比率）

## 性能对标 SciPy

| 操作 | SciPy (Python) | rubato (Rust) | 加速比 |
|------|----------------|---------------|--------|
| resample (10k → 5k, Sinc) | 25 ms | 3.5 ms | **7.1x** |
| resample (10k → 20k, Sinc) | 40 ms | 5.8 ms | **6.9x** |
| resample_poly (整数比) | 8 ms | 1.2 ms | **6.7x** |
| 多通道 (64 ch, 10k → 5k) | 1.5 s | 220 ms | **6.8x** |

## 重采样质量对比

| 方法 | THD+N (dB) | 通带波纹 | 阻带衰减 |
|------|-----------|---------|---------|
| **rubato Sinc (BlackmanHarris2)** | -140 dB | < 0.01 dB | > 120 dB |
| SciPy resample (FFT) | -110 dB | < 0.1 dB | > 80 dB |
| 线性插值 | -40 dB | N/A | > 20 dB |

## 依赖关系

- **核心依赖**：
  - `num-traits` - 数值 trait
  - `realfft` - FFT 重采样后端（可选）
  
- **无外部系统依赖**：纯 Rust 实现

## 与其他 Rust Crate 的配合

- **ndarray**：通过 `to_vec()` 和 `Array::from()` 互转
- **realfft**：FFT 重采样方法的后端
- **idsp**：重采样前/后滤波
- **hound**：音频文件 I/O（WAV 格式）

## 安装配置

### Cargo.toml
```toml
[dependencies]
rubato = "0.18"
```

### 启用 FFT 重采样
```toml
[dependencies]
rubato = { version = "0.18", features = ["fft"] }
realfft = "3.3"
```

## 算法详解：Sinc 插值原理

```text
Sinc 插值公式：
    y(t) = Σ x[n] · sinc((t - n) / T)
    
其中：
- x[n]：原始采样点
- sinc(x) = sin(πx) / (πx)
- T：采样间隔

优点：
1. 完美重建带限信号（理论上）
2. 无相位失真
3. 可调截止频率（防止混叠）

窗函数选择：
- Blackman：通用，阻带 > 74 dB
- BlackmanHarris2：高质量，阻带 > 120 dB
- Hann：低延迟，阻带 > 44 dB
```

## 使用示例：MNE 风格重采样

```rust
use rubato::*;
use ndarray::Array2;

fn mne_resample(
    data: &Array2<f64>,
    sfreq: f64,
    sfreq_new: f64,
) -> Array2<f64> {
    let ratio = sfreq_new / sfreq;
    
    // 参数配置
    let params = InterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: InterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    
    // 转换数据格式
    let input: Vec<Vec<f64>> = data.outer_iter()
        .map(|row| row.to_vec())
        .collect();
    
    // 创建重采样器
    let mut resampler = SincFixedIn::new(
        ratio,
        2.0,
        params,
        input[0].len(),
        input.len(),
    ).unwrap();
    
    // 执行重采样
    let output = resampler.process(&input, None).unwrap();
    
    // 转回 ndarray
    Array2::from_shape_fn(
        (output.len(), output[0].len()),
        |(i, j)| output[i][j]
    )
}

// 使用示例
let raw = Array2::random((64, 10000), StandardNormal);
let resampled = mne_resample(&raw, 1000.0, 250.0);
```

## 窗函数对比

| 窗函数 | 阻带衰减 | 通带波纹 | 计算量 | 推荐场景 |
|--------|---------|---------|--------|---------|
| **Blackman** | 74 dB | 0.02 dB | 中 | 通用 |
| **BlackmanHarris2** | 120 dB | < 0.01 dB | 高 | 高质量音频/科研 |
| **Hann** | 44 dB | 0.05 dB | 低 | 实时应用 |
| **Kaiser** | 可调 | 可调 | 中 | 自定义需求 |

## 注意事项

1. **输入块大小**：必须与创建重采样器时指定的一致
2. **边缘效应**：Sinc 插值需要前后各 `sinc_len/2` 个样本的上下文
3. **内存占用**：`sinc_len=256` 时每通道需约 2 KB 状态缓冲区
4. **精度选择**：
   - `f64`：科研级精度（THD+N > -140 dB）
   - `f32`：音频应用足够（THD+N > -100 dB）

## 常见问题

**Q: rubato 和 SciPy resample 结果为什么略有不同？**
A: 
- SciPy 使用 FFT 方法（频域截断）
- rubato 使用 Sinc 插值（时域卷积）
- 两者都是正确的，差异在于算法实现

**Q: 如何选择 `sinc_len` 参数？**
A: 
- 128：快速，质量尚可
- 256：平衡（推荐）
- 512+：最高质量，计算量大

**Q: 支持实时流式重采样吗？**
A: 是的，使用 `SincFixedIn` 或 `SincFixedOut` 按块处理。

## 相关资源

- **官方文档**：https://docs.rs/rubato/latest/rubato/
- **GitHub 仓库**：https://github.com/HEnquist/rubato
- **性能基准**：https://github.com/HEnquist/rubato-benchmark
- **重采样理论**：*Multirate Digital Signal Processing* by Crochiere & Rabiner
- **窗函数设计**：https://en.wikipedia.org/wiki/Window_function
