# Rust AI-Friendly Coding Guideline (Embedded / no_std)
# Rust AI 友好编码规范（嵌入式 / no_std）

> **Edition**: Rust 2021  
> **Target**: ARM Cortex-M, RISC-V, Bare-metal, no_std  
> **Philosophy**: Deterministic, explicit, zero-allocation when possible.  
> **核心理念**: 确定性、显式、尽可能零分配。

---

## Table of Contents / 目录

1. [Core Principles / 核心原则](#1-core-principles--核心原则)
2. [Project Setup / 项目配置](#2-project-setup--项目配置)
3. [Type System / 类型系统](#3-type-system--类型系统)
4. [Memory Management / 内存管理](#4-memory-management--内存管理)
5. [Error Handling / 错误处理](#5-error-handling--错误处理)
6. [Comments & Documentation / 注释与文档](#6-comments--documentation--注释与文档)
7. [Hardware Abstraction / 硬件抽象](#7-hardware-abstraction--硬件抽象)
8. [Interrupts & Concurrency / 中断与并发](#8-interrupts--concurrency--中断与并发)
9. [Control Flow / 控制流](#9-control-flow--控制流)
10. [Forbidden Patterns / 禁止项](#10-forbidden-patterns--禁止项)
11. [Recommended Crates / 推荐依赖](#11-recommended-crates--推荐依赖)
12. [Code Examples / 代码示例](#12-code-examples--代码示例)

---

## 1. Core Principles / 核心原则

### English

1. **Stack over heap** - Prefer stack allocation. Avoid dynamic allocation entirely if possible.
2. **Static over dynamic** - Use fixed-size arrays and buffers.
3. **Explicit over implicit** - Always annotate types, especially integer widths.
4. **Deterministic over flexible** - Predictable timing is more important than flexibility.
5. **No panic** - Never allow panics in production code.
6. **AI-readable** - Keep code patterns simple and recognizable.

### 中文

1. **栈优于堆** - 优先栈分配。尽可能完全避免动态分配。
2. **静态优于动态** - 使用固定大小的数组和缓冲区。
3. **显式优于隐式** - 始终标注类型，特别是整数位宽。
4. **确定性优于灵活性** - 可预测的时序比灵活性更重要。
5. **禁止 panic** - 生产代码中绝不允许 panic。
6. **AI 可读** - 保持代码模式简单且可识别。

### The Golden Rule / 黄金法则

```
显式 > 隐式
Explicit > Implicit

栈 > 堆
Stack > Heap

静态 > 动态
Static > Dynamic

确定性 > 灵活性
Deterministic > Flexible
```

---

## 2. Project Setup / 项目配置

### Cargo.toml

```toml
[package]
name = "bcif-embedded"
version = "0.1.0"
edition = "2021"

[dependencies]
# Cortex-M runtime / Cortex-M 运行时
cortex-m = "0.7"
cortex-m-rt = "0.7"

# Hardware abstraction / 硬件抽象
embedded-hal = "1.0"

# Fixed-point math (optional) / 定点数学（可选）
fixed = "1.28"

# Fixed-capacity collections / 固定容量集合
heapless = "0.8"

# Critical section support / 临界区支持
critical-section = "1.1"

# Defmt logging (development) / Defmt 日志（开发）
defmt = "0.3"
defmt-rtt = "0.4"

# Panic handler / Panic 处理器
panic-halt = "1.0"

[profile.release]
opt-level = "s"        # Optimize for size / 优化大小
lto = true             # Link-time optimization / 链接时优化
codegen-units = 1      # Single codegen unit / 单一代码生成单元
debug = false          # No debug info / 无调试信息
panic = "abort"        # Abort on panic / panic 时中止
strip = true           # Strip symbols / 剥离符号
```

### .cargo/config.toml

```toml
[build]
# Target: Cortex-M4 with hardware FPU
# 目标: 带硬件 FPU 的 Cortex-M4
target = "thumbv7em-none-eabihf"

[target.thumbv7em-none-eabihf]
runner = "probe-run --chip STM32F407VGTx"
rustflags = [
    "-C", "link-arg=-Tlink.x",
]

[target.'cfg(all(target_arch = "arm", target_os = "none"))']
runner = "probe-run --chip STM32F407VGTx"
```

### main.rs Entry Point / main.rs 入口点

```rust
//! BCIF Embedded Entry Point
//! BCIF 嵌入式入口点

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;

/// Main entry point.
/// 主入口点。
#[entry]
fn main() -> ! {
    // Initialize hardware.
    // 初始化硬件。
    
    // Main loop - never returns.
    // 主循环 - 永不返回。
    loop {
        // Application logic here.
        // 应用逻辑在这里。
        cortex_m::asm::wfi(); // Wait for interrupt / 等待中断
    }
}
```

---

## 3. Type System / 类型系统

### Fixed-Width Integer Types / 固定宽度整数类型

```rust
// ✅ GOOD: Use explicit integer widths matching hardware
// ✅ 好: 使用与硬件匹配的显式整数宽度

// For 8-bit registers / 用于 8 位寄存器
let register_value: u8 = 0xFF;

// For 16-bit ADC / 用于 16 位 ADC
let adc_raw: u16 = 4095;

// For 32-bit timers / 用于 32 位定时器
let timer_count: u32 = 0xFFFF_FFFF;

// For signed sensor data / 用于有符号传感器数据
let temperature_raw: i16 = -100;

// For array indexing / 用于数组索引
let index: usize = 0;

// ❌ BAD: Platform-dependent types for hardware
// ❌ 坏: 硬件使用平台相关类型
let value: i32 = 100;  // Use u8/u16/u32 instead
                        // 改用 u8/u16/u32
```

### Type Annotation Rules / 类型标注规则

```rust
// ✅ GOOD: Always annotate types
// ✅ 好: 始终标注类型

let sample_rate_hz: u32 = 256;
let channel_count: u8 = 32;
let buffer: [u8; 64] = [0u8; 64];
let voltage_uv: i32 = 1000;  // microvolts / 微伏

// ✅ GOOD: Use type suffix for literals
// ✅ 好: 字面量使用类型后缀

let timeout_ms: u32 = 100u32;
let gain: f32 = 24.0f32;

// ❌ BAD: Rely on type inference
// ❌ 坏: 依赖类型推导

let value = 100;  // What type? / 什么类型？
let data = [0; 64];  // Element type unclear / 元素类型不清
```

### Struct Definition (no_std) / 结构体定义 (no_std)

```rust
/// ADC sample buffer with fixed capacity.
/// 固定容量的 ADC 采样缓冲区。
pub struct AdcBuffer {
    /// Sample data buffer.
    /// 采样数据缓冲区。
    data: [i16; 256],
    
    /// Number of valid samples.
    /// 有效采样点数。
    len: usize,
    
    /// Sampling rate in Hz.
    /// 采样率（赫兹）。
    sample_rate: u32,
}

impl AdcBuffer {
    /// Create a new empty buffer.
    /// 创建新的空缓冲区。
    pub const fn new(sample_rate: u32) -> Self {
        Self {
            data: [0i16; 256],
            len: 0,
            sample_rate,
        }
    }
    
    /// Add a sample to the buffer.
    /// 向缓冲区添加采样点。
    pub fn push(&mut self, sample: i16) -> bool {
        if self.len < self.data.len() {
            self.data[self.len] = sample;
            self.len += 1;
            true
        } else {
            false  // Buffer full / 缓冲区已满
        }
    }
    
    /// Get buffer contents as slice.
    /// 获取缓冲区内容切片。
    pub fn as_slice(&self) -> &[i16] {
        &self.data[..self.len]
    }
    
    /// Clear the buffer.
    /// 清空缓冲区。
    pub fn clear(&mut self) {
        self.len = 0;
    }
}
```

---

## 4. Memory Management / 内存管理

### Static Allocation / 静态分配

```rust
use core::cell::RefCell;
use critical_section::Mutex;

/// Global signal buffer.
/// 全局信号缓冲区。
static SIGNAL_BUFFER: Mutex<RefCell<[i16; 512]>> = 
    Mutex::new(RefCell::new([0i16; 512]));

/// Write data to global buffer.
/// 向全局缓冲区写入数据。
fn write_to_buffer(data: &[i16]) {
    critical_section::with(|cs| {
        let mut buffer = SIGNAL_BUFFER.borrow_ref_mut(cs);
        let len: usize = data.len().min(buffer.len());
        buffer[..len].copy_from_slice(&data[..len]);
    });
}

/// Read data from global buffer.
/// 从全局缓冲区读取数据。
fn read_from_buffer(output: &mut [i16]) {
    critical_section::with(|cs| {
        let buffer = SIGNAL_BUFFER.borrow_ref(cs);
        let len: usize = output.len().min(buffer.len());
        output[..len].copy_from_slice(&buffer[..len]);
    });
}
```

### Heapless Collections / 无堆集合

```rust
use heapless::Vec;
use heapless::String;

/// Fixed-capacity vector (max 64 elements).
/// 固定容量向量（最多 64 个元素）。
type SampleVec = Vec<i16, 64>;

/// Fixed-capacity string (max 32 bytes).
/// 固定容量字符串（最多 32 字节）。
type NameString = String<32>;

/// Channel configuration.
/// 通道配置。
pub struct ChannelConfig {
    /// Channel name.
    /// 通道名称。
    name: NameString,
    
    /// Recent samples.
    /// 最近的采样点。
    samples: SampleVec,
    
    /// Gain setting.
    /// 增益设置。
    gain: u8,
}

impl ChannelConfig {
    /// Create new channel configuration.
    /// 创建新的通道配置。
    pub fn new(name: &str, gain: u8) -> Self {
        let mut name_string: NameString = NameString::new();
        // Ignore error if name is too long.
        // 如果名称太长则忽略错误。
        let _ = name_string.push_str(name);
        
        Self {
            name: name_string,
            samples: SampleVec::new(),
            gain,
        }
    }
    
    /// Add sample (returns false if buffer full).
    /// 添加采样点（缓冲区满时返回 false）。
    pub fn add_sample(&mut self, sample: i16) -> bool {
        self.samples.push(sample).is_ok()
    }
    
    /// Get samples as slice.
    /// 获取采样点切片。
    pub fn samples(&self) -> &[i16] {
        &self.samples
    }
}
```

### Fixed-Size Arrays / 固定大小数组

```rust
/// Fixed buffer sizes.
/// 固定缓冲区大小。
const SAMPLE_BUFFER_SIZE: usize = 256;
const CHANNEL_COUNT: usize = 8;
const MAX_PACKET_SIZE: usize = 64;

/// Multi-channel sample buffer.
/// 多通道采样缓冲区。
pub struct MultiChannelBuffer {
    /// Sample data [channel][sample].
    /// 采样数据 [通道][采样点]。
    data: [[i16; SAMPLE_BUFFER_SIZE]; CHANNEL_COUNT],
    
    /// Write index for each channel.
    /// 每个通道的写入索引。
    write_index: [usize; CHANNEL_COUNT],
}

impl MultiChannelBuffer {
    /// Create empty buffer.
    /// 创建空缓冲区。
    pub const fn new() -> Self {
        Self {
            data: [[0i16; SAMPLE_BUFFER_SIZE]; CHANNEL_COUNT],
            write_index: [0usize; CHANNEL_COUNT],
        }
    }
    
    /// Add sample to specific channel.
    /// 向特定通道添加采样点。
    pub fn add_sample(&mut self, channel: usize, sample: i16) -> bool {
        if channel >= CHANNEL_COUNT {
            return false;
        }
        
        let idx: usize = self.write_index[channel];
        if idx >= SAMPLE_BUFFER_SIZE {
            return false;
        }
        
        self.data[channel][idx] = sample;
        self.write_index[channel] = idx + 1;
        true
    }
    
    /// Get channel data as slice.
    /// 获取通道数据切片。
    pub fn get_channel(&self, channel: usize) -> Option<&[i16]> {
        if channel >= CHANNEL_COUNT {
            return None;
        }
        
        let len: usize = self.write_index[channel];
        Some(&self.data[channel][..len])
    }
}
```

---

## 5. Error Handling / 错误处理

### Error Enum (no panic) / 错误枚举（无 panic）

```rust
/// Hardware error types.
/// 硬件错误类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareError {
    /// I2C communication timeout.
    /// I2C 通信超时。
    I2cTimeout,
    
    /// SPI communication error.
    /// SPI 通信错误。
    SpiError,
    
    /// ADC conversion timeout.
    /// ADC 转换超时。
    AdcTimeout,
    
    /// Buffer overflow.
    /// 缓冲区溢出。
    BufferOverflow,
    
    /// Invalid parameter.
    /// 无效参数。
    InvalidParameter,
    
    /// Device not initialized.
    /// 设备未初始化。
    NotInitialized,
}

/// Processing result type.
/// 处理结果类型。
pub type HwResult<T> = Result<T, HardwareError>;
```

### Error Handling Patterns / 错误处理模式

```rust
/// Read ADC value with error handling.
/// 带错误处理的 ADC 读取。
pub fn read_adc(channel: u8) -> HwResult<u16> {
    // Validate channel.
    // 验证通道。
    if channel > 7 {
        return Err(HardwareError::InvalidParameter);
    }
    
    // Attempt read with timeout.
    // 带超时尝试读取。
    let mut timeout_count: u32 = 0;
    const MAX_TIMEOUT: u32 = 10000;
    
    loop {
        // Check if conversion complete.
        // 检查转换是否完成。
        if is_adc_ready() {
            return Ok(get_adc_value());
        }
        
        timeout_count += 1;
        if timeout_count > MAX_TIMEOUT {
            return Err(HardwareError::AdcTimeout);
        }
    }
}

// Placeholder functions for example.
// 示例占位函数。
fn is_adc_ready() -> bool { true }
fn get_adc_value() -> u16 { 0 }

/// Process multiple channels.
/// 处理多个通道。
pub fn read_all_channels(output: &mut [u16; 8]) -> HwResult<()> {
    for (i, value) in output.iter_mut().enumerate() {
        // Use ? to propagate errors.
        // 使用 ? 传播错误。
        *value = read_adc(i as u8)?;
    }
    Ok(())
}
```

### Safe Array Access / 安全数组访问

```rust
/// Get array element safely.
/// 安全获取数组元素。
fn get_sample(buffer: &[i16], index: usize) -> Option<i16> {
    // ✅ GOOD: Use get() for safe access
    // ✅ 好: 使用 get() 进行安全访问
    buffer.get(index).copied()
}

/// Set array element safely.
/// 安全设置数组元素。
fn set_sample(buffer: &mut [i16], index: usize, value: i16) -> bool {
    // ✅ GOOD: Use get_mut() for safe access
    // ✅ 好: 使用 get_mut() 进行安全访问
    if let Some(slot) = buffer.get_mut(index) {
        *slot = value;
        true
    } else {
        false
    }
}

// ❌ BAD: Direct indexing can panic
// ❌ 坏: 直接索引会 panic
fn get_sample_bad(buffer: &[i16], index: usize) -> i16 {
    buffer[index]  // Panics if out of bounds!
                   // 越界时会 panic！
}
```

---

## 6. Comments & Documentation / 注释与文档

### Bilingual Documentation / 双语文档

```rust
/// Initialize the SPI peripheral for ADC communication.
/// 初始化用于 ADC 通信的 SPI 外设。
///
/// # Arguments / 参数
///
/// * `spi` - SPI peripheral instance.
///           SPI 外设实例。
/// * `cs_pin` - Chip select GPIO pin.
///              片选 GPIO 引脚。
///
/// # Returns / 返回
///
/// Configured ADC driver instance.
/// 配置好的 ADC 驱动实例。
///
/// # Errors / 错误
///
/// Returns `HardwareError::SpiError` if SPI initialization fails.
/// 如果 SPI 初始化失败则返回 `HardwareError::SpiError`。
///
/// # Safety / 安全性
///
/// Must be called only once during system initialization.
/// 只能在系统初始化期间调用一次。
pub fn init_adc<SPI, CS>(spi: SPI, cs_pin: CS) -> HwResult<AdcDriver<SPI, CS>>
where
    SPI: embedded_hal::spi::SpiDevice,
    CS: embedded_hal::digital::OutputPin,
{
    // Implementation here...
    // 实现代码...
    todo!()
}

pub struct AdcDriver<SPI, CS> {
    _spi: SPI,
    _cs: CS,
}
```

### Register Access Comments / 寄存器访问注释

```rust
/// Configure ADC control register.
/// 配置 ADC 控制寄存器。
fn configure_adc_register(value: u8) {
    // Register address: 0x01 (CTRL_REG1)
    // 寄存器地址: 0x01 (CTRL_REG1)
    //
    // Bit layout / 位布局:
    // [7:6] - Reserved / 保留
    // [5:4] - Gain setting (00=1x, 01=2x, 10=4x, 11=8x)
    //         增益设置 (00=1x, 01=2x, 10=4x, 11=8x)
    // [3:2] - Sample rate (00=125Hz, 01=250Hz, 10=500Hz, 11=1kHz)
    //         采样率 (00=125Hz, 01=250Hz, 10=500Hz, 11=1kHz)
    // [1]   - Enable continuous mode / 使能连续模式
    // [0]   - Start conversion / 开始转换
    
    const CTRL_REG1_ADDR: u8 = 0x01;
    write_register(CTRL_REG1_ADDR, value);
}

fn write_register(_addr: u8, _value: u8) {
    // Placeholder / 占位符
}
```

### State Machine Comments / 状态机注释

```rust
/// ADC state machine states.
/// ADC 状态机状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdcState {
    /// Uninitialized state.
    /// 未初始化状态。
    Idle,
    
    /// Waiting for conversion to complete.
    /// 等待转换完成。
    Converting,
    
    /// Data ready to read.
    /// 数据准备好可读取。
    DataReady,
    
    /// Error state.
    /// 错误状态。
    Error,
}
```

---

## 7. Hardware Abstraction / 硬件抽象

### GPIO Operations / GPIO 操作

```rust
use embedded_hal::digital::{OutputPin, InputPin, PinState};

/// LED controller.
/// LED 控制器。
pub struct LedController<P: OutputPin> {
    pin: P,
    is_on: bool,
}

impl<P: OutputPin> LedController<P> {
    /// Create new LED controller.
    /// 创建新的 LED 控制器。
    pub fn new(pin: P) -> Self {
        Self { pin, is_on: false }
    }
    
    /// Turn LED on.
    /// 打开 LED。
    pub fn on(&mut self) -> Result<(), P::Error> {
        self.pin.set_high()?;
        self.is_on = true;
        Ok(())
    }
    
    /// Turn LED off.
    /// 关闭 LED。
    pub fn off(&mut self) -> Result<(), P::Error> {
        self.pin.set_low()?;
        self.is_on = false;
        Ok(())
    }
    
    /// Toggle LED state.
    /// 切换 LED 状态。
    pub fn toggle(&mut self) -> Result<(), P::Error> {
        if self.is_on {
            self.off()
        } else {
            self.on()
        }
    }
}
```

### SPI Communication / SPI 通信

```rust
use embedded_hal::spi::SpiDevice;

/// Read register via SPI.
/// 通过 SPI 读取寄存器。
pub fn spi_read_register<SPI: SpiDevice>(
    spi: &mut SPI,
    register: u8,
) -> Result<u8, SPI::Error> {
    // Read command: register address with read bit set.
    // 读取命令: 寄存器地址设置读取位。
    let cmd: u8 = register | 0x80;
    
    let mut buffer: [u8; 2] = [cmd, 0x00];
    spi.transfer_in_place(&mut buffer)?;
    
    Ok(buffer[1])
}

/// Write register via SPI.
/// 通过 SPI 写入寄存器。
pub fn spi_write_register<SPI: SpiDevice>(
    spi: &mut SPI,
    register: u8,
    value: u8,
) -> Result<(), SPI::Error> {
    // Write command: register address with write bit clear.
    // 写入命令: 寄存器地址清除写入位。
    let cmd: u8 = register & 0x7F;
    
    let buffer: [u8; 2] = [cmd, value];
    spi.write(&buffer)?;
    
    Ok(())
}
```

### Timer/Delay Operations / 定时器/延迟操作

```rust
use embedded_hal::delay::DelayNs;

/// Wait for specified milliseconds.
/// 等待指定毫秒数。
pub fn delay_ms<D: DelayNs>(delay: &mut D, ms: u32) {
    delay.delay_ms(ms);
}

/// Simple timeout counter.
/// 简单超时计数器。
pub struct TimeoutCounter {
    /// Maximum count before timeout.
    /// 超时前的最大计数。
    max_count: u32,
    
    /// Current count.
    /// 当前计数。
    current: u32,
}

impl TimeoutCounter {
    /// Create new timeout counter.
    /// 创建新的超时计数器。
    pub const fn new(max_count: u32) -> Self {
        Self {
            max_count,
            current: 0,
        }
    }
    
    /// Increment counter, returns true if timeout.
    /// 增加计数器，超时返回 true。
    pub fn tick(&mut self) -> bool {
        self.current += 1;
        self.current >= self.max_count
    }
    
    /// Reset counter.
    /// 重置计数器。
    pub fn reset(&mut self) {
        self.current = 0;
    }
}
```

---

## 8. Interrupts & Concurrency / 中断与并发

### Critical Section Pattern / 临界区模式

```rust
use core::cell::RefCell;
use critical_section::Mutex;

/// Shared counter protected by critical section.
/// 受临界区保护的共享计数器。
static SAMPLE_COUNT: Mutex<RefCell<u32>> = Mutex::new(RefCell::new(0));

/// Increment sample count (interrupt-safe).
/// 增加采样计数（中断安全）。
pub fn increment_sample_count() {
    critical_section::with(|cs| {
        let mut count = SAMPLE_COUNT.borrow_ref_mut(cs);
        *count = count.wrapping_add(1);
    });
}

/// Get current sample count.
/// 获取当前采样计数。
pub fn get_sample_count() -> u32 {
    critical_section::with(|cs| {
        *SAMPLE_COUNT.borrow_ref(cs)
    })
}
```

### Ring Buffer for ISR / 中断服务程序的环形缓冲区

```rust
use core::sync::atomic::{AtomicUsize, Ordering};

/// Lock-free ring buffer for ISR communication.
/// 用于 ISR 通信的无锁环形缓冲区。
pub struct RingBuffer<const N: usize> {
    buffer: [i16; N],
    write_idx: AtomicUsize,
    read_idx: AtomicUsize,
}

impl<const N: usize> RingBuffer<N> {
    /// Create new ring buffer.
    /// 创建新的环形缓冲区。
    pub const fn new() -> Self {
        Self {
            buffer: [0i16; N],
            write_idx: AtomicUsize::new(0),
            read_idx: AtomicUsize::new(0),
        }
    }
    
    /// Push value (called from ISR).
    /// 推入值（从 ISR 调用）。
    ///
    /// # Safety / 安全性
    ///
    /// Only one producer (ISR) should call this.
    /// 只应有一个生产者（ISR）调用此函数。
    pub fn push(&self, value: i16) -> bool {
        let write: usize = self.write_idx.load(Ordering::Relaxed);
        let next_write: usize = (write + 1) % N;
        
        let read: usize = self.read_idx.load(Ordering::Acquire);
        if next_write == read {
            return false;  // Buffer full / 缓冲区满
        }
        
        // Safety: Only ISR writes to buffer[write].
        // 安全: 只有 ISR 写入 buffer[write]。
        unsafe {
            let ptr = self.buffer.as_ptr() as *mut i16;
            ptr.add(write).write_volatile(value);
        }
        
        self.write_idx.store(next_write, Ordering::Release);
        true
    }
    
    /// Pop value (called from main loop).
    /// 弹出值（从主循环调用）。
    pub fn pop(&self) -> Option<i16> {
        let read: usize = self.read_idx.load(Ordering::Relaxed);
        let write: usize = self.write_idx.load(Ordering::Acquire);
        
        if read == write {
            return None;  // Buffer empty / 缓冲区空
        }
        
        // Safety: Only main loop reads from buffer[read].
        // 安全: 只有主循环从 buffer[read] 读取。
        let value: i16 = unsafe {
            let ptr = self.buffer.as_ptr();
            ptr.add(read).read_volatile()
        };
        
        let next_read: usize = (read + 1) % N;
        self.read_idx.store(next_read, Ordering::Release);
        
        Some(value)
    }
}
```

---

## 9. Control Flow / 控制流

### State Machine Pattern / 状态机模式

```rust
/// Processing state machine.
/// 处理状态机。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessState {
    /// Waiting for trigger.
    /// 等待触发。
    Idle,
    
    /// Collecting samples.
    /// 采集采样点。
    Sampling,
    
    /// Processing collected data.
    /// 处理采集的数据。
    Processing,
    
    /// Transmitting results.
    /// 传输结果。
    Transmitting,
}

/// State machine controller.
/// 状态机控制器。
pub struct StateMachine {
    state: ProcessState,
    sample_count: u32,
}

impl StateMachine {
    /// Create new state machine.
    /// 创建新的状态机。
    pub const fn new() -> Self {
        Self {
            state: ProcessState::Idle,
            sample_count: 0,
        }
    }
    
    /// Process one tick of state machine.
    /// 处理状态机一个时钟周期。
    pub fn tick(&mut self, trigger: bool, sample_ready: bool) {
        // ✅ GOOD: Explicit state transitions
        // ✅ 好: 显式状态转换
        match self.state {
            ProcessState::Idle => {
                if trigger {
                    // Transition: Idle -> Sampling
                    // 转换: 空闲 -> 采样
                    self.state = ProcessState::Sampling;
                    self.sample_count = 0;
                }
            }
            
            ProcessState::Sampling => {
                if sample_ready {
                    self.sample_count += 1;
                }
                
                if self.sample_count >= 256 {
                    // Transition: Sampling -> Processing
                    // 转换: 采样 -> 处理
                    self.state = ProcessState::Processing;
                }
            }
            
            ProcessState::Processing => {
                // Process data...
                // 处理数据...
                
                // Transition: Processing -> Transmitting
                // 转换: 处理 -> 传输
                self.state = ProcessState::Transmitting;
            }
            
            ProcessState::Transmitting => {
                // Transmit data...
                // 传输数据...
                
                // Transition: Transmitting -> Idle
                // 转换: 传输 -> 空闲
                self.state = ProcessState::Idle;
            }
        }
    }
    
    /// Get current state.
    /// 获取当前状态。
    pub fn state(&self) -> ProcessState {
        self.state
    }
}
```

### Loop Patterns / 循环模式

```rust
/// Process buffer with explicit bounds.
/// 带显式边界处理缓冲区。
fn process_buffer(buffer: &mut [i16; 256]) {
    // ✅ GOOD: Explicit loop with clear bounds
    // ✅ 好: 带清晰边界的显式循环
    for i in 0..256 {
        buffer[i] = buffer[i].saturating_mul(2);
    }
}

/// Find maximum value with early exit.
/// 带提前退出的最大值查找。
fn find_max(buffer: &[i16]) -> Option<i16> {
    if buffer.is_empty() {
        return None;
    }
    
    let mut max: i16 = buffer[0];
    
    // ✅ GOOD: Simple iteration
    // ✅ 好: 简单迭代
    for &value in buffer.iter().skip(1) {
        if value > max {
            max = value;
        }
    }
    
    Some(max)
}
```

---

## 10. Forbidden Patterns / 禁止项

### ❌ NEVER USE / 禁止使用

| Pattern / 模式 | Reason / 原因 | Alternative / 替代方案 |
|---------------|--------------|----------------------|
| `unwrap()` | Panics / 会 panic | Return `Result` or `Option` |
| `expect()` | Panics / 会 panic | Return `Result` or `Option` |
| `panic!()` | System halt / 系统停止 | Return error / 返回错误 |
| `Vec<T>` (std) | Heap allocation / 堆分配 | `heapless::Vec` or arrays |
| `String` (std) | Heap allocation / 堆分配 | `heapless::String` or `&str` |
| `Box<T>` | Heap allocation / 堆分配 | Stack allocation / 栈分配 |
| `Rc`/`Arc` | Heap + complexity | Static + critical section |
| `async`/`await` | Complex for beginners | Sequential + state machine |
| `dyn Trait` | Dynamic dispatch + alloc | Generics / 泛型 |
| Complex macros | AI cannot parse | Functions / 函数 |
| Deep generics | Hard to read | Concrete types / 具体类型 |
| Inline assembly | Portability | HAL / PAC |

### Forbidden Code Examples / 禁止代码示例

```rust
// ❌ BAD: All of these are forbidden
// ❌ 坏: 以下全部禁止

fn bad_examples() {
    // let data = vec![1, 2, 3];      // ❌ Heap allocation
    // let name = String::from("test"); // ❌ Heap allocation
    // let value = some_option.unwrap(); // ❌ Can panic
    // panic!("Error!");               // ❌ System halt
    // let boxed = Box::new(42);       // ❌ Heap allocation
}

// ✅ GOOD: Correct alternatives
// ✅ 好: 正确的替代方案

fn good_examples() {
    let data: [i32; 3] = [1, 2, 3];  // ✅ Stack array
    let name: &str = "test";          // ✅ Static string
    
    let some_option: Option<i32> = Some(42);
    if let Some(value) = some_option {  // ✅ Safe unwrap
        let _ = value;
    }
    
    // Return error instead of panic.
    // 返回错误而非 panic。
}
```

---

## 11. Recommended Crates / 推荐依赖

### Core Dependencies / 核心依赖

```toml
[dependencies]
# Runtime / 运行时
cortex-m = "0.7"          # Cortex-M intrinsics / Cortex-M 内部函数
cortex-m-rt = "0.7"       # Runtime / 运行时

# HAL / 硬件抽象层
embedded-hal = "1.0"      # Hardware abstraction traits
                          # 硬件抽象特征

# Collections / 集合
heapless = "0.8"          # Fixed-capacity collections
                          # 固定容量集合

# Synchronization / 同步
critical-section = "1.1"  # Critical section abstraction
                          # 临界区抽象

# Math / 数学
fixed = "1.28"            # Fixed-point arithmetic
                          # 定点运算
libm = "0.2"              # Math functions for no_std
                          # no_std 数学函数

# Panic handler / Panic 处理
panic-halt = "1.0"        # Halt on panic
                          # panic 时停止

# Debug / 调试
defmt = "0.3"             # Efficient logging
                          # 高效日志
defmt-rtt = "0.4"         # RTT transport
                          # RTT 传输
```

### Optional Signal Processing / 可选信号处理

```toml
[dependencies]
# Note: Only use these if they support no_std!
# 注意: 只有支持 no_std 时才使用！

# cmsis-dsp = "0.1"       # ARM CMSIS-DSP bindings (if available)
                          # ARM CMSIS-DSP 绑定（如果可用）
```

---

## 12. Code Examples / 代码示例

### Complete Example: Simple ADC Driver
### 完整示例: 简单 ADC 驱动

```rust
//! Simple ADC Driver Example
//! 简单 ADC 驱动示例

#![no_std]
#![no_main]

use core::cell::RefCell;
use cortex_m_rt::entry;
use critical_section::Mutex;
use heapless::Vec;
use panic_halt as _;

// ============================================
// Constants / 常量
// ============================================

/// Maximum number of samples in buffer.
/// 缓冲区最大采样点数。
const MAX_SAMPLES: usize = 256;

/// Number of ADC channels.
/// ADC 通道数。
const NUM_CHANNELS: usize = 8;

/// ADC reference voltage in millivolts.
/// ADC 参考电压（毫伏）。
const VREF_MV: u32 = 3300;

/// ADC resolution (12-bit).
/// ADC 分辨率（12 位）。
const ADC_MAX: u32 = 4095;

// ============================================
// Error Types / 错误类型
// ============================================

/// ADC error types.
/// ADC 错误类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdcError {
    /// Conversion timeout.
    /// 转换超时。
    Timeout,
    
    /// Invalid channel number.
    /// 无效通道号。
    InvalidChannel,
    
    /// Buffer overflow.
    /// 缓冲区溢出。
    BufferFull,
}

/// Result type for ADC operations.
/// ADC 操作的结果类型。
pub type AdcResult<T> = Result<T, AdcError>;

// ============================================
// Data Structures / 数据结构
// ============================================

/// ADC sample buffer.
/// ADC 采样缓冲区。
pub struct SampleBuffer {
    /// Sample storage.
    /// 采样存储。
    data: Vec<u16, MAX_SAMPLES>,
    
    /// Channel number.
    /// 通道号。
    channel: u8,
}

impl SampleBuffer {
    /// Create new buffer for specified channel.
    /// 为指定通道创建新缓冲区。
    pub const fn new(channel: u8) -> Self {
        Self {
            data: Vec::new(),
            channel,
        }
    }
    
    /// Add sample to buffer.
    /// 向缓冲区添加采样点。
    pub fn push(&mut self, value: u16) -> AdcResult<()> {
        self.data.push(value).map_err(|_| AdcError::BufferFull)
    }
    
    /// Get samples as slice.
    /// 获取采样点切片。
    pub fn samples(&self) -> &[u16] {
        &self.data
    }
    
    /// Clear buffer.
    /// 清空缓冲区。
    pub fn clear(&mut self) {
        self.data.clear();
    }
    
    /// Get number of samples.
    /// 获取采样点数。
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if buffer is empty.
    /// 检查缓冲区是否为空。
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ============================================
// ADC Functions / ADC 函数
// ============================================

/// Convert raw ADC value to millivolts.
/// 将原始 ADC 值转换为毫伏。
pub fn adc_to_millivolts(raw: u16) -> u32 {
    // Formula: V = (raw * Vref) / ADC_MAX
    // 公式: V = (raw * Vref) / ADC_MAX
    (raw as u32 * VREF_MV) / ADC_MAX
}

/// Calculate mean of samples.
/// 计算采样均值。
pub fn calculate_mean(samples: &[u16]) -> Option<u16> {
    if samples.is_empty() {
        return None;
    }
    
    let sum: u32 = samples.iter().map(|&x| x as u32).sum();
    let mean: u32 = sum / samples.len() as u32;
    
    Some(mean as u16)
}

/// Find minimum and maximum values.
/// 查找最小和最大值。
pub fn find_min_max(samples: &[u16]) -> Option<(u16, u16)> {
    if samples.is_empty() {
        return None;
    }
    
    let mut min: u16 = samples[0];
    let mut max: u16 = samples[0];
    
    for &value in samples.iter().skip(1) {
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
    }
    
    Some((min, max))
}

// ============================================
// Global State / 全局状态
// ============================================

/// Global sample counter (interrupt-safe).
/// 全局采样计数器（中断安全）。
static SAMPLE_COUNTER: Mutex<RefCell<u32>> = Mutex::new(RefCell::new(0));

/// Increment and get sample counter.
/// 增加并获取采样计数器。
fn next_sample_id() -> u32 {
    critical_section::with(|cs| {
        let mut counter = SAMPLE_COUNTER.borrow_ref_mut(cs);
        let id: u32 = *counter;
        *counter = counter.wrapping_add(1);
        id
    })
}

// ============================================
// Main Entry / 主入口
// ============================================

/// Main entry point.
/// 主入口点。
#[entry]
fn main() -> ! {
    // Initialize buffer for channel 0.
    // 为通道 0 初始化缓冲区。
    let mut buffer: SampleBuffer = SampleBuffer::new(0);
    
    // Simulated ADC readings.
    // 模拟 ADC 读数。
    let test_values: [u16; 8] = [100, 200, 300, 400, 500, 600, 700, 800];
    
    // Fill buffer with test data.
    // 用测试数据填充缓冲区。
    for &value in test_values.iter() {
        if buffer.push(value).is_err() {
            // Handle buffer full error.
            // 处理缓冲区满错误。
            break;
        }
    }
    
    // Calculate statistics.
    // 计算统计数据。
    if let Some(mean) = calculate_mean(buffer.samples()) {
        let _mean_mv: u32 = adc_to_millivolts(mean);
        // Use mean_mv...
        // 使用 mean_mv...
    }
    
    if let Some((min, max)) = find_min_max(buffer.samples()) {
        let _min_mv: u32 = adc_to_millivolts(min);
        let _max_mv: u32 = adc_to_millivolts(max);
        // Use min_mv, max_mv...
        // 使用 min_mv, max_mv...
    }
    
    // Main loop.
    // 主循环。
    loop {
        // Get sample ID.
        // 获取采样 ID。
        let _sample_id: u32 = next_sample_id();
        
        // Wait for interrupt.
        // 等待中断。
        cortex_m::asm::wfi();
    }
}
```

---

## Quick Reference Card / 快速参考卡

```
┌─────────────────────────────────────────────────────────────────┐
│              EMBEDDED RUST AI CODING CHECKLIST                  │
│              嵌入式 RUST AI 编码检查清单                          │
├─────────────────────────────────────────────────────────────────┤
│ ✅ #![no_std] declared      ✅ 已声明 #![no_std]                 │
│ ✅ #![no_main] declared     ✅ 已声明 #![no_main]                │
│ ✅ Types explicitly sized   ✅ 类型显式指定大小                   │
│ ✅ No heap allocation       ✅ 无堆分配                          │
│ ✅ No unwrap/expect         ✅ 无 unwrap/expect                  │
│ ✅ No panic possible        ✅ 不可能 panic                      │
│ ✅ Fixed-size buffers       ✅ 固定大小缓冲区                     │
│ ✅ Bilingual comments       ✅ 双语注释                          │
│ ✅ Critical sections used   ✅ 使用临界区                        │
│ ✅ Error types defined      ✅ 错误类型已定义                     │
│ ✅ State machine explicit   ✅ 状态机显式                        │
│ ✅ Release profile tuned    ✅ 发布配置已调优                     │
└─────────────────────────────────────────────────────────────────┘

Memory Rule / 内存规则:
┌─────────────────────────────────────────────────────────────────┐
│   栈 > 静态 > 堆（禁止）                                          │
│   Stack > Static > Heap (Forbidden)                             │
└─────────────────────────────────────────────────────────────────┘
```

---

*Document Version: 1.0.0 / 文档版本: 1.0.0*  
*Last Updated: 2026-02-01 / 最后更新: 2026-02-01*  
*For BCIF Embedded Project / 用于 BCIF 嵌入式项目*
