# C++ AI-Friendly Coding Guideline (Embedded / Bare-metal)
# C++ AI 友好编码规范（嵌入式 / 裸机）

> **Standard**: C++17  
> **Target**: ARM Cortex-M, RISC-V, Bare-metal, Resource-constrained  
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
11. [Recommended Libraries / 推荐库](#11-recommended-libraries--推荐库)
12. [Code Examples / 代码示例](#12-code-examples--代码示例)

---

## 1. Core Principles / 核心原则

### English

1. **Stack over heap** - Prefer stack allocation. Avoid dynamic allocation entirely if possible.
2. **Static over dynamic** - Use fixed-size arrays and compile-time constants.
3. **Explicit over implicit** - Always annotate types, especially integer widths.
4. **No exceptions, no RTTI** - Compile with `-fno-exceptions -fno-rtti`.
5. **Deterministic timing** - Predictable execution time is more important than average performance.
6. **AI-readable** - Keep code patterns simple and recognizable.
7. **Volatile for hardware** - Always use `volatile` for hardware registers.

### 中文

1. **栈优于堆** - 优先栈分配。尽可能完全避免动态分配。
2. **静态优于动态** - 使用固定大小的数组和编译时常量。
3. **显式优于隐式** - 始终标注类型，特别是整数位宽。
4. **禁用异常和 RTTI** - 使用 `-fno-exceptions -fno-rtti` 编译。
5. **确定性时序** - 可预测的执行时间比平均性能更重要。
6. **AI 可读** - 保持代码模式简单且可识别。
7. **硬件寄存器使用 volatile** - 硬件寄存器始终使用 `volatile`。

### The Golden Rule / 黄金法则

```
显式 > 隐式
Explicit > Implicit

栈 > 静态 > 堆（禁止）
Stack > Static > Heap (Forbidden)

确定性 > 灵活性
Deterministic > Flexible

编译时 > 运行时
Compile-time > Runtime
```

---

## 2. Project Setup / 项目配置

### Why Zig Build? / 为什么使用 Zig 构建？

```
Advantages / 优势:
1. Built-in cross-compilation (no arm-none-eabi-gcc needed)
   内置交叉编译（无需 arm-none-eabi-gcc）
2. Reproducible builds across platforms
   跨平台可重现构建
3. Can compile C/C++/Rust in one unified system
   可在一个统一系统中编译 C/C++/Rust
4. No CMake, no Makefile - just `zig build`
   无需 CMake、无需 Makefile - 只需 `zig build`
5. Package management via build.zig.zon
   通过 build.zig.zon 进行包管理
```

### build.zig for Embedded / 嵌入式 build.zig

```zig
// build.zig - Embedded C++17 firmware build
// build.zig - 嵌入式 C++17 固件构建

const std = @import("std");

pub fn build(b: *std.Build) void {
    // Target: ARM Cortex-M4 with hardware FPU
    // 目标: 带硬件 FPU 的 ARM Cortex-M4
    const target = b.resolveTargetQuery(.{
        .cpu_arch = .thumb,
        .cpu_model = .{ .explicit = &std.Target.arm.cpu.cortex_m4 },
        .os_tag = .freestanding,
        .abi = .eabihf,  // Hard-float ABI / 硬浮点 ABI
    });

    // Optimize for size (embedded)
    // 针对大小优化（嵌入式）
    const optimize = b.standardOptimizeOption(.{});

    // Create executable / 创建可执行文件
    const exe = b.addExecutable(.{
        .name = "firmware",
        .target = target,
        .optimize = optimize,
        .link_libc = false,  // Freestanding / 独立环境
    });

    // Add C++ source files / 添加 C++ 源文件
    exe.addCSourceFiles(.{
        .files = &.{
            "src/main.cpp",
            "src/startup.cpp",
            "src/drivers/gpio.cpp",
            "src/drivers/spi.cpp",
        },
        .flags = &.{
            "-std=c++17",
            "-fno-exceptions",
            "-fno-rtti",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
        },
    });

    // Linker script / 链接脚本
    exe.setLinkerScript(.{ .cwd_relative = "linker.ld" });

    // Strip dead code / 移除死代码
    exe.link_gc_sections = true;

    // Install artifact / 安装产物
    b.installArtifact(exe);

    // Generate .bin file / 生成 .bin 文件
    const bin = exe.addObjCopy(.{
        .format = .bin,
    });
    const install_bin = b.addInstallBinFile(bin.getOutput(), "firmware.bin");
    b.getInstallStep().dependOn(&install_bin.step);

    // Generate .hex file / 生成 .hex 文件
    const hex = exe.addObjCopy(.{
        .format = .hex,
    });
    const install_hex = b.addInstallBinFile(hex.getOutput(), "firmware.hex");
    b.getInstallStep().dependOn(&install_hex.step);
}
```

### build.zig.zon (Package Manifest) / 包清单

```zig
// build.zig.zon - Package dependencies
// build.zig.zon - 包依赖

.{
    .name = "bcif_embedded",
    .version = "0.1.0",
    .dependencies = .{
        // ETL (Embedded Template Library) - if available
        // ETL（嵌入式模板库）- 如果可用
        // .etl = .{
        //     .url = "https://example.com/etl.tar.gz",
        //     .hash = "...",
        // },
    },
    .paths = .{
        "build.zig",
        "build.zig.zon",
        "src",
        "linker.ld",
    },
}
```

### Build Commands / 构建命令

```bash
# Build firmware / 构建固件
zig build

# Build with release optimization / 发布优化构建
zig build -Doptimize=ReleaseSmall

# Clean build / 清理构建
zig build --clean

# Output location / 输出位置
# zig-out/bin/firmware
# zig-out/bin/firmware.bin
# zig-out/bin/firmware.hex
```

---

## 3. Type System / 类型系统

### Fixed-Width Integer Types / 固定宽度整数类型

```cpp
#include <cstdint>
#include <cstddef>

// ✅ GOOD: Use fixed-width types
// ✅ 好: 使用固定宽度类型

// For 8-bit registers / 用于 8 位寄存器
uint8_t register_value = 0xFF;

// For 16-bit ADC / 用于 16 位 ADC
uint16_t adc_raw = 4095;

// For 32-bit timers / 用于 32 位定时器
uint32_t timer_count = 0xFFFF'FFFF;

// For signed sensor data / 用于有符号传感器数据
int16_t temperature_raw = -100;

// For array indexing / 用于数组索引
std::size_t index = 0;

// ❌ BAD: Platform-dependent types
// ❌ 坏: 平台相关类型
int value = 100;       // Size varies! / 大小不定！
long big_value = 0;    // Size varies! / 大小不定！
```

### Always Annotate Types / 始终标注类型

```cpp
// ✅ GOOD: Explicit type annotations
// ✅ 好: 显式类型标注
uint32_t sample_rate_hz = 256;
uint8_t channel_count = 32;
int16_t buffer[64] = {};
float voltage_v = 3.3f;

// ✅ GOOD: Use type suffix for literals
// ✅ 好: 字面量使用类型后缀
uint32_t timeout_ms = 100u;
float gain = 24.0f;

// ❌ BAD: Rely on type inference in embedded
// ❌ 坏: 嵌入式中依赖类型推导
auto value = 100;     // What type? / 什么类型？
auto data = get_data();  // Type unclear / 类型不清
```

### Struct Definition / 结构体定义

```cpp
/// ADC sample buffer with fixed capacity.
/// 固定容量的 ADC 采样缓冲区。
struct AdcBuffer {
    /// Maximum buffer size.
    /// 最大缓冲区大小。
    static constexpr std::size_t MAX_SIZE = 256;
    
    /// Sample data buffer.
    /// 采样数据缓冲区。
    int16_t data[MAX_SIZE] = {};
    
    /// Number of valid samples.
    /// 有效采样点数。
    std::size_t len = 0;
    
    /// Sampling rate in Hz.
    /// 采样率（赫兹）。
    uint32_t sample_rate = 0;
    
    /// Add a sample to the buffer.
    /// 向缓冲区添加采样点。
    [[nodiscard]] bool push(int16_t sample) {
        if (len >= MAX_SIZE) {
            return false;  // Buffer full / 缓冲区满
        }
        data[len] = sample;
        ++len;
        return true;
    }
    
    /// Clear the buffer.
    /// 清空缓冲区。
    void clear() {
        len = 0;
    }
    
    /// Check if buffer is full.
    /// 检查缓冲区是否已满。
    [[nodiscard]] bool is_full() const {
        return len >= MAX_SIZE;
    }
};
```

### Enum Class (Strong Enum) / 枚举类（强枚举）

```cpp
// ✅ GOOD: Use enum class for type safety
// ✅ 好: 使用 enum class 以保证类型安全
enum class GpioMode : uint8_t {
    Input = 0,
    Output = 1,
    AlternateFunction = 2,
    Analog = 3
};

enum class GpioPull : uint8_t {
    None = 0,
    PullUp = 1,
    PullDown = 2
};

// Usage / 用法
void configure_pin(uint8_t pin, GpioMode mode, GpioPull pull);

// ❌ BAD: Old-style enum
// ❌ 坏: 旧式枚举
enum Mode { INPUT, OUTPUT };  // Pollutes namespace / 污染命名空间
```

---

## 4. Memory Management / 内存管理

### Static Allocation / 静态分配

```cpp
/// Global signal buffer (statically allocated).
/// 全局信号缓冲区（静态分配）。
static int16_t g_signal_buffer[1024];
static std::size_t g_signal_len = 0;

/// Get buffer pointer (read-only).
/// 获取缓冲区指针（只读）。
const int16_t* get_signal_buffer() {
    return g_signal_buffer;
}

/// Get buffer length.
/// 获取缓冲区长度。
std::size_t get_signal_length() {
    return g_signal_len;
}

/// Add sample to buffer.
/// 向缓冲区添加采样点。
bool add_sample(int16_t sample) {
    if (g_signal_len >= 1024) {
        return false;
    }
    g_signal_buffer[g_signal_len] = sample;
    ++g_signal_len;
    return true;
}
```

### Fixed-Size Containers (std::array) / 固定大小容器 (std::array)

```cpp
#include <array>

/// Multi-channel sample buffer.
/// 多通道采样缓冲区。
struct MultiChannelBuffer {
    static constexpr std::size_t NUM_CHANNELS = 8;
    static constexpr std::size_t SAMPLES_PER_CHANNEL = 256;
    
    /// Sample data [channel][sample].
    /// 采样数据 [通道][采样点]。
    std::array<std::array<int16_t, SAMPLES_PER_CHANNEL>, NUM_CHANNELS> data = {};
    
    /// Write index for each channel.
    /// 每个通道的写入索引。
    std::array<std::size_t, NUM_CHANNELS> write_index = {};
    
    /// Add sample to specific channel.
    /// 向特定通道添加采样点。
    [[nodiscard]] bool add_sample(std::size_t channel, int16_t sample) {
        if (channel >= NUM_CHANNELS) {
            return false;
        }
        
        std::size_t idx = write_index[channel];
        if (idx >= SAMPLES_PER_CHANNEL) {
            return false;
        }
        
        data[channel][idx] = sample;
        write_index[channel] = idx + 1;
        return true;
    }
    
    /// Get channel data.
    /// 获取通道数据。
    [[nodiscard]] const int16_t* get_channel(std::size_t channel) const {
        if (channel >= NUM_CHANNELS) {
            return nullptr;
        }
        return data[channel].data();
    }
    
    /// Get number of samples in channel.
    /// 获取通道中的采样点数。
    [[nodiscard]] std::size_t get_channel_length(std::size_t channel) const {
        if (channel >= NUM_CHANNELS) {
            return 0;
        }
        return write_index[channel];
    }
};
```

### Placement New (Advanced) / 放置 new（高级）

```cpp
#include <new>

/// Memory pool for fixed-size objects.
/// 固定大小对象的内存池。
template<typename T, std::size_t N>
class MemoryPool {
public:
    /// Allocate one object from pool.
    /// 从池中分配一个对象。
    [[nodiscard]] T* allocate() {
        for (std::size_t i = 0; i < N; ++i) {
            if (!m_used[i]) {
                m_used[i] = true;
                // Construct in place / 就地构造
                return new (&m_storage[i]) T();
            }
        }
        return nullptr;  // Pool exhausted / 池耗尽
    }
    
    /// Deallocate object back to pool.
    /// 将对象归还给池。
    void deallocate(T* ptr) {
        for (std::size_t i = 0; i < N; ++i) {
            if (&m_storage[i] == ptr) {
                ptr->~T();  // Call destructor / 调用析构函数
                m_used[i] = false;
                return;
            }
        }
    }

private:
    std::array<T, N> m_storage = {};
    std::array<bool, N> m_used = {};
};
```

---

## 5. Error Handling / 错误处理

### Error Enum (No Exceptions) / 错误枚举（无异常）

```cpp
/// Hardware error types.
/// 硬件错误类型。
enum class HwError : uint8_t {
    /// Success (no error).
    /// 成功（无错误）。
    Ok = 0,
    
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
    NotInitialized
};

/// Check if error code indicates success.
/// 检查错误码是否表示成功。
[[nodiscard]] inline bool is_ok(HwError err) {
    return err == HwError::Ok;
}
```

### Result Type Pattern / 结果类型模式

```cpp
/// Result type combining value and error.
/// 组合值和错误的结果类型。
template<typename T>
struct Result {
    T value = {};
    HwError error = HwError::Ok;
    
    /// Check if operation succeeded.
    /// 检查操作是否成功。
    [[nodiscard]] bool ok() const {
        return error == HwError::Ok;
    }
    
    /// Create success result.
    /// 创建成功结果。
    static Result success(T val) {
        return Result{val, HwError::Ok};
    }
    
    /// Create error result.
    /// 创建错误结果。
    static Result fail(HwError err) {
        return Result{{}, err};
    }
};

/// Read ADC value with error handling.
/// 带错误处理的 ADC 读取。
Result<uint16_t> read_adc(uint8_t channel) {
    // Validate channel.
    // 验证通道。
    if (channel > 7) {
        return Result<uint16_t>::fail(HwError::InvalidParameter);
    }
    
    // Attempt read with timeout.
    // 带超时尝试读取。
    uint32_t timeout_count = 0;
    constexpr uint32_t MAX_TIMEOUT = 10000;
    
    while (!is_adc_ready()) {
        ++timeout_count;
        if (timeout_count > MAX_TIMEOUT) {
            return Result<uint16_t>::fail(HwError::AdcTimeout);
        }
    }
    
    uint16_t value = get_adc_value();
    return Result<uint16_t>::success(value);
}
```

### Safe Array Access / 安全数组访问

```cpp
/// Get array element safely (returns default if out of bounds).
/// 安全获取数组元素（越界时返回默认值）。
template<typename T, std::size_t N>
T safe_get(const std::array<T, N>& arr, std::size_t index, T default_val = {}) {
    if (index >= N) {
        return default_val;
    }
    return arr[index];
}

/// Set array element safely (returns false if out of bounds).
/// 安全设置数组元素（越界时返回 false）。
template<typename T, std::size_t N>
bool safe_set(std::array<T, N>& arr, std::size_t index, T value) {
    if (index >= N) {
        return false;
    }
    arr[index] = value;
    return true;
}

// Usage / 用法
void example() {
    std::array<int16_t, 10> data = {};
    
    // ✅ GOOD: Safe access
    // ✅ 好: 安全访问
    int16_t val = safe_get(data, 5, 0);
    bool ok = safe_set(data, 5, 100);
    
    // ❌ BAD: Direct access can crash
    // ❌ 坏: 直接访问可能崩溃
    // int16_t val = data[100];  // Undefined behavior!
}
```

---

## 6. Comments & Documentation / 注释与文档

### Bilingual Documentation / 双语文档

```cpp
/// Initialize the SPI peripheral for ADC communication.
/// 初始化用于 ADC 通信的 SPI 外设。
///
/// @param spi_instance SPI peripheral instance (1, 2, or 3).
///                     SPI 外设实例（1、2 或 3）。
/// @param clock_div Clock divider for SPI speed.
///                  SPI 速度的时钟分频器。
///
/// @return HwError::Ok on success, error code otherwise.
///         成功返回 HwError::Ok，否则返回错误码。
///
/// @note Must be called before any ADC operations.
///       必须在任何 ADC 操作之前调用。
///
/// @warning This function is NOT interrupt-safe.
///          此函数不是中断安全的。
HwError init_spi(uint8_t spi_instance, uint8_t clock_div);
```

### Register Documentation / 寄存器文档

```cpp
/// GPIO control register structure.
/// GPIO 控制寄存器结构。
///
/// Memory layout matches STM32F4 GPIO peripheral.
/// 内存布局匹配 STM32F4 GPIO 外设。
struct GpioRegs {
    /// Mode register.
    /// 模式寄存器。
    /// Bits [1:0] for pin 0, [3:2] for pin 1, etc.
    /// 位 [1:0] 用于引脚 0，位 [3:2] 用于引脚 1，依此类推。
    volatile uint32_t MODER;
    
    /// Output type register.
    /// 输出类型寄存器。
    /// 0 = Push-pull, 1 = Open-drain.
    /// 0 = 推挽，1 = 开漏。
    volatile uint32_t OTYPER;
    
    /// Output speed register.
    /// 输出速度寄存器。
    volatile uint32_t OSPEEDR;
    
    /// Pull-up/pull-down register.
    /// 上拉/下拉寄存器。
    volatile uint32_t PUPDR;
    
    /// Input data register (read-only).
    /// 输入数据寄存器（只读）。
    volatile uint32_t IDR;
    
    /// Output data register.
    /// 输出数据寄存器。
    volatile uint32_t ODR;
    
    /// Bit set/reset register (write-only).
    /// 位设置/复位寄存器（只写）。
    volatile uint32_t BSRR;
};

// Base addresses / 基地址
constexpr uintptr_t GPIOA_BASE = 0x4002'0000;
constexpr uintptr_t GPIOB_BASE = 0x4002'0400;

// Peripheral pointers / 外设指针
inline auto* const GPIOA = reinterpret_cast<GpioRegs*>(GPIOA_BASE);
inline auto* const GPIOB = reinterpret_cast<GpioRegs*>(GPIOB_BASE);
```

### State Machine Documentation / 状态机文档

```cpp
/// ADC state machine states.
/// ADC 状态机状态。
///
/// State transitions / 状态转换:
/// ```
/// [Idle] --start()--> [Configuring]
/// [Configuring] --config_done()--> [Sampling]
/// [Sampling] --conversion_done()--> [DataReady]
/// [DataReady] --read_data()--> [Idle]
/// [Any] --error()--> [Error]
/// [Error] --reset()--> [Idle]
/// ```
enum class AdcState : uint8_t {
    /// Idle, waiting for start command.
    /// 空闲，等待启动命令。
    Idle,
    
    /// Configuring ADC parameters.
    /// 配置 ADC 参数。
    Configuring,
    
    /// Sampling in progress.
    /// 采样进行中。
    Sampling,
    
    /// Data ready to read.
    /// 数据准备好可读取。
    DataReady,
    
    /// Error state, needs reset.
    /// 错误状态，需要复位。
    Error
};
```

---

## 7. Hardware Abstraction / 硬件抽象

### GPIO Operations / GPIO 操作

```cpp
/// GPIO pin abstraction.
/// GPIO 引脚抽象。
class GpioPin {
public:
    /// Create GPIO pin reference.
    /// 创建 GPIO 引脚引用。
    constexpr GpioPin(GpioRegs* port, uint8_t pin)
        : m_port(port), m_pin(pin) {}
    
    /// Set pin high.
    /// 设置引脚高电平。
    void set_high() const {
        m_port->BSRR = (1u << m_pin);
    }
    
    /// Set pin low.
    /// 设置引脚低电平。
    void set_low() const {
        m_port->BSRR = (1u << (m_pin + 16));
    }
    
    /// Toggle pin state.
    /// 切换引脚状态。
    void toggle() const {
        m_port->ODR ^= (1u << m_pin);
    }
    
    /// Read pin state.
    /// 读取引脚状态。
    [[nodiscard]] bool read() const {
        return (m_port->IDR & (1u << m_pin)) != 0;
    }
    
    /// Configure pin mode.
    /// 配置引脚模式。
    void set_mode(GpioMode mode) const {
        uint32_t shift = m_pin * 2;
        uint32_t mask = 0x3u << shift;
        m_port->MODER = (m_port->MODER & ~mask) | 
                        (static_cast<uint32_t>(mode) << shift);
    }

private:
    GpioRegs* m_port;
    uint8_t m_pin;
};

// Usage / 用法
constexpr GpioPin led_pin(GPIOA, 5);

void blink_led() {
    led_pin.set_mode(GpioMode::Output);
    led_pin.toggle();
}
```

### SPI Communication / SPI 通信

```cpp
/// SPI transaction result.
/// SPI 事务结果。
struct SpiResult {
    HwError error = HwError::Ok;
    uint8_t rx_byte = 0;
};

/// Read register via SPI.
/// 通过 SPI 读取寄存器。
///
/// @param cs_pin Chip select pin.
///               片选引脚。
/// @param reg Register address.
///            寄存器地址。
/// @return SPI result with data or error.
///         带数据或错误的 SPI 结果。
SpiResult spi_read_register(const GpioPin& cs_pin, uint8_t reg) {
    SpiResult result;
    
    // Assert chip select (active low).
    // 断言片选（低电平有效）。
    cs_pin.set_low();
    
    // Send read command (register | 0x80).
    // 发送读取命令（寄存器 | 0x80）。
    uint8_t cmd = reg | 0x80;
    result.error = spi_transfer(cmd, nullptr);
    if (result.error != HwError::Ok) {
        cs_pin.set_high();
        return result;
    }
    
    // Read response.
    // 读取响应。
    result.error = spi_transfer(0x00, &result.rx_byte);
    
    // Deassert chip select.
    // 取消片选。
    cs_pin.set_high();
    
    return result;
}

/// Write register via SPI.
/// 通过 SPI 写入寄存器。
HwError spi_write_register(const GpioPin& cs_pin, uint8_t reg, uint8_t value) {
    cs_pin.set_low();
    
    // Send write command (register & 0x7F).
    // 发送写入命令（寄存器 & 0x7F）。
    HwError err = spi_transfer(reg & 0x7F, nullptr);
    if (err != HwError::Ok) {
        cs_pin.set_high();
        return err;
    }
    
    // Send value.
    // 发送值。
    err = spi_transfer(value, nullptr);
    
    cs_pin.set_high();
    return err;
}
```

### Timer/Delay Operations / 定时器/延迟操作

```cpp
/// Get system tick count (milliseconds since boot).
/// 获取系统滴答计数（启动后的毫秒数）。
[[nodiscard]] uint32_t get_tick_ms();

/// Simple blocking delay.
/// 简单阻塞延迟。
///
/// @param ms Delay time in milliseconds.
///           延迟时间（毫秒）。
void delay_ms(uint32_t ms) {
    uint32_t start = get_tick_ms();
    while ((get_tick_ms() - start) < ms) {
        // Busy wait / 忙等待
    }
}

/// Timeout helper class.
/// 超时辅助类。
class Timeout {
public:
    /// Create timeout with duration.
    /// 创建带持续时间的超时。
    explicit Timeout(uint32_t timeout_ms)
        : m_start(get_tick_ms())
        , m_timeout(timeout_ms)
    {}
    
    /// Check if timeout has expired.
    /// 检查超时是否已过期。
    [[nodiscard]] bool expired() const {
        return (get_tick_ms() - m_start) >= m_timeout;
    }
    
    /// Reset timeout.
    /// 重置超时。
    void reset() {
        m_start = get_tick_ms();
    }

private:
    uint32_t m_start;
    uint32_t m_timeout;
};

// Usage / 用法
HwError wait_for_ready() {
    Timeout timeout(100);  // 100ms timeout / 100ms 超时
    
    while (!is_device_ready()) {
        if (timeout.expired()) {
            return HwError::AdcTimeout;
        }
    }
    
    return HwError::Ok;
}
```

---

## 8. Interrupts & Concurrency / 中断与并发

### Critical Section / 临界区

```cpp
#include <cstdint>

// ARM Cortex-M specific / ARM Cortex-M 特定
extern "C" uint32_t __get_PRIMASK();
extern "C" void __set_PRIMASK(uint32_t);

/// Critical section guard (RAII).
/// 临界区守卫（RAII）。
class CriticalSection {
public:
    /// Enter critical section (disable interrupts).
    /// 进入临界区（禁用中断）。
    CriticalSection() : m_primask(__get_PRIMASK()) {
        __disable_irq();
    }
    
    /// Exit critical section (restore interrupts).
    /// 退出临界区（恢复中断）。
    ~CriticalSection() {
        __set_PRIMASK(m_primask);
    }
    
    // Non-copyable / 不可拷贝
    CriticalSection(const CriticalSection&) = delete;
    CriticalSection& operator=(const CriticalSection&) = delete;

private:
    uint32_t m_primask;
};

// Usage / 用法
void update_shared_data(int16_t value) {
    CriticalSection cs;  // Interrupts disabled / 中断已禁用
    g_shared_value = value;
    ++g_update_count;
}  // Interrupts restored here / 中断在此恢复
```

### Ring Buffer for ISR / 中断服务程序的环形缓冲区

```cpp
#include <atomic>
#include <array>

/// Lock-free ring buffer for ISR to main communication.
/// 用于 ISR 到主程序通信的无锁环形缓冲区。
template<typename T, std::size_t N>
class RingBuffer {
public:
    /// Push value (called from ISR).
    /// 推入值（从 ISR 调用）。
    [[nodiscard]] bool push(T value) {
        std::size_t write = m_write.load(std::memory_order_relaxed);
        std::size_t next = (write + 1) % N;
        
        if (next == m_read.load(std::memory_order_acquire)) {
            return false;  // Buffer full / 缓冲区满
        }
        
        m_buffer[write] = value;
        m_write.store(next, std::memory_order_release);
        return true;
    }
    
    /// Pop value (called from main loop).
    /// 弹出值（从主循环调用）。
    [[nodiscard]] bool pop(T& value) {
        std::size_t read = m_read.load(std::memory_order_relaxed);
        
        if (read == m_write.load(std::memory_order_acquire)) {
            return false;  // Buffer empty / 缓冲区空
        }
        
        value = m_buffer[read];
        m_read.store((read + 1) % N, std::memory_order_release);
        return true;
    }
    
    /// Check if buffer is empty.
    /// 检查缓冲区是否为空。
    [[nodiscard]] bool empty() const {
        return m_read.load(std::memory_order_acquire) == 
               m_write.load(std::memory_order_acquire);
    }

private:
    std::array<T, N> m_buffer = {};
    std::atomic<std::size_t> m_read{0};
    std::atomic<std::size_t> m_write{0};
};

// Global ring buffer for ADC samples
// ADC 采样的全局环形缓冲区
RingBuffer<int16_t, 256> g_adc_buffer;

// Called from ADC interrupt handler
// 从 ADC 中断处理程序调用
extern "C" void ADC_IRQHandler() {
    int16_t sample = read_adc_data_register();
    g_adc_buffer.push(sample);  // Non-blocking / 非阻塞
}
```

### Volatile for Hardware Registers / 硬件寄存器使用 volatile

```cpp
// ✅ GOOD: Always use volatile for hardware registers
// ✅ 好: 硬件寄存器始终使用 volatile

// Register definition / 寄存器定义
volatile uint32_t* const ADC_DR = 
    reinterpret_cast<volatile uint32_t*>(0x4001'2000 + 0x4C);

uint32_t read_adc_blocking() {
    // Start conversion / 开始转换
    *ADC_CR2 |= (1u << 30);  // SWSTART
    
    // Wait for completion (volatile read in loop)
    // 等待完成（循环中的 volatile 读取）
    while ((*ADC_SR & (1u << 1)) == 0) {
        // EOC bit not set / EOC 位未设置
    }
    
    // Read result / 读取结果
    return *ADC_DR;
}

// ❌ BAD: Compiler may optimize away the loop
// ❌ 坏: 编译器可能优化掉循环
// uint32_t* adc_sr = ...;  // Missing volatile!
```

---

## 9. Control Flow / 控制流

### State Machine Pattern / 状态机模式

```cpp
/// Processing state machine.
/// 处理状态机。
class StateMachine {
public:
    /// Process one tick of state machine.
    /// 处理状态机一个时钟周期。
    void tick(bool trigger, bool sample_ready) {
        switch (m_state) {
            case State::Idle:
                handle_idle(trigger);
                break;
                
            case State::Sampling:
                handle_sampling(sample_ready);
                break;
                
            case State::Processing:
                handle_processing();
                break;
                
            case State::Transmitting:
                handle_transmitting();
                break;
        }
    }
    
    /// Get current state.
    /// 获取当前状态。
    [[nodiscard]] State state() const {
        return m_state;
    }

private:
    enum class State : uint8_t {
        Idle,
        Sampling,
        Processing,
        Transmitting
    };
    
    void handle_idle(bool trigger) {
        if (trigger) {
            // Transition: Idle -> Sampling
            // 转换: 空闲 -> 采样
            m_state = State::Sampling;
            m_sample_count = 0;
        }
    }
    
    void handle_sampling(bool sample_ready) {
        if (sample_ready) {
            ++m_sample_count;
        }
        
        if (m_sample_count >= 256) {
            // Transition: Sampling -> Processing
            // 转换: 采样 -> 处理
            m_state = State::Processing;
        }
    }
    
    void handle_processing() {
        // Process data...
        // 处理数据...
        
        // Transition: Processing -> Transmitting
        // 转换: 处理 -> 传输
        m_state = State::Transmitting;
    }
    
    void handle_transmitting() {
        // Transmit data...
        // 传输数据...
        
        // Transition: Transmitting -> Idle
        // 转换: 传输 -> 空闲
        m_state = State::Idle;
    }
    
    State m_state = State::Idle;
    uint32_t m_sample_count = 0;
};
```

### Loop Patterns / 循环模式

```cpp
/// Process buffer with explicit bounds.
/// 带显式边界处理缓冲区。
void process_buffer(int16_t* buffer, std::size_t len) {
    // ✅ GOOD: Explicit loop with clear bounds
    // ✅ 好: 带清晰边界的显式循环
    for (std::size_t i = 0; i < len; ++i) {
        buffer[i] = static_cast<int16_t>(buffer[i] * 2);
    }
}

/// Find maximum value.
/// 查找最大值。
int16_t find_max(const int16_t* buffer, std::size_t len) {
    if (len == 0) {
        return 0;
    }
    
    int16_t max_val = buffer[0];
    
    for (std::size_t i = 1; i < len; ++i) {
        if (buffer[i] > max_val) {
            max_val = buffer[i];
        }
    }
    
    return max_val;
}
```

---

## 10. Forbidden Patterns / 禁止项

### ❌ NEVER USE / 禁止使用

| Pattern / 模式 | Reason / 原因 | Alternative / 替代方案 |
|---------------|--------------|----------------------|
| `new`/`delete` | Heap fragmentation | Static/stack allocation |
| `malloc`/`free` | Heap fragmentation | Static/stack allocation |
| Exceptions (`throw`) | Code bloat, non-deterministic | Error codes, Result type |
| RTTI (`dynamic_cast`) | Code bloat | Static polymorphism |
| `std::string` | Heap allocation | `char[]`, `std::string_view` |
| `std::vector` | Heap allocation | `std::array`, C arrays |
| `std::function` | Heap allocation possible | Function pointers, templates |
| Virtual functions (excessive) | Vtable overhead | CRTP, templates |
| Complex templates | Code bloat | Simple templates |
| Recursion (deep) | Stack overflow | Iteration |

### Forbidden Code Examples / 禁止代码示例

```cpp
// ❌ BAD: All of these are forbidden in embedded
// ❌ 坏: 以下在嵌入式中全部禁止

void forbidden_examples() {
    // Dynamic allocation / 动态分配
    // int* ptr = new int[100];      // ❌
    // delete[] ptr;                  // ❌
    // auto vec = std::vector<int>(); // ❌
    // auto str = std::string("hi");  // ❌
    
    // Exceptions / 异常
    // throw std::runtime_error("error"); // ❌
    
    // RTTI / 运行时类型信息
    // dynamic_cast<Derived*>(base);  // ❌
}

// ✅ GOOD: Correct alternatives
// ✅ 好: 正确的替代方案

void good_examples() {
    // Static allocation / 静态分配
    int buffer[100] = {};                    // ✅
    std::array<int, 100> arr = {};           // ✅
    char str[32] = "hello";                  // ✅
    
    // Error codes / 错误码
    HwError err = do_something();            // ✅
    if (err != HwError::Ok) { /* handle */ } // ✅
}
```

---

## 11. Recommended Libraries / 推荐库

### Embedded-Friendly Libraries / 嵌入式友好库

| Library / 库 | Purpose / 用途 | Notes / 注意 |
|-------------|----------------|--------------|
| ETL (Embedded Template Library) | STL replacement | No heap, fixed-size containers |
| CMSIS | ARM Cortex support | Standard ARM abstraction |
| FreeRTOS | RTOS | Optional, adds complexity |
| libfixmath | Fixed-point math | No FPU required |
| printf/sprintf | Formatting | Use lightweight version |

### ETL Example / ETL 示例

```cpp
#include <etl/vector.h>
#include <etl/string.h>
#include <etl/queue.h>

// Fixed-capacity vector (no heap)
// 固定容量向量（无堆）
etl::vector<int16_t, 256> samples;

// Fixed-capacity string (no heap)
// 固定容量字符串（无堆）
etl::string<32> channel_name;

// Fixed-capacity queue (no heap)
// 固定容量队列（无堆）
etl::queue<uint8_t, 64> command_queue;

void use_etl() {
    samples.push_back(100);
    channel_name = "Fp1";
    command_queue.push(0x01);
}
```

---

## 12. Code Examples / 代码示例

### Complete Example: ADC Driver
### 完整示例: ADC 驱动

```cpp
/**
 * @file adc_driver.cpp
 * @brief Simple ADC Driver Example
 *        简单 ADC 驱动示例
 */

#include <array>
#include <atomic>
#include <cstdint>

// ============================================
// Constants / 常量
// ============================================

/// Maximum number of samples in buffer.
/// 缓冲区最大采样点数。
constexpr std::size_t MAX_SAMPLES = 256;

/// Number of ADC channels.
/// ADC 通道数。
constexpr std::size_t NUM_CHANNELS = 8;

/// ADC reference voltage in millivolts.
/// ADC 参考电压（毫伏）。
constexpr uint32_t VREF_MV = 3300;

/// ADC resolution (12-bit).
/// ADC 分辨率（12 位）。
constexpr uint32_t ADC_MAX = 4095;

// ============================================
// Error Types / 错误类型
// ============================================

/// ADC error types.
/// ADC 错误类型。
enum class AdcError : uint8_t {
    Ok = 0,
    Timeout,
    InvalidChannel,
    BufferFull
};

// ============================================
// Data Structures / 数据结构
// ============================================

/// ADC sample buffer.
/// ADC 采样缓冲区。
struct SampleBuffer {
    std::array<uint16_t, MAX_SAMPLES> data = {};
    std::size_t len = 0;
    uint8_t channel = 0;
    
    /// Add sample to buffer.
    /// 向缓冲区添加采样点。
    [[nodiscard]] bool push(uint16_t sample) {
        if (len >= MAX_SAMPLES) {
            return false;
        }
        data[len] = sample;
        ++len;
        return true;
    }
    
    /// Clear buffer.
    /// 清空缓冲区。
    void clear() {
        len = 0;
    }
};

// ============================================
// ADC Functions / ADC 函数
// ============================================

/// Convert raw ADC value to millivolts.
/// 将原始 ADC 值转换为毫伏。
///
/// @param raw Raw ADC value (0-4095).
///            原始 ADC 值（0-4095）。
/// @return Voltage in millivolts.
///         电压（毫伏）。
[[nodiscard]] constexpr uint32_t adc_to_millivolts(uint16_t raw) {
    return (static_cast<uint32_t>(raw) * VREF_MV) / ADC_MAX;
}

/// Calculate mean of samples.
/// 计算采样均值。
///
/// @param buffer Sample buffer.
///               采样缓冲区。
/// @return Mean value, or 0 if empty.
///         均值，如果为空则返回 0。
[[nodiscard]] uint16_t calculate_mean(const SampleBuffer& buffer) {
    if (buffer.len == 0) {
        return 0;
    }
    
    uint32_t sum = 0;
    for (std::size_t i = 0; i < buffer.len; ++i) {
        sum += buffer.data[i];
    }
    
    return static_cast<uint16_t>(sum / buffer.len);
}

/// Find minimum and maximum values.
/// 查找最小和最大值。
///
/// @param buffer Sample buffer.
///               采样缓冲区。
/// @param[out] min_val Minimum value.
///                     最小值。
/// @param[out] max_val Maximum value.
///                     最大值。
/// @return true if buffer is not empty.
///         如果缓冲区非空则返回 true。
[[nodiscard]] bool find_min_max(
    const SampleBuffer& buffer,
    uint16_t& min_val,
    uint16_t& max_val)
{
    if (buffer.len == 0) {
        return false;
    }
    
    min_val = buffer.data[0];
    max_val = buffer.data[0];
    
    for (std::size_t i = 1; i < buffer.len; ++i) {
        if (buffer.data[i] < min_val) {
            min_val = buffer.data[i];
        }
        if (buffer.data[i] > max_val) {
            max_val = buffer.data[i];
        }
    }
    
    return true;
}

// ============================================
// Global State / 全局状态
// ============================================

/// Global sample counter (atomic for ISR safety).
/// 全局采样计数器（原子以保证 ISR 安全）。
static std::atomic<uint32_t> g_sample_counter{0};

/// Increment and get sample counter.
/// 增加并获取采样计数器。
[[nodiscard]] uint32_t next_sample_id() {
    return g_sample_counter.fetch_add(1, std::memory_order_relaxed);
}

// ============================================
// Main Entry / 主入口
// ============================================

/// Main entry point.
/// 主入口点。
int main() {
    // Initialize buffer for channel 0.
    // 为通道 0 初始化缓冲区。
    SampleBuffer buffer;
    buffer.channel = 0;
    
    // Simulated ADC readings.
    // 模拟 ADC 读数。
    constexpr std::array<uint16_t, 8> test_values = {
        100, 200, 300, 400, 500, 600, 700, 800
    };
    
    // Fill buffer with test data.
    // 用测试数据填充缓冲区。
    for (uint16_t value : test_values) {
        if (!buffer.push(value)) {
            // Handle buffer full.
            // 处理缓冲区满。
            break;
        }
    }
    
    // Calculate statistics.
    // 计算统计数据。
    uint16_t mean = calculate_mean(buffer);
    uint32_t mean_mv = adc_to_millivolts(mean);
    
    uint16_t min_val = 0;
    uint16_t max_val = 0;
    if (find_min_max(buffer, min_val, max_val)) {
        uint32_t min_mv = adc_to_millivolts(min_val);
        uint32_t max_mv = adc_to_millivolts(max_val);
        // Use min_mv, max_mv...
        // 使用 min_mv, max_mv...
        (void)min_mv;
        (void)max_mv;
    }
    
    // Use mean_mv...
    // 使用 mean_mv...
    (void)mean_mv;
    
    // Main loop.
    // 主循环。
    while (true) {
        // Get sample ID.
        // 获取采样 ID。
        uint32_t sample_id = next_sample_id();
        (void)sample_id;
        
        // Wait for interrupt (low power).
        // 等待中断（低功耗）。
        __WFI();
    }
    
    return 0;
}
```

---

## Quick Reference Card / 快速参考卡

```
┌─────────────────────────────────────────────────────────────────┐
│           EMBEDDED C++17 AI CODING CHECKLIST                    │
│           嵌入式 C++17 AI 编码检查清单                           │
├─────────────────────────────────────────────────────────────────┤
│ ✅ C++17 standard          ✅ C++17 标准                         │
│ ✅ -fno-exceptions         ✅ 禁用异常                           │
│ ✅ -fno-rtti               ✅ 禁用 RTTI                          │
│ ✅ No heap allocation      ✅ 无堆分配                           │
│ ✅ Fixed-width integers    ✅ 固定宽度整数                        │
│ ✅ volatile for registers  ✅ 寄存器使用 volatile                 │
│ ✅ Bilingual comments      ✅ 双语注释                           │
│ ✅ Error codes used        ✅ 使用错误码                          │
│ ✅ [[nodiscard]] used      ✅ 使用 [[nodiscard]]                 │
│ ✅ Critical sections       ✅ 临界区                              │
│ ✅ Static allocation       ✅ 静态分配                           │
│ ✅ -Os -flto enabled       ✅ 启用 -Os -flto                     │
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
