# C++ AI-Friendly Coding Guideline (Standard Environment)
# C++ AI 友好编码规范（标准环境）

> **Standard**: C++17  
> **Target**: Desktop / Server / CLI / Library (with full std)  
> **Philosophy**: Keep it simple. Readable code > clever code.  
> **核心理念**: 简单至上。可读性 > 炫技。

---

## Table of Contents / 目录

1. [Core Principles / 核心原则](#1-core-principles--核心原则)
2. [Naming Conventions / 命名规范](#2-naming-conventions--命名规范)
3. [Type System / 类型系统](#3-type-system--类型系统)
4. [Memory Management / 内存管理](#4-memory-management--内存管理)
5. [Error Handling / 错误处理](#5-error-handling--错误处理)
6. [Comments & Documentation / 注释与文档](#6-comments--documentation--注释与文档)
7. [Collections & Containers / 集合与容器](#7-collections--containers--集合与容器)
8. [Control Flow / 控制流](#8-control-flow--控制流)
9. [Functions & Classes / 函数与类](#9-functions--classes--函数与类)
10. [Forbidden Patterns / 禁止项](#10-forbidden-patterns--禁止项)
11. [Recommended Libraries / 推荐库](#11-recommended-libraries--推荐库)
12. [Code Examples / 代码示例](#12-code-examples--代码示例)

---

## 1. Core Principles / 核心原则

### English

1. **Explicit over implicit** - Always prefer clear, explicit code over clever shortcuts.
2. **RAII everywhere** - Resources should be managed by objects, not manually.
3. **One thing per function** - Functions should do one thing and do it well.
4. **No raw pointers for ownership** - Use smart pointers for memory management.
5. **AI-readable patterns** - Stick to common patterns that AI tools understand well.
6. **Const by default** - Mark everything `const` unless it needs to change.

### 中文

1. **显式优于隐式** - 始终选择清晰、显式的代码，而非巧妙的捷径。
2. **到处使用 RAII** - 资源应由对象管理，而非手动管理。
3. **单一职责** - 函数只做一件事，并把它做好。
4. **禁止裸指针管理所有权** - 使用智能指针进行内存管理。
5. **AI 可读模式** - 坚持使用 AI 工具能够良好理解的常见模式。
6. **默认使用 const** - 除非需要修改，否则标记为 `const`。

---

## 2. Naming Conventions / 命名规范

### Summary Table / 总结表

| Entity / 实体 | Style / 风格 | Example / 示例 |
|--------------|-------------|----------------|
| Namespace | `snake_case` | `signal_processing` |
| Class / Struct | `PascalCase` | `SignalBuffer`, `ChannelInfo` |
| Function / Method | `snake_case` or `camelCase` | `calculate_fft()`, `getData()` |
| Variable / Parameter | `snake_case` | `sample_rate`, `channel_count` |
| Constant / Macro | `SCREAMING_SNAKE_CASE` | `MAX_CHANNELS`, `DEFAULT_RATE` |
| Member variable | `m_` prefix or `_` suffix | `m_data`, `data_` |
| Template parameter | Single uppercase | `T`, `U`, `Container` |

### Rules / 规则

```cpp
// ✅ GOOD: Descriptive names with units
// ✅ 好: 描述性名称，带单位
double sample_rate_hz = 256.0;
double duration_seconds = 10.0;
double voltage_microvolts = 5.5;

// ❌ BAD: Cryptic abbreviations
// ❌ 坏: 神秘的缩写
double sr = 256.0;
double dur = 10.0;
double v = 5.5;
```

### Boolean Naming / 布尔值命名

```cpp
// ✅ GOOD: Use is_, has_, can_, should_ prefix
// ✅ 好: 使用 is_, has_, can_, should_ 前缀
bool is_valid = true;
bool has_data = false;
bool can_process = true;

// ❌ BAD: Ambiguous boolean names
// ❌ 坏: 模糊的布尔值名称
bool valid = true;
bool data = false;
```

---

## 3. Type System / 类型系统

### Use Modern Type Aliases / 使用现代类型别名

```cpp
#include <cstdint>
#include <cstddef>

// ✅ GOOD: Fixed-width types for clarity
// ✅ 好: 固定宽度类型以清晰
int32_t signed_value = -100;
uint32_t unsigned_value = 100;
size_t array_index = 0;
std::ptrdiff_t offset = -5;

// ❌ BAD: Platform-dependent types
// ❌ 坏: 平台相关类型
int value = 100;      // Size varies by platform
                      // 大小因平台而异
long big_value = 0;   // Ambiguous size
                      // 大小不明确
```

### Auto Usage Rules / auto 使用规则

```cpp
// ✅ GOOD: Use auto when type is obvious from right-hand side
// ✅ 好: 当类型在右侧明显时使用 auto
auto ptr = std::make_unique<SignalBuffer>();
auto it = container.begin();
auto [key, value] = *map_iter;  // C++17 structured binding
                                 // C++17 结构化绑定

// ✅ GOOD: Explicit type for basic types and clarity
// ✅ 好: 基本类型和需要清晰时使用显式类型
int count = 0;
double rate = 256.0;
std::string name = "Channel1";

// ❌ BAD: auto hiding important type information
// ❌ 坏: auto 隐藏重要类型信息
auto result = some_complex_function();  // What type is result?
                                         // result 是什么类型？
```

### Struct Definition / 结构体定义

```cpp
/// Signal buffer for EEG data processing.
/// EEG 数据处理的信号缓冲区。
struct SignalBuffer {
    /// Raw sample data in microvolts.
    /// 原始采样数据（微伏）。
    std::vector<double> data;
    
    /// Sampling rate in Hz.
    /// 采样率（赫兹）。
    double sample_rate = 0.0;
    
    /// Channel names.
    /// 通道名称。
    std::vector<std::string> channel_names;
    
    /// Get number of samples.
    /// 获取采样点数。
    [[nodiscard]] size_t sample_count() const {
        return data.size();
    }
};
```

### Enum Class (Strong Enum) / 枚举类（强枚举）

```cpp
// ✅ GOOD: Use enum class for type safety
// ✅ 好: 使用 enum class 以保证类型安全
enum class ProcessingState {
    Idle,
    Sampling,
    Processing,
    Complete,
    Error
};

// Usage / 用法
ProcessingState state = ProcessingState::Idle;

// ❌ BAD: Old-style enum (pollutes namespace)
// ❌ 坏: 旧式枚举（污染命名空间）
enum State { Idle, Sampling };  // Avoid / 避免
```

---

## 4. Memory Management / 内存管理

### Smart Pointer Rules / 智能指针规则

```cpp
#include <memory>

// ✅ GOOD: Use unique_ptr for exclusive ownership
// ✅ 好: 独占所有权使用 unique_ptr
std::unique_ptr<SignalBuffer> create_buffer() {
    return std::make_unique<SignalBuffer>();
}

// ✅ GOOD: Use shared_ptr only when truly shared
// ✅ 好: 只在真正需要共享时使用 shared_ptr
std::shared_ptr<Config> shared_config = std::make_shared<Config>();

// ❌ BAD: Raw new/delete
// ❌ 坏: 裸 new/delete
SignalBuffer* buffer = new SignalBuffer();  // NO!
delete buffer;                               // NEVER!
```

### Ownership Patterns / 所有权模式

```cpp
/// Process buffer (takes ownership).
/// 处理缓冲区（获取所有权）。
void consume_buffer(std::unique_ptr<Buffer> buffer) {
    // buffer is moved in, ownership transferred
    // buffer 被移入，所有权已转移
    process(buffer->data);
    // buffer automatically deleted when function exits
    // 函数退出时 buffer 自动删除
}

/// Read buffer (borrows, does not own).
/// 读取缓冲区（借用，不拥有）。
void read_buffer(const Buffer& buffer) {
    // Reference: no ownership transfer
    // 引用: 无所有权转移
    for (const auto& sample : buffer.data) {
        process_sample(sample);
    }
}

/// Modify buffer (borrows mutably).
/// 修改缓冲区（可变借用）。
void modify_buffer(Buffer& buffer) {
    // Mutable reference: can modify but doesn't own
    // 可变引用: 可以修改但不拥有
    buffer.data.push_back(0.0);
}
```

### RAII Example / RAII 示例

```cpp
/// File handle with automatic cleanup.
/// 带自动清理的文件句柄。
class FileHandle {
public:
    /// Open file for reading.
    /// 打开文件用于读取。
    explicit FileHandle(const std::string& path)
        : m_file(std::fopen(path.c_str(), "r"))
    {
        if (!m_file) {
            throw std::runtime_error("Failed to open file");
        }
    }
    
    /// Destructor automatically closes file.
    /// 析构函数自动关闭文件。
    ~FileHandle() {
        if (m_file) {
            std::fclose(m_file);
        }
    }
    
    // Delete copy (prevent double-free).
    // 删除拷贝（防止重复释放）。
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
    
    // Allow move.
    // 允许移动。
    FileHandle(FileHandle&& other) noexcept
        : m_file(other.m_file)
    {
        other.m_file = nullptr;
    }
    
private:
    FILE* m_file;
};
```

---

## 5. Error Handling / 错误处理

### Use std::optional for Missing Values / 使用 std::optional 表示缺失值

```cpp
#include <optional>

/// Find channel by name.
/// 按名称查找通道。
///
/// Returns channel index if found, std::nullopt otherwise.
/// 如果找到返回通道索引，否则返回 std::nullopt。
std::optional<size_t> find_channel(
    const std::vector<std::string>& channels,
    std::string_view name)
{
    for (size_t i = 0; i < channels.size(); ++i) {
        if (channels[i] == name) {
            return i;  // Found / 找到
        }
    }
    return std::nullopt;  // Not found / 未找到
}

// Usage / 用法
void process_channel(const std::vector<std::string>& channels) {
    // ✅ GOOD: Check before use
    // ✅ 好: 使用前检查
    if (auto index = find_channel(channels, "Fp1")) {
        std::cout << "Found at index: " << *index << "\n";
    } else {
        std::cout << "Channel not found\n";
    }
}
```

### Exception Usage / 异常使用

```cpp
#include <stdexcept>

/// Calculate power spectral density.
/// 计算功率谱密度。
///
/// @throws std::invalid_argument if sample_rate <= 0
/// @throws std::invalid_argument 如果 sample_rate <= 0
std::vector<double> calculate_psd(
    const std::vector<double>& data,
    double sample_rate)
{
    // ✅ GOOD: Validate input with clear exception
    // ✅ 好: 用清晰的异常验证输入
    if (sample_rate <= 0) {
        throw std::invalid_argument(
            "Sample rate must be positive, got: " + 
            std::to_string(sample_rate));
    }
    
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    
    // Process...
    // 处理...
    return {};
}

// Usage with try-catch / 使用 try-catch
void safe_process() {
    try {
        auto psd = calculate_psd(data, 256.0);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid input: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}
```

### Error Code Pattern / 错误码模式

```cpp
/// Error codes for signal processing.
/// 信号处理的错误码。
enum class ErrorCode {
    Success,
    InvalidParameter,
    BufferOverflow,
    FileNotFound,
    ParseError
};

/// Result type combining value and error.
/// 组合值和错误的结果类型。
template<typename T>
struct Result {
    std::optional<T> value;
    ErrorCode error = ErrorCode::Success;
    std::string message;
    
    /// Check if operation succeeded.
    /// 检查操作是否成功。
    [[nodiscard]] bool ok() const {
        return error == ErrorCode::Success;
    }
    
    /// Get value or default.
    /// 获取值或默认值。
    [[nodiscard]] T value_or(T default_val) const {
        return value.value_or(default_val);
    }
};

/// Read signal file with result type.
/// 用结果类型读取信号文件。
Result<SignalBuffer> read_signal_file(const std::string& path) {
    Result<SignalBuffer> result;
    
    std::ifstream file(path);
    if (!file) {
        result.error = ErrorCode::FileNotFound;
        result.message = "Could not open: " + path;
        return result;
    }
    
    // Parse file...
    // 解析文件...
    result.value = SignalBuffer{};
    return result;
}
```

---

## 6. Comments & Documentation / 注释与文档

### Bilingual Documentation Style / 双语文档风格

```cpp
/// Calculate the power spectral density using Welch's method.
/// 使用 Welch 方法计算功率谱密度。
///
/// @param data Input signal samples in microvolts.
///             输入信号采样（微伏）。
/// @param sample_rate Sampling frequency in Hz.
///                    采样频率（赫兹）。
/// @param window_size FFT window size (must be power of 2).
///                    FFT 窗口大小（必须是 2 的幂）。
///
/// @return Power spectral density array.
///         功率谱密度数组。
///
/// @throws std::invalid_argument if parameters are invalid.
///         如果参数无效则抛出 std::invalid_argument。
///
/// @example
/// @code
/// std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
/// auto psd = calculate_psd(data, 256.0, 256);
/// @endcode
std::vector<double> calculate_psd(
    const std::vector<double>& data,
    double sample_rate,
    size_t window_size);
```

### Inline Comments / 行内注释

```cpp
std::vector<double> process_epoch(
    const std::vector<double>& data,
    size_t baseline_samples)
{
    // Step 1: Calculate baseline mean.
    // 步骤 1: 计算基线均值。
    double sum = 0.0;
    for (size_t i = 0; i < baseline_samples && i < data.size(); ++i) {
        sum += data[i];
    }
    double baseline_mean = sum / static_cast<double>(baseline_samples);
    
    // Step 2: Subtract baseline from all samples.
    // 步骤 2: 从所有采样中减去基线。
    std::vector<double> corrected;
    corrected.reserve(data.size());
    
    for (double sample : data) {
        corrected.push_back(sample - baseline_mean);
    }
    
    // Step 3: Return corrected data.
    // 步骤 3: 返回校正后的数据。
    return corrected;
}
```

### File Header Comment / 文件头注释

```cpp
/**
 * @file signal_processing.hpp
 * @brief Signal processing utilities for EEG data.
 *        EEG 数据的信号处理工具。
 *
 * This module provides functions for filtering, FFT, and
 * artifact removal in EEG signal processing pipelines.
 *
 * 本模块提供 EEG 信号处理管道中的滤波、FFT 和伪影去除函数。
 *
 * @author BCIF Team
 * @date 2026-02-01
 */

#pragma once

#include <vector>
#include <string>

namespace bcif::signal {
    // ... declarations ...
}
```

---

## 7. Collections & Containers / 集合与容器

### std::vector - The Default Choice / std::vector - 默认选择

```cpp
#include <vector>

// ✅ GOOD: Pre-allocate when size is known
// ✅ 好: 已知大小时预分配
std::vector<double> create_buffer(size_t sample_count) {
    std::vector<double> buffer;
    buffer.reserve(sample_count);  // Pre-allocate / 预分配
    return buffer;
}

// ✅ GOOD: Use range-based for loop
// ✅ 好: 使用基于范围的 for 循环
void process_samples(const std::vector<double>& samples) {
    for (const double& sample : samples) {
        // Process each sample / 处理每个采样点
    }
}

// ✅ GOOD: Use at() for bounds checking in non-critical code
// ✅ 好: 非关键代码中使用 at() 进行边界检查
double get_safe(const std::vector<double>& v, size_t index) {
    return v.at(index);  // Throws if out of bounds
                         // 越界时抛出异常
}
```

### std::map and std::unordered_map / std::map 和 std::unordered_map

```cpp
#include <map>
#include <unordered_map>

// ✅ GOOD: Use unordered_map for fast lookup (O(1) average)
// ✅ 好: 快速查找使用 unordered_map（平均 O(1)）
std::unordered_map<std::string, size_t> channel_indices;

// ✅ GOOD: Use map when order matters
// ✅ 好: 需要顺序时使用 map
std::map<double, std::string> frequency_bands = {
    {4.0, "Delta"},
    {8.0, "Theta"},
    {13.0, "Alpha"},
    {30.0, "Beta"}
};

// ✅ GOOD: Use structured binding (C++17)
// ✅ 好: 使用结构化绑定 (C++17)
void print_bands(const std::map<double, std::string>& bands) {
    for (const auto& [freq, name] : bands) {
        std::cout << name << ": " << freq << " Hz\n";
    }
}

// ✅ GOOD: Use insert_or_assign (C++17)
// ✅ 好: 使用 insert_or_assign (C++17)
void update_channel(
    std::unordered_map<std::string, double>& gains,
    const std::string& channel,
    double gain)
{
    gains.insert_or_assign(channel, gain);
}
```

### std::string and std::string_view / std::string 和 std::string_view

```cpp
#include <string>
#include <string_view>

// ✅ GOOD: Use string_view for read-only parameters
// ✅ 好: 只读参数使用 string_view
bool starts_with(std::string_view str, std::string_view prefix) {
    return str.substr(0, prefix.size()) == prefix;
}

// ✅ GOOD: Use string for owned data
// ✅ 好: 拥有数据时使用 string
struct ChannelInfo {
    std::string name;  // Owned / 拥有
    std::string unit;  // Owned / 拥有
};

// ❌ BAD: Use const char* for API
// ❌ 坏: API 使用 const char*
void bad_function(const char* str);  // Avoid / 避免
```

---

## 8. Control Flow / 控制流

### If with Initializer (C++17) / 带初始化的 if (C++17)

```cpp
// ✅ GOOD: Limit scope with if initializer
// ✅ 好: 用 if 初始化限制作用域
void process_map(const std::map<std::string, int>& data) {
    if (auto it = data.find("key"); it != data.end()) {
        // 'it' only exists in this scope
        // 'it' 只存在于此作用域
        std::cout << "Found: " << it->second << "\n";
    }
    // 'it' no longer accessible here
    // 'it' 在这里不再可访问
}
```

### Early Return Pattern / 提前返回模式

```cpp
// ✅ GOOD: Use early returns to reduce nesting
// ✅ 好: 使用提前返回减少嵌套
std::optional<double> calculate_mean(const std::vector<double>& data) {
    // Guard clause 1 / 守卫子句 1
    if (data.empty()) {
        return std::nullopt;
    }
    
    // Main logic with minimal nesting
    // 主逻辑，最小嵌套
    double sum = 0.0;
    for (double value : data) {
        sum += value;
    }
    
    return sum / static_cast<double>(data.size());
}

// ❌ BAD: Deep nesting
// ❌ 坏: 深层嵌套
std::optional<double> calculate_mean_bad(const std::vector<double>& data) {
    if (!data.empty()) {
        double sum = 0.0;
        for (double value : data) {
            sum += value;
        }
        return sum / static_cast<double>(data.size());
    } else {
        return std::nullopt;
    }
}
```

### Switch with Enum / switch 与枚举

```cpp
/// Get state name as string.
/// 获取状态名称字符串。
std::string_view get_state_name(ProcessingState state) {
    switch (state) {
        case ProcessingState::Idle:
            return "Idle";
        case ProcessingState::Sampling:
            return "Sampling";
        case ProcessingState::Processing:
            return "Processing";
        case ProcessingState::Complete:
            return "Complete";
        case ProcessingState::Error:
            return "Error";
    }
    // No default - compiler warns if case missing
    // 无 default - 如果缺少 case 编译器会警告
    return "Unknown";
}
```

### Range-based For Loop / 基于范围的 for 循环

```cpp
// ✅ GOOD: Range-based for loop
// ✅ 好: 基于范围的 for 循环
void process_channels(const std::vector<ChannelInfo>& channels) {
    for (const auto& channel : channels) {
        std::cout << channel.name << "\n";
    }
}

// ✅ GOOD: With index using enumerate pattern
// ✅ 好: 使用枚举模式带索引
void print_with_index(const std::vector<std::string>& items) {
    size_t index = 0;
    for (const auto& item : items) {
        std::cout << index << ": " << item << "\n";
        ++index;
    }
}

// ❌ BAD: Old-style index loop when not needed
// ❌ 坏: 不需要时使用旧式索引循环
void process_bad(const std::vector<double>& data) {
    for (size_t i = 0; i < data.size(); ++i) {
        process(data[i]);  // Unnecessary indexing
                           // 不必要的索引
    }
}
```

---

## 9. Functions & Classes / 函数与类

### Function Parameter Guidelines / 函数参数指南

```cpp
// ✅ GOOD: Clear parameter passing conventions
// ✅ 好: 清晰的参数传递约定

// Small types: pass by value
// 小类型: 传值
void set_rate(double rate);
void set_count(int count);

// Large types (read-only): pass by const reference
// 大类型（只读）: 传常引用
void process(const std::vector<double>& data);
void analyze(const SignalBuffer& buffer);

// Large types (need to modify): pass by reference
// 大类型（需要修改）: 传引用
void normalize(std::vector<double>& data);

// Strings (read-only): use string_view
// 字符串（只读）: 使用 string_view
void log_message(std::string_view message);

// Transfer ownership: use unique_ptr
// 转移所有权: 使用 unique_ptr
void take_ownership(std::unique_ptr<Buffer> buffer);

// Optional parameters: use pointer (nullable)
// 可选参数: 使用指针（可为空）
void configure(const Config* config = nullptr);
```

### Class Design / 类设计

```cpp
/// Signal processor class.
/// 信号处理器类。
class SignalProcessor {
public:
    // ============================================
    // Constructors / 构造函数
    // ============================================
    
    /// Create processor with sample rate.
    /// 用采样率创建处理器。
    explicit SignalProcessor(double sample_rate)
        : m_sample_rate(sample_rate)
    {
        if (sample_rate <= 0) {
            throw std::invalid_argument("Sample rate must be positive");
        }
    }
    
    // Default destructor is fine (Rule of Zero).
    // 默认析构函数即可（零法则）。
    ~SignalProcessor() = default;
    
    // Delete copy (or explicitly default if copyable).
    // 删除拷贝（或如果可拷贝则显式 default）。
    SignalProcessor(const SignalProcessor&) = delete;
    SignalProcessor& operator=(const SignalProcessor&) = delete;
    
    // Allow move.
    // 允许移动。
    SignalProcessor(SignalProcessor&&) = default;
    SignalProcessor& operator=(SignalProcessor&&) = default;
    
    // ============================================
    // Public Methods / 公共方法
    // ============================================
    
    /// Process signal data.
    /// 处理信号数据。
    [[nodiscard]] std::vector<double> process(
        const std::vector<double>& input) const;
    
    /// Get sample rate.
    /// 获取采样率。
    [[nodiscard]] double sample_rate() const noexcept {
        return m_sample_rate;
    }

private:
    // ============================================
    // Private Members / 私有成员
    // ============================================
    
    double m_sample_rate;
    std::vector<double> m_buffer;
};
```

### Use [[nodiscard]] / 使用 [[nodiscard]]

```cpp
// ✅ GOOD: Mark functions whose return value should not be ignored
// ✅ 好: 标记返回值不应被忽略的函数
[[nodiscard]] bool is_valid() const;
[[nodiscard]] std::optional<int> find_index() const;
[[nodiscard]] std::unique_ptr<Buffer> create_buffer();

// Compiler warns if return value is ignored
// 如果返回值被忽略，编译器会警告
```

### Use noexcept / 使用 noexcept

```cpp
// ✅ GOOD: Mark non-throwing functions
// ✅ 好: 标记不抛出异常的函数
class Buffer {
public:
    // Simple getters should be noexcept
    // 简单的 getter 应该是 noexcept
    [[nodiscard]] size_t size() const noexcept {
        return m_data.size();
    }
    
    [[nodiscard]] bool empty() const noexcept {
        return m_data.empty();
    }
    
    // Move operations should be noexcept
    // 移动操作应该是 noexcept
    Buffer(Buffer&&) noexcept = default;
    Buffer& operator=(Buffer&&) noexcept = default;
    
private:
    std::vector<double> m_data;
};
```

---

## 10. Forbidden Patterns / 禁止项

### ❌ DO NOT USE / 禁止使用

| Pattern / 模式 | Reason / 原因 | Alternative / 替代方案 |
|---------------|--------------|----------------------|
| Raw `new`/`delete` | Memory leaks | `std::make_unique`, `std::make_shared` |
| C-style casts `(int)x` | Unsafe | `static_cast<int>(x)` |
| C arrays `int arr[10]` | No bounds info | `std::array<int, 10>` |
| `#define` for constants | No type safety | `constexpr` |
| `using namespace std;` | Name pollution | Explicit `std::` prefix |
| `goto` | Unstructured flow | Loops, functions |
| Complex macros | AI cannot parse | Inline functions, templates |
| Deep inheritance (>2) | Hard to follow | Composition |
| Multiple inheritance | Complex | Single inheritance + interfaces |
| Global mutable state | Hard to reason | Dependency injection |

### Code Examples of Forbidden Patterns / 禁止模式代码示例

```cpp
// ❌ BAD: All of these are forbidden
// ❌ 坏: 以下全部禁止

// Raw memory management / 裸内存管理
int* ptr = new int[100];  // ❌
delete[] ptr;             // ❌

// C-style cast / C 风格转换
double d = 3.14;
int i = (int)d;  // ❌ Use static_cast<int>(d)

// C-style array / C 风格数组
int arr[10];  // ❌ Use std::array<int, 10>

// Macro constant / 宏常量
#define MAX_SIZE 100  // ❌ Use constexpr size_t MAX_SIZE = 100;

// Using namespace / 使用命名空间
using namespace std;  // ❌ Never in headers, avoid in .cpp

// ============================================

// ✅ GOOD: Correct alternatives
// ✅ 好: 正确的替代方案

auto arr = std::make_unique<int[]>(100);  // ✅
int i = static_cast<int>(d);              // ✅
std::array<int, 10> arr{};                // ✅
constexpr size_t MAX_SIZE = 100;          // ✅
```

---

## 11. Recommended Libraries / 推荐库

### Why Zig Build? / 为什么使用 Zig 构建？

```
Advantages / 优势:
1. Cross-compilation out of the box
   开箱即用的交叉编译
2. No CMake, no Makefile - just `zig build`
   无需 CMake、无需 Makefile - 只需 `zig build`
3. Can compile C/C++/Rust in unified system
   可在统一系统中编译 C/C++/Rust
4. Reproducible builds
   可重现构建
5. Package management via build.zig.zon
   通过 build.zig.zon 进行包管理
```

### Core Dependencies for BCIF / BCIF 核心依赖

```zig
// build.zig - Standard C++17 library build
// build.zig - 标准 C++17 库构建

const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target / 标准目标
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create static library / 创建静态库
    const lib = b.addStaticLibrary(.{
        .name = "bcif_signal",
        .target = target,
        .optimize = optimize,
    });

    // Add C++ source files / 添加 C++ 源文件
    lib.addCSourceFiles(.{
        .files = &.{
            "src/signal_processing.cpp",
            "src/fft.cpp",
            "src/filter.cpp",
        },
        .flags = &.{
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
        },
    });

    // Include paths / 包含路径
    lib.addIncludePath(.{ .cwd_relative = "include" });
    lib.addIncludePath(.{ .cwd_relative = "third_party/eigen" });
    lib.addIncludePath(.{ .cwd_relative = "third_party/fmt/include" });
    lib.addIncludePath(.{ .cwd_relative = "third_party/spdlog/include" });
    lib.addIncludePath(.{ .cwd_relative = "third_party/nlohmann_json/include" });

    // Link C++ standard library / 链接 C++ 标准库
    lib.linkLibCpp();

    // Install artifact / 安装产物
    b.installArtifact(lib);

    // Create executable (optional) / 创建可执行文件（可选）
    const exe = b.addExecutable(.{
        .name = "bcif_cli",
        .target = target,
        .optimize = optimize,
    });
    exe.addCSourceFile(.{
        .file = .{ .cwd_relative = "src/main.cpp" },
        .flags = &.{ "-std=c++17" },
    });
    exe.linkLibrary(lib);
    exe.linkLibCpp();
    b.installArtifact(exe);

    // Run step / 运行步骤
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    const run_step = b.step("run", "Run the CLI / 运行 CLI");
    run_step.dependOn(&run_cmd.step);
}
```

### build.zig.zon (Package Manifest) / 包清单

```zig
// build.zig.zon - Package dependencies
// build.zig.zon - 包依赖

.{
    .name = "bcif_signal",
    .version = "1.0.0",
    .dependencies = .{
        // Header-only libraries are vendored in third_party/
        // 纯头文件库放在 third_party/ 目录中
    },
    .paths = .{
        "build.zig",
        "build.zig.zon",
        "src",
        "include",
        "third_party",
    },
}
```

### Build Commands / 构建命令

```bash
# Build library and CLI / 构建库和 CLI
zig build

# Build with release optimization / 发布优化构建
zig build -Doptimize=ReleaseFast

# Run CLI / 运行 CLI
zig build run

# Cross-compile for Linux from macOS / 从 macOS 交叉编译到 Linux
zig build -Dtarget=x86_64-linux-gnu

# Cross-compile for Windows / 交叉编译到 Windows
zig build -Dtarget=x86_64-windows-msvc
```

### Library Recommendations / 库推荐

| Category / 类别 | Library / 库 | Purpose / 用途 |
|----------------|-------------|----------------|
| Linear Algebra | Eigen3 | Matrix operations / 矩阵运算 |
| FFT | FFTW, KFR | Fourier transform / 傅里叶变换 |
| Logging | spdlog | Fast logging / 快速日志 |
| JSON | nlohmann_json | JSON parsing / JSON 解析 |
| Formatting | fmt | String formatting / 字符串格式化 |
| Testing | Catch2, GoogleTest | Unit testing / 单元测试 |
| CLI | CLI11 | Command line / 命令行 |

---

## 12. Code Examples / 代码示例

### Complete Example: Signal Processing Pipeline
### 完整示例: 信号处理管道

```cpp
/**
 * @file signal_example.cpp
 * @brief EEG Signal Processing Example
 *        EEG 信号处理示例
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// ============================================
// Constants / 常量
// ============================================

/// Default sampling rate in Hz.
/// 默认采样率（赫兹）。
constexpr double DEFAULT_SAMPLE_RATE = 256.0;

/// Maximum number of channels.
/// 最大通道数。
constexpr size_t MAX_CHANNELS = 64;

// ============================================
// Error Types / 错误类型
// ============================================

/// Processing error codes.
/// 处理错误码。
enum class ErrorCode {
    Success,
    EmptyData,
    InvalidSampleRate,
    ChannelMismatch
};

// ============================================
// Data Structures / 数据结构
// ============================================

/// EEG signal buffer.
/// EEG 信号缓冲区。
struct EegBuffer {
    /// Sample data [samples].
    /// 采样数据 [采样点]。
    std::vector<double> data;
    
    /// Sampling rate in Hz.
    /// 采样率（赫兹）。
    double sample_rate = DEFAULT_SAMPLE_RATE;
    
    /// Channel name.
    /// 通道名称。
    std::string channel_name;
    
    /// Get number of samples.
    /// 获取采样点数。
    [[nodiscard]] size_t sample_count() const noexcept {
        return data.size();
    }
    
    /// Check if buffer is empty.
    /// 检查缓冲区是否为空。
    [[nodiscard]] bool empty() const noexcept {
        return data.empty();
    }
};

// ============================================
// Processing Functions / 处理函数
// ============================================

/// Calculate mean of signal.
/// 计算信号均值。
///
/// @param data Input signal data.
///             输入信号数据。
/// @return Mean value, or nullopt if data is empty.
///         均值，如果数据为空则返回 nullopt。
[[nodiscard]] std::optional<double> calculate_mean(
    const std::vector<double>& data)
{
    // Guard clause: check for empty data.
    // 守卫子句: 检查空数据。
    if (data.empty()) {
        return std::nullopt;
    }
    
    // Calculate sum using accumulate.
    // 使用 accumulate 计算总和。
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    
    return sum / static_cast<double>(data.size());
}

/// Calculate standard deviation.
/// 计算标准差。
///
/// @param data Input signal data.
///             输入信号数据。
/// @return Standard deviation, or nullopt if data is empty.
///         标准差，如果数据为空则返回 nullopt。
[[nodiscard]] std::optional<double> calculate_std(
    const std::vector<double>& data)
{
    // Get mean first.
    // 首先获取均值。
    auto mean_opt = calculate_mean(data);
    if (!mean_opt) {
        return std::nullopt;
    }
    
    double mean = *mean_opt;
    
    // Calculate variance.
    // 计算方差。
    double variance = 0.0;
    for (double value : data) {
        double diff = value - mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(data.size());
    
    return std::sqrt(variance);
}

/// Apply baseline correction.
/// 应用基线校正。
///
/// @param data Input signal data.
///             输入信号数据。
/// @param baseline_samples Number of samples to use for baseline.
///                         用于基线的采样点数。
/// @return Corrected data.
///         校正后的数据。
/// @throws std::invalid_argument if data is empty.
///         如果数据为空则抛出 std::invalid_argument。
[[nodiscard]] std::vector<double> baseline_correct(
    const std::vector<double>& data,
    size_t baseline_samples)
{
    // Validate input.
    // 验证输入。
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    
    // Calculate baseline mean.
    // 计算基线均值。
    size_t baseline_end = std::min(baseline_samples, data.size());
    double baseline_sum = std::accumulate(
        data.begin(),
        data.begin() + static_cast<std::ptrdiff_t>(baseline_end),
        0.0);
    double baseline_mean = baseline_sum / static_cast<double>(baseline_end);
    
    // Subtract baseline.
    // 减去基线。
    std::vector<double> corrected;
    corrected.reserve(data.size());
    
    for (double sample : data) {
        corrected.push_back(sample - baseline_mean);
    }
    
    return corrected;
}

/// Downsample signal by integer factor.
/// 按整数因子降采样信号。
///
/// @param data Input signal data.
///             输入信号数据。
/// @param factor Downsampling factor.
///               降采样因子。
/// @return Downsampled data.
///         降采样后的数据。
[[nodiscard]] std::vector<double> downsample(
    const std::vector<double>& data,
    size_t factor)
{
    if (factor == 0) {
        factor = 1;
    }
    
    std::vector<double> result;
    result.reserve(data.size() / factor + 1);
    
    for (size_t i = 0; i < data.size(); i += factor) {
        result.push_back(data[i]);
    }
    
    return result;
}

// ============================================
// Main Entry Point / 主入口
// ============================================

int main() {
    // Create buffer with test data.
    // 创建带测试数据的缓冲区。
    EegBuffer buffer;
    buffer.channel_name = "Fp1";
    buffer.sample_rate = DEFAULT_SAMPLE_RATE;
    
    // Generate sine wave test data.
    // 生成正弦波测试数据。
    constexpr size_t SAMPLE_COUNT = 1000;
    buffer.data.reserve(SAMPLE_COUNT);
    
    for (size_t i = 0; i < SAMPLE_COUNT; ++i) {
        double t = static_cast<double>(i) / buffer.sample_rate;
        double sample = std::sin(2.0 * M_PI * 10.0 * t);  // 10 Hz sine
        buffer.data.push_back(sample);
    }
    
    // Calculate statistics.
    // 计算统计数据。
    if (auto mean = calculate_mean(buffer.data)) {
        std::cout << "Mean: " << *mean << "\n";
    }
    
    if (auto std_dev = calculate_std(buffer.data)) {
        std::cout << "Std: " << *std_dev << "\n";
    }
    
    // Apply baseline correction.
    // 应用基线校正。
    try {
        auto corrected = baseline_correct(buffer.data, 100);
        std::cout << "Corrected samples: " << corrected.size() << "\n";
        
        // Downsample.
        // 降采样。
        auto downsampled = downsample(corrected, 2);
        std::cout << "Downsampled samples: " << downsampled.size() << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
```

---

## Quick Reference Card / 快速参考卡

```
┌─────────────────────────────────────────────────────────────────┐
│                   C++17 AI CODING CHECKLIST                     │
│                   C++17 AI 编码检查清单                          │
├─────────────────────────────────────────────────────────────────┤
│ ✅ C++17 standard          ✅ C++17 标准                         │
│ ✅ No raw new/delete       ✅ 无裸 new/delete                    │
│ ✅ Use smart pointers      ✅ 使用智能指针                        │
│ ✅ Bilingual comments      ✅ 双语注释                           │
│ ✅ [[nodiscard]] used      ✅ 使用 [[nodiscard]]                 │
│ ✅ noexcept where safe     ✅ 安全处使用 noexcept                 │
│ ✅ enum class used         ✅ 使用 enum class                    │
│ ✅ No C-style casts        ✅ 无 C 风格转换                       │
│ ✅ const by default        ✅ 默认使用 const                      │
│ ✅ Functions < 50 lines    ✅ 函数 < 50 行                       │
│ ✅ -Wall -Wextra -Werror   ✅ 启用严格警告                        │
│ ✅ No using namespace std  ✅ 无 using namespace std             │
└─────────────────────────────────────────────────────────────────┘
```

---

*Document Version: 1.0.0 / 文档版本: 1.0.0*  
*Last Updated: 2026-02-01 / 最后更新: 2026-02-01*  
*For BCIF Project / 用于 BCIF 项目*
