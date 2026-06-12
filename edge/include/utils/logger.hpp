// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#pragma once

#include <memory>
#include <string>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

namespace semaforo::utils {

/// @brief Initialize the project-wide logger.
///
/// Creates a multi-sink logger that outputs to both stdout (with color)
/// and a rotating log file. Should be called once at application startup.
///
/// @param name       Logger name (typically "semaforo").
/// @param level      Log level string: "trace", "debug", "info", "warn", "error", "critical".
/// @param log_file   Path to the log file (default: "semaforo_edge.log").
inline void init_logger(const std::string& name = "semaforo",
                        const std::string& level = "info",
                        const std::string& log_file = "semaforo_edge.log") {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");

    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        log_file, 1024 * 1024 * 10, 3); // 10MB, 3 rotated files
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] [%t] %v");

    auto logger = std::make_shared<spdlog::logger>(
        name, spdlog::sinks_init_list{console_sink, file_sink});

    logger->set_level(spdlog::level::from_str(level));
    logger->flush_on(spdlog::level::warn);

    spdlog::set_default_logger(logger);
    spdlog::info("Logger initialized: level={}, file={}", level, log_file);
}

/// @brief Get the project logger instance.
/// @return Shared pointer to the default spdlog logger.
inline std::shared_ptr<spdlog::logger> get_logger() {
    return spdlog::default_logger();
}

/// @brief Convenience macros for logging (optional, can also use spdlog directly).
#define LOG_TRACE(...)    SPDLOG_TRACE(__VA_ARGS__)
#define LOG_DEBUG(...)    SPDLOG_DEBUG(__VA_ARGS__)
#define LOG_INFO(...)     SPDLOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)     SPDLOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...)    SPDLOG_ERROR(__VA_ARGS__)
#define LOG_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)

} // namespace semaforo::utils
