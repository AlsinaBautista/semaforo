// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#pragma once

#include <string>
#include <vector>

namespace semaforo {

/// @brief Configuration for a single edge node at an intersection.
///
/// Loaded from a YAML configuration file. All fields are required unless
/// noted otherwise. See configs/node_config.yaml for an example.
struct NodeConfig {
    // ── Identity ────────────────────────────────────────────────────────
    std::string node_id;          ///< Unique node identifier (e.g., "edge-001")
    std::string zone;             ///< Deployment zone/region (e.g., "zona-centro")
    std::string intersection_id;  ///< Intersection identifier (e.g., "int-001")

    // ── Video source ────────────────────────────────────────────────────
    std::string camera_uri;       ///< Camera source URI (RTSP, file path, or device)

    // ── ML models ───────────────────────────────────────────────────────
    std::string model_path;       ///< Path to detection ONNX model
    std::string policy_path;      ///< Path to policy ONNX model

    // ── MQTT ────────────────────────────────────────────────────────────
    std::string mqtt_broker;      ///< MQTT broker hostname or IP
    int mqtt_port = 1883;         ///< MQTT broker port (default: 1883)
    std::string mqtt_username;    ///< MQTT username (optional, empty = no auth)
    std::string mqtt_password;    ///< MQTT password (optional, empty = no auth)
    std::string mqtt_tls_ca;      ///< Path to CA certificate for TLS
    std::string mqtt_tls_cert;    ///< Path to client certificate for TLS
    std::string mqtt_tls_key;     ///< Path to client private key for TLS

    // ── Detection tuning ────────────────────────────────────────────────
    float conf_threshold = 0.5f;  ///< Detection confidence threshold
    float nms_threshold = 0.45f;  ///< NMS IoU threshold

    // ── Signal controller ───────────────────────────────────────────────
    std::string controller_type = "simulation"; ///< "simulation" or "gpio"

    /// Corridor main axis ("NS" or "EW") for the SafetyLayer's terminal
    /// FLASH mode: the major approach flashes yellow, the crossing approach
    /// flashes red. Optional on purpose — missing/invalid values are NOT a
    /// config error; they degrade FLASH to the safe all-red flash fallback.
    std::string major_approach;

    // ── Hardware dead-man / MMU heartbeat ───────────────────────────────
    bool heartbeat_enabled = false; ///< Drive the cabinet MMU heartbeat GPIO
    int heartbeat_gpio_pin = 17;     ///< sysfs GPIO pin for the dead-man pulse

    // ── Logging ─────────────────────────────────────────────────────────
    std::string log_level = "info"; ///< Log level: trace, debug, info, warn, error, critical

    // ── Telemetry ───────────────────────────────────────────────────────
    float telemetry_interval_s = 1.0f; ///< Telemetry publish interval in seconds
};

/// @brief Load a NodeConfig from a YAML file.
///
/// Parses the YAML file at the given path and populates a NodeConfig struct.
/// Throws std::runtime_error if the file cannot be read or required fields
/// are missing.
///
/// @param yaml_path  Filesystem path to the YAML configuration file.
/// @return Populated NodeConfig struct.
/// @throws std::runtime_error on parse error or missing required fields.
NodeConfig load_config(const std::string& yaml_path);

/// @brief Validate a NodeConfig, checking for logical consistency.
///
/// Ensures that required string fields are non-empty, port numbers are
/// in valid range, and file paths are accessible.
///
/// @param config  The configuration to validate.
/// @return true if the configuration is valid.
/// @throws std::runtime_error with details if validation fails.
bool validate_config(const NodeConfig& config);

} // namespace semaforo
