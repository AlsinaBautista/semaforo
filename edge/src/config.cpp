// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#include "semaforo/config.hpp"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <filesystem>

namespace semaforo {

NodeConfig load_config(const std::string& yaml_path) {
    if (!std::filesystem::exists(yaml_path)) {
        throw std::runtime_error("Configuration file not found: " + yaml_path);
    }

    YAML::Node yaml;
    try {
        yaml = YAML::LoadFile(yaml_path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error parsing YAML file: " + std::string(e.what()));
    }

    NodeConfig config;

    // Helper macro to read required fields
    auto read_required_string = [&](const std::string& key) {
        if (!yaml[key] || !yaml[key].IsScalar()) {
            throw std::runtime_error("Missing required configuration field: " + key);
        }
        return yaml[key].as<std::string>();
    };

    // Identity
    config.node_id = read_required_string("node_id");
    config.zone = yaml["zone"] ? yaml["zone"].as<std::string>() : "default";
    config.intersection_id = read_required_string("intersection_id");

    // Video source
    config.camera_uri = read_required_string("camera_uri");

    // ML models
    config.model_path = read_required_string("model_path");
    config.policy_path = yaml["policy_path"] ? yaml["policy_path"].as<std::string>() : "";

    // MQTT
    config.mqtt_broker = yaml["mqtt_broker"].as<std::string>("tcp://localhost");
    config.mqtt_port = yaml["mqtt_port"] ? yaml["mqtt_port"].as<int>() : 1883;
    config.mqtt_username = yaml["mqtt_username"] ? yaml["mqtt_username"].as<std::string>() : "";
    config.mqtt_password = yaml["mqtt_password"] ? yaml["mqtt_password"].as<std::string>() : "";
    config.mqtt_tls_ca = yaml["mqtt_tls_ca"] ? yaml["mqtt_tls_ca"].as<std::string>() : "";
    config.mqtt_tls_cert = yaml["mqtt_tls_cert"] ? yaml["mqtt_tls_cert"].as<std::string>() : "";
    config.mqtt_tls_key = yaml["mqtt_tls_key"] ? yaml["mqtt_tls_key"].as<std::string>() : "";

    // Detection tuning
    if (yaml["conf_threshold"]) config.conf_threshold = yaml["conf_threshold"].as<float>();
    if (yaml["nms_threshold"]) config.nms_threshold = yaml["nms_threshold"].as<float>();

    // Signal controller
    if (yaml["controller_type"]) config.controller_type = yaml["controller_type"].as<std::string>();
    // Optional and deliberately not validated as an error: an absent or
    // unparsable major_approach must degrade the terminal FLASH mode to the
    // all-red flash fallback, never abort the node.
    if (yaml["major_approach"]) config.major_approach = yaml["major_approach"].as<std::string>();

    // Hardware dead-man / MMU heartbeat
    if (yaml["heartbeat_enabled"]) config.heartbeat_enabled = yaml["heartbeat_enabled"].as<bool>();
    if (yaml["heartbeat_gpio_pin"]) config.heartbeat_gpio_pin = yaml["heartbeat_gpio_pin"].as<int>();

    // Logging & Telemetry
    if (yaml["log_level"]) config.log_level = yaml["log_level"].as<std::string>();
    if (yaml["telemetry_interval_s"]) config.telemetry_interval_s = yaml["telemetry_interval_s"].as<float>();

    return config;
}

bool validate_config(const NodeConfig& config) {
    if (config.node_id.empty()) throw std::runtime_error("node_id cannot be empty");
    if (config.intersection_id.empty()) throw std::runtime_error("intersection_id cannot be empty");
    if (config.camera_uri.empty()) throw std::runtime_error("camera_uri cannot be empty");
    if (config.model_path.empty()) throw std::runtime_error("model_path cannot be empty");
    if (config.mqtt_broker.empty()) throw std::runtime_error("mqtt_broker cannot be empty");

    if (config.mqtt_port <= 0 || config.mqtt_port > 65535) {
        throw std::runtime_error("Invalid mqtt_port: " + std::to_string(config.mqtt_port));
    }

    if (config.conf_threshold < 0.0f || config.conf_threshold > 1.0f) {
        throw std::runtime_error("conf_threshold must be between 0 and 1");
    }
    
    if (config.nms_threshold < 0.0f || config.nms_threshold > 1.0f) {
        throw std::runtime_error("nms_threshold must be between 0 and 1");
    }

    return true;
}

} // namespace semaforo
