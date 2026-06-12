// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "semaforo/policy_engine.hpp"
#include "semaforo/queue_estimator.hpp"

namespace semaforo {

/// @brief Telemetry payload published to the MQTT broker.
struct TelemetryPayload {
    std::string node_id;          ///< Edge node identifier
    std::string intersection_id;  ///< Intersection identifier
    QueueState queue_state;       ///< Current queue state
    Phase current_phase;          ///< Active signal phase
    float uptime_s;               ///< Node uptime in seconds
    float fps;                    ///< Current processing frames per second
};

/// @brief Command received from the cloud/backend via MQTT.
struct Command {
    std::string type;      ///< Command type (e.g., "set_phase", "update_policy", "reboot")
    std::string payload;   ///< JSON payload with command parameters
    std::string reply_to;  ///< Topic to send response to (if any)
};

/// @brief Callback type for received commands.
using CommandCallback = std::function<void(const Command&)>;

/// @brief MQTT client wrapper for cloud communication using Paho MQTT C++.
///
/// Handles bidirectional communication between the edge node and the
/// cloud backend. Publishes periodic telemetry and subscribes to
/// command topics for remote control.
///
/// Supports automatic reconnection and Last Will and Testament (LWT)
/// for node liveness detection.
class MqttClient {
public:
    MqttClient();
    ~MqttClient();

    // Non-copyable, movable
    MqttClient(const MqttClient&) = delete;
    MqttClient& operator=(const MqttClient&) = delete;
    MqttClient(MqttClient&&) noexcept;
    MqttClient& operator=(MqttClient&&) noexcept;

    /// @brief Connect to the MQTT broker.
    ///
    /// Establishes a persistent connection with automatic reconnection.
    /// The client ID is derived from the node_id for uniqueness.
    ///
    /// @param broker   Broker URI (e.g., "ssl://broker.example.com:8883").
    /// @param port     Broker port (default: 8883).
    /// @param node_id  Unique node identifier used as part of the client ID.
    /// @param tls_ca   Path to CA cert.
    /// @param tls_cert Path to client cert.
    /// @param tls_key  Path to client private key.
    /// @return true if connection was established successfully.
    bool connect(const std::string& broker, int port, const std::string& node_id,
                 const std::string& tls_ca = "", const std::string& tls_cert = "", const std::string& tls_key = "");

    /// @brief Publish a telemetry update to the broker.
    ///
    /// Serializes the telemetry payload to JSON and publishes it to the
    /// topic: semaforo/telemetry/{node_id}
    ///
    /// @param telemetry  The telemetry data to publish.
    /// @param qos        MQTT QoS level (0=at most once, 1=at least once, 2=exactly once).
    /// @return true if the message was published successfully.
    bool publish_telemetry(const TelemetryPayload& telemetry, int qos = 1);

    /// @brief Publish an alert status message to the broker.
    ///
    /// Publishes to: semaforo/status/{node_id}
    ///
    /// @param node_id  Node identifier
    /// @param status   Status string (e.g. "online", "offline")
    /// @param reason   Reason for the alert
    /// @return true if the message was published successfully.
    bool publish_alert(const std::string& node_id, const std::string& status, const std::string& reason);

    /// @brief Subscribe to command topics for this node.
    ///
    /// Subscribes to: semaforo/commands/{node_id}
    /// Received commands are dispatched to the registered callback.
    ///
    /// @param node_id   Node identifier to subscribe commands for.
    /// @param callback  Function to call when a command is received.
    /// @return true if subscription was established successfully.
    bool subscribe_commands(const std::string& node_id, CommandCallback callback);

    /// @brief Set the Last Will and Testament message.
    ///
    /// Configures a message that the broker will publish on behalf of this
    /// client if the connection is lost unexpectedly. Published to:
    /// semaforo/status/{node_id}
    ///
    /// @param node_id  Node identifier for the LWT topic.
    /// @return true if LWT was configured successfully.
    bool set_lwt(const std::string& node_id);

    /// @brief Check if the client is currently connected to the broker.
    /// @return true if connected.
    bool is_connected() const;

    /// @brief Disconnect from the broker gracefully.
    void disconnect();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_; ///< PIMPL for Paho MQTT internals
};

} // namespace semaforo
