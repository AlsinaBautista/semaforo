// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#include "semaforo/mqtt_client.hpp"

#include "semaforo/telemetry_outbox.hpp"

#include <spdlog/spdlog.h>

// Paho MQTT C++ headers
#include <mqtt/async_client.h>

#include <atomic>
#include <sstream>

namespace semaforo {

// ── PIMPL implementation ────────────────────────────────────────────────────

struct MqttClient::Impl : public virtual mqtt::callback,
                           public virtual mqtt::iaction_listener {
    std::unique_ptr<mqtt::async_client> client;
    mqtt::connect_options conn_opts;
    CommandCallback command_callback;
    std::string node_id;
    // Written from the Paho callback thread (connected/connection_lost) and read
    // from the main thread (is_connected/publish_*) — must be atomic.
    std::atomic<bool> is_connected_{false};
    // Bounded outbound telemetry buffer (M-4 backpressure). Staged here so a
    // downed broker link can never let the backlog grow without limit; the
    // queue is internally synchronised and caps memory via drop-oldest.
    TelemetryOutbox outbox_;

    // ── mqtt::callback overrides ────────────────────────────────────────
    void connected(const std::string& cause) override {
        spdlog::info("MQTT connected: {}", cause);
        this->is_connected_ = true;
    }

    void connection_lost(const std::string& cause) override {
        spdlog::warn("MQTT connection lost: {}", cause);
        this->is_connected_ = false;
        // Auto-reconnect is handled by Paho
    }

    void message_arrived(mqtt::const_message_ptr msg) override {
        if (!command_callback) return;

        spdlog::debug("MQTT message on {}: {}", msg->get_topic(),
                      msg->to_string());

        Command cmd;
        cmd.type = msg->get_topic();
        cmd.payload = msg->to_string();
        command_callback(cmd);
    }

    // ── mqtt::iaction_listener overrides ────────────────────────────────
    void on_failure(const mqtt::token& tok) override {
        spdlog::error("MQTT action failed: token={}", tok.get_message_id());
    }

    void on_success(const mqtt::token& /*tok*/) override {
        // Success is expected, no logging needed
    }

    /// @brief Serialize telemetry payload to JSON string.
    static std::string telemetry_to_json(const TelemetryPayload& t) {
        std::ostringstream ss;
        ss << "{"
           << "\"node_id\":\"" << t.node_id << "\","
           << "\"intersection_id\":\"" << t.intersection_id << "\","
           << "\"phase\":\"" << phase_to_string(t.current_phase) << "\","
           << "\"total_vehicles\":" << t.queue_state.total_count << ","
           << "\"max_density\":" << t.queue_state.max_density << ","
           << "\"max_wait_s\":" << t.queue_state.max_wait_s << ","
           << "\"uptime_s\":" << t.uptime_s << ","
           << "\"fps\":" << t.fps << ","
           << "\"lanes\":[";

        for (size_t i = 0; i < t.queue_state.lanes.size(); ++i) {
            const auto& lane = t.queue_state.lanes[i];
            if (i > 0) ss << ",";
            ss << "{\"id\":\"" << lane.lane_id << "\","
               << "\"count\":" << lane.count << ","
               << "\"density\":" << lane.density << ","
               << "\"avg_wait_s\":" << lane.avg_wait_s << "}";
        }

        ss << "]}";
        return ss.str();
    }

    /// @brief Drain the bounded outbox to the broker, oldest-first.
    ///
    /// Only attempts delivery while connected. If a publish fails the message
    /// is re-buffered and draining stops; the bound still guarantees O(1)
    /// memory if the link stays down across many ticks.
    /// @return true if the outbox was fully drained, false otherwise.
    bool flush_outbox() {
        if (!is_connected_ || !client) return false;
        OutboundMessage msg;
        while (outbox_.try_pop(msg)) {
            try {
                client->publish(msg.topic, msg.payload.c_str(),
                                msg.payload.size(), msg.qos, msg.retained);
            } catch (const mqtt::exception& ex) {
                spdlog::error("MQTT publish error: {}", ex.what());
                outbox_.push(std::move(msg));  // retry next tick (drop-oldest bounds RAM)
                return false;
            }
        }
        return true;
    }
};

// ── Constructor / Destructor ────────────────────────────────────────────────

MqttClient::MqttClient() : impl_(std::make_unique<Impl>()) {}
MqttClient::~MqttClient() {
    disconnect();
}
MqttClient::MqttClient(MqttClient&&) noexcept = default;
MqttClient& MqttClient::operator=(MqttClient&&) noexcept = default;

// ── Public methods ──────────────────────────────────────────────────────────

bool MqttClient::connect(const std::string& broker,
                          int port,
                          const std::string& node_id,
                          const std::string& tls_ca,
                          const std::string& tls_cert,
                          const std::string& tls_key) {
    try {
        std::string server_uri = broker + ":" + std::to_string(port);
        std::string client_id = "semaforo_edge_" + node_id;

        impl_->node_id = node_id;
        impl_->client = std::make_unique<mqtt::async_client>(
            server_uri, client_id);

        impl_->client->set_callback(*impl_);

        auto builder = mqtt::connect_options_builder()
            .clean_session(true)
            .automatic_reconnect(true)
            .keep_alive_interval(std::chrono::seconds(20))
            .connect_timeout(std::chrono::seconds(10));
            
        if (!tls_ca.empty()) {
            mqtt::ssl_options ssl;
            ssl.set_trust_store(tls_ca);
            if (!tls_cert.empty() && !tls_key.empty()) {
                ssl.set_key_store(tls_cert);
                ssl.set_private_key(tls_key);
            }
            builder.ssl(std::move(ssl));
            spdlog::info("MQTT: TLS enabled using CA: {}", tls_ca);
        }

        impl_->conn_opts = builder.finalize();

        spdlog::info("MQTT: connecting to {} as {}", server_uri, client_id);
        auto tok = impl_->client->connect(impl_->conn_opts);
        tok->wait();
        impl_->is_connected_ = true;
        return true;

    } catch (const mqtt::exception& ex) {
        spdlog::error("MQTT connect error: {}", ex.what());
        impl_->is_connected_ = false;
        return false;
    }
}

bool MqttClient::publish_telemetry(const TelemetryPayload& telemetry,
                                    int qos) {
    std::string topic = "semaforo/telemetry/" + telemetry.node_id;
    std::string payload = Impl::telemetry_to_json(telemetry);

    // Backpressure (M-4): stage every sample into the bounded outbox first, so
    // a downed link can never grow the backlog without limit — the queue is
    // O(1) in space and silently drops the oldest telemetry on overflow. Then
    // drain as much as the link will accept this tick.
    impl_->outbox_.push({std::move(topic), std::move(payload), qos, false});
    return impl_->flush_outbox();
}

bool MqttClient::publish_alert(const std::string& node_id, const std::string& status, const std::string& reason) {
    if (!impl_->is_connected_ || !impl_->client) return false;

    try {
        std::string topic = "semaforo/status/" + node_id;
        std::string payload = "{\"status\":\"" + status + "\",\"reason\":\"" + reason + "\",\"node_id\":\"" + node_id + "\"}";

        // Alerts use QoS 1 and retained flag
        impl_->client->publish(topic, payload.c_str(), payload.size(), 1, true);
        spdlog::info("MQTT: published alert to {}: status={}", topic, status);
        return true;

    } catch (const mqtt::exception& ex) {
        spdlog::error("MQTT publish alert error: {}", ex.what());
        return false;
    }
}

bool MqttClient::subscribe_commands(const std::string& node_id,
                                     CommandCallback callback) {
    if (!impl_->is_connected_ || !impl_->client) return false;

    try {
        impl_->command_callback = std::move(callback);
        std::string topic = "semaforo/commands/" + node_id;

        impl_->client->subscribe(topic, 1);
        spdlog::info("MQTT: subscribed to {}", topic);
        return true;

    } catch (const mqtt::exception& ex) {
        spdlog::error("MQTT subscribe error: {}", ex.what());
        return false;
    }
}

bool MqttClient::set_lwt(const std::string& node_id) {
    try {
        std::string topic = "semaforo/status/" + node_id;
        std::string payload = "{\"status\":\"offline\",\"node_id\":\""
                              + node_id + "\"}";

        auto lwt = mqtt::message(topic, payload, 1, true);
        impl_->conn_opts = mqtt::connect_options_builder()
            .clean_session(true)
            .automatic_reconnect(true)
            .keep_alive_interval(std::chrono::seconds(20))
            .connect_timeout(std::chrono::seconds(10))
            .will(std::move(lwt))
            .finalize();

        spdlog::info("MQTT: LWT configured on {}", topic);
        return true;

    } catch (const mqtt::exception& ex) {
        spdlog::error("MQTT LWT error: {}", ex.what());
        return false;
    }
}

bool MqttClient::is_connected() const {
    return impl_->is_connected_ && impl_->client && impl_->client->is_connected();
}

void MqttClient::disconnect() {
    if (impl_->client && impl_->client->is_connected()) {
        try {
            // Publish online→offline status before disconnecting
            std::string topic = "semaforo/status/" + impl_->node_id;
            std::string payload = "{\"status\":\"offline\",\"node_id\":\""
                                  + impl_->node_id + "\"}";
            impl_->client->publish(topic, payload.c_str(), payload.size(), 1, true)->wait();

            impl_->client->disconnect()->wait();
            spdlog::info("MQTT: disconnected gracefully");
        } catch (const mqtt::exception& ex) {
            spdlog::error("MQTT disconnect error: {}", ex.what());
        }
    }
    impl_->is_connected_ = false;
}

} // namespace semaforo
