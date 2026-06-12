// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

#include <spdlog/spdlog.h>

#include "semaforo/command_latch.hpp"
#include "semaforo/command_parser.hpp"
#include "semaforo/command_queue.hpp"
#include "semaforo/config.hpp"
#include "semaforo/detector.hpp"
#include "semaforo/heartbeat.hpp"
#include "semaforo/mqtt_client.hpp"
#include "semaforo/object_pool.hpp"
#include "semaforo/pipeline.hpp"
#include "semaforo/policy_engine.hpp"
#include "semaforo/queue_estimator.hpp"
#include "semaforo/safety_layer.hpp"
#include "semaforo/signal_controller.hpp"
#include "semaforo/tracker.hpp"
#include "utils/logger.hpp"
#include "utils/timer.hpp"

namespace {

/// Global flag for graceful shutdown via signal handlers.
std::atomic<bool> g_running{true};

/// @brief Signal handler for SIGINT and SIGTERM.
void signal_handler(int signum) {
    spdlog::warn("Received signal {} — initiating graceful shutdown...", signum);
    g_running.store(false);
}

/// @brief Install signal handlers for graceful shutdown.
void install_signal_handlers() {
    struct sigaction sa{};
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}

/// @brief Print usage information.
void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <config.yaml>\n"
              << "\n"
              << "  Semáforo Inteligente - Edge Perception & Control Module\n"
              << "  Runs the real-time traffic detection, queue estimation,\n"
              << "  and adaptive signal control pipeline.\n"
              << "\n"
              << "Arguments:\n"
              << "  config.yaml   Path to the node configuration file.\n";
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    // ── Parse command line ──────────────────────────────────────────────
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string config_path = argv[1];

    // ── Load configuration ──────────────────────────────────────────────
    semaforo::NodeConfig config;
    try {
        config = semaforo::load_config(config_path);
        semaforo::validate_config(config);
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: Failed to load config: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    // ── Initialize logger ───────────────────────────────────────────────
    semaforo::utils::init_logger("semaforo", config.log_level);
    spdlog::info("=== Semáforo Inteligente - Edge Module ===");
    spdlog::info("Node: {}  Intersection: {}  Zone: {}",
                 config.node_id, config.intersection_id, config.zone);

    // ── Install signal handlers ─────────────────────────────────────────
    install_signal_handlers();

    // ── Initialize components ───────────────────────────────────────────
    semaforo::VideoPipeline pipeline;
    if (!pipeline.init(config.camera_uri)) {
        spdlog::critical("Failed to open video source: {}", config.camera_uri);
        return EXIT_FAILURE;
    }
    spdlog::info("Video pipeline opened: {} @ {:.1f} FPS",
                 config.camera_uri, pipeline.get_fps());

    semaforo::Detector detector;
    if (!detector.init(config.model_path, config.conf_threshold, config.nms_threshold)) {
        spdlog::critical("Failed to load detection model: {}", config.model_path);
        return EXIT_FAILURE;
    }
    spdlog::info("Detector initialized: {}", config.model_path);

    semaforo::VehicleTracker tracker;
    semaforo::QueueEstimator queue_estimator;
    semaforo::LaneConfig lane_config; // TODO: load from config file

    // Per-frame perception buffers, allocated ONCE and recycled every tick so
    // the 30 FPS loop performs no heap allocation in steady state (audit M-5).
    constexpr std::size_t kMaxDetectionsPerFrame = 512;
    constexpr std::size_t kMaxTracksPerFrame = 256;
    cv::Mat frame;                                              // reused capture buffer
    semaforo::ObjectPool<semaforo::Detection> det_pool(kMaxDetectionsPerFrame);
    semaforo::ObjectPool<semaforo::TrackedObject> track_pool(kMaxTracksPerFrame);
    semaforo::QueueState queue_state;                          // reused queue-state buffer

    semaforo::PolicyEngine policy;
    if (!policy.init(config.policy_path)) {
        spdlog::warn("Policy model not loaded, using fallback heuristic");
    } else {
        spdlog::info("Policy engine initialized: {}", config.policy_path);
    }

    semaforo::SafetyLayer safety;

    // Hardened command ingestion: the parser rejects any malformed/stale/spoofed
    // payload, and the guard decides when the Brain can no longer be trusted and
    // the SafetyLayer must drop to a deterministic fixed cycle (and when to
    // resume) using hysteresis so transient glitches do not flap the signal.
    semaforo::CommandParser cmd_parser(/*max_age_ms=*/2000, /*max_future_skew_ms=*/1000);
    semaforo::CommandGuard  cmd_guard;
    // SPSC hand-off: the MQTT callback (producer) only parses + enqueues; the
    // main loop (single consumer/writer) owns the guard, SafetyLayer and actuator.
    semaforo::CommandQueue  cmd_queue;
    // Latched remote directive: a valid Brain command stays the primary control
    // input across ticks until superseded or until its TTL (5s) lapses, at which
    // point control degrades to the local policy.
    semaforo::DirectiveLatch directive_latch(semaforo::DirectiveLatch::DEFAULT_TTL_MS);

    auto signal_ctrl = semaforo::create_signal_controller(config.controller_type);
    if (!signal_ctrl) {
        spdlog::critical("Failed to create signal controller: {}", config.controller_type);
        return EXIT_FAILURE;
    }
    spdlog::info("Signal controller ready: {}", config.controller_type);

    // ── Hardware dead-man / MMU heartbeat ───────────────────────────────
    // The *hardware* fail-safe, independent of every software interlock: a
    // square wave on a GPIO pin that the cabinet MMU watches. It is kicked
    // INLINE from the hot loop below (never a separate thread), so if this
    // thread freezes, dead-locks, or segfaults the pulse train mathematically
    // stops and the MMU opens the control relay → flashing red, in hardware.
    std::optional<semaforo::Heartbeat> heartbeat;
    if (config.heartbeat_enabled) {
        auto line = semaforo::make_gpio_line(config.heartbeat_gpio_pin);
        if (line->ok()) {
            spdlog::info("MMU heartbeat armed on GPIO {} ({} ms half-period)",
                         config.heartbeat_gpio_pin,
                         semaforo::Heartbeat::DEFAULT_PERIOD.count());
        } else {
            spdlog::error("Heartbeat GPIO {} unavailable — hardware MMU dead-man "
                          "INACTIVE (software interlocks still enforced).",
                          config.heartbeat_gpio_pin);
        }
        heartbeat.emplace(std::move(line));
    } else {
        spdlog::warn("MMU heartbeat disabled in config — no hardware dead-man.");
    }

    semaforo::MqttClient mqtt;
    mqtt.set_lwt(config.node_id);
    if (!mqtt.connect(config.mqtt_broker, config.mqtt_port, config.node_id,
                      config.mqtt_tls_ca, config.mqtt_tls_cert, config.mqtt_tls_key)) {
        spdlog::error("Failed to connect to MQTT broker: {}:{}",
                      config.mqtt_broker, config.mqtt_port);
        // Continue without MQTT — local operation only
    } else {
        spdlog::info("MQTT connected: {}:{}", config.mqtt_broker, config.mqtt_port);
        // Arm the command watchdog so the silence timeout starts counting now
        // that we are online and expecting Brain commands.
        cmd_guard.arm(semaforo::CommandParser::now_ms());

        mqtt.subscribe_commands(config.node_id, [&](const semaforo::Command& cmd) {
            // PRODUCER ONLY. This runs on the Paho callback thread. Its sole job
            // is to parse the untrusted payload (never throws) and hand it to the
            // main loop. It must NOT touch the SafetyLayer, the guard, or the
            // actuator — that is the single writer's exclusive responsibility.
            const std::int64_t now_ms = semaforo::CommandParser::now_ms();
            semaforo::ParsedCommand parsed = cmd_parser.parse(cmd.payload, now_ms);
            cmd_queue.push(parsed);
            spdlog::debug("MQTT cmd queued [{}] (qsize~{})",
                          semaforo::command_status_to_string(parsed.status),
                          cmd_queue.size());
        });
    }

    // ── Main processing loop ────────────────────────────────────────────
    spdlog::info("Entering main loop...");
    semaforo::utils::Timer uptime_timer;
    semaforo::utils::Timer telemetry_timer;
    semaforo::utils::Timer frame_timer;
    semaforo::utils::Timer last_frame_timer; // Watchdog timer
    uint64_t frame_count = 0;
    float fps = 0.0f;
    bool is_camera_online = true;        // Tracks global camera status
    bool inference_online = true;        // E-1: tracks ONNX inference health
    bool prev_directive_active = false;  // Edge-trigger for latch acquire/stale logs

    // Single-writer helpers: only the main loop ever drives the SafetyLayer's
    // emergency state and publishes status, so these are race-free by design.
    auto exit_fallback_if_safe = [&]() {
        if (is_camera_online && inference_online) {
            spdlog::info("BRAIN STABILISED. Releasing AI control from fixed cycle.");
            safety.set_emergency_mode(false);
            if (mqtt.is_connected()) {
                mqtt.publish_alert(config.node_id, "online", "ai_restored");
            }
        } else {
            spdlog::info("BRAIN STABILISED but camera/inference still down — "
                         "holding fixed cycle.");
        }
    };

    while (g_running.load()) {
        frame_timer.reset();
        const std::int64_t tick_now_ms = semaforo::CommandParser::now_ms();

        // 0. SINGLE-WRITER command processing. Drain everything the producer
        //    enqueued, feed each result to the freshness/fallback guard, and
        //    latch the most recent *actionable* phase as the primary directive.
        //    All SafetyLayer and actuator writes happen here, on this one thread.
        {
            semaforo::ParsedCommand parsed;
            while (cmd_queue.try_pop(parsed)) {
                auto decision = cmd_guard.on_command(parsed, tick_now_ms);
                if (decision == semaforo::GuardDecision::ENTER_FALLBACK) {
                    spdlog::critical("BRAIN UNTRUSTWORTHY (last reject: {}). Blocking "
                                     "AI, engaging deterministic fixed cycle.",
                                     semaforo::command_status_to_string(parsed.status));
                    safety.set_emergency_mode(true);
                    if (mqtt.is_connected()) {
                        mqtt.publish_alert(config.node_id, "degraded",
                                           "ai_blocked_fixed_cycle");
                    }
                } else if (decision == semaforo::GuardDecision::EXIT_FALLBACK) {
                    exit_fallback_if_safe();
                }

                if (parsed.is_actionable()) {
                    // Idempotency gate (A-2): drop replayed/out-of-order
                    // sequences before they can be re-applied to the hardware.
                    if (cmd_guard.accept_sequence(parsed)) {
                        // Latch the directive and (re)start its TTL window.
                        directive_latch.update(parsed.phase, tick_now_ms);
                    } else {
                        spdlog::warn("Discarded replayed command seq={} "
                                     "(<= last applied {})",
                                     parsed.sequence_number,
                                     cmd_guard.last_applied_seq());
                    }
                } else {
                    spdlog::warn("Rejected command [{}]: {}",
                                 semaforo::command_status_to_string(parsed.status),
                                 parsed.detail);
                }
            }

            // Command-silence watchdog: total silence (broker down, link
            // saturated, Brain crashed) is invisible to the callback; only the
            // loop can detect it. If commands stop arriving, drop to fixed cycle.
            if (mqtt.is_connected()) {
                if (cmd_guard.tick(tick_now_ms) == semaforo::GuardDecision::ENTER_FALLBACK) {
                    spdlog::critical("COMMAND SILENCE TIMEOUT. Brain unreachable — "
                                     "engaging deterministic fixed cycle.");
                    safety.set_emergency_mode(true);
                    mqtt.publish_alert(config.node_id, "degraded",
                                       "command_silence_timeout");
                }
            }
        }

        // 1. Acquire frame into the hoisted, recycled buffer (M-5).
        const bool have_frame = pipeline.read_frame(frame);
        if (!have_frame) {
            if (last_frame_timer.elapsed_s() > 3.0) {
                if (is_camera_online) {
                    spdlog::critical("CAMERA TIMEOUT (>3s). Entering EMERGENCY MODE.");
                    is_camera_online = false;
                    safety.set_emergency_mode(true);

                    if (mqtt.is_connected()) {
                        mqtt.publish_alert(config.node_id, "offline", "camera_timeout");
                    }
                }
                
                if (!pipeline.is_reconnecting()) {
                    pipeline.trigger_reconnect_async();
                }
            }
            
            // Allow the loop to continue to let SafetyLayer process the fixed cycle
            // but we skip detection/tracking
        } else {
            // Frame received!
            if (!is_camera_online) {
                is_camera_online = true;
                // Only leave emergency mode if the Brain is also trusted AND
                // inference is healthy; any other active fallback source (a
                // command-silence/parse fallback, or an inference fault) must
                // keep the fixed cycle.
                if (!cmd_guard.in_fallback() && inference_online) {
                    spdlog::info("CAMERA RECOVERED. Exiting EMERGENCY MODE.");
                    safety.set_emergency_mode(false);
                    if (mqtt.is_connected()) {
                        mqtt.publish_alert(config.node_id, "online", "camera_recovered");
                    }
                } else {
                    spdlog::info("CAMERA RECOVERED but Brain still untrusted — "
                                 "holding fixed cycle.");
                }
            }
            last_frame_timer.reset();
        }
        
        // 2. Run detection into the recycled pool (only if online).
        //    E-1: detect() never throws; a false return signals an inference
        //    fault (ONNX threw, or the model emitted a malformed tensor). On a
        //    fault we force the deterministic emergency ring rather than driving
        //    the signal on garbage perception. Edge-triggered, mirroring camera.
        det_pool.clear();
        if (is_camera_online) {
            const bool inference_ok = detector.detect(frame, det_pool);
            if (!inference_ok) {
                if (inference_online) {
                    spdlog::critical("INFERENCE FAULT (ONNX exception / malformed "
                                     "model output). Forcing EMERGENCY MODE — "
                                     "process continues, signal fails safe.");
                    inference_online = false;
                    safety.set_emergency_mode(true);
                    if (mqtt.is_connected()) {
                        mqtt.publish_alert(config.node_id, "degraded",
                                           "inference_fault");
                    }
                }
            } else if (!inference_online) {
                inference_online = true;
                // Only leave emergency if the camera is up AND the Brain is
                // trusted; another active fallback source must keep the cycle.
                if (is_camera_online && !cmd_guard.in_fallback()) {
                    spdlog::info("INFERENCE RECOVERED. Exiting EMERGENCY MODE.");
                    safety.set_emergency_mode(false);
                    if (mqtt.is_connected()) {
                        mqtt.publish_alert(config.node_id, "online",
                                           "inference_restored");
                    }
                } else {
                    spdlog::info("INFERENCE RECOVERED but camera/Brain still "
                                 "degraded — holding fixed cycle.");
                }
            }
        }

        // 3. Update tracker into the recycled pool (clears over time if offline).
        tracker.update(det_pool.view(), track_pool);

        // 4. Estimate queue state into the hoisted, reused buffer (M-5).
        queue_estimator.estimate(track_pool.view(), lane_config, queue_state);

        // 5. Decide the requested phase (single source of desire this tick).
        //    The latched remote directive is the PRIMARY control input: it
        //    persists across ticks (the Brain need not re-send it). It is used
        //    only while AI-trusted AND within its TTL; once the latch goes stale
        //    (>5s with no Brain update) the node degrades to the local policy.
        const std::optional<semaforo::Phase> directive =
            (is_camera_online && !cmd_guard.in_fallback())
                ? directive_latch.get(tick_now_ms)
                : std::nullopt;

        // Edge-triggered logging of latch acquisition / staleness degradation.
        const bool directive_active = directive.has_value();
        if (directive_active != prev_directive_active) {
            if (directive_active) {
                spdlog::info("Remote directive LATCHED as primary control");
            } else if (directive_latch.has_directive()) {
                spdlog::warn("Remote directive STALE (>{}ms without Brain update) — "
                             "degrading to local policy",
                             directive_latch.ttl_ms());
            }
            prev_directive_active = directive_active;
        }

        semaforo::Phase proposed_phase;
        if (!is_camera_online) {
            proposed_phase = semaforo::Phase::GREEN_NS;  // ignored by emergency ring
        } else if (directive) {
            proposed_phase = *directive;                 // latched primary directive
        } else {
            proposed_phase = policy.decide(queue_state); // degraded / local control
        }

        // 6. Safety validation. The SafetyLayer is the single source of truth:
        //    it self-times min/max-green, amber and all-red clearance, so the
        //    proposed phase is just a *request*. It returns the phase to drive.
        auto validated = safety.validate(proposed_phase);

        if (validated.was_modified) {
            spdlog::debug("Safety override: requested {} -> applying {} ({})",
                          semaforo::phase_to_string(proposed_phase),
                          semaforo::phase_to_string(validated.phase),
                          validated.reason);
        }

        // 7. Apply phase
        signal_ctrl->set_phase(validated.phase);

        // 7b. Kick the hardware dead-man. INLINE on the hot path BY DESIGN:
        //     this is the only thing that toggles the MMU heartbeat pin, so a
        //     stall/segfault here stops the pulse and the MMU forces flashing
        //     red. Gated on inference_online — an inference fault also drops the
        //     pulse, tripping the relay independently of the SafetyLayer.
        if (heartbeat) {
            heartbeat->tick(inference_online);
        }

        // 8. Compute FPS
        frame_count++;
        double frame_ms = frame_timer.elapsed_ms();
        fps = static_cast<float>(1000.0 / std::max(frame_ms, 0.001));

        // 9. Publish telemetry at configured interval
        if (telemetry_timer.elapsed_s() >= config.telemetry_interval_s) {
            telemetry_timer.reset();

            if (mqtt.is_connected()) {
                semaforo::TelemetryPayload telemetry;
                telemetry.node_id = config.node_id;
                telemetry.intersection_id = config.intersection_id;
                telemetry.queue_state = queue_state;
                telemetry.current_phase = validated.phase;
                telemetry.uptime_s = static_cast<float>(uptime_timer.elapsed_s());
                telemetry.fps = fps;

                mqtt.publish_telemetry(telemetry);
            }

            spdlog::info("Frame #{}: detections={}, tracked={}, phase={}, fps={:.1f}",
                         frame_count, det_pool.size(), track_pool.size(),
                         semaforo::phase_to_string(validated.phase), fps);
        }
    }

    // ── Graceful shutdown ───────────────────────────────────────────────
    spdlog::info("Shutting down...");
    // Drop the dead-man pulse first: the cabinet MMU sees the train stop and
    // independently forces flashing red, matching the software emergency flash.
    if (heartbeat) {
        heartbeat->stop();
    }
    signal_ctrl->set_emergency_flash();
    mqtt.disconnect();
    pipeline.release();

    spdlog::info("Semáforo Inteligente edge module stopped. Total frames: {}, Uptime: {:.1f}s",
                 frame_count, uptime_timer.elapsed_s());
    return EXIT_SUCCESS;
}
