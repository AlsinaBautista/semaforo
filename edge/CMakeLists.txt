# Semáforo Inteligente - Edge Perception Module
# CMake build configuration

cmake_minimum_required(VERSION 3.20)
project(semaforo_edge
    VERSION 0.1.0
    DESCRIPTION "Semáforo Inteligente - Edge Perception & Control Module"
    LANGUAGES CXX
)

# ── C++17 Standard ───────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Compiler warnings ───────────────────────────────────────────────────────
add_compile_options(
    -Wall -Wextra -Wpedantic -Werror
    $<$<CONFIG:Debug>:-g> $<$<CONFIG:Debug>:-O0>
    $<$<CONFIG:Release>:-O2> $<$<CONFIG:Release>:-DNDEBUG>
)

# ── Conan integration (if available) ────────────────────────────────────────
if(EXISTS "${CMAKE_BINARY_DIR}/conanbuildinfo.cmake")
    include("${CMAKE_BINARY_DIR}/conanbuildinfo.cmake")
    conan_basic_setup(TARGETS)
endif()

# ── Find required packages ──────────────────────────────────────────────────
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc highgui videoio dnn)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
find_package(onnxruntime CONFIG QUIET)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

include(FetchContent)

FetchContent_Declare(
    paho_mqtt_cpp
    GIT_REPOSITORY https://github.com/eclipse/paho.mqtt.cpp.git
    GIT_TAG v1.4.0
)
set(PAHO_BUILD_SHARED ON CACHE BOOL "" FORCE)
set(PAHO_BUILD_STATIC OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(paho_mqtt_cpp)

# nlohmann/json — header-only, used by the hardened command parser to safely
# validate untrusted MQTT payloads from the Brain/Coordinator.
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

# ── Resolve OnnxRuntime → imported target ───────────────────────────────────
# The neural inference paths in detector.cpp / policy_engine.cpp are guarded by
# the ONNXRUNTIME_AVAILABLE macro. If we do not both (a) define that macro and
# (b) link the runtime, every Ort:: path is compiled out and the local ONNX
# policy is silently dead code. Prefer the CMake config package (exports the
# onnxruntime::onnxruntime imported target, which carries its own include dirs
# and library path); fall back to pkg-config and synthesize the same target so
# the rest of this file links against one canonical name.
if(NOT TARGET onnxruntime::onnxruntime)
    pkg_check_modules(ONNXRUNTIME QUIET IMPORTED_TARGET libonnxruntime)
    if(TARGET PkgConfig::ONNXRUNTIME)
        add_library(onnxruntime::onnxruntime ALIAS PkgConfig::ONNXRUNTIME)
    endif()
endif()

if(TARGET onnxruntime::onnxruntime)
    add_compile_definitions(ONNXRUNTIME_AVAILABLE)
    message(STATUS "OnnxRuntime found — neural inference ENABLED (ONNXRUNTIME_AVAILABLE)")
else()
    message(WARNING "OnnxRuntime not found via cmake config or pkg-config. "
                    "Neural inference is compiled out; the Edge node will run "
                    "fail-safe (fixed-cycle / emergency ring) only. "
                    "Set onnxruntime_DIR to enable inference.")
endif()

# ── Source files ─────────────────────────────────────────────────────────────
set(SEMAFORO_SOURCES
    src/main.cpp
    src/detector.cpp
    src/queue_estimator.cpp
    src/policy_engine.cpp
    src/safety_layer.cpp
    src/mqtt_client.cpp
    src/pipeline.cpp
    src/config.cpp
    src/tracker.cpp
    src/signal_controller.cpp
)

# ── Main executable ─────────────────────────────────────────────────────────
add_executable(semaforo_edge ${SEMAFORO_SOURCES})

target_include_directories(semaforo_edge
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${GSTREAMER_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(semaforo_edge
    PRIVATE
        ${OpenCV_LIBS}
        PkgConfig::GSTREAMER
        $<TARGET_NAME_IF_EXISTS:onnxruntime::onnxruntime>
        paho-mqttpp3
        spdlog::spdlog
        yaml-cpp::yaml-cpp
        nlohmann_json::nlohmann_json
        pthread
)

# ── Tests ────────────────────────────────────────────────────────────────────
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()
    find_package(GTest 1.14 REQUIRED)

    # signal_controller.cpp is compiled in so the F-1 no-double-yellow
    # invariant is asserted against the REAL phase→lamp mapping, not a copy.
    add_executable(test_safety_layer
        tests/test_safety_layer.cpp
        src/safety_layer.cpp
        src/policy_engine.cpp
        src/signal_controller.cpp
    )

    target_include_directories(test_safety_layer
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${OpenCV_INCLUDE_DIRS}
            ${ONNXRUNTIME_INCLUDE_DIRS}
    )

    target_link_libraries(test_safety_layer
        PRIVATE
            ${OpenCV_LIBS}
            $<TARGET_NAME_IF_EXISTS:onnxruntime::onnxruntime>
            GTest::gtest
            GTest::gtest_main
            spdlog::spdlog
    )

    include(GoogleTest)
    gtest_discover_tests(test_safety_layer)

    # Hardened command-parser / fallback-guard tests (chaos resilience).
    add_executable(test_command_parser
        tests/test_command_parser.cpp
        src/policy_engine.cpp
    )

    target_include_directories(test_command_parser
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${OpenCV_INCLUDE_DIRS}
            ${ONNXRUNTIME_INCLUDE_DIRS}
    )

    target_link_libraries(test_command_parser
        PRIVATE
            ${OpenCV_LIBS}
            $<TARGET_NAME_IF_EXISTS:onnxruntime::onnxruntime>
            nlohmann_json::nlohmann_json
            GTest::gtest
            GTest::gtest_main
            spdlog::spdlog
    )

    gtest_discover_tests(test_command_parser)

    # TTL-latched remote directive tests.
    add_executable(test_command_latch
        tests/test_command_latch.cpp
        src/policy_engine.cpp
    )

    target_include_directories(test_command_latch
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${OpenCV_INCLUDE_DIRS}
            ${ONNXRUNTIME_INCLUDE_DIRS}
    )

    target_link_libraries(test_command_latch
        PRIVATE
            ${OpenCV_LIBS}
            $<TARGET_NAME_IF_EXISTS:onnxruntime::onnxruntime>
            GTest::gtest
            GTest::gtest_main
            spdlog::spdlog
    )

    gtest_discover_tests(test_command_latch)

    # Hardware dead-man / MMU heartbeat tests: the 500 ms GPIO pulse train is
    # emitted ONLY while inference is online, ceases when the hot loop freezes,
    # and de-asserts on an inference fault — validated against a mock GPIO line
    # with an injected clock (no real hardware, no sleeping). Header-only unit.
    add_executable(test_heartbeat
        tests/test_heartbeat.cpp
    )

    target_include_directories(test_heartbeat
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(test_heartbeat
        PRIVATE
            GTest::gtest
            GTest::gtest_main
    )

    gtest_discover_tests(test_heartbeat)

    # Bounded-outbox backpressure tests (M-4): outbound telemetry memory stays
    # O(1) under a downed link, with drop-oldest eviction.
    add_executable(test_telemetry_outbox
        tests/test_telemetry_outbox.cpp
    )

    target_include_directories(test_telemetry_outbox
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(test_telemetry_outbox
        PRIVATE
            GTest::gtest
            GTest::gtest_main
    )

    gtest_discover_tests(test_telemetry_outbox)

    # SORT tracker tests (M-3): Kalman-coasted ID persistence under occlusion
    # and optimal (Hungarian) association.
    add_executable(test_tracker
        tests/test_tracker.cpp
        src/tracker.cpp
    )

    target_include_directories(test_tracker
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(test_tracker
        PRIVATE
            ${OpenCV_LIBS}
            GTest::gtest
            GTest::gtest_main
    )

    gtest_discover_tests(test_tracker)

    # Thermal-throttling resilience (dynamic Δt): when the pipeline drops from
    # 30 FPS to 10 FPS, the SORT motion model must coast each track by the real
    # elapsed time (predict(dt)) so IDs survive the ~100 ms inter-frame gaps.
    add_executable(test_thermal_drops
        tests/test_thermal_drops.cpp
        src/tracker.cpp
    )

    target_include_directories(test_thermal_drops
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(test_thermal_drops
        PRIVATE
            ${OpenCV_LIBS}
            GTest::gtest
            GTest::gtest_main
    )

    gtest_discover_tests(test_thermal_drops)

    # Zero-allocation proofs for the 30 FPS inference loop (M-5): the ObjectPool,
    # preallocated preprocessing buffers (cv::Mat + tensor), and the tracker must
    # not touch the heap in steady state. A program-wide operator new/delete
    # counter asserts the allocation count stays constant across 10,000 frames,
    # and cv::Mat .data pointers are asserted byte-stable (buffers reused, not
    # reallocated). Exercises the real detector/tracker code paths.
    add_executable(test_inference_pool
        tests/test_inference_pool.cpp
        src/detector.cpp
        src/tracker.cpp
        src/queue_estimator.cpp
    )

    target_include_directories(test_inference_pool
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${OpenCV_INCLUDE_DIRS}
            ${ONNXRUNTIME_INCLUDE_DIRS}
    )

    target_link_libraries(test_inference_pool
        PRIVATE
            ${OpenCV_LIBS}
            $<TARGET_NAME_IF_EXISTS:onnxruntime::onnxruntime>
            GTest::gtest
            GTest::gtest_main
            spdlog::spdlog
    )

    gtest_discover_tests(test_inference_pool)

    # YOLO decode hardening proofs (E-1/E-2): a corrupt/unexpected model output
    # tensor (shape inconsistent with the buffer) must be rejected without an
    # out-of-bounds read, and an inference fault must drive the SafetyLayer's
    # deterministic emergency ring — never a crash, never an unsafe phase.
    add_executable(test_detector_decode
        tests/test_detector_decode.cpp
        src/detector.cpp
        src/safety_layer.cpp
        src/policy_engine.cpp
    )

    target_include_directories(test_detector_decode
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${OpenCV_INCLUDE_DIRS}
            ${ONNXRUNTIME_INCLUDE_DIRS}
    )

    target_link_libraries(test_detector_decode
        PRIVATE
            ${OpenCV_LIBS}
            $<TARGET_NAME_IF_EXISTS:onnxruntime::onnxruntime>
            GTest::gtest
            GTest::gtest_main
            spdlog::spdlog
    )

    gtest_discover_tests(test_detector_decode)
endif()

# ── Install rules ────────────────────────────────────────────────────────────
install(TARGETS semaforo_edge RUNTIME DESTINATION bin)
install(DIRECTORY configs/ DESTINATION etc/semaforo)
