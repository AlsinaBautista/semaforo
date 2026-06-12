# 🚦 Semáforo Inteligente — API Reference (MQTT Topics)

> Complete reference for all MQTT topics, message schemas, and usage patterns.

---

## Topic Hierarchy

```
semaforo/
├── {intersection_id}/
│   ├── telemetry          # Intersection telemetry data
│   ├── command            # Phase control commands
│   └── emergency          # Emergency vehicle preemption
├── {zone_id}/
│   ├── timing/plan        # Coordinated timing plans
│   └── sync/pulse         # Synchronization pulses
```

---

## 1. Telemetry

**Topic:** `semaforo/{intersection_id}/telemetry`
**Direction:** Edge → Broker → Brain / Coordinator / Dashboard
**QoS:** 1
**Retain:** No
**Frequency:** Every 1–2 seconds

### Schema

```json
{
  "timestamp": 1719432000000,
  "intersection_id": "int-001",
  "phase": "NS_GREEN",
  "phase_elapsed_s": 15,
  "queues": {
    "north": {
      "count": 8,
      "density": 0.45,
      "avg_wait_s": 22.3
    },
    "south": {
      "count": 5,
      "density": 0.28,
      "avg_wait_s": 14.1
    },
    "east": {
      "count": 12,
      "density": 0.67,
      "avg_wait_s": 35.8
    },
    "west": {
      "count": 3,
      "density": 0.15,
      "avg_wait_s": 8.2
    }
  },
  "flow_rate": 0.85,
  "vehicle_count": 28,
  "pedestrian_count": 4
}
```

### Field Descriptions

| Field              | Type              | Description                                    |
|--------------------|-------------------|------------------------------------------------|
| `timestamp`        | `uint64`          | Unix timestamp in milliseconds                 |
| `intersection_id`  | `string`          | Unique intersection identifier                 |
| `phase`            | `string`          | Current signal phase                           |
| `phase_elapsed_s`  | `uint32`          | Seconds since current phase started            |
| `queues`           | `map<str, obj>`   | Queue state per approach direction             |
| `queues.*.count`   | `uint32`          | Vehicles in queue                              |
| `queues.*.density` | `float`           | Vehicles per meter                             |
| `queues.*.avg_wait_s` | `float`        | Average wait time in seconds                   |
| `flow_rate`        | `float`           | Vehicles per second through intersection       |
| `vehicle_count`    | `uint32`          | Total vehicles in detection area               |
| `pedestrian_count` | `uint32`          | Total pedestrians in detection area            |

---

## 2. Commands

**Topic:** `semaforo/{intersection_id}/command`
**Direction:** Brain / Coordinator → Broker → Edge
**QoS:** 1 (normal), 2 (emergency)
**Retain:** No

### Schema

```json
{
  "timestamp": 1719432015000,
  "action": "set_phase",
  "phase": "EW_GREEN",
  "duration_s": 30,
  "priority": "normal",
  "plan_id": "plan-1719432000"
}
```

### Actions

| Action         | Description                                        |
|----------------|----------------------------------------------------|
| `set_phase`    | Switch to the specified phase                      |
| `extend_phase` | Extend the current phase by `duration_s` seconds   |
| `skip_phase`   | Skip to the next phase in the cycle                |
| `override`     | Emergency override — ignores normal cycle          |

### Priority Levels

| Priority    | QoS | Description                                     |
|-------------|-----|-------------------------------------------------|
| `normal`    | 1   | Standard RL agent or coordinator command         |
| `high`      | 1   | Manual override from dashboard                   |
| `emergency` | 2   | Emergency vehicle preemption                      |

---

## 3. Timing Plans

**Topic:** `semaforo/{zone_id}/timing/plan`
**Direction:** Coordinator → Broker → Edge
**QoS:** 1
**Retain:** Yes (latest plan is always available)

### Schema

```json
{
  "plan_id": "plan-1719432000",
  "timestamp": 1719432000000,
  "zone_id": "zone-centro",
  "cycle_length_s": 90,
  "corridors": [
    {
      "corridor_id": "corridor-int001-int003",
      "name": "Av. Principal",
      "design_speed_kmh": 50.0,
      "intersections": [
        {
          "intersection_id": "int-001",
          "offset_s": 0,
          "cycle_length_s": 90
        },
        {
          "intersection_id": "int-002",
          "offset_s": 25,
          "cycle_length_s": 90
        },
        {
          "intersection_id": "int-003",
          "offset_s": 54,
          "cycle_length_s": 90
        }
      ]
    }
  ],
  "valid_for_s": 300,
  "source": "coordinator"
}
```

### Offset Calculation

The offset for each intersection is computed as:

```
offset_s = cumulative_distance_m / (design_speed_kmh / 3.6)
```

This ensures a vehicle traveling at the design speed encounters green lights progressively (green wave).

---

## 4. Sync Pulses

**Topic:** `semaforo/{zone_id}/sync/pulse`
**Direction:** Coordinator → Broker → Edge
**QoS:** 0
**Retain:** No
**Frequency:** Every 10 seconds

### Schema

```json
{
  "timestamp": 1719432010000,
  "cycle_ref": 42,
  "zone_id": "zone-centro",
  "uptime_s": 3600
}
```

### Field Descriptions

| Field        | Type     | Description                                |
|--------------|----------|--------------------------------------------|
| `timestamp`  | `uint64` | Coordinator's current time (ms)            |
| `cycle_ref`  | `uint32` | Monotonically increasing cycle counter     |
| `zone_id`    | `string` | Zone this pulse applies to                 |
| `uptime_s`   | `uint32` | Coordinator uptime in seconds              |

---

## 5. Emergency

**Topic:** `semaforo/{intersection_id}/emergency`
**Direction:** Emergency System → Broker → Coordinator
**QoS:** 2 (guaranteed delivery)
**Retain:** No

### Schema

```json
{
  "emergency_id": "em-001",
  "vehicle_type": "ambulance",
  "origin_intersection": "int-001",
  "destination_intersection": "int-003",
  "direction": "north",
  "eta_s": 45
}
```

### Vehicle Types

| Type         | Description              |
|--------------|--------------------------|
| `ambulance`  | Medical emergency        |
| `fire`       | Fire department          |
| `police`     | Law enforcement          |

---

## Phase Identifiers

Standard phase identifiers used across the system:

| Phase                  | Description                                |
|------------------------|--------------------------------------------|
| `NS_GREEN`             | North-South green, East-West red           |
| `EW_GREEN`             | East-West green, North-South red           |
| `ALL_RED`              | All directions red (clearance)             |
| `NS_LEFT`              | North-South left turn arrow                |
| `EW_LEFT`              | East-West left turn arrow                  |
| `PEDESTRIAN_NS`        | North-South pedestrian crossing            |
| `PEDESTRIAN_EW`        | East-West pedestrian crossing              |
| `EMERGENCY_GREEN_*`    | Emergency override in specified direction  |
| `RESTORE_NORMAL`       | Return to normal cycle after emergency     |

---

## Connection Parameters

| Parameter     | Development       | Production                |
|---------------|-------------------|---------------------------|
| Host          | `localhost`        | `mosquitto.semaforo.local`|
| MQTT Port     | `1883`             | `8883` (TLS)              |
| WS Port       | `9001`             | `9443` (WSS)              |
| Protocol      | MQTT 5.0           | MQTT 5.0                  |
| Auth          | Anonymous          | Username/Password + ACL   |
| TLS           | Disabled           | Required                  |
