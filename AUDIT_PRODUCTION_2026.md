# AUDIT_PRODUCTION_2026 — Pre-Flight Check

| | |
|---|---|
| **Fecha** | 2026-06-11 |
| **Alcance** | Edge (C++), Brain/Coordinator (Python), Infraestructura. Dashboard Flutter fuera de alcance. |
| **Metodología** | 3 subagentes exploratorios paralelos (uno por dominio) + verificación manual de cada referencia crítica `archivo:línea`. Modo solo lectura: ningún archivo fuente fue alterado. |
| **Veredicto** | **NO-GO** — 8 bloqueantes críticos activos repartidos en los 3 dominios. |

---

## 1. Matriz de bloqueantes (críticos — bloquean el pase a producción)

| ID | Dominio | Bloqueante | Modo de fallo en producción |
|----|---------|-----------|------------------------------|
| E-1 | Edge | Excepción de ONNX Runtime no capturada en el hot loop | Crash del proceso; cabezales quedan en el último estado aplicado |
| E-2 | Edge | Parsing del tensor YOLO sin validación de límites | Lectura out-of-bounds → segfault no controlado con modelo corrupto o de otra versión |
| B-1 | Brain | Cadena de despliegue del modelo rota (exporter solo SB3) | La política MAPPO entrenada nunca puede llegar al Edge |
| B-2 | Brain | `AIPolicyWrapper` del Coordinator es un stub hardcodeado | El sistema en vivo opera con una heurística trivial, no con la IA entrenada |
| B-3 | Brain | Topic de emergencia divergente del contrato | La preempción para ambulancias/bomberos jamás llega al Edge |
| I-1 | Infra | Sin control de versiones (no existe `.git/`) | Sin rollback, sin trazabilidad, sin forense post-incidente |
| I-2 | Infra | Edge sin supervisión de proceso (systemd/watchdog) | Crash del binario = intersección oscura hasta intervención manual |
| I-3 | Infra | Broker MQTT con acceso anónimo readwrite | Cualquier cliente de la red inyecta comandos de fase a los semáforos |

---

## 2. Dominio Edge (C++)

### E-1 — CRÍTICO · Excepción de inferencia no contenida
`edge/src/detector.cpp:282-285` · `edge/src/main.cpp:305`

`session->Run()` en `Detector::detect()` no está envuelto en try/catch (el único catch de `Ort::Exception` cubre la carga del modelo, líneas 156-188). El hot loop de `main.cpp` tampoco tiene contención: el único try/catch del main cubre solo la carga de config (líneas 75-78). Un fallo del runtime ONNX (modelo corrupto, OOM, fallo de dispositivo) propaga la excepción hasta `main`, termina el proceso y deja los cabezales en el último estado aplicado. Combinado con I-2 (sin supervisor), la intersección queda sin control indefinidamente.

### E-2 — CRÍTICO · Indexación del tensor de salida sin validar
`edge/src/detector.cpp:296-315`

El parser asume formato YOLOv8 `[1, 84, 8400]` verificando solo `shape.size()==3 && shape[1]>=4`, y luego indexa `out_data[(4+c)*num_anchors + i]` sin comprobar que el buffer real contenga `shape[1]*shape[2]` elementos. Un modelo de otra arquitectura (p. ej. YOLOv5 `[1, 25200, 85]`) o corrupto pasa el check de shape y produce lecturas fuera de límites → corrupción de memoria / segfault no controlado en el nodo de seguridad crítica.

### E-3 — ALTO · Conexión MQTT bloqueante en arranque
`edge/src/mqtt_client.cpp:148`

`connect()` espera síncronamente con timeout hardcodeado de 10 s. Con el broker inalcanzable en el arranque, el hilo principal queda bloqueado y SIGINT/SIGTERM se difieren: el nodo aparece colgado ante cualquier orquestación externa.

### E-4 — ALTO · Carrera del hilo de reconexión de video en shutdown
`edge/src/pipeline.cpp:244-286`

El hilo de reconexión captura `this` y duerme 2 s antes de reintentar. No hay coordinación entre `trigger_reconnect_async()` invocado desde el main loop y la destrucción de `VideoPipeline` durante el shutdown: una señal que termine el loop mientras el hilo de reconexión está vivo puede derivar en use-after-free. El patrón doble-buffer + stop-token es correcto en operación normal; la grieta es exclusivamente la ventana de teardown.

### E-5 — ALTO · El anillo de emergencia no tiene watchdog max-green
`edge/src/safety_layer.cpp:157-164` (watchdog en `step_normal`) vs `227-245` (`step_emergency` sin él)

El invariante anti-freeze (forzar cambio si un verde se sostiene demasiado) solo se aplica en el camino normal. En modo emergencia el anillo avanza por duraciones fijas sin esa red de seguridad: una misconfiguración de duraciones del anillo no sería detenida por ningún watchdog. Gap de cobertura del invariante central del sistema.

**Nota (no reabrir):** el parser endurecido (`command_parser.hpp`), la cola SPSC con drop-oldest, los object pools de inferencia y el modelo actor single-writer (callback MQTT = productor puro; main loop = único escritor de estado de control) están correctamente implementados y verificados por tests de propiedades.

---

## 3. Dominio Brain (Python) / Coordinator

### B-1 — CRÍTICO · No existe camino de despliegue para la política entrenada
`brain/src/export/onnx_exporter.py:17,56` · `brain/scripts/train_mappo.py:281` · `brain/src/training/centralized_critic_module.py:41-84`

El entrenamiento MAPPO produce checkpoints RLlib (RLModule con `CentralizedCriticPPOTorchRLModule`), pero el exporter ONNX hace `PPO.load()` de stable-baselines3 — formatos estructuralmente incompatibles. No hay ninguna ruta por la cual el modelo multi-agente entrenado se convierta en el artefacto ONNX que consume el Edge. El producto central del sistema (la política coordinada) es indesplegable.

### B-2 — CRÍTICO · El Coordinator no usa el modelo: stub hardcodeado
`network/coordinator/src/coordinator.py:47-54, 276-283`

`AIPolicyWrapper.predict()` es `vehicle_count > 10 → GREEN_NS, si no GREEN_EW`. Nunca se carga checkpoint alguno. En producción, todas las decisiones de fase provendrían de esta heurística de dos ramas, haciendo irrelevantes las corridas de entrenamiento.

### B-3 — CRÍTICO · Las órdenes de emergencia se publican en un topic que nadie escucha
`network/schemas/topics.yaml:84-88` · `network/coordinator/src/emergency.py:180`

`emergency.py` publica overrides en `semaforo/{intersection_id}/command`; el Edge solo se suscribe a `semaforo/commands/{node_id}`. El propio contrato lo documenta como "KNOWN DIVERGENCE". Resultado: la preempción para vehículos de emergencia se publica y se pierde silenciosamente. Funcionalidad de seguridad pública inoperante.

### B-4 — ALTO · Fallos silenciosos en el callback MQTT del Coordinator
`network/coordinator/src/coordinator.py:217-245`

paho-mqtt no propaga excepciones de `on_message`: un payload malformado se loguea y la ejecución continúa, dejando al Coordinator "conectado" pero no funcional, sin métrica ni alerta que lo delate. `_on_disconnect` tampoco implementa watchdog propio de reconexión; durante flapping del broker los comandos en vuelo se pierden sin reintento.

### B-5 — ALTO · Fallo asintótico de entrenamiento: fugas de SUMO y de escenarios
`brain/src/environments/multi_intersection.py:115, 222-239` · `brain/src/training/scenarios.py:290-322`

Cada `reset()` recrea `SumoEnvironment` sin garantía de teardown del subproceso SUMO/conexión TraCI (acumulación de procesos huérfanos y puertos ocupados entre corridas), y cada episodio escribe route files en `/tmp/semaforo_scenarios` sin recolección: una corrida larga (300-400 iteraciones × 6 workers) fuga del orden de GB en disco y memoria. El entrenamiento degrada y termina fallando con el tiempo, no de inmediato.

### B-6 — ALTO · El RLModule CTDE es incompatible con ejecución descentralizada
`brain/src/training/centralized_critic_module.py:67-75`

El forward exige un Dict con `obs` **y** `state` global. En el Edge (ejecución descentralizada) solo existe la observación local: aun resolviendo B-1, el actor debe extraerse/trazarse por separado del crítico o la inferencia falla con KeyError. Restricción de diseño a resolver junto con la cadena de export.

### B-7 — ALTO · Comandos stale rechazados sin visibilidad ni reintento
`network/coordinator/src/coordinator.py:301-321` · `network/schemas/topics.yaml:50`

El Edge rechaza comandos con más de 2000 ms de antigüedad (`max_age_ms`). Con latencia de broker o ciclo de planificación lento, los comandos rechazados se descartan silenciosamente del lado Edge y el Coordinator no se entera ni reemite. Brechas de coordinación recurrentes bajo carga de red.

### ✓ RESUELTO — Mismatch de telemetría (corrige a CLAUDE.md)
`network/coordinator/src/coordinator.py:211, 256` · `edge/src/mqtt_client.cpp:178`

El defecto documentado en CLAUDE.md (Coordinator suscrito a `semaforo/+/telemetry` parseando protobuf) **ya no existe**: el Coordinator carga `topics.yaml` al arranque (y rehúsa arrancar sin él), se suscribe a `semaforo/telemetry/+` y parsea JSON — alineado con lo que publica el Edge. El bucle de telemetría está cerrado. Residuo menor: `scripts/mock_telemetry.py` aún emite protobuf (tooling de prueba desalineado, no afecta producción). **No re-arreglar lo ya resuelto; actualizar CLAUDE.md.**

---

## 4. Dominio Infraestructura

### I-1 — CRÍTICO · Sin control de versiones
No existe `.git/` en la raíz del proyecto (verificado). Sin historial no hay rollback de despliegues, ni atribución de cambios, ni forense tras un incidente en vía pública. Existe `.github/workflows/test.yml`, lo que sugiere que el repo remoto existe pero la copia de trabajo está desconectada de él.

### I-2 — CRÍTICO · Edge sin supervisión de proceso
`scripts/deploy_edge.sh:63` ("TODO: Implement deployment steps") · `docs/deployment-guide.md:38-50` (sección systemd en TODO)

No hay unidad systemd, watchdog ni wrapper de reinicio para el binario del Edge. Cualquier crash (incluidos E-1/E-2) deja la intersección oscura hasta intervención manual. Es el multiplicador de severidad de todos los demás hallazgos del Edge: con `Restart=always` + watchdog, un crash es un parpadeo; sin él, es un incidente de tráfico.

### I-3 — CRÍTICO · Broker MQTT abierto a anónimos
`network/configs/mosquitto.conf:24-25, 33` · `network/configs/mosquitto.acl:8, 16`

El listener 1883 (plaintext) tiene `allow_anonymous true` —marcado "TODO: Eliminar en Producción" pero activo— y el listener WebSockets también (línea 33). El ACL otorga `readwrite semaforo/#` a anónimos. Cualquier cliente de la red puede publicar comandos de fase, inundar telemetría o suplantar al Coordinator, rodeando por completo el mTLS del puerto 8883. El parser endurecido del Edge mitiga payloads malformados, pero un comando *bien formado* malicioso es indistinguible de uno legítimo.

### I-4 — ALTO · Cero observabilidad
`docs/deployment-guide.md:101-114` (monitoring en TODO) · `docker-compose.yml` (healthcheck solo para Mosquitto)

El Edge publica status + LWT, pero ningún componente los consume con lógica de timeout; no hay métricas (Prometheus/statsd), ni alerting, ni health endpoint en el Coordinator. Un nodo muerto se detecta por quejas ciudadanas, no por el sistema (MTTD en horas).

### I-5 — ALTO · Coordinator: supervisión solo-Docker y SPOF
`docker-compose.yml:42`

`restart: unless-stopped` no cubre caída del daemon Docker ni del host; no hay supervisor a nivel host ni redundancia. Single point of failure de la coordinación de toda la red.

### I-6 — ALTO · Logs no aptos para operación
`edge/include/utils/logger.hpp:18-28` · `network/coordinator/src/coordinator.py:467-471`

El Edge escribe `semaforo_edge.log` con **path relativo al CWD** (no configurable desde `node_config.yaml`), rotación 3×10 MB; en una Jetson con eMMC pequeña esto es pérdida de logs o disco lleno según desde dónde se invoque. El Coordinator loguea solo a stdout sin límite del log-driver de Docker (json-file sin max-size = crecimiento sin cota en el host).

### I-7 — ALTO · CI no compila el componente de seguridad crítica
`.github/workflows/test.yml` (único workflow)

El CI ejecuta solo tests Python. El binario C++ —el componente con los invariantes de seguridad— nunca se compila ni se testea en CI (incluye los tests de propiedades de `test_safety_layer` / `test_command_parser`), y TSan, roto localmente en macOS, tampoco corre en el runner Linux donde sí funcionaría. Sin build de imágenes ni artefactos versionados.

### I-8 — MEDIO · Edge sin Dockerfile ni despliegue reproducible
Solo Mosquitto y Coordinator tienen Dockerfile. El despliegue del Edge es manual (scp + cmake en la Jetson): sin versionado del binario desplegado, rollback = recompilar a mano, drift de configuración entre nodos.

---

## 5. Criterios de salida (gate de pase a producción)

Cada criterio es verificable y mapea 1:1 con un bloqueante:

1. **[E-1]** El hot loop del Edge contiene fallos de inferencia: ante excepción de ONNX, el nodo pasa a modo emergencia (fixed cycle) en lugar de morir. Verificable con un test que inyecte un modelo corrupto.
2. **[E-2]** El parser del tensor valida `element_count == shape[1]*shape[2]` (y tipo float) antes de indexar; test con modelo de shape incompatible.
3. **[B-1 + B-6]** Existe un camino probado checkpoint RLlib → actor ONNX (solo rama actor, sin `state` global) → carga en el Edge, con test de contrato de entrada/salida.
4. **[B-2]** `AIPolicyWrapper` carga el checkpoint real y sus predicciones se usan en el ciclo de decisión (o se elimina y se documenta que el Coordinator no decide fases).
5. **[B-3]** Las órdenes de emergencia llegan al Edge: topic unificado a `semaforo/commands/{node_id}` y test de contrato extremo a extremo (publicar override → Edge lo recibe).
6. **[I-1]** Repositorio bajo git con historial, y la copia desplegada referencia un SHA.
7. **[I-2]** Unidad systemd para el Edge con `Restart=always` + watchdog; matar el proceso en pruebas y verificar reinicio y estado seguro.
8. **[I-3]** Listener 1883 eliminado (o restringido a localhost), `allow_anonymous false` en todos los listeners, ACL sin entradas anónimas; verificar que un cliente sin certificado no puede publicar en `semaforo/commands/#`.

Recomendado antes del despliegue (no bloqueante estricto, pero los ALTOS E-3/E-4/E-5, B-4/B-5/B-7 e I-4/I-5/I-6/I-7 elevan materialmente el riesgo de incidentes bajo red inestable y deben tener plan con fecha).

---

*Generado por auditoría paralela de 3 subagentes + verificación manual. Ningún archivo fuente fue modificado durante esta auditoría.*
