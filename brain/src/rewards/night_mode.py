"""Night-mode reward via potential-based reward shaping (PBRS).

Replaces the previous unbounded penalty (``-penalty_multiplier * queue``,
an infinite "cliff" that could collapse the value function) with a shaping
term ``F(s, s') = gamma * phi(s') - phi(s)`` over a bounded potential
``phi(s) in [-1, 0]``.  Potential-based shaping preserves the optimal
policy (Ng et al. invariance), and the returned reward is always within
``[-1.0, 1.0]``.
"""

import numpy as np


def compute_night_potential(observation, avenue_lanes, cross_lanes, *, max_queue=20.0):
    """Potential phi(s) for the night-mode condition, bounded to [-1, 0].

    The potential is negative when the main avenue has vehicles waiting while
    the cross streets are completely empty (the night-mode failure case), and
    zero otherwise.

    Args:
        observation (dict): Diccionario con las colas actuales {'queue': [q0, q1, ...]}
        avenue_lanes (list): Índices de los carriles correspondientes a la avenida principal.
        cross_lanes (list): Índices de los carriles correspondientes a la calle transversal.
        max_queue (float): Cola de saturación usada para normalizar el potencial.

    Returns:
        float: phi(s) en [-1.0, 0.0].
    """
    queues = observation.get('queue', [])
    if not queues:
        return 0.0

    # Suma de vehículos en carriles de la avenida (flujo principal)
    avenue_queue_total = sum(queues[i] for i in avenue_lanes if i < len(queues))
    # Suma de vehículos en carriles transversales (flujo secundario)
    cross_queue_total = sum(queues[i] for i in cross_lanes if i < len(queues))

    # Condición crítica: la avenida principal tiene autos esperando,
    # pero las calles transversales están completamente vacías.
    if cross_queue_total == 0 and avenue_queue_total > 0:
        norm = max(float(max_queue), 1.0)
        return -float(np.clip(avenue_queue_total / norm, 0.0, 1.0))

    return 0.0


def compute_night_mode_reward(
    observation,
    current_phase,
    avenue_lanes,
    cross_lanes,
    *,
    prev_observation=None,
    gamma=0.99,
    max_queue=20.0,
):
    """Bounded night-mode shaping reward F(s, s') = gamma*phi(s') - phi(s).

    Penaliza mantener en rojo la avenida principal si las transversales están
    vacías ('Modo Nocturno'), sin el cliff no acotado de la versión anterior:
    el resultado siempre está en [-1.0, 1.0] y, al ser shaping basado en
    potencial, no altera la política óptima.

    Args:
        observation (dict): Observación actual {'queue': [q0, q1, ...]}.
        current_phase (int): Fase actual del semáforo (conservado por compatibilidad).
        avenue_lanes (list): Índices de los carriles de la avenida principal.
        cross_lanes (list): Índices de los carriles transversales.
        prev_observation (dict | None): Observación del paso anterior. Si es
            ``None`` (primer paso) se devuelve phi(s) directamente.
        gamma (float): Factor de descuento del agente (debe coincidir con PPO).
        max_queue (float): Cola de saturación para normalizar el potencial.

    Returns:
        float: Recompensa de shaping acotada en [-1.0, 1.0].
    """
    phi_now = compute_night_potential(
        observation, avenue_lanes, cross_lanes, max_queue=max_queue
    )
    if prev_observation is None:
        return phi_now

    phi_prev = compute_night_potential(
        prev_observation, avenue_lanes, cross_lanes, max_queue=max_queue
    )
    # Con phi en [-1, 0], F está en [-gamma, 1]; el clip es cinturón de seguridad.
    return float(np.clip(gamma * phi_now - phi_prev, -1.0, 1.0))
