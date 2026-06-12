# Semáforo Inteligente - Brain Module
"""M/D/1 queueing model for signalised intersections.

The **M/D/1** queue assumes Poisson arrivals (rate λ) and deterministic
(constant) service times (rate μ).  This is a reasonable first
approximation for a traffic lane served by a fixed green phase.

Key formulae (Pollaczek–Khinchine):

* Utilisation: ρ = λ / μ
* Average queue length: Lq = ρ² / (2 (1 − ρ))
* Average delay (waiting time): Wq = ρ / (2 μ (1 − ρ))
"""

from __future__ import annotations


class MD1Queue:
    """Deterministic-service single-server queue (M/D/1).

    Args:
        arrival_rate: Mean arrival rate λ (vehicles / second).
        service_rate: Deterministic service rate μ (vehicles / second).
            Must be strictly greater than ``arrival_rate`` for stability.

    Raises:
        ValueError: If ``arrival_rate >= service_rate`` (unstable queue)
            or if either rate is non-positive.

    Example::

        q = MD1Queue(arrival_rate=0.3, service_rate=0.5)
        print(q.utilization())      # 0.6
        print(q.avg_queue_length()) # 0.45
        print(q.avg_delay())        # 1.5
    """

    def __init__(self, arrival_rate: float, service_rate: float) -> None:
        if arrival_rate <= 0 or service_rate <= 0:
            raise ValueError("Arrival and service rates must be positive.")
        if arrival_rate >= service_rate:
            raise ValueError(
                f"Queue is unstable: λ ({arrival_rate}) >= μ ({service_rate}). "
                "Arrival rate must be strictly less than service rate."
            )
        self.arrival_rate = arrival_rate  # λ
        self.service_rate = service_rate  # μ

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def utilization(self) -> float:
        """Server utilisation ρ = λ / μ.

        Returns:
            Utilisation factor in [0, 1).
        """
        return self.arrival_rate / self.service_rate

    def avg_queue_length(self) -> float:
        """Average number of vehicles waiting in the queue (Lq).

        Uses the M/D/1 formula: Lq = ρ² / (2 (1 − ρ)).

        Returns:
            Expected queue length (vehicles).
        """
        rho = self.utilization()
        return rho**2 / (2.0 * (1.0 - rho))

    def avg_delay(self) -> float:
        """Average waiting time in the queue (Wq) in seconds.

        Uses the M/D/1 formula: Wq = ρ / (2 μ (1 − ρ)).

        Returns:
            Expected delay per vehicle (seconds).
        """
        rho = self.utilization()
        return rho / (2.0 * self.service_rate * (1.0 - rho))

    def avg_system_time(self) -> float:
        """Average total time in the system (W = Wq + 1/μ).

        Returns:
            Expected sojourn time (seconds).
        """
        return self.avg_delay() + 1.0 / self.service_rate

    def avg_vehicles_in_system(self) -> float:
        """Average number of vehicles in the system (L = Lq + ρ).

        Returns:
            Expected number of vehicles (waiting + being served).
        """
        return self.avg_queue_length() + self.utilization()

    def __repr__(self) -> str:
        return (
            f"MD1Queue(λ={self.arrival_rate}, μ={self.service_rate}, "
            f"ρ={self.utilization():.3f})"
        )


# ------------------------------------------------------------------
# Webster's delay as a standalone function (for convenience)
# ------------------------------------------------------------------

def webster_uniform_delay(
    cycle_length: float,
    green_ratio: float,
) -> float:
    """Compute Webster's uniform (deterministic) delay component.

    This is the *d₁* term in Webster's delay formula, representing the
    average delay assuming perfectly uniform arrivals:

        d₁ = C (1 − g/C)² / (2 (1 − g/C × x))

    where *x* is the degree of saturation (assumed = 1 here for the
    simplified form).

    Args:
        cycle_length: Total cycle length C (seconds).
        green_ratio: Effective green ratio g/C (dimensionless, 0–1).

    Returns:
        Uniform delay d₁ (seconds per vehicle).
    """
    if not 0.0 < green_ratio < 1.0:
        raise ValueError("green_ratio must be in (0, 1).")
    return cycle_length * (1.0 - green_ratio) ** 2 / (2.0 * (1.0 - green_ratio))
