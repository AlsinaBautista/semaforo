#!/usr/bin/env python3
# Semáforo Inteligente - Brain Module
"""Train a coordinated MAPPO policy on the Necochea corridor with curriculum
learning and domain randomization.

This is the *advanced training loop* for the 3-intersection corridor on Avenida
59.  It uses Ray RLlib's multi-agent PPO with **parameter sharing** (one shared
policy drives all three signals — the practical MAPPO recipe for homogeneous
agents) and feeds the agent an ever-changing world so the learned controller
generalises to reality instead of overfitting one demand pattern:

* **Curriculum learning.**  Training starts on Stage 0 (calm, near-symmetric
  winter flow) and only unlocks the chaotic stages — heavy asymmetric summer
  exodus toward the coast, sudden sudestada storms, incident spikes — once the
  shared policy reaches a competence threshold (:class:`CurriculumCallback`).

* **Domain randomization.**  Every episode the environment samples a fresh,
  jittered scenario from the unlocked stages (see ``src/training/scenarios.py``):
  demand scale, directional asymmetry, turn ratios, weather-driven vehicle
  dynamics and the SUMO seed all vary, so no two episodes are identical.

Examples
--------
Quick functional smoke test (builds the algo, runs 1 short iteration)::

    python scripts/train_mappo.py --smoke

Full run::

    python scripts/train_mappo.py \\
        --iterations 400 --num-env-runners 6 \\
        --checkpoint-dir models/mappo_corridor
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the brain root (this file's grandparent) is importable as ``src.*``.
BRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(BRAIN_ROOT))

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from src.environments.multi_intersection import MultiIntersectionEnv
from src.training.centralized_critic_module import CentralizedCriticPPOTorchRLModule
from src.training.scenarios import CURRICULUM

ENV_NAME = "necochea_corridor"
SHARED_POLICY = "shared_signal_policy"
NUM_STAGES = len(CURRICULUM)


# --------------------------------------------------------------------------- #
# Env registration + multi-agent wiring
# --------------------------------------------------------------------------- #
def _env_creator(config: dict) -> ParallelPettingZooEnv:
    """Build the RLlib-facing corridor env from an ``env_config`` dict."""
    return ParallelPettingZooEnv(MultiIntersectionEnv(**config))


def _policy_mapping_fn(agent_id, *args, **kwargs) -> str:
    """Map every signal agent onto the single shared policy (parameter sharing)."""
    return SHARED_POLICY


def _set_stage_on_env(env, stage: int) -> None:
    """Set the curriculum stage on a (possibly wrapped) corridor env."""
    target = getattr(env, "par_env", env)  # unwrap ParallelPettingZooEnv
    if hasattr(target, "set_curriculum_stage"):
        target.set_curriculum_stage(stage)


# --------------------------------------------------------------------------- #
# Curriculum callback
# --------------------------------------------------------------------------- #
class CurriculumCallback(RLlibCallback):
    """Escalate the curriculum stage once the shared policy is competent.

    After each training iteration we read the mean episode return; once it stays
    above ``promote_threshold`` for ``patience`` consecutive iterations we unlock
    the next (harder) stage and broadcast it to every rollout environment. The
    threshold is intentionally negative because the reward (diff-waiting-time) is
    a cost the agent drives toward zero.
    """

    def __init__(
        self,
        promote_threshold: float = -20.0,
        patience: int = 5,
        max_stage: int = NUM_STAGES - 1,
    ) -> None:
        super().__init__()
        self.promote_threshold = promote_threshold
        self.patience = patience
        self.max_stage = max_stage
        self._stage = 0
        self._good_iters = 0

    def on_train_result(self, *, algorithm=None, result=None, **kwargs) -> None:  # noqa: D102
        if result is None or algorithm is None:
            return
        ret = _episode_return_mean(result)
        result["curriculum_stage"] = self._stage
        if ret is None or self._stage >= self.max_stage:
            return

        if ret >= self.promote_threshold:
            self._good_iters += 1
        else:
            self._good_iters = 0

        if self._good_iters >= self.patience:
            self._stage += 1
            self._good_iters = 0
            stage = self._stage
            try:
                algorithm.env_runner_group.foreach_env(
                    lambda env: _set_stage_on_env(env, stage)
                )
            except Exception:  # noqa: BLE001 – never let curriculum break training
                pass
            print(f"\n>>> CURRICULUM: promoted to stage {stage}/{self.max_stage} "
                  f"(return_mean={ret:.1f})\n")


def _episode_return_mean(result: dict) -> float | None:
    """Pull the mean episode return out of an RLlib result dict (API-tolerant)."""
    env_runners = result.get("env_runners", {}) if isinstance(result, dict) else {}
    for key in ("episode_return_mean", "episode_reward_mean"):
        if key in env_runners and env_runners[key] is not None:
            return float(env_runners[key])
        if key in result and result[key] is not None:
            return float(result[key])
    return None


# --------------------------------------------------------------------------- #
# Config builder
# --------------------------------------------------------------------------- #
def build_config(args: argparse.Namespace) -> PPOConfig:
    """Assemble the PPOConfig for parameter-shared multi-agent training.

    CTDE: the env exposes Dict observations (local ``"obs"`` + global
    ``"state"``) and the shared policy is a custom RLModule whose critic is
    centralized over the corridor state while the actor stays decentralized.
    """
    probe = MultiIntersectionEnv(include_global_state=True)
    obs_space = probe.observation_space("c0")
    act_space = probe.action_space("c0")

    env_config = {
        "num_seconds": args.episode_seconds,
        "delta_time": args.delta_time,
        "reward_fn": args.reward_fn,
        "auto_randomize": True,
        "curriculum_stage": 0,
        # Mixed-task rehearsal: each iteration's train batch mixes episodes
        # from past curriculum stages (asymmetric distribution) to mitigate
        # catastrophic forgetting.
        "rehearsal_ratio": args.rehearsal_ratio,
        "include_global_state": True,
    }

    config = (
        PPOConfig()
        .environment(env=ENV_NAME, env_config=env_config)
        .framework("torch")
        .multi_agent(
            policies={SHARED_POLICY},
            policy_mapping_fn=_policy_mapping_fn,
        )
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=args.train_batch_size,
            num_epochs=10,
            entropy_coeff=0.01,
            clip_param=0.2,
            grad_clip=0.5,
        )
        .env_runners(
            num_env_runners=args.num_env_runners,
            rollout_fragment_length="auto",
            # RLlib's default sample_timeout_s=60 livelocks full-size runs: a
            # 12000-step batch over 4 runners is ~3000 steps/runner at ~28 ms
            # SUMO-step plus several per-episode SUMO relaunches — well past
            # 60 s — so every sampling round was discarded and retried forever
            # ("No samples returned from remote workers", 0 iterations in 24
            # min). Infra timeout only; no effect on the learning math.
            sample_timeout_s=600.0,
        )
        # Learner pinned to the local CPU: RLlib 2.55 device resolution is
        # CUDA-only (no MPS on Apple Silicon), and a hidden_dim=256 MLP trains
        # faster on CPU than through MPS dispatch overhead anyway.
        .learners(num_learners=0, num_gpus_per_learner=0)
        .callbacks(CurriculumCallback)
        # Centralized critic (CTDE): spaces are bound via the RLModuleSpec.
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    SHARED_POLICY: RLModuleSpec(
                        module_class=CentralizedCriticPPOTorchRLModule,
                        observation_space=obs_space,
                        action_space=act_space,
                        model_config={"hidden_dim": 256},
                    ),
                },
            ),
        )
    )
    return config


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=300,
                   help="Number of RLlib training iterations.")
    p.add_argument("--num-env-runners", type=int, default=None,
                   help="Parallel rollout workers (each runs its own SUMO). "
                        "Default: cpu_count - 2 (cores reserved for the "
                        "driver and the local learner).")
    p.add_argument("--episode-seconds", type=int, default=3600,
                   help="Simulated seconds per episode.")
    p.add_argument("--delta-time", type=int, default=5,
                   help="Seconds between agent decisions.")
    p.add_argument("--train-batch-size", type=int, default=None,
                   help="Environment steps per training iteration. "
                        "Default: 3000 x env runners.")
    p.add_argument("--reward-fn", type=str, default="diff-waiting-time-spillback",
                   help="sumo_rl reward key, or 'diff-waiting-time-spillback' "
                        "(default: diff-waiting-time plus a bounded penalty for "
                        "holding green toward a saturated downstream block).")
    p.add_argument("--rehearsal-ratio", type=float, default=0.25,
                   help="Probability of replaying a past-curriculum-stage "
                        "scenario per episode (mixed-task rehearsal against "
                        "catastrophic forgetting). 0 disables.")
    p.add_argument("--checkpoint-dir", type=str, default="models/mappo_corridor",
                   help="Directory to write checkpoints into.")
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="Checkpoint every N iterations.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny functional run: 1 short iteration, 1 worker.")
    return p.parse_args()


def main() -> None:
    """Build the algorithm and run the curriculum training loop."""
    args = parse_args()

    if args.smoke:
        args.iterations = 1
        args.num_env_runners = 0      # run rollouts in the local process
        args.episode_seconds = 200
        args.train_batch_size = 400

    # Resolve dynamic defaults after the smoke override so smoke wins and an
    # explicit CLI value is never overridden.
    if args.num_env_runners is None:
        args.num_env_runners = max((os.cpu_count() or 4) - 2, 1)
    if args.train_batch_size is None:
        args.train_batch_size = 3000 * max(args.num_env_runners, 1)

    register_env(ENV_NAME, _env_creator)
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    print("=" * 64)
    print("  Semáforo Inteligente — MAPPO Corridor Training (Necochea Av. 59)")
    print("=" * 64)
    print(f"  Iterations      : {args.iterations}")
    print(f"  Env runners     : {args.num_env_runners}")
    print(f"  Episode seconds : {args.episode_seconds}")
    print(f"  Curriculum      : {NUM_STAGES} stages, domain randomization ON")
    print("=" * 64)

    algo = build_config(args).build_algo()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i in range(1, args.iterations + 1):
            result = algo.train()
            ret = _episode_return_mean(result)
            stage = result.get("curriculum_stage", 0)
            ret_str = f"{ret:8.2f}" if ret is not None else "    n/a"
            print(f"[iter {i:4d}/{args.iterations}] return_mean={ret_str}  "
                  f"stage={stage}/{NUM_STAGES - 1}")

            if i % args.checkpoint_every == 0 or i == args.iterations:
                path = algo.save(str(checkpoint_dir))
                print(f"           checkpoint -> {path}")
    finally:
        final = algo.save(str(checkpoint_dir))
        print(f"\nFinal checkpoint: {final}")
        try:
            # Lazy import: onnx/onnxruntime load at that module's import time.
            from src.export.onnx_exporter import export_to_onnx

            onnx_path = export_to_onnx(checkpoint_dir, checkpoint_dir / "policy.onnx")
            print(f"ONNX policy exported + verified: {onnx_path}")
        except Exception as exc:  # noqa: BLE001 – never mask a training error
            print(f"ONNX export failed (checkpoint is intact): {exc}")
        algo.stop()
        ray.shutdown()
        print("Done ✓")


if __name__ == "__main__":
    main()
