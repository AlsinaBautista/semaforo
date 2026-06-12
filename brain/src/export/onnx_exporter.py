# Semáforo Inteligente - Brain Module
"""Export the MAPPO actor network to ONNX format.

The trained policy is a Ray RLlib :class:`CentralizedCriticPPOTorchRLModule`
(CTDE): a decentralized **actor** over the 17-dim local observation and a
centralized **critic** over the 51-dim global state. Only the actor runs at
execution time, so the export traces *exclusively* the actor's forward pass —
the critic is pruned entirely and never enters the ONNX graph.

The exported graph is served via ``onnxruntime`` by the Coordinator's
``AIPolicyWrapper`` (``network/coordinator/src/coordinator.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ray.rllib.core.rl_module.rl_module import RLModule

# RLlib 2.55.1: RLModule.__init__ unconditionally logs a DeprecationWarning
# through this logger; it is not caused by project code.
logging.getLogger("ray._common.deprecation").setLevel(logging.ERROR)


def export_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    module_id: str = "shared_signal_policy",
    opset_version: int = 17,
    verify: bool = True,
) -> Path:
    """Export the MAPPO RLModule's actor network to ONNX.

    Restores the RLModule from an RLlib checkpoint, isolates the actor
    submodule, and traces it with a dummy local observation. The centralized
    critic (global-state branch) is not part of the traced graph.

    Args:
        checkpoint_path: RLlib Algorithm checkpoint root (the directory
            containing ``learner_group/``) or the RLModule checkpoint
            directory itself.
        output_path: Destination path for the ``.onnx`` file.
        module_id: Module id inside the checkpoint's MultiRLModule.
        opset_version: ONNX opset version.
        verify: If ``True``, validate the exported model with
            ``onnx.checker`` and check numerical parity against the
            PyTorch actor with ``onnxruntime``.

    Returns:
        Resolved ``Path`` to the written ONNX file.

    Raises:
        FileNotFoundError: If *checkpoint_path* is not an RLModule checkpoint.
        TypeError: If the restored module has no ``actor`` submodule.
        onnx.checker.ValidationError: If the exported graph is invalid
            (only when *verify* is ``True``).
    """
    # pyarrow (used by RLlib checkpoint loading) rejects relative paths.
    ckpt = Path(checkpoint_path).resolve()
    output_path = Path(output_path)

    if (ckpt / "learner_group").is_dir():
        ckpt = ckpt / "learner_group" / "learner" / "rl_module" / module_id
    if not (ckpt / "class_and_ctor_args.pkl").is_file():
        raise FileNotFoundError(f"Not an RLModule checkpoint dir: {ckpt}")

    module = RLModule.from_checkpoint(str(ckpt))
    actor = getattr(module, "actor", None)
    if actor is None:
        raise TypeError(
            f"Restored module {type(module).__name__} has no 'actor' submodule; "
            "expected CentralizedCriticPPOTorchRLModule. Note: brain/models/"
            "mappo_corridor is a stale Default-PPO checkpoint — use the "
            "repo-root models/mappo_corridor."
        )

    obs_dim = module.observation_space["obs"].shape[0]

    actor.cpu()
    actor.eval()
    dummy_input = torch.zeros(1, obs_dim, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # dynamo=False: keep the legacy single-file TorchScript exporter; the
    # dynamo path (torch>=2.12 default) writes weights to a sidecar file.
    torch.onnx.export(
        actor,
        (dummy_input,),
        str(output_path),
        dynamo=False,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["action_logits"],
        dynamic_axes={
            "obs": {0: "batch"},
            "action_logits": {0: "batch"},
        },
    )

    if verify:
        _verify_onnx(output_path, actor, obs_dim)

    return output_path.resolve()


def _verify_onnx(onnx_path: Path, actor: torch.nn.Module, obs_dim: int) -> None:
    """Validate the exported ONNX actor.

    Args:
        onnx_path: Path to the ONNX file.
        actor: The PyTorch actor used as numerical reference.
        obs_dim: Local observation size for the test inference.

    Raises:
        onnx.checker.ValidationError: If the graph is structurally invalid.
        RuntimeError: If the graph kept extra inputs (critic not pruned) or
            inference diverges from the PyTorch actor.
    """
    # Structural check.
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )

    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(
            f"Expected a single 'obs' graph input (critic pruned); "
            f"got {[i.name for i in inputs]}."
        )

    dummy = np.random.rand(1, obs_dim).astype(np.float32)
    (ort_logits,) = session.run(None, {inputs[0].name: dummy})

    with torch.no_grad():
        torch_logits = actor(torch.from_numpy(dummy)).numpy()

    if ort_logits.shape != torch_logits.shape:
        raise RuntimeError(
            f"ONNX output shape {ort_logits.shape} != actor output "
            f"shape {torch_logits.shape}."
        )
    if not np.allclose(ort_logits, torch_logits, atol=1e-5):
        raise RuntimeError(
            "ONNX inference diverges from the PyTorch actor "
            f"(max abs diff {np.abs(ort_logits - torch_logits).max():.3e})."
        )
