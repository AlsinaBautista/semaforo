# Semáforo Inteligente — repo-level convenience targets.

# Regenerate the gitignored pre-retrain ONNX stub (brain/models/policy_stub.onnx).
# Needed after every fresh clone: the Coordinator and verify_brain_inference.py
# refuse to start without a model on disk.
.PHONY: brain-stub
brain-stub:
	.venv/bin/python scripts/bootstrap_brain_stub.py
