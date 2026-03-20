"""
C-simulation verification for the hls4ml MLP HLS project.

Runs without Vivado — uses g++ to compile the generated HLS C++ and
verify that outputs match the PyTorch reference model within fixed-point
tolerance (ap_fixed<16,6> has ~1/64 ≈ 0.016 LSB).

Usage:
    python csim_mlp.py
    python csim_mlp.py --hidden 64 --n-samples 100

The script:
  1. Loads the trained FlatRegimeClassifier checkpoint
  2. Reconstructs the SimpleMLP (hls4ml-compatible version with nn.Linear)
  3. Re-runs hls4ml conversion if needed (or reuses existing project)
  4. Calls hls_model.compile() → runs C-sim via g++
  5. Calls hls_model.predict() on random samples
  6. Compares class predictions and raw logits vs PyTorch
  7. Logs everything to results/logs/csim_mlp.log
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "model"))

from model_mlp import FlatRegimeClassifier, FLAT_DIM, N_REGIMES, SEQ_LEN, N_FEATURES

RESULTS_DIR = ROOT / "results"
LOG_PATH    = RESULTS_DIR / "logs" / "csim_mlp.log"
HLS_DIR     = RESULTS_DIR / "hls4ml_prj" / "mlp_regime_h64"


# ---------------------------------------------------------------------------
# SimpleMLP — identical to what convert.py used for hls4ml PyTorch frontend.
# hls4ml requires plain nn.Linear; FlatRegimeClassifier uses MatMulLinear.
# Weights are copied from the trained FlatRegimeClassifier checkpoint.
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int = FLAT_DIM, hidden: int = 64,
                 num_classes: int = N_REGIMES):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def copy_weights(flat_model: FlatRegimeClassifier, simple_model: SimpleMLP):
    """Copy MatMulLinear weights → nn.Linear weights (same layout)."""
    for src_layer, dst_layer in [
        (flat_model.fc1, simple_model.fc1),
        (flat_model.fc2, simple_model.fc2),
        (flat_model.fc3, simple_model.fc3),
    ]:
        dst_layer.weight.data.copy_(src_layer.weight.data)
        dst_layer.bias.data.copy_(src_layer.bias.data)


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def run(args, log):
    import hls4ml

    # 1. Load trained FlatRegimeClassifier
    ckpt_path = RESULTS_DIR / "models" / f"mlp_regime_AAPL_h{args.hidden}.pth"
    flat_model = FlatRegimeClassifier(input_dim=FLAT_DIM, hidden=args.hidden,
                                      num_classes=N_REGIMES)
    flat_model.load_state_dict(torch.load(ckpt_path, weights_only=True,
                                           map_location="cpu"))
    flat_model.eval()
    log.info(f"Loaded FlatRegimeClassifier from {ckpt_path}")

    # 2. Build SimpleMLP and copy weights
    simple = SimpleMLP(input_dim=FLAT_DIM, hidden=args.hidden,
                       num_classes=N_REGIMES)
    copy_weights(flat_model, simple)
    simple.eval()

    # Sanity-check weight copy on random input
    dummy = torch.randn(8, FLAT_DIM)
    with torch.no_grad():
        out_flat   = flat_model(dummy)
        out_simple = simple(dummy)
    max_diff = (out_flat - out_simple).abs().max().item()
    log.info(f"Weight-copy sanity check: max logit diff = {max_diff:.2e}")
    assert max_diff < 1e-4, f"Weight copy failed: {max_diff}"

    # 3. Build or reuse HLS project
    hls_dir_str = str(HLS_DIR)
    if HLS_DIR.exists() and not args.rebuild:
        log.info(f"Reusing existing HLS project at {HLS_DIR}")
        config = hls4ml.utils.config_from_pytorch_model(
            simple,
            input_shape=(FLAT_DIM,),
            granularity="name",
            default_precision="ap_fixed<16,6>",
            default_reuse_factor=1,
            backend="Vivado",
        )
        hls_model = hls4ml.converters.convert_from_pytorch_model(
            simple,
            input_shape=(FLAT_DIM,),
            output_dir=hls_dir_str,
            backend="Vivado",
            hls_config=config,
        )
        hls_model.write()
        log.info("HLS project (re)written")
    else:
        log.info("Building HLS project from scratch...")
        config = hls4ml.utils.config_from_pytorch_model(
            simple,
            input_shape=(FLAT_DIM,),
            granularity="name",
            default_precision="ap_fixed<16,6>",
            default_reuse_factor=1,
            backend="Vivado",
        )
        hls_model = hls4ml.converters.convert_from_pytorch_model(
            simple,
            input_shape=(FLAT_DIM,),
            output_dir=hls_dir_str,
            backend="Vivado",
            hls_config=config,
        )
        hls_model.write()
        log.info(f"HLS project written to {HLS_DIR}")

    # 4. Compile (C-simulation via g++)
    log.info("Compiling HLS project (g++ C-simulation)...")
    try:
        hls_model.compile()
        log.info("C-simulation compile: PASSED")
    except Exception as e:
        log.error(f"C-simulation compile FAILED: {e}")
        raise

    # 5. Run predict on random test samples
    rng = np.random.default_rng(42)
    X_test = rng.standard_normal((args.n_samples, FLAT_DIM)).astype(np.float32)

    log.info(f"Running hls4ml predict on {args.n_samples} random samples...")
    hls_out = hls_model.predict(X_test)           # shape: (N, 4) — ap_fixed logits

    with torch.no_grad():
        pt_out  = simple(torch.from_numpy(X_test)).numpy()  # float32 logits

    # 6. Compare
    # Logit-level comparison (will differ due to quantization — expect ~0.1 max diff)
    logit_max_diff = np.abs(pt_out - hls_out).max()
    logit_mean_diff = np.abs(pt_out - hls_out).mean()

    # Classification accuracy (argmax should agree)
    pt_cls  = pt_out.argmax(axis=1)
    hls_cls = hls_out.argmax(axis=1)
    agree   = (pt_cls == hls_cls).mean() * 100.0

    log.info(f"Logit max  diff (PyTorch vs HLS C-sim): {logit_max_diff:.4f}")
    log.info(f"Logit mean diff (PyTorch vs HLS C-sim): {logit_mean_diff:.4f}")
    log.info(f"Class agreement: {agree:.1f}%  ({(pt_cls == hls_cls).sum()}/{len(pt_cls)})")

    # Per-class breakdown
    for k in range(N_REGIMES):
        mask = pt_cls == k
        if mask.sum() == 0:
            continue
        agree_k = (pt_cls[mask] == hls_cls[mask]).mean() * 100
        log.info(f"  Regime {k}: {agree_k:.1f}% agreement ({mask.sum()} samples)")

    # Pass/fail threshold: >90% class agreement and logit diff < 0.5
    LOGIT_TOL = 0.5
    AGREE_TOL = 90.0
    passed = (logit_max_diff < LOGIT_TOL) and (agree >= AGREE_TOL)
    status = "PASSED" if passed else "FAILED"
    log.info(f"\nC-simulation functional check: {status}")
    log.info(f"  (thresholds: logit_max_diff < {LOGIT_TOL}, agreement >= {AGREE_TOL}%)")
    log.info(f"Full log saved to {LOG_PATH}")

    print(f"\n{'='*50}")
    print(f"C-SIM {status}")
    print(f"  Logit max diff:    {logit_max_diff:.4f}  (tol {LOGIT_TOL})")
    print(f"  Class agreement:   {agree:.1f}%  (tol {AGREE_TOL}%)")
    print(f"{'='*50}\n")

    return passed


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hidden",    type=int,  default=64)
    p.add_argument("--n-samples", type=int,  default=100)
    p.add_argument("--rebuild",   action="store_true",
                   help="Force rebuild of HLS project even if it already exists")
    args = p.parse_args()

    log = setup_logging()
    ok = run(args, log)
    sys.exit(0 if ok else 1)
