"""
Fixed-point quantization simulation for the MLP regime classifier.

Replaces hls4ml C-simulation (which fails on macOS due to Xilinx ap_types /
libc++ 'complex' ambiguity). Simulates ap_fixed<W, I> arithmetic in Python
to measure quantization error before the model goes to a Linux machine for
actual C-sim and synthesis.

ap_fixed<W, I>: W total bits, I integer bits, W-I fractional bits.
Default: ap_fixed<16, 6> → 6 integer + 10 fractional bits, resolution 1/1024.

Usage:
    python quantize_sim.py                 # uses saved test data
    python quantize_sim.py --symbol AAPL   # reloads real LOB data
    python quantize_sim.py --w 8 --i 4    # test different precision

Paper table: compare F1 at float32, ap_fixed<16,6>, ap_fixed<8,4>.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "model"))
sys.path.insert(0, str(ROOT / "src" / "data"))

from model_mlp import FlatRegimeClassifier, FLAT_DIM, N_REGIMES, SEQ_LEN, N_FEATURES

RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# Fixed-point simulation
# ---------------------------------------------------------------------------

def quantize(x: np.ndarray, W: int, I: int) -> np.ndarray:
    """
    Simulate ap_fixed<W, I> truncation+saturation (AP_TRN, AP_WRAP default).
    Resolution: 2^(-(W-I))
    Range: [-2^(I-1), 2^(I-1) - 2^(-(W-I))]
    """
    frac_bits = W - I
    scale     = 2.0 ** frac_bits
    max_val   =  2.0 ** (I - 1) - 1.0 / scale
    min_val   = -2.0 ** (I - 1)
    # Truncate (round toward -inf), then saturate
    quantized = np.floor(x * scale) / scale
    return np.clip(quantized, min_val, max_val)


def relu_fixed(x: np.ndarray, W: int, I: int) -> np.ndarray:
    return quantize(np.maximum(0.0, x), W, I)


def dense_fixed(x: np.ndarray, W_mat: np.ndarray, b: np.ndarray,
                W: int, I: int, relu: bool = True) -> np.ndarray:
    """One Dense layer with fixed-point accumulation and optional ReLU."""
    acc = x @ W_mat.T + b
    acc = quantize(acc, W, I)
    if relu:
        acc = relu_fixed(acc, W, I)
    return acc


def run_fixed_point(model: FlatRegimeClassifier, X: np.ndarray,
                    W: int, I: int) -> np.ndarray:
    """
    Run the MLP forward pass with ap_fixed<W,I> simulation.
    X: (N, FLAT_DIM) float32
    Returns: (N, N_REGIMES) quantized logits
    """
    # Extract weights (fp32 → quantize weights too)
    w1 = quantize(model.fc1.weight.detach().numpy(), W, I)
    b1 = quantize(model.fc1.bias.detach().numpy(),   W, I)
    w2 = quantize(model.fc2.weight.detach().numpy(), W, I)
    b2 = quantize(model.fc2.bias.detach().numpy(),   W, I)
    w3 = quantize(model.fc3.weight.detach().numpy(), W, I)
    b3 = quantize(model.fc3.bias.detach().numpy(),   W, I)

    # Forward
    x = quantize(X.astype(np.float64), W, I)  # quantize input
    x = dense_fixed(x, w1, b1, W, I, relu=True)
    x = dense_fixed(x, w2, b2, W, I, relu=True)
    x = dense_fixed(x, w3, b3, W, I, relu=False)  # output layer, no ReLU
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# SimpleMLP helper (nn.Linear, same weights as FlatRegimeClassifier)
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=FLAT_DIM, hidden=64, num_classes=N_REGIMES):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def load_model(hidden: int) -> FlatRegimeClassifier:
    ckpt = RESULTS_DIR / "models" / f"mlp_regime_AAPL_h{hidden}.pth"
    model = FlatRegimeClassifier(input_dim=FLAT_DIM, hidden=hidden,
                                  num_classes=N_REGIMES)
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    from sklearn.metrics import f1_score, classification_report
    from sklearn.preprocessing import StandardScaler

    model = load_model(args.hidden)
    print(f"Loaded mlp_regime_AAPL_h{args.hidden}.pth")

    # Load real test data (reproduce 70/15/15 split)
    if args.symbol:
        from lobster_loader import load_lobster
        from features import compute_features
        from regime_labels import load_gmm, predict_regimes

        ob, msg = load_lobster(args.symbol, "2012-06-21", 5)
        feats_raw = compute_features(ob, n_levels=5, msg=msg)

        gmm_path = RESULTS_DIR / "models" / "gmm_regime.pkl"
        gmm, scaler_gmm = load_gmm(gmm_path)
        labels = predict_regimes(feats_raw, gmm, scaler_gmm)

        scaler = StandardScaler()
        feats  = scaler.fit_transform(feats_raw).astype(np.float32)

        X_all, y_all = [], []
        for i in range(SEQ_LEN, len(feats)):
            X_all.append(feats[i - SEQ_LEN:i].flatten())
            y_all.append(labels[i])
        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.int64)

        n = len(X_all)
        n_train = int(0.7 * n)
        n_val   = int(0.15 * n)
        X_test  = X_all[n_train + n_val:]
        y_test  = y_all[n_train + n_val:]
        print(f"Test samples: {len(X_test)}")
    else:
        # Random data — useful for quantization error analysis alone
        rng    = np.random.default_rng(42)
        X_test = rng.standard_normal((args.n_samples, FLAT_DIM)).astype(np.float32)
        y_test = None
        print(f"Using {args.n_samples} random samples (no labels)")

    # Float32 reference
    with torch.no_grad():
        pt_logits = model(torch.from_numpy(X_test)).numpy()
    pt_cls = pt_logits.argmax(axis=1)

    # Fixed-point configs to test
    # (16,6): production choice — negligible F1 drop
    # (12,6): intermediate — 6 integer bits, 6 fractional bits, resolution 1/64
    # (10,4): intermediate — 4 integer bits, 6 fractional bits, resolution 1/64
    # (8,4):  aggressive — 4 fractional bits, resolution 1/16
    configs = [(16, 6), (12, 6), (10, 4), (8, 4)]
    if (args.w, args.i) not in configs:
        configs.insert(0, (args.w, args.i))

    print(f"\n{'Precision':<16} {'MaxDiff':>8} {'MeanDiff':>9} {'Agreement':>10}")
    print("-" * 45)

    results = {}
    for W, I in configs:
        fxp_logits = run_fixed_point(model, X_test.copy(), W, I)
        fxp_cls    = fxp_logits.argmax(axis=1)
        max_diff   = np.abs(pt_logits - fxp_logits).max()
        mean_diff  = np.abs(pt_logits - fxp_logits).mean()
        agree      = (pt_cls == fxp_cls).mean() * 100.0
        label      = f"ap_fixed<{W},{I}>"
        print(f"{label:<16} {max_diff:>8.4f} {mean_diff:>9.4f} {agree:>9.1f}%")
        results[(W, I)] = {"max_diff": max_diff, "mean_diff": mean_diff,
                            "agree": agree, "cls": fxp_cls, "logits": fxp_logits}

    if y_test is not None:
        names = ["Directional", "Mean-Rev.", "Toxic", "Illiquid"]
        print(f"\n{'=== Float32 F1 ==='}")
        print(classification_report(y_test, pt_cls, target_names=names, zero_division=0))
        f1_float = f1_score(y_test, pt_cls, average="macro", zero_division=0)

        print(f"\n{'Precision':<16} {'Macro F1':>10} {'F1 Drop':>9}")
        print("-" * 35)
        print(f"{'float32':<16} {f1_float:>10.4f} {'—':>9}")
        for W, I in configs:
            f1_fxp  = f1_score(y_test, results[(W, I)]["cls"],
                               average="macro", zero_division=0)
            drop    = f1_float - f1_fxp
            label   = f"ap_fixed<{W},{I}>"
            print(f"{label:<16} {f1_fxp:>10.4f} {drop:>+9.4f}")

    print(f"\nQuantization simulation complete.")
    print(f"ap_fixed<16,6>: resolution 1/1024 ≈ 0.001, range [-32, +32)")
    print(f"ap_fixed<12,6>: resolution 1/64   ≈ 0.016, range [-32, +32)")
    print(f"ap_fixed<10,4>: resolution 1/64   ≈ 0.016, range [-8,  +8)")
    print(f"ap_fixed<8,4>:  resolution 1/16   ≈ 0.063, range [-8,  +8)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hidden",    type=int,   default=64)
    p.add_argument("--symbol",    default="AAPL",
                   help="Load real LOBSTER test data (default AAPL). "
                        "Set to '' for random samples only.")
    p.add_argument("--w",         type=int,   default=16, help="Total bit width")
    p.add_argument("--i",         type=int,   default=6,  help="Integer bits")
    p.add_argument("--n-samples", type=int,   default=500)
    args = p.parse_args()
    if args.symbol == "":
        args.symbol = None
    run(args)
