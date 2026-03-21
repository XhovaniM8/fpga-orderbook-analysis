"""
Training script for the flat MLP regime classifier (FPGA deployment target).
Identical pipeline to train.py but uses FlatRegimeClassifier.

Usage:
    # single symbol
    python train_mlp.py --symbol AAPL --levels 5 --hidden 64 --epochs 30
    # multi-symbol
    python train_mlp.py --symbols AAPL AMZN INTC --levels 5 --hidden 64 --epochs 30
    # synthetic data (e.g. data/synthetic/)
    python train_mlp.py --synthetic data/synthetic/ --levels 5 --hidden 64 --epochs 30
    # with HMM label smoothing
    python train_mlp.py --symbol AAPL --use-hmm --epochs 30
"""

import argparse, sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "model"))

from lobster_loader import load_lobster, load_synthetic
from features import compute_features, N_FEATURES
from regime_labels import fit_gmm, fit_hmm, load_gmm, predict_regimes, save_gmm, print_regime_summary, N_REGIMES
from model_mlp import FlatRegimeClassifier, SEQ_LEN, FLAT_DIM

RESULTS_DIR = ROOT / "results"


def build_sequences(features, labels, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len:i].flatten())  # flatten here
        y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def _load_and_label(symbol_or_path: str, is_synthetic: bool, args,
                    gmm, scaler_gmm) -> tuple[np.ndarray, np.ndarray]:
    """Load one symbol (or synthetic dir), compute features + GMM labels."""
    if is_synthetic:
        ob, msg = load_synthetic(symbol_or_path, n_levels=args.levels)
    else:
        ob, msg = load_lobster(symbol_or_path, "2012-06-21", args.levels)
    feats_raw = compute_features(ob, n_levels=args.levels, msg=msg)
    labels = predict_regimes(feats_raw, gmm, scaler_gmm)
    if args.use_hmm:
        print(f"  Applying HMM smoothing to {symbol_or_path}...")
        labels = fit_hmm(feats_raw, labels, gmm=gmm, scaler=scaler_gmm)
    return feats_raw, labels


def _split_symbol(feats_raw: np.ndarray, labels: np.ndarray,
                  fit_scaler: bool = True, scaler: StandardScaler | None = None):
    """Scale + build sequences + 70/15/15 time-ordered split for one symbol."""
    if fit_scaler:
        scaler = StandardScaler()
        feats = scaler.fit_transform(feats_raw).astype(np.float32)
    else:
        feats = scaler.transform(feats_raw).astype(np.float32)
    X, y = build_sequences(feats, labels)
    n = len(X)
    n_tr, n_v = int(0.7 * n), int(0.15 * n)
    return (X[:n_tr], y[:n_tr],
            X[n_tr:n_tr + n_v], y[n_tr:n_tr + n_v],
            X[n_tr + n_v:], y[n_tr + n_v:],
            scaler)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Determine symbols list
    if args.symbols:
        symbols = args.symbols
        is_synthetic = False
        tag = "_".join(symbols)
    elif args.synthetic:
        symbols = [args.synthetic]
        is_synthetic = True
        tag = "synthetic"
    else:
        symbols = [args.symbol]
        is_synthetic = False
        tag = args.symbol

    # Fit or load GMM — real and synthetic use separate pickle files
    gmm_path = (RESULTS_DIR / "models" / "gmm_regime_synthetic.pkl"
                if is_synthetic else RESULTS_DIR / "models" / "gmm_regime.pkl")
    if gmm_path.exists():
        print(f"Loading saved GMM from {gmm_path.name}...")
        gmm, scaler_gmm = load_gmm(gmm_path)
    else:
        print(f"Fitting GMM on {symbols[0]}...")
        first_ob, first_msg = (load_synthetic(symbols[0], n_levels=args.levels)
                               if is_synthetic
                               else load_lobster(symbols[0], "2012-06-21", args.levels))
        first_feats = compute_features(first_ob, n_levels=args.levels, msg=first_msg)
        gmm, scaler_gmm, _ = fit_gmm(first_feats)
        save_gmm(gmm, scaler_gmm, gmm_path)

    # Load and split all symbols
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    per_symbol_test: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for i, sym in enumerate(symbols):
        print(f"Loading {sym}  (snapshots→)")
        feats_raw, labels = _load_and_label(sym, is_synthetic, args, gmm, scaler_gmm)
        print(f"  Snapshots: {len(feats_raw)}")
        print_regime_summary(feats_raw, labels)

        fit_sc = (i == 0)  # fit scaler on first symbol
        if i == 0:
            Xtr, ytr, Xv, yv, Xte, yte, feature_scaler = _split_symbol(
                feats_raw, labels, fit_scaler=True)
        else:
            Xtr, ytr, Xv, yv, Xte, yte, _ = _split_symbol(
                feats_raw, labels, fit_scaler=False, scaler=feature_scaler)

        all_train_X.append(Xtr); all_train_y.append(ytr)
        all_val_X.append(Xv);   all_val_y.append(yv)
        per_symbol_test[sym] = (Xte, yte)
        print(f"  Train: {len(Xtr)}  Val: {len(Xv)}  Test: {len(Xte)}")

    X_train = np.concatenate(all_train_X)
    y_train = np.concatenate(all_train_y)
    X_val   = np.concatenate(all_val_X)
    y_val   = np.concatenate(all_val_y)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=256, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=512,
    )

    model = FlatRegimeClassifier(input_dim=FLAT_DIM, hidden=args.hidden,
                                  num_classes=N_REGIMES).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    counts = np.bincount(y_train, minlength=N_REGIMES).astype(np.float32)
    weights = torch.tensor(1.0 / (counts + 1e-6)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights / weights.sum(),
                                    label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5)

    best_f1 = 0.0
    model_path = RESULTS_DIR / "models" / f"mlp_regime_{tag}_h{args.hidden}.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                trues.extend(yb.numpy())
        val_f1 = f1_score(trues, preds, average="macro", zero_division=0)
        scheduler.step(val_f1)
        print(f"Epoch {epoch:3d}  loss={total_loss/len(train_loader):.4f}  val_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_path)

    print(f"\nBest val F1: {best_f1:.4f}")

    # Test — report per-symbol F1 when multi-symbol
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    names = ["Directional", "Mean-Rev.", "Toxic", "Illiquid"]

    test_f1_per_sym = {}
    for sym, (Xte, yte) in per_symbol_test.items():
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)), batch_size=512)
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                trues.extend(yb.numpy())
        sym_f1 = f1_score(trues, preds, average="macro", zero_division=0)
        test_f1_per_sym[sym] = sym_f1
        print(f"\nTest results — {sym}:")
        print(classification_report(trues, preds, target_names=names, zero_division=0))
        print(f"Test macro F1 [{sym}]: {sym_f1:.4f}")

        cm = confusion_matrix(trues, preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=names, yticklabels=names, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"MLP Regime Classifier — {sym} (h={args.hidden})")
        fig_path = RESULTS_DIR / "figures" / f"confusion_mlp_{sym}_h{args.hidden}.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix: {fig_path}")

    if len(symbols) > 1:
        print("\nPer-symbol test F1 summary:")
        for sym, f1 in test_f1_per_sym.items():
            print(f"  {sym}: {f1:.4f}")

    test_f1 = float(np.mean(list(test_f1_per_sym.values())))
    print(f"\nMean test macro F1: {test_f1:.4f}")
    return model, test_f1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="AAPL",
                   help="Single symbol to train on (ignored if --symbols is set)")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="Train on multiple symbols, e.g. --symbols AAPL AMZN INTC")
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--synthetic", default=None,
                   help="Path to synthetic data directory (e.g. data/synthetic/)")
    p.add_argument("--use-hmm", action="store_true",
                   help="Apply Gaussian HMM temporal smoothing to GMM regime labels")
    args = p.parse_args()
    train(args)
