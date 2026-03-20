"""
Unsupervised regime labeling via Gaussian Mixture Model.

Fits a 4-component GMM on rolling LOB features and assigns a regime label
to each snapshot. Labels are interpretable post-hoc but not hand-crafted.

Regime interpretation (after inspection of cluster centroids):
  0: Directional/momentum     — persistent OFI, asymmetric depth, widening spread
  1: Mean-reverting/balanced  — oscillating OFI, tight spread, symmetric depth
  2: Toxic/adverse-selection  — high volatility, rapid spread widening, thin depth
  3: Illiquid/wide-spread     — wide spread, thin total volume, low OFI magnitude

Run this script standalone to fit and save the GMM, then visualize regime distribution.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

N_REGIMES = 4
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Features to use for GMM fitting (exclude raw mid_price and mid_return — too noisy)
# Indices into the feature matrix from features.py:
#   0=mid_price, 1=spread_norm, 2=ofi, 3=vol_imbalance, 4=depth_ratio,
#   5=mid_return, 6=volatility, 7=spread_trend, 8=ofi_ma
GMM_FEATURE_IDX = [1, 2, 3, 4, 6, 7, 8]  # skip mid_price and raw mid_return


def fit_gmm(features: np.ndarray, seed: int = 42) -> tuple[GaussianMixture, StandardScaler, np.ndarray]:
    """
    Fit GMM on selected features. Returns (gmm, scaler, labels).
    """
    X = features[:, GMM_FEATURE_IDX]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(
        n_components=N_REGIMES,
        covariance_type="full",
        n_init=5,
        random_state=seed,
        max_iter=200,
    )
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    return gmm, scaler, labels


def predict_regimes(features: np.ndarray, gmm: GaussianMixture, scaler: StandardScaler) -> np.ndarray:
    X = features[:, GMM_FEATURE_IDX]
    X_scaled = scaler.transform(X)
    return gmm.predict(X_scaled)


def save_gmm(gmm: GaussianMixture, scaler: StandardScaler, path: Path | None = None):
    if path is None:
        path = RESULTS_DIR / "models" / "gmm_regime.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"gmm": gmm, "scaler": scaler}, path)
    print(f"Saved GMM to {path}")


def load_gmm(path: Path | None = None) -> tuple[GaussianMixture, StandardScaler]:
    if path is None:
        path = RESULTS_DIR / "models" / "gmm_regime.pkl"
    obj = joblib.load(path)
    return obj["gmm"], obj["scaler"]


def print_regime_summary(features: np.ndarray, labels: np.ndarray):
    print(f"\nRegime distribution (n={len(labels)}):")
    for r in range(N_REGIMES):
        mask = labels == r
        count = mask.sum()
        pct = 100 * count / len(labels)
        # Show mean of key features for this regime
        spread = features[mask, 1].mean()
        ofi = features[mask, 2].mean()
        vol_imb = features[mask, 3].mean()
        vol = features[mask, 6].mean()
        print(f"  Regime {r}: {count:6d} ({pct:5.1f}%)  "
              f"spread_norm={spread:.4f}  ofi_mean={ofi:.3f}  "
              f"vol_imb={vol_imb:.3f}  volatility={vol:.6f}")


def fit_hmm(features: np.ndarray, gmm_labels: np.ndarray,
            gmm: GaussianMixture | None = None,
            scaler: StandardScaler | None = None,
            seed: int = 42) -> np.ndarray:
    """
    Fit a Gaussian HMM on GMM-selected features, warm-started from GMM parameters,
    and return Viterbi-decoded regime labels with temporal smoothing.

    This enforces regime persistence — adjacent snapshots cannot flip regimes
    arbitrarily, which reduces label noise at regime boundaries.

    FPGA impact: none — HMM runs only at label-generation time; FPGA sees only the MLP.

    Args:
        features:   (T, N_FEATURES) raw feature array from compute_features()
        gmm_labels: (T,) integer labels from fit_gmm() / predict_regimes()
        gmm:        fitted GaussianMixture (used for warm start). If None, uses
                    empirical means/covars from gmm_labels.
        scaler:     StandardScaler used for GMM fitting; applied before HMM if provided.
        seed:       random seed for reproducibility.

    Returns:
        (T,) integer array of HMM-smoothed regime labels.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise ImportError("hmmlearn is required for HMM smoothing: pip install hmmlearn")

    X = features[:, GMM_FEATURE_IDX]
    if scaler is not None:
        X = scaler.transform(X)

    # Build warm-start parameters from GMM or empirical cluster stats
    if gmm is not None:
        means_init = gmm.means_                          # (K, d)
        covars_init = np.stack([np.diag(np.diag(c))     # force diagonal for stability
                                for c in gmm.covariances_])
    else:
        d = X.shape[1]
        means_init = np.array([X[gmm_labels == k].mean(axis=0)
                               if (gmm_labels == k).sum() > 0 else np.zeros(d)
                               for k in range(N_REGIMES)])
        covars_init = np.array([np.diag(np.where(
                                    (gmm_labels == k).sum() > 1,
                                    X[gmm_labels == k].var(axis=0) + 1e-6,
                                    np.ones(d)))
                                for k in range(N_REGIMES)])

    # Empirical transition matrix from GMM labels
    trans_init = np.full((N_REGIMES, N_REGIMES), 1.0 / N_REGIMES)
    for t in range(len(gmm_labels) - 1):
        trans_init[gmm_labels[t], gmm_labels[t + 1]] += 1.0
    row_sums = trans_init.sum(axis=1, keepdims=True)
    trans_init /= row_sums

    hmm = GaussianHMM(
        n_components=N_REGIMES,
        covariance_type="diag",
        n_iter=50,
        random_state=seed,
        init_params="s",   # only re-init startprob; means/covars/transmat are warm-started
        params="stmc",     # learn all parameters
    )
    hmm.means_   = means_init
    hmm.covars_  = np.stack([np.diag(c) for c in covars_init])  # diag format
    hmm.transmat_ = trans_init

    hmm.fit(X)
    smoothed_labels = hmm.predict(X)
    return smoothed_labels


def plot_regimes(features: np.ndarray, labels: np.ndarray, mid_prices: np.ndarray,
                 save_path: Path | None = None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    colors = ["steelblue", "darkorange", "crimson", "forestgreen"]
    regime_names = ["Directional", "Mean-Rev.", "Toxic", "Illiquid"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Top: mid price colored by regime
    for r in range(N_REGIMES):
        mask = labels == r
        idx = np.where(mask)[0]
        ax1.scatter(idx, mid_prices[mask], c=colors[r], s=1, alpha=0.4, label=regime_names[r])
    ax1.set_ylabel("Mid Price ($)")
    ax1.set_title("Order Flow Regime Classification — AAPL 2012-06-21")
    patches = [mpatches.Patch(color=colors[r], label=regime_names[r]) for r in range(N_REGIMES)]
    ax1.legend(handles=patches, loc="upper right", markerscale=5)

    # Bottom: regime label as a stepped line
    ax2.plot(labels, linewidth=0.5, color="gray")
    ax2.set_ylabel("Regime")
    ax2.set_xlabel("Snapshot index")
    ax2.set_yticks(range(N_REGIMES))
    ax2.set_yticklabels(regime_names)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved regime plot to {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from lobster_loader import load_lobster
    from features import compute_features

    ob, msg = load_lobster("AAPL", "2012-06-21", 5)
    feats = compute_features(ob, n_levels=5)
    mid_prices = feats[:, 0]

    print(f"Fitting GMM on {len(feats)} snapshots...")
    gmm, scaler, labels = fit_gmm(feats)
    print_regime_summary(feats, labels)
    save_gmm(gmm, scaler)

    fig_path = RESULTS_DIR / "figures" / "regime_classification.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_regimes(feats, labels, mid_prices, save_path=fig_path)
