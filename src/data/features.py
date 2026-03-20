"""
Rolling LOB feature extraction.

Input: orderbook DataFrame from lobster_loader.load_lobster()
Output: numpy array of shape (T, N_FEATURES) with per-snapshot features

Features (all computed over a rolling window):
  0  mid_price          — (best_ask + best_bid) / 2
  1  spread_norm        — spread / mid_price
  2  ofi                — order flow imbalance (best level delta)
  3  vol_imbalance      — (total_bid_vol - total_ask_vol) / total_vol  across all levels
  4  depth_ratio        — total_bid_vol / total_ask_vol
  5  mid_return         — log return of mid_price vs previous snapshot
  6  volatility         — rolling std of mid_return over WINDOW snapshots
  7  spread_trend       — rolling mean of spread change over WINDOW snapshots
  8  ofi_ma             — rolling mean of OFI over WINDOW snapshots
  9  ofi_l2             — OFI at level 2 (sign(ΔBidSize_2) - sign(ΔAskSize_2))
  10 ofi_l3             — OFI at level 3
  11 weighted_ofi       — size-weighted OFI: (ΔBid_1 - ΔAsk_1) / (BidSize_1 + AskSize_1)
  12 trade_intensity    — rolling fraction of trade executions (msg types 4+5) in WINDOW
"""

import numpy as np
import pandas as pd

WINDOW = 20  # rolling window size
N_FEATURES = 13


def compute_features(ob: pd.DataFrame, n_levels: int = 5,
                     msg: pd.DataFrame | None = None) -> np.ndarray:
    """
    Returns float32 array of shape (T, N_FEATURES).
    First WINDOW rows will have NaN-derived zeros from rolling stats.
    Pass msg (LOBSTER message DataFrame) to populate trade_intensity; otherwise zeros.
    """
    T = len(ob)

    best_ask = ob["ask_price_1"].values
    best_bid = ob["bid_price_1"].values
    best_ask_sz = ob["ask_size_1"].fillna(0).values
    best_bid_sz = ob["bid_size_1"].fillna(0).values

    mid = (best_ask + best_bid) / 2.0
    spread = best_ask - best_bid

    # Spread normalized by mid
    spread_norm = np.where(mid > 0, spread / mid, 0.0)

    # OFI: sign(ΔBidSize) - sign(ΔAskSize) at best level
    d_bid = np.diff(best_bid_sz, prepend=best_bid_sz[0])
    d_ask = np.diff(best_ask_sz, prepend=best_ask_sz[0])
    ofi = np.sign(d_bid) - np.sign(d_ask)

    # Volume imbalance and depth ratio across all levels
    total_bid_vol = np.zeros(T)
    total_ask_vol = np.zeros(T)
    for i in range(1, n_levels + 1):
        bid_sz_col = f"bid_size_{i}"
        ask_sz_col = f"ask_size_{i}"
        if bid_sz_col in ob.columns:
            total_bid_vol += ob[bid_sz_col].fillna(0).values
        if ask_sz_col in ob.columns:
            total_ask_vol += ob[ask_sz_col].fillna(0).values

    total_vol = total_bid_vol + total_ask_vol + 1e-9
    vol_imbalance = (total_bid_vol - total_ask_vol) / total_vol
    depth_ratio = np.where(total_ask_vol > 0, total_bid_vol / (total_ask_vol + 1e-9), 1.0)

    # Mid log returns
    mid_safe = np.where(mid > 0, mid, 1.0)
    log_mid = np.log(mid_safe)
    mid_return = np.diff(log_mid, prepend=log_mid[0])

    # Rolling stats using pandas for simplicity
    s_mid_return = pd.Series(mid_return)
    s_spread = pd.Series(spread_norm)
    s_ofi = pd.Series(ofi)

    volatility = s_mid_return.rolling(WINDOW, min_periods=1).std().fillna(0).values
    spread_trend = s_spread.diff().rolling(WINDOW, min_periods=1).mean().fillna(0).values
    ofi_ma = s_ofi.rolling(WINDOW, min_periods=1).mean().fillna(0).values

    # OFI at level 2
    bid_sz_2 = ob["bid_size_2"].fillna(0).values if "bid_size_2" in ob.columns else np.zeros(T)
    ask_sz_2 = ob["ask_size_2"].fillna(0).values if "ask_size_2" in ob.columns else np.zeros(T)
    ofi_l2 = np.sign(np.diff(bid_sz_2, prepend=bid_sz_2[0])) - \
              np.sign(np.diff(ask_sz_2, prepend=ask_sz_2[0]))

    # OFI at level 3
    bid_sz_3 = ob["bid_size_3"].fillna(0).values if "bid_size_3" in ob.columns else np.zeros(T)
    ask_sz_3 = ob["ask_size_3"].fillna(0).values if "ask_size_3" in ob.columns else np.zeros(T)
    ofi_l3 = np.sign(np.diff(bid_sz_3, prepend=bid_sz_3[0])) - \
              np.sign(np.diff(ask_sz_3, prepend=ask_sz_3[0]))

    # Size-weighted OFI at level 1
    size_sum = best_bid_sz + best_ask_sz + 1e-9
    weighted_ofi = (d_bid - d_ask) / size_sum

    # Trade intensity: rolling fraction of trade executions (types 4+5) in window
    if msg is not None and "type" in msg.columns:
        is_trade = msg["type"].isin([4, 5]).astype(np.float32).values
        if len(is_trade) == T:
            trade_intensity = pd.Series(is_trade).rolling(WINDOW, min_periods=1).mean().values
        else:
            trade_intensity = np.zeros(T, dtype=np.float32)
    else:
        trade_intensity = np.zeros(T, dtype=np.float32)

    features = np.stack([
        mid,
        spread_norm,
        ofi,
        vol_imbalance,
        depth_ratio,
        mid_return,
        volatility,
        spread_trend,
        ofi_ma,
        ofi_l2,
        ofi_l3,
        weighted_ofi,
        trade_intensity,
    ], axis=1).astype(np.float32)

    # Replace any remaining inf/nan with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


if __name__ == "__main__":
    from lobster_loader import load_lobster
    ob, msg = load_lobster("AAPL", "2012-06-21", 5)
    feats = compute_features(ob, n_levels=5, msg=msg)
    print(f"Features shape: {feats.shape}")
    print(f"Feature means: {feats.mean(axis=0).round(4)}")
    print(f"Feature stds:  {feats.std(axis=0).round(4)}")
