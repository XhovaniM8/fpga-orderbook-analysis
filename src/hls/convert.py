"""
hls4ml conversion pipeline.

Converts the trained Brevitas LSTM to HLS and runs synthesis.
All errors are logged to results/logs/hls4ml_errors.log — these become paper content
(documenting what the automated tool handles vs. what required manual intervention).

Usage:
    python convert.py --model-path results/models/regime_lstm_AAPL_h32.pth \
                      --symbol AAPL --hidden 32 --backend VivadoAccelerator
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "model"))

from model import RegimeClassifier, SEQ_LEN, N_FEATURES, N_REGIMES
from features import N_FEATURES as NF

RESULTS_DIR = ROOT / "results"
LOG_PATH = RESULTS_DIR / "logs" / "hls4ml_conversion.log"


def setup_logging():
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


def export_onnx(model: RegimeClassifier, onnx_path: Path):
    model.eval()
    dummy = torch.randn(1, SEQ_LEN, N_FEATURES)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )


def convert(args, log):
    import hls4ml

    # Load model
    model = RegimeClassifier(input_size=N_FEATURES, hidden_size=args.hidden, num_classes=N_REGIMES)
    state = torch.load(args.model_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    log.info(f"Loaded model from {args.model_path}")

    # Export ONNX
    onnx_path = RESULTS_DIR / "models" / f"regime_lstm_{args.symbol}_h{args.hidden}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        export_onnx(model, onnx_path)
        log.info(f"ONNX export successful: {onnx_path}")
    except Exception as e:
        log.error(f"ONNX export FAILED: {e}")
        raise

    # hls4ml conversion
    hls_output_dir = str(RESULTS_DIR / "hls4ml_prj" / f"{args.symbol}_h{args.hidden}")
    try:
        config = hls4ml.converters.get_config_from_onnx(str(onnx_path))
        config["Model"]["Precision"] = "ap_fixed<8,4>"
        config["Model"]["ReuseFactor"] = 1
        config["Model"]["Strategy"] = "Resource"
        log.info(f"hls4ml config: {config}")

        hls_model = hls4ml.converters.convert_from_onnx(
            str(onnx_path),
            output_dir=hls_output_dir,
            backend=args.backend,
            hls_config=config,
        )
        log.info("hls4ml conversion successful")
    except Exception as e:
        log.error(f"hls4ml conversion FAILED: {e}")
        log.error("--- Document this error for the paper ---")
        raise

    # Functional check: compare HLS model output vs PyTorch on 10 samples
    try:
        dummy_np = np.random.randn(10, SEQ_LEN, N_FEATURES).astype(np.float32)
        with torch.no_grad():
            pt_out = model(torch.from_numpy(dummy_np)).numpy()
        hls_out = hls_model.predict(dummy_np)
        max_diff = np.abs(pt_out - hls_out).max()
        log.info(f"Functional check max output diff (PyTorch vs HLS): {max_diff:.6f}")
    except Exception as e:
        log.warning(f"Functional check skipped: {e}")

    # C-simulation
    try:
        hls_model.compile()
        log.info("C-simulation passed")
    except Exception as e:
        log.error(f"C-simulation FAILED: {e}")

    # Synthesis (optional — takes ~10-30 min)
    if args.synthesize:
        try:
            report = hls_model.build(csim=False, synth=True, export=True)
            log.info(f"Synthesis report: {report}")
        except Exception as e:
            log.error(f"Synthesis FAILED: {e}")
    else:
        log.info("Skipping synthesis (pass --synthesize to run Vivado)")

    log.info(f"Full conversion log saved to {LOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="Path to .pth checkpoint")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--backend", default="VivadoAccelerator",
                        choices=["Vivado", "VivadoAccelerator", "Vitis", "VitisAccelerator"])
    parser.add_argument("--synthesize", action="store_true",
                        help="Run full Vivado synthesis (slow)")
    args = parser.parse_args()

    log = setup_logging()
    convert(args, log)
