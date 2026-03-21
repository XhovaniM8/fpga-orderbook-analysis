# Working Notes

## Project Goal
IEEE CIFEr 2026 paper. Deadline: May 15, 2026.
Pivot from "spoofing detection on FPGA with synthetic data" → "real-time order flow regime classification via quantized LSTM on FPGA, validated on real NASDAQ LOB data."

## Folder Structure
```
data/raw/          — LOBSTER zip files (5 symbols × 3 levels)
data/processed/    — extracted numpy arrays (gitignored)
src/data/          — LOBSTER loader, feature extraction, GMM labeling
src/model/         — Brevitas LSTM, training, evaluation
src/hls/           — hls4ml conversion pipeline
src/orderbook-simulator/  — original C++ simulator (keep for synthetic ablation)
paper/             — IEEEtran LaTeX
results/           — figures, model checkpoints, logs (mostly gitignored)
```

## Data
- LOBSTER sample data already in data/raw/ — 5 symbols (AAPL, AMZN, GOOG, INTC, MSFT)
- Each symbol has levels 1, 5, 10 snapshots for 2012-06-21
- **Primary training symbol: AAPL level 5** (liquid, well-studied)
- No headers in CSVs. Orderbook cols: [AskP1, AskS1, BidP1, BidS1, AskP2, AskS2, BidP2, BidS2, ...]
- Message cols: [Time, Type, OrderID, Size, Price, Direction]
- Prices are in units of $×10000 (e.g. 5859400 = $585.94)

## Architecture Decisions
- Model: Brevitas QuantLSTM, 8-bit weights/activations
  - 32 hidden units if KV260 available
  - 16 hidden units if Z7-10 only
- Labels: Unsupervised 4-class GMM on rolling LOB features (no hand-crafted manipulation events)
- FPGA toolchain: hls4ml first (not FINN — FINN had ONNX shape issues in capstone)
- Fallback: hand-HLS for LSTM cell if hls4ml fails

## Key Features (rolling window = 20 snapshots)
1. OFI (Order Flow Imbalance) — best bid/ask volume delta
2. Spread (normalized)
3. Bid/ask volume imbalance across all levels
4. Mid-price return volatility
5. Depth ratio (total bid vol / total ask vol)

## GMM Regime Labels (4 classes)
- 0: Directional/momentum — persistent OFI, widening spread, asymmetric depth
- 1: Mean-reverting/balanced — oscillating OFI, tight spread, symmetric depth
- 2: Toxic/high-adverse-selection — large trades, rapid depth evaporation
- 3: Illiquid/wide-spread — thin book, slow updates
(These are interpretations — actual cluster assignment comes from data)

## Paper Novel Claim
"First end-to-end characterization of quantized LSTM order book regime classifier on
resource-constrained Zynq FPGA — measuring PS/PL DMA handoff overhead that FINN
simulation systematically underestimates — validated on real NASDAQ ITCH data."

## CIFEr Fit
- Squarely in: ML & Deep Learning in Finance, Algorithmic Trading & Trade Execution Systems
- CIFEr 2026 emphasis is GenAI/LLMs — our angle: why inference-latency constraints rule
  out LLM-based approaches for this use case (turns potential weakness into a strength)
- Short paper is a realistic acceptance target if hardware results are real

## Progress Log
- [2026-03-18] **Model v2 — 13 features, extended quant sweep, multi-symbol, HMM ablation:**
              Features expanded 9→13: added ofi_l2, ofi_l3, weighted_ofi, trade_intensity (msg types 4+5).
              FLAT_DIM 180→260. Separate GMM pickle for synthetic (gmm_regime_synthetic.pkl).
              AAPL real  (13 feat, label_smooth=0.1): macro F1 ≈ 0.485–0.50 → baseline maintained.
              AAPL + HMM smoothing (hmmlearn GaussianHMM, warm-start from GMM):
                F1 = 0.43 — NEGATIVE RESULT: HMM collapses 67% of data into Regime 3 (over-smoothing).
                Note for paper: HMM helpful only if regime persistence >> snapshot frequency.
              Multi-symbol AAPL+AMZN+INTC joint model:
                val F1 = 0.60 (joint), but per-symbol test: AAPL=0.43, AMZN=0.44, INTC=0.35, mean=0.41.
                Joint training hurts AAPL — INTC's balanced regime distribution pulls GMM labels off.
                INTC regime 1 (Mean-Rev): 43/87k test samples — essentially absent in INTC.
              Extended quantization sweep (updated with 13-feat model):
                float32:         F1=0.485
                ap_fixed<16,6>:  F1=0.486  ΔF1=-0.001  agreement=98.7% ← keep this
                ap_fixed<12,6>:  F1=0.474  ΔF1=+0.011  agreement=80.2% ← useful midpoint for table
                ap_fixed<10,4>:  F1=0.474  ΔF1=+0.011  agreement=80.1% ← same as <12,6>
                ap_fixed<8,4>:   F1=0.378  ΔF1=+0.107  agreement=42.8% ← too aggressive
              Folder cleanup: root CSVs deleted, data/processed removed, capstone code → archive/,
                docs/references.bib merged into paper/refs.bib (added 11 new citations),
                .gitignore updated (LaTeX artifacts, data/synthetic/, src/model/archive/).
- [2026-03-18] Multi-symbol results (OLD): AMZN F1=0.46, INTC F1=0.39 (AAPL GMM applied cross-symbol)
              INTC: Regime 1 has only 43/87k samples → essentially absent in INTC order book
              MLP synthetic ablation: F1=0.86 (vs LSTM synthetic 0.93; vs MLP real 0.50)
- [2026-03-18] Quantization simulation (src/hls/quantize_sim.py):
              ap_fixed<16,6>: 99% class agreement, ΔF1=-0.0001 → USE THIS
              ap_fixed<8,4>:  50% class agreement, ΔF1=+0.091 → logit saturation, avoid
- [2026-03-18] C-sim blocked on macOS (Error 6): Xilinx ap_types complex ambiguity
              → Documented; C-sim to run on Linux; Python quantize_sim.py is functional proxy
- [2026-03-18] Paper scaffolded: paper/main.tex + paper/refs.bib → compiles to 4 pages, 0 errors
              All results tables filled (except hardware latency — pending Vivado machine)
- [2026-03-18] Folder structure created. LOBSTER data moved to data/raw/.
              Data pipeline scripts written (lobster_loader, features, regime_labels, train).
- [2026-03-18] Python env: 3.14.3 venv (pyenv 3.12.9 fails on macOS 26 Tahoe - libHacl_Hash_SHA2.a linker bug).
              torch 2.10, brevitas 0.12.0 (install --no-deps), scikit-learn, pandas, seaborn, hls4ml 1.2.0.
- [2026-03-18] eXSimulator lob_export built and tested. Fixed histogram.hpp type mismatch (uint64_t vs size_t).
              Fixed agent BBO init (agents default to $100 mid; fixed by broadcasting synthetic BBOUpdate before tick 0).
              Synthetic data: 50k snapshots at correct ~$585 AAPL-scale prices.
- [2026-03-18] GMM regime labeling: 301,587 LOBSTER AAPL snapshots → 4 regimes.
              Regime 0 (14%): OFI=-1.0 (sell pressure) | Regime 2 (13%): OFI=+1.0 (buy pressure)
              Regime 1 (8%): vol_imb=0.775 (bid-heavy) | Regime 3 (65%): balanced/base
- [2026-03-18] KEY RESULTS:
              LSTM (h=32) on real LOBSTER AAPL:    macro F1 = 0.52
              LSTM (h=32) on synthetic (eXSim):    macro F1 = 0.93
              MLP  (h=64) on real LOBSTER AAPL:    macro F1 = 0.50
              MLP  (h=64) on synthetic (eXSim):    macro F1 = 0.86
              MLP  (h=64) on real LOBSTER AMZN:    macro F1 = 0.46
              MLP  (h=64) on real LOBSTER INTC:    macro F1 = 0.39
              → Sim-to-reality gap (LSTM): 0.41 F1 (paper finding #1)
              → LSTM→MLP FPGA tax (real): 0.02 F1 (paper finding #2)
              → Cross-symbol: AAPL GMM generalizes to AMZN/INTC at lower F1
                INTC note: Regime 1 has only 43/87k samples (regime absent in INTC)
              → Quantization (ap_fixed<16,6>): ΔF1 = -0.001 (paper finding #4)
              → Quantization (ap_fixed<8,4>):  ΔF1 = +0.107  (too aggressive)
- [2026-03-18] QUANTIZATION SIMULATION (src/hls/quantize_sim.py) — v2 with 13-feat model:
              ap_fixed<16,6>: class agreement 98.7%, ΔF1 = -0.001  (negligible) ← SELECTED
              ap_fixed<12,6>: class agreement 80.2%, ΔF1 = +0.011  (useful midpoint)
              ap_fixed<10,4>: class agreement 80.1%, ΔF1 = +0.011  (same as 12,6)
              ap_fixed<8,4>:  class agreement 42.8%, ΔF1 = +0.107  (too aggressive)
              → 16-bit is sufficient; 8-bit loses >half of correct classifications
              → This replaces C-sim (blocked on macOS by Xilinx ap_types/libc++ conflict)
- [2026-03-18] hls4ml conversion pipeline — documented errors (paper finding #3):
              Error 1: hls4ml ONNX frontend does not support LSTM op
              Fix: re-implemented LSTM as explicit gate ops (MatMul+Sigmoid+Tanh) → unrolled model verified
              Error 2: Gather op (from seq indexing) not supported by hls4ml ONNX
              Fix: switch to MLP (no sequential indexing) for FPGA-deployed model
              Error 3: Gemm op (from nn.Linear) not supported by hls4ml ONNX frontend
              Fix: use MatMulLinear (x @ W.T + b) → exports as MatMul+Add
              Error 4: hls4ml ONNX Merge/Add bias shape bug (NoneType.shape)
              Fix: use hls4ml PyTorch native frontend instead of ONNX
              Error 5: input_shape=(1,180) triggers Conv1D code path
              Fix: input_shape=(180,) for flat Dense network
              FINAL: hls4ml PyTorch frontend → SimpleMLP → Vivado backend HLS project GENERATED ✓
              Output: results/hls4ml_prj/mlp_regime_h64/ (myproject.cpp, build_prj.tcl, etc.)
              HLS function signature: void myproject(input_t x[260], result_t layer6_out[4])
              (Updated: FLAT_DIM 180→260 after 13-feature expansion; hls4ml project must be regenerated)
              Pragmas: HLS PIPELINE (II=1 target), ARRAY_PARTITION complete

## Toolchain Notes (for paper Section III)
- Brevitas 0.12.0: incompatible with Python 3.14 — _dependencies DependencyError on Magic methods
  → Decision: train float32, apply ap_fixed<16,6> in hls4ml config (standard practice)
- hls4ml 1.2.0 ONNX frontend: does not support LSTM, Gemm, or Gather ops
  → LSTM→MLP architecture change is documented as a hardware-driven design decision
  → The 0.02 F1 cost is the measurable price of FPGA deployability
- hls4ml PyTorch frontend: requires input_shape=(N,) not (1,N) for Dense layers
  → write() must be called explicitly after convert_from_pytorch_model()
- Error 6: hls4ml C-simulation fails on macOS (Apple Silicon, Xcode clang)
  → Xilinx bundled ap_types headers: 'complex' is ambiguous (std::__1::complex vs std::complex)
  → ap_int_special.h and ap_fixed_special.h use unqualified 'complex' inside namespace
  → Fix: run C-sim on Linux (Ubuntu 20.04 + Vivado); macOS is dev-only environment
  → Workaround for paper: Python fixed-point quantization simulation (src/hls/quantize_sim.py)

## Open Questions
- [ ] KV260 availability confirmed? (changes hidden_size: 32 vs 16)
- [ ] Run Vivado synthesis on Linux machine → get real LUT/DSP/BRAM counts
- [ ] Measure PL inference latency (ILA or hardware timer) and DMA round-trip
- [ ] Generate paper figures: regime visualization on mid-price chart (regime_labels.py has this)
- [ ] Add confusion matrix figures to paper/figures/ and include in paper
- [ ] GOOG and MSFT symbols: run for completeness (optional, paper table already has 3 symbols)

## Hardware Milestones
- April 13: Hard checkpoint — working bitstream or switch to projected estimates
- Target latency: beat/characterize FINN estimate of 2.52µs (Z7-20 baseline)
