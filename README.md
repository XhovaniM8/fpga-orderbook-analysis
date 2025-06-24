# fpga-orderbook-analysis

NYU FRE Capstone project implementing a low-latency system for detecting market patterns using LSTM networks deployed on FPGA hardware. Combines deep learning with hardware acceleration to analyze high-frequency order book data in microseconds.

## Real-Time Order Book Analysis with LSTM on FPGA

This capstone project explores the integration of deep learning and reconfigurable hardware for ultra-low-latency financial signal detection. It leverages an LSTM-based neural network to analyze synthetic limit order book (LOB) data and flags directional movements, spoofing, and liquidity shifts. The trained model is optimized and deployed onto FPGA using the FINN compiler and Xilinx toolchain.

Developed for the Financial and Risk Engineering Capstone at NYU Tandon.

---

## Project Status

**Actively in development.**

### Completed

- **Synthetic Order Book Simulator**  
  C++-based simulator replicating realistic HFT behaviors like sweeps, spoofing, and cancellations.

- **Feature Extraction & Preprocessing**  
  Structured binary dataset pipeline using rolling window statistics and engineered market features.

- **LSTM Model Training (PyTorch)**  
  Compact, quantized LSTM model trained on synthetic data with class balancing, weighted loss, and t-SNE/CM evaluations.

- **Model Export to ONNX**  
  Final quantized model saved as `quant_lstm.onnx` in `src/model/`.

- **Organized Codebase**  
  All relevant source code and assets moved to the `src/` directory.

### In Progress

- **FPGA Hardware Generation with FINN Compiler**  
  Streamlining, quantization, and HLS-ready transformations using FINN's partial Python flow on Ubuntu VM.

- **ZynqBuild Deployment (Vivado 2019.2)**  
  Generate stitched IP and prepare for optional manual inference via custom DMA logic or PYNQ runtime.

- **Performance Evaluation**  
  Comparison of inference latency and accuracy across CPU, GPU, and FPGA targets.

---

