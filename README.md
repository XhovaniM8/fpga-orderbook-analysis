# Real-Time Market Behavior Classification with Deep Learning on FPGA Hardware

note: will return back to this sometime soon in 2026

NYU Financial & Risk Engineering Capstone Project

This repository implements a system for analyzing financial market data in real-time using LSTM neural networks deployed on FPGA hardware. The goal is to detect market behaviors like price manipulation and liquidity changes fast enough to be useful in high-frequency trading environments.

## Project Overview

Traditional trading algorithms use simple rules to make decisions, but these struggle to catch subtle patterns that modern markets exhibit. This project explores whether we can embed machine learning models directly into hardware to get both speed and intelligence.

The system uses an LSTM neural network to analyze order book data and predict short-term price movements. The trained model is then optimized and compiled for FPGA deployment using Xilinx's FINN framework.

## Key Results

**Model Performance:**
- Macro F1 Score: 0.41 on synthetic market data
- Handles severe class imbalance (98% "no change" samples)
- Successfully detects directional price movements 5 timesteps ahead

**Hardware Performance (Estimated):**
- Inference latency: 2.52 microseconds
- Throughput: 1+ million frames per second
- Resource usage: 71.7% LUTs, 47.1% BRAM on Zynq-7020

## Technical Approach

**Data Generation:**
Built a C++ order book simulator that creates realistic market scenarios including spoofing, liquidity sweeps, and momentum ignition. This gives us labeled training data with known market behaviors.

**Model Architecture:**
- Input: 5 timesteps of 64 market features each
- LSTM layer with 32 hidden units
- 8-bit quantization for hardware efficiency
- Output: 3-class prediction (UP/DOWN/NO_CHANGE)

**FPGA Implementation:**
1. Train quantized model using PyTorch + Brevitas
2. Export to ONNX format with quantization metadata
3. Use FINN compiler to generate hardware-optimized implementation
4. Target Xilinx Zynq-7020 FPGA platform

## Repository Structure

```
src/
├── simulation/              # C++ order book simulator
├── model/                   # LSTM training and quantization
├── fpga/                    # FINN compilation pipeline
└── evaluation/              # Performance analysis and visualization

docs/                        # Project documentation and paper
results/                     # Model outputs and hardware estimates
```

## Current Status

**Completed:**
- Order book simulator with realistic market behaviors
- LSTM model training and quantization
- Feature extraction pipeline
- Model export to ONNX format
- FINN compilation estimates

**In Progress:**
- Full FPGA synthesis (encountering ONNX shape compatibility issues)
- Hardware validation on actual FPGA board

## Technical Challenges

The main bottleneck has been getting the quantized LSTM model through the complete FINN compilation pipeline. Issues with tensor shape mismatches during ONNX transformations prevented full hardware synthesis, though the estimation pipeline completed successfully.

## Results Summary

While full hardware implementation wasn't completed, the project demonstrates that:
1. LSTM models can learn meaningful patterns from order book data
2. Quantization preserves model performance while enabling hardware deployment
3. FPGA acceleration could theoretically achieve microsecond-level inference latency
4. The approach is feasible for real-time financial applications

The projected performance levels would be competitive with current high-frequency trading systems, where each microsecond of latency reduction has significant value.

## Future Work

- Resolve ONNX compatibility issues for complete hardware synthesis
- Test on live market data feeds
- Explore hybrid CNN-LSTM architectures
- Investigate online learning for model adaptation
- Integrate with actual trading infrastructure

## Academic Context

This work was developed as part of the Financial and Risk Engineering capstone program at NYU Tandon. It combines concepts from machine learning, computer architecture, and financial markets to address practical problems in algorithmic trading.

The project builds on recent research showing that FPGAs can provide significant latency advantages for deep learning inference while consuming less power than traditional GPU-based systems.
