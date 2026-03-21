"""
Flat MLP regime classifier for FPGA deployment.

Takes the rolling LOB feature window as a flattened vector:
  Input: (SEQ_LEN * N_FEATURES,) = (20 * 13,) = 260 features
  Hidden: 64 units, ReLU
  Output: 4 regime classes

Why MLP instead of LSTM for FPGA:
  - No sequential data dependencies → can be fully pipelined
  - All ops are MatMul + bias + ReLU: directly supported by hls4ml's Vivado backend
  - Estimated II=1 achievable (LSTM requires II=SEQ_LEN at minimum for the state loop)
  - 95% fewer cycles for same window size

Accuracy tradeoff is documented in the paper and measured explicitly.
"""

import torch
from torch import nn

N_FEATURES = 13   # must match features.N_FEATURES (9 base + ofi_l2, ofi_l3, weighted_ofi, trade_intensity)
N_REGIMES = 4
SEQ_LEN = 20
FLAT_DIM = SEQ_LEN * N_FEATURES  # 20 * 13 = 260


class MatMulLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that exports as MatMul + Add in ONNX.
    nn.Linear exports as Gemm, which hls4ml's ONNX frontend does not support.
    MatMul and Add are both in hls4ml's supported op set.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t() + self.bias


class FlatRegimeClassifier(nn.Module):
    def __init__(self, input_dim: int = FLAT_DIM, hidden: int = 64,
                 num_classes: int = N_REGIMES):
        super().__init__()
        # Use MatMulLinear so ONNX export produces MatMul+Add, not Gemm
        self.fc1 = MatMulLinear(input_dim, hidden)
        self.fc2 = MatMulLinear(hidden, hidden // 2)
        self.fc3 = MatMulLinear(hidden // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.reshape(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


if __name__ == "__main__":
    model = FlatRegimeClassifier()
    dummy = torch.randn(4, SEQ_LEN, N_FEATURES)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
    print(model)
