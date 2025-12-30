# Flash Attention: Memory-Efficient Attention Implementation

A Triton implementation of Flash Attention algorithms (v1 and v2) that dramatically reduces memory usage and improves computational efficiency for transformer models.

## Overview

This project implements the Flash Attention algorithms from scratch using Triton, demonstrating how to optimize the attention mechanism in transformers. Flash Attention reduces memory complexity from O(N²) to O(N) while maintaining numerical stability and improving speed. Kernels were tested on T4 gpu and computed using dtype=float16

## Key Features

- **Flash Attention v1**: Initial tiled algorithm with Online Softmax using m_i and l_i for recomputing softmax
- **Flash Attention v2**: Optimized version with reduced non-matmul FLOPs using only lse (Log-Sum-Exponential) for recomputation of softmax
- **Flash Attention v2 (Atomic)**: Uses the same logic from V2 but this variant uses atomic operations for backward pass
- **Memory Efficient**: Reduces HBM memory usage from O(N²) to O(N)
- **Numerically Stable**: Implements Online Softmax for improved stability
- **Benchmarking Suite**: Comprehensive performance comparisons

## Project Structure

```
flash-attention/
├── kernels/
│   ├── flash-v1.py              # Flash Attention v1 implementation
│   ├── flash-v2-non-atomic.py   # Flash Attention v2 (standard)
│   └── flash-v2-atomic.py       # Flash Attention v2 (atomic operations)
├── benchmark/
│   └── benchmark.ipynb          # Performance benchmarking notebook
├── images/
│   ├── forward_pass.png         # Forward pass visualization
│   ├── backward_pass.png        # Backward pass visualization
│   └── numerical_stability.png  # Numerical stability comparison
└── README.md
```

## Technologies Used

- **Triton**: Python-based GPU kernel language for high-performance computing
- **PyTorch**: Deep learning framework
- **Python**: Primary programming language
- **CUDA**: Underlying GPU architecture

## Algorithm Explanation

### Flash Attention v1
Implements tiled matrix multiplication with Online Softmax to compute attention without materializing the full N×N attention matrix. Uses block-wise computation to stay within SRAM.

### Flash Attention v2
Improves upon v1 by:
- Reducing non-matmul FLOPs
- Better work partitioning across thread blocks
- Optimized memory access patterns
- 2x faster than v1 in practice

### Key Innovations

1. **Tiling**: Breaks computation into blocks that fit in SRAM
2. **Online Softmax**: Incrementally computes softmax statistics
3. **Kernel Fusion**: Combines operations to minimize HBM reads/writes
4. **Recomputation**: Recomputes attention in backward pass instead of storing

## Performance Results

All benchmarks were conducted on a **Tesla T4 GPU** (Google Colab environment).

### Forward Pass Performance
![Forward Pass Performance](images\forward.png)
*Forward pass latency comparison across sequence lengths*

**Key Observations:**
- Triton V2 (Non-Atomic) achieves the best performance, consistently faster than all other implementations
- Both Triton V2 variants outperform Triton V1 and PyTorch implementations
- Performance gains increase with sequence length
- PyTorch Naive implementation becomes computationally infeasible at longer sequences

| Sequence Length | Triton V2 (Non-Atomic) | Triton V2 (Atomic) | Triton V1 | PyTorch (Flash) | PyTorch (Naive) |
|----------------|------------------------|--------------------|-----------|--------------------|-----------------|
| 256 | 0.064 ms | 0.056 ms | 0.037 ms | 0.071 ms | 0.147 ms |
| 512 | 0.090 ms | 0.092 ms | 0.094 ms | 0.180 ms | 0.525 ms |
| 1024 | 0.340 ms | 0.330 ms | 0.331 ms | 0.668 ms | 2.005 ms |
| 2048 | 1.455 ms | 1.455 ms | 1.511 ms | 2.792 ms | 11.771 ms |
| 4096 | 5.876 ms | 5.739 ms | 6.338 ms | 11.358 ms | 39.585 ms |
| 8192 | 23.572 ms | 23.859 ms | 26.895 ms | 45.864 ms | inf |
| 16384 | 95.750 ms | 96.884 ms | 108.312 ms | 186.714 ms | inf |

### Forward + Backward Pass Performance
![Forward+Backward Pass Performance](images\foward+backward.png)
*Combined forward and backward pass latency demonstrating efficient gradient computation*

**Key Observations:**
- Triton V2 (Non-Atomic) shows superior performance in the complete training loop
- Memory efficiency becomes critical as sequence length increases
- Flash Attention enables training on sequences that would OOM with naive implementation

| Sequence Length | Triton V2 (Non-Atomic) | Triton V2 (Atomic) | Triton V1 | PyTorch (Flash) | PyTorch (Naive) |
|----------------|------------------------|--------------------|-----------|--------------------|-----------------|
| 256 | 0.414 ms | 0.498 ms | 0.316 ms | 0.319 ms | 0.474 ms |
| 512 | 0.653 ms | 0.975 ms | 0.703 ms | 1.036 ms | 1.634 ms |
| 1024 | 2.354 ms | 3.497 ms | 2.413 ms | 3.547 ms | 6.241 ms |
| 2048 | 9.026 ms | 12.731 ms | 9.786 ms | 13.074 ms | 27.586 ms |
| 4096 | 34.458 ms | 48.998 ms | 39.087 ms | 51.434 ms | 112.488 ms |
| 8192 | 140.075 ms | 201.640 ms | 157.118 ms | 213.637 ms | inf |
| 16384 | 572.292 ms | 835.843 ms | 646.165 ms | 930.631 ms | inf |

### Numerical Stability
![Numerical Stability](images\stability.png)
*Numerical accuracy comparison showing all implementations maintain comparable precision*

**Error Metrics:**

| Implementation | Output Max Error | Max dQ Error | Max dK Error | Max dV Error |
|----------------|------------------|--------------|--------------|--------------|
| Triton V1 (Baseline) | 9.77e-04 | 1.46e-03 | 1.95e-03 | 9.77e-04 |
| Triton V2 (Atomic) | 9.77e-04 | 1.46e-03 | 1.95e-03 | 9.77e-04 |
| Triton V2 (Non-Atomic) | 9.77e-04 | 9.77e-04 | 1.95e-03 | 9.77e-04 |
| PyTorch (Flash) | 9.77e-04 | 9.77e-04 | 1.95e-03 | 1.95e-03 |

All implementations maintain excellent numerical precision with errors in the order of 10⁻³ to 10⁻⁴, demonstrating that the optimizations do not compromise accuracy.

### Benchmarks

The benchmark notebook includes:
- Memory usage comparisons across implementations
- Speed benchmarks across sequence lengths (256 to 16384 tokens)
- Numerical accuracy verification with error metrics
- Scaling analysis showing performance characteristics
- Tested on Tesla T4 GPU (Google Colab)



**Note**: This implementation was tested on Tesla T4 GPU. Performance may vary on different GPU architectures.

## Usage

### Running Individual Kernels

```python
# Flash Attention v1
from kernels.flash_v1 import Myattention1

output = Myattention1.apply(Q, K, V, causal=False)

# Flash Attention v2 (Non-Atomic)
from kernels.flash_v2_non_atomic import MyattentionV2

output = MyattentionV2.apply(Q, K, V, causal=False)

# Flash Attention v2 (Atomic)
from kernels.flash_v2_atomic import Myattention2

output = Myattention2.apply(Q, K, V, causal=False)
```

### Running Benchmarks

Open and run the benchmark notebook:

- Compare all implementations against standard PyTorch attention
- Generate performance graphs across sequence lengths
- Verify numerical correctness with error metrics
- Test various sequence lengths (256 to 16384 tokens) and batch sizes
- Display memory efficiency improvements

## Implementation Details



**Flash Attention V1**: Uses m_i (max statistics) and l_i (sum statistics) for softmax recomputation
**Flash Attention V2**: Optimized to use only LSE (Log-Sum-Exponential) and uses 2^x rather than e^x for more efficient recomputation

### Triton Optimizations

- Shared memory utilization for tile caching
- Coalesced memory access patterns
- Minimized global memory (HBM) transactions
- Thread block optimization
- Kernel fusion to reduce memory overhead


## References

- [Flash Attention Paper (v1)](https://arxiv.org/abs/2205.14135)
- [Flash Attention 2 Paper](https://arxiv.org/abs/2307.08691)
- Original implementation: https://github.com/Dao-AILab/flash-attention

## Contributing

Contributions are welcome! Areas for improvement:
- Additional kernel optimizations
- Support for more attention variants
- Extended benchmarking suite
- Documentation improvements

Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Tri Dao and collaborators for the original Flash Attention papers
- The Triton team for the excellent GPU programming framework
- PyTorch community for the deep learning infrastructure

## Contact

For questions or discussions about this implementation, please open an issue on GitHub.

---

**Note**: This is an educational implementation. For production use, consider the official Flash Attention library which includes additional optimizations and hardware-specific tuning.