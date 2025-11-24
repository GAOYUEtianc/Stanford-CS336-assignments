## nsys_profile (Small model with context length 256 on A100 as an example)
- The total forward pass time measured by Nsight Systems is 258.7 ms (25.87 ms/iteration), which matches within 4% of the inference-only measurement of 268.7 ms (26.87 ms/iteration), validating the accuracy of our Python timeit benchmarking with proper CUDA synchronization.
- The GEMM (matrix multiplication) kernel dominates GPU time during forward pass with approximately 96-144 invocations, and this same kernel family remains dominant during full training with roughly double the invocations (~288 times) due to gradient computations in the backward pass.
- Beyond matrix multiplications, softmax kernels consume 31.9 ms (12.3% of forward time) and elementwise operations for LayerNorm, GELU, and dropout account for another 10-15%, with softmax being particularly expensive despite low FLOPs due to its memory-bandwidth-bound nature.
-  The fraction of time spent on matrix multiplication decreases from approximately 75-78% during inference to 59% during full training, as the backward pass and AdamW optimizer introduce substantial elementwise operations (135 ms optimizer step) and memory-intensive gradient computations that grow faster than matmul operations.
- Attention matmuls (188 ms total) take 5.9× longer than softmax (31.9 ms), despite having ~206× more FLOPs, demonstrating that softmax's memory-bound nature (requiring multiple passes with low arithmetic intensity) makes it disproportionately expensive compared to compute-bound GEMMs that efficiently utilize tensor cores.

## Mixed precision 
```bash
Running mixed precision benchmarks...
==========================================
Testing SMALL model (124M params)...
-----------------------------------
  a) FP32 baseline...
Using CUDA: NVIDIA A100 80GB PCIe

============================================================
Benchmark Configuration
============================================================
Model: 12L, 768D, 12H, 3072FF
Batch size: 64
Context length: 256
Dtype: float32
Mixed Precision: Disabled
Warmup steps: 5
Measurement steps: 10
Mode: Forward only
============================================================

Initializing model...
Model parameters: 128,625,408
Generating random data...

Starting benchmark...
Using full precision (FP32)
Running 5 warmup iterations ...
Running 10 measurement iterations ...
    Iteration 1/10: 117.66 ms
    Iteration 2/10: 118.55 ms
    Iteration 3/10: 117.41 ms
    Iteration 4/10: 116.67 ms
    Iteration 5/10: 117.24 ms
    Iteration 6/10: 117.54 ms
    Iteration 7/10: 117.66 ms
    Iteration 8/10: 117.32 ms
    Iteration 9/10: 117.17 ms
    Iteration 10/10: 117.38 ms

============================================================
Forward Pass Timing Results

============================================================
Mean:  117.46 ms (0.117459 s)
Std:    0.46 ms (0.000455 s)
Min:    116.67 ms (0.116667 s)
Max:    118.55 ms (0.118553 s)
Median: 117.40 ms (0.117396 s)
CV:     0.39%
============================================================

Throughput: 139,487 tokens/second
Throughput: 139.49K tokens/second
GPU Memory Allocated: 0.53 GB
GPU Memory Reserved: 47.73 GB


  a) FP32 baseline with backward...
Using CUDA: NVIDIA A100 80GB PCIe

============================================================
Benchmark Configuration
============================================================
Model: 12L, 768D, 12H, 3072FF
Batch size: 64
Context length: 256
Dtype: float32
Mixed Precision: Disabled
Warmup steps: 5
Measurement steps: 10
Mode: Forward + Backward
============================================================

Initializing model...
Model parameters: 128,625,408
Generating random data...

Starting benchmark...
Using full precision (FP32)
Running 5 warmup iterations ...
Running 10 measurement iterations ...
    Iteration 1/10: 346.37 ms
    Iteration 2/10: 347.29 ms
    Iteration 3/10: 348.98 ms
    Iteration 4/10: 349.27 ms
    Iteration 5/10: 348.53 ms
    Iteration 6/10: 349.59 ms
    Iteration 7/10: 348.58 ms
    Iteration 8/10: 348.48 ms
    Iteration 9/10: 348.70 ms
    Iteration 10/10: 349.76 ms

============================================================
Forward+Backward Pass Timing Results

============================================================
Mean:  348.56 ms (0.348556 s)
Std:    0.98 ms (0.000980 s)
Min:    346.37 ms (0.346372 s)
Max:    349.76 ms (0.349761 s)
Median: 348.64 ms (0.348641 s)
CV:     0.28%
============================================================

Throughput: 47,005 tokens/second
Throughput: 47.01K tokens/second
GPU Memory Allocated: 0.54 GB
GPU Memory Reserved: 26.18 GB


  b) BF16 mixed precision...
Using CUDA: NVIDIA A100 80GB PCIe

============================================================
Benchmark Configuration
============================================================
Model: 12L, 768D, 12H, 3072FF
Batch size: 64
Context length: 256
Dtype: float32
Mixed Precision: Enabled (bfloat16)
Warmup steps: 5
Measurement steps: 10
Mode: Forward only
============================================================

Initializing model...
Model parameters: 128,625,408
Generating random data...

Starting benchmark...
Using AMP with torch.bfloat16
Running 5 warmup iterations ...
Running 10 measurement iterations ...
    Iteration 1/10: 100.91 ms
    Iteration 2/10: 108.45 ms
    Iteration 3/10: 101.71 ms
    Iteration 4/10: 102.16 ms
    Iteration 5/10: 101.69 ms
    Iteration 6/10: 107.95 ms
    Iteration 7/10: 102.90 ms
    Iteration 8/10: 101.86 ms
    Iteration 9/10: 102.14 ms
    Iteration 10/10: 101.51 ms

============================================================
Forward Pass Timing Results

============================================================
Mean:  103.13 ms (0.103129 s)
Std:    2.58 ms (0.002582 s)
Min:    100.91 ms (0.100913 s)
Max:    108.45 ms (0.108448 s)
Median: 102.00 ms (0.102004 s)
CV:     2.50%
============================================================

Throughput: 158,870 tokens/second
Throughput: 158.87K tokens/second
GPU Memory Allocated: 0.53 GB
GPU Memory Reserved: 33.61 GB


  b) BF16 mixed precision with backward...
Using CUDA: NVIDIA A100 80GB PCIe

============================================================
Benchmark Configuration
============================================================
Model: 12L, 768D, 12H, 3072FF
Batch size: 64
Context length: 256
Dtype: float32
Mixed Precision: Enabled (bfloat16)
Warmup steps: 5
Measurement steps: 10
Mode: Forward + Backward
============================================================

Initializing model...
Model parameters: 128,625,408
Generating random data...

Starting benchmark...
Using AMP with torch.bfloat16
Running 5 warmup iterations ...
Running 10 measurement iterations ...
    Iteration 1/10: 282.32 ms
    Iteration 2/10: 284.41 ms
    Iteration 3/10: 304.81 ms
    Iteration 4/10: 297.49 ms
    Iteration 5/10: 293.70 ms
    Iteration 6/10: 292.76 ms
    Iteration 7/10: 296.74 ms
    Iteration 8/10: 310.78 ms
    Iteration 9/10: 293.27 ms
    Iteration 10/10: 298.99 ms

============================================================
Forward+Backward Pass Timing Results

============================================================
Mean:  295.53 ms (0.295528 s)
Std:    8.06 ms (0.008065 s)
Min:    282.32 ms (0.282321 s)
Max:    310.78 ms (0.310779 s)
Median: 295.22 ms (0.295223 s)
CV:     2.73%
============================================================

Throughput: 55,440 tokens/second
Throughput: 55.44K tokens/second
GPU Memory Allocated: 0.54 GB
GPU Memory Reserved: 19.46 GB


-----------------------------------
Small model complete!

Testing LARGE model...
-----------------------------------
  a) FP32 baseline...
Using CUDA: NVIDIA A100 80GB PCIe

============================================================
Benchmark Configuration
============================================================
Model: 36L, 1280D, 20H, 5120FF
Batch size: 32
Context length: 256
Dtype: float32
Mixed Precision: Disabled
Warmup steps: 5
Measurement steps: 10
Mode: Forward only
============================================================

Initializing model...
Model parameters: 969,411,840
Generating random data...

Starting benchmark...
Using full precision (FP32)
Running 5 warmup iterations ...
Out of Memory !!!

  a) FP32 baseline with backward...
Using CUDA: NVIDIA A100 80GB PCIe

============================================================
Benchmark Configuration
============================================================
Model: 36L, 1280D, 20H, 5120FF
Batch size: 32
Context length: 256
Dtype: float32
Mixed Precision: Disabled
Warmup steps: 5
Measurement steps: 10
Mode: Forward + Backward
============================================================

Initializing model...
Model parameters: 969,411,840
Generating random data...

Starting benchmark...
Using full precision (FP32)
Running 5 warmup iterations ...
Running 10 measurement iterations ...
    Iteration 1/10: 1022.66 ms
    Iteration 2/10: 1023.38 ms
    Iteration 3/10: 1019.72 ms
    Iteration 4/10: 1030.57 ms
    Iteration 5/10: 1041.70 ms
    Iteration 6/10: 1023.79 ms
    Iteration 7/10: 1025.23 ms
    Iteration 8/10: 1021.86 ms
    Iteration 9/10: 1028.52 ms
    Iteration 10/10: 1024.11 ms

============================================================
Forward+Backward Pass Timing Results

============================================================
Mean:  1026.15 ms (1.026152 s)
Std:    5.98 ms (0.005975 s)
Min:    1019.72 ms (1.019721 s)
Max:    1041.70 ms (1.041698 s)
Median: 1023.95 ms (1.023950 s)
CV:     0.58%
============================================================

Throughput: 7,983 tokens/second
Throughput: 7.98K tokens/second
GPU Memory Allocated: 4.02 GB
GPU Memory Reserved: 58.26 GB


  b) BF16 mixed precision...
Using CUDA: NVIDIA A100 80GB PCIe

============================================================
Benchmark Configuration
============================================================
Model: 36L, 1280D, 20H, 5120FF
Batch size: 32
Context length: 256
Dtype: float32
Mixed Precision: Enabled (bfloat16)
Warmup steps: 5
Measurement steps: 10
Mode: Forward only
============================================================

Initializing model...
Model parameters: 969,411,840
Generating random data...

Starting benchmark...
Using AMP with torch.bfloat16
Running 5 warmup iterations ...
Running 10 measurement iterations ...
    Iteration 1/10: 280.72 ms
    Iteration 2/10: 277.58 ms
    Iteration 3/10: 308.03 ms
    Iteration 4/10: 311.77 ms
    Iteration 5/10: 299.62 ms
    Iteration 6/10: 285.97 ms
    Iteration 7/10: 278.07 ms
    Iteration 8/10: 300.87 ms
    Iteration 9/10: 307.34 ms
    Iteration 10/10: 293.39 ms

============================================================
Forward Pass Timing Results

============================================================
Mean:  294.34 ms (0.294336 s)
Std:    12.38 ms (0.012382 s)
Min:    277.58 ms (0.277577 s)
Max:    311.77 ms (0.311772 s)
Median: 296.51 ms (0.296508 s)
CV:     4.21%
============================================================

Throughput: 27,832 tokens/second
Throughput: 27.83K tokens/second
GPU Memory Allocated: 4.01 GB
GPU Memory Reserved: 80.44 GB


  b) BF16 mixed precision...
Using CUDA: NVIDIA A100 80GB PCIe

============================================================
Benchmark Configuration
============================================================
Model: 36L, 1280D, 20H, 5120FF
Batch size: 32
Context length: 256
Dtype: float32
Mixed Precision: Enabled (bfloat16)
Warmup steps: 5
Measurement steps: 10
Mode: Forward + Backward
============================================================

Initializing model...
Model parameters: 969,411,840
Generating random data...

Starting benchmark...
Using AMP with torch.bfloat16
Running 5 warmup iterations ...
Running 10 measurement iterations ...
    Iteration 1/10: 794.96 ms
    Iteration 2/10: 793.65 ms
    Iteration 3/10: 796.80 ms
    Iteration 4/10: 796.64 ms
    Iteration 5/10: 796.23 ms
    Iteration 6/10: 806.78 ms
    Iteration 7/10: 825.42 ms
    Iteration 8/10: 819.23 ms
    Iteration 9/10: 812.03 ms
    Iteration 10/10: 813.30 ms

============================================================
Forward+Backward Pass Timing Results

============================================================
Mean:  805.50 ms (0.805505 s)
Std:    10.87 ms (0.010872 s)
Min:    793.65 ms (0.793654 s)
Max:    825.42 ms (0.825417 s)
Median: 801.79 ms (0.801790 s)
CV:     1.35%
============================================================

Throughput: 10,170 tokens/second
Throughput: 10.17K tokens/second
GPU Memory Allocated: 4.02 GB
GPU Memory Reserved: 43.48 GB


-----------------------------------
Large model complete!
```
### Performance Comparison
### Small Model (124M params) - Batch Size 64
|Mode	| FP32	| BF16 |	Speedup|
|---|---|---|---|
Forward Pass|	117.46 ms|	103.13 ms|	BF16 13.9% faster|
|Forward + Backward	|348.56 ms	|295.53 ms|	BF16 17.9% faster|
|Forward Throughput|	139.49K/s|	158.87K/s|	BF16 13.9% higher|
|Training Throughput|	47.01K/s|55.44K/s|	BF16 17.9% higher

### Large Model (969M params) - Batch Size 32

|Mode	|FP32|	BF16|	Speedup|
|---|---|---|---|
|Forward Pass|	OOM	|294.34 ms|	BF16 avoids OOM|
|Forward + Backward|	1026.15 ms|	805.50 ms|	BF16 27.4% faster|
|Training Throughput	|7.98K/s	|10.17K/s|	BF16 27.4% higher|

## Dramatic Memory Efficiency Improvement
Small Model Memory Usage Comparison:
FP32 Reserved Memory: 47.73GB → 26.18GB
BF16 Reserved Memory: 33.61GB → 19.46GB
BF16 Memory Saving: ~30-35%

Large Model Memory Usage:
BF16 enables batch_size=32 (FP32 causes OOM)

## Memory Profiling
The forward pass memory timeline shows the highest peak as PyTorch must retain all intermediate activations for later gradient computation, with memory steadily increasing layer-by-layer through the model. The full training step timeline displays three distinct peaks: the tallest during the forward pass (all activations retained), followed by two lower, roughly equal peaks during the backward pass (gradients + progressively freed activations) and optimizer step (parameter updates using gradients and optimizer momentum/variance states). The forward peak is highest because activations are fully accumulated and not yet freed, while backward and optimizer peaks are similar in height as they both maintain parameters, gradients, and optimizer states but no longer need the full activation memory.

```
Memory
  ^
  |     Forward peak
  |        /\
  |       /  \
  |      /     \  Backward  Optimizer
  |     /        \ —— —— —— —— —— \
  |    /                            \
  |___/____________________________> Time
       ↑         ↑        ↑
    Forward  Backward  Optimizer
```
AMP saves memory compared with FP32

## Memory scaling with sequence length
```
====================================================================================================
Attention Benchmarking (vanilla PyTorch)
====================================================================================================
Batch size: 8
d_model values: [16, 32, 64, 128]
Sequence lengths: [256, 1024, 4096, 8192, 16384]
====================================================================================================

d_model    seq_len    Forward (ms)    Backward (ms)   Mem Before Bwd (GB)  Peak Mem (GB)   Status         
----------------------------------------------------------------------------------------------------
16         256        0.258           0.806           0.020                0.026           OK             
16         1024       0.611           1.423           0.053                0.155           OK             
16         4096       4.221           10.235          0.562                2.179           OK             
16         8192       14.186          34.607          2.181                8.636           OK             
16         16384      56.908          140.362         8.641                34.435          OK             
32         256        0.334           0.897           0.020                0.027           OK             
32         1024       0.587           1.435           0.055                0.159           OK             
32         4096       4.395           10.450          0.571                2.194           OK             
32         8192       15.077          35.515          2.198                8.666           OK             
32         16384      60.498          144.024         8.674                34.494          OK             
64         256        0.345           0.898           0.021                0.029           OK             
64         1024       0.575           1.426           0.059                0.166           OK             
64         4096       4.854           10.964          0.587                2.223           OK             
64         8192       16.891          37.418          2.232                8.724           OK             
64         16384      67.748          151.426         8.741                34.612          OK             
128        256        0.344           0.920           0.023                0.033           OK             
128        1024       0.664           1.502           0.067                0.181           OK             
128        4096       5.740           11.845          0.621                2.282           OK             
128        8192       20.332          40.973          2.299                8.842           OK             
128        16384      81.476          165.394         8.875                34.847          OK             
====================================================================================================

```
Memory for backward scales as O(seq_len²) due to attention matrix!

Compare with torch.compile : 
```

====================================================================================================
Attention Benchmarking (torch.compile)
====================================================================================================
Batch size: 8
d_model values: [16, 32, 64, 128]
Sequence lengths: [256, 1024, 4096, 8192, 16384]
====================================================================================================

Compiling attention with torch.compile()...
Compilation complete.

d_model    seq_len    Forward (ms)    Backward (ms)   Mem Before Bwd (GB)  Peak Mem (GB)   Status         
----------------------------------------------------------------------------------------------------
/workspace/Stanford-CS336-assignments/assignment2-systems/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py:194: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
  warnings.warn(
16         256        0.453           1.162           0.020                0.025           OK             
16         1024       0.603           1.296           0.053                0.123           OK             
16         4096       3.662           9.759           0.562                1.649           OK             
16         8192       13.448          34.864          2.181                6.501           OK             
16         16384      53.928          131.751         8.641                25.871          OK             
32         256        0.303           0.861           0.020                0.026           OK             
32         1024       0.522           1.354           0.055                0.128           OK             
32         4096       4.481           10.577          0.571                1.670           OK             
32         8192       15.660          35.994          2.198                6.543           OK             
32         16384      57.566          135.394         8.674                25.955          OK             
64         256        0.334           0.662           0.021                0.029           OK             
64         1024       0.607           1.420           0.059                0.139           OK             
64         4096       4.945           11.085          0.587                1.712           OK             
64         8192       17.464          37.859          2.232                6.627           OK             
64         16384      64.844          142.773         8.741                26.122          OK             
128        256        0.373           0.961           0.023                0.034           OK             
128        1024       0.707           1.542           0.067                0.160           OK             
128        4096       5.819           11.984          0.621                1.795           OK             
128        8192       20.923          41.536          2.299                6.795           OK             
128        16384      81.510          165.312         8.875                34.847          OK             
====================================================================================================
```
It can be observed that when seq is long, torch.compile saves memory significantly (around 25%). For very short sequence, the forward time of torch.compile is even slightly worse than vanilla, that is because when sequence is very short, compiling cost might be higher than the benefit of compiling. For backward process, torch.compile saved time. 

## Transformer compilation benchmark
```
================================================================================
Transformer Compilation Benchmark
================================================================================

Benchmarking VANILLA Transformer
Config: Large (774M)
--------------------------------------------------------------------------------
Parameters: 969,411,840

Benchmarking forward pass...
Benchmarking full training step...

Benchmarking COMPILED Transformer
Config: Large (774M)
--------------------------------------------------------------------------------
Parameters: 969,411,840
Compiling model...
Compilation complete.

Benchmarking forward pass...
Benchmarking full training step...

================================================================================
RESULTS COMPARISON
================================================================================

Metric                         Vanilla (ms)         Compiled (ms)        Speedup        
--------------------------------------------------------------------------------
Forward Pass                   318.78               283.66               1.12           x
Full Training Step             1045.14              922.44               1.13           x
Backward + Optimizer           723.68               640.30               1.13           x
================================================================================

ANALYSIS:
- Forward pass speedup: 1.12x
- Full training step speedup: 1.13x
✓ torch.compile provides meaningful speedup!

Reasons for speedup:
- Kernel fusion (reduces memory traffic)
- Operator specialization (optimized for specific shapes)
- Reduced Python overhead
```

## DDP Results
### 2.2 DDP flatten on \& after backward
This is the result running on A-100, 2 GPU, communicating with gloo, concatenating all parameter gradients into a single flat tensor, performing only ONE all-reduce operation (instead of one per parameter), hence reducing communication time : 
```
================================================================================
Method               Total Time (ms)      Comm Time (ms)       Compute Time (ms)   
--------------------------------------------------------------------------------
Naive (Individual)   285.480              110.965              174.515             
Flattened (Batched)  244.271              68.141               176.130             
================================================================================

Speedup (Flattened vs Naive):
  Total Time: 1.17x faster
  Communication Time: 1.63x faster
  Communication Overhead: Naive 38.9% vs Flattened 27.9%
```
