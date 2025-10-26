import sys
import os

# Add the assignment1 directory to Python path
assignment1_dir = os.path.join(os.path.dirname(__file__), '..', 'assignment1')
sys.path.insert(0, assignment1_dir)

# Now import from transformer
from transformer import TransformerLM, count_parameters, cross_entropy_loss, AdamW

# Import what we need from train_transformer if available
try:
    from train_transformer import *
except ImportError:
    print("Warning: Could not import from train_transformer, some features may not be available")


import argparse
import timeit
import time
import numpy as np 
import torch 
from itertools import product

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Standard scaled dot-product attention.
    
    Args:
        Q: Query [batch, seq_len, d_k]
        K: Key [batch, seq_len, d_k]
        V: Value [batch, seq_len, d_k]
        mask: Optional mask
    
    Returns:
        output, attention_weights
    """
    d_k = Q.size(-1)
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)
    
    # Weighted sum of values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


def benchmark_attention(
    attention_fn,
    Q, K, V,
    num_warmup=10,
    num_iters=100,
    device='cuda'
):
    """
    Benchmark attention forward and backward passes.
    
    Returns:
        dict with forward_time, backward_time, peak_memory
    """
    
    # Warmup
    for _ in range(num_warmup):
        output, _ = attention_fn(Q, K, V)
        loss = output.sum()
        loss.backward()
        Q.grad = None
        K.grad = None
        V.grad = None
        torch.cuda.synchronize()
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    
    # Benchmark forward pass
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_iters):
        output, _ = attention_fn(Q, K, V)
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    forward_time = (end - start) / num_iters
    
    # Measure memory before backward
    memory_before_backward = torch.cuda.memory_allocated() / 1e9
    
    # Benchmark backward pass
    output, _ = attention_fn(Q, K, V)
    loss = output.sum()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_iters):
        loss.backward(retain_graph=True)
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    backward_time = (end - start) / num_iters
    
    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    
    torch.cuda.reset_peak_memory_stats()
    
    return {
        'forward_time': forward_time,
        'backward_time': backward_time,
        'memory_before_backward_gb': memory_before_backward,
        'peak_memory_gb': peak_memory
    }


def run_benchmarks(use_compile=False):
    """Run attention benchmarks across different configurations."""
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    
    device = 'cuda'
    batch_size = 8
    
    # Configuration ranges
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    
    print("="*100)
    print(f"Attention Benchmarking {'(torch.compile)' if use_compile else '(vanilla PyTorch)'}")
    print("="*100)
    print(f"Batch size: {batch_size}")
    print(f"d_model values: {d_models}")
    print(f"Sequence lengths: {seq_lens}")
    print("="*100)
    print()
    
    # Prepare attention function
    attention_fn = scaled_dot_product_attention
    if use_compile:
        print("Compiling attention with torch.compile()...")
        attention_fn = torch.compile(attention_fn)
        print("Compilation complete.")
        print()
    
    # Results storage
    results = []
    
    # Header
    print(f"{'d_model':<10} {'seq_len':<10} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Mem Before Bwd (GB)':<20} {'Peak Mem (GB)':<15} {'Status':<15}")
    print("-"*100)
    
    for d_model, seq_len in product(d_models, seq_lens):
        try:
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Create inputs
            Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            
            # Run benchmark
            result = benchmark_attention(attention_fn, Q, K, V, device=device)
            
            # Store results
            results.append({
                'd_model': d_model,
                'seq_len': seq_len,
                'forward_ms': result['forward_time'] * 1000,
                'backward_ms': result['backward_time'] * 1000,
                'memory_before_backward_gb': result['memory_before_backward_gb'],
                'peak_memory_gb': result['peak_memory_gb'],
                'status': 'OK'
            })
            
            print(f"{d_model:<10} {seq_len:<10} {result['forward_time']*1000:<15.3f} "
                  f"{result['backward_time']*1000:<15.3f} {result['memory_before_backward_gb']:<20.3f} "
                  f"{result['peak_memory_gb']:<15.3f} {'OK':<15}")
            
            # Clean up
            del Q, K, V
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                results.append({
                    'd_model': d_model,
                    'seq_len': seq_len,
                    'forward_ms': None,
                    'backward_ms': None,
                    'memory_before_backward_gb': None,
                    'peak_memory_gb': None,
                    'status': 'OOM'
                })
                print(f"{d_model:<10} {seq_len:<10} {'N/A':<15} {'N/A':<15} {'N/A':<20} {'N/A':<15} {'OOM':<15}")
                torch.cuda.empty_cache()
            else:
                raise e
    
    print("="*100)
    print()
    
    return results


def analyze_memory_usage(batch_size=8, seq_len=8192, d_model=128):
    """
    Analyze memory usage for attention.
    This helps answer the memory accounting question.
    """
    print("="*80)
    print("Memory Analysis for Attention")
    print("="*80)
    print(f"Configuration: batch={batch_size}, seq_len={seq_len}, d_model={d_model}")
    print()
    
    # Input tensors: Q, K, V
    input_size = batch_size * seq_len * d_model * 4  # FP32 = 4 bytes
    total_input = 3 * input_size  # Q, K, V
    print(f"Input tensors (Q, K, V):")
    print(f"  Each: {batch_size} × {seq_len} × {d_model} × 4 bytes = {input_size/1e9:.3f} GB")
    print(f"  Total: {total_input/1e9:.3f} GB")
    print()
    
    # Attention scores: [batch, seq_len, seq_len]
    scores_size = batch_size * seq_len * seq_len * 4
    print(f"Attention scores matrix (Q @ K^T):")
    print(f"  {batch_size} × {seq_len} × {seq_len} × 4 bytes = {scores_size/1e9:.3f} GB")
    print()
    
    # Attention weights (after softmax): same size as scores
    weights_size = scores_size
    print(f"Attention weights (after softmax):")
    print(f"  {batch_size} × {seq_len} × {seq_len} × 4 bytes = {weights_size/1e9:.3f} GB")
    print()
    
    # Output: [batch, seq_len, d_model]
    output_size = batch_size * seq_len * d_model * 4
    print(f"Output tensor:")
    print(f"  {batch_size} × {seq_len} × {d_model} × 4 bytes = {output_size/1e9:.3f} GB")
    print()
    
    # Gradients (backward pass)
    print("Gradients (backward pass):")
    print(f"  dQ, dK, dV: {total_input/1e9:.3f} GB (same as inputs)")
    print(f"  d(scores): {scores_size/1e9:.3f} GB")
    print(f"  d(weights): {weights_size/1e9:.3f} GB")
    print()
    
    # Total forward memory (activations to save for backward)
    forward_activations = scores_size + weights_size  # Need to save for backward
    print(f"Activations saved for backward:")
    print(f"  Scores + Weights: {forward_activations/1e9:.3f} GB")
    print()
    
    # Peak memory estimate
    peak_memory = total_input + scores_size + weights_size + output_size + total_input  # +gradients
    print(f"Estimated peak memory: {peak_memory/1e9:.3f} GB")
    print()
    
    # Scaling analysis
    print("="*80)
    print("Memory Scaling with Sequence Length")
    print("="*80)
    print()
    print("Key observation: Attention scores are O(seq_len²)")
    print()
    print(f"{'Seq Len':<12} {'Scores Memory (GB)':<25} {'Scaling':<15}")
    print("-"*50)
    
    for s in [1024, 2048, 4096, 8192, 16384]:
        mem = batch_size * s * s * 4 / 1e9
        scaling = (s / 1024) ** 2
        print(f"{s:<12} {mem:<25.3f} {scaling:<15.1f}x")
    
    print()
    print("Memory for backward scales as O(seq_len²) due to attention matrix!")
    print()
    print("="*80)
    print("How to Eliminate This Memory Cost")
    print("="*80)
    print("""
1. FlashAttention: Recompute attention scores during backward instead of saving
   - Trade computation for memory
   - Fused kernel makes recomputation fast
   - Memory: O(seq_len) instead of O(seq_len²)

2. Gradient Checkpointing: Save only some activations, recompute others
   - Selective recomputation during backward
   - Reduces memory at cost of ~30% more compute

3. Memory-Efficient Attention: Compute attention in tiles/blocks
   - Never materialize full seq_len × seq_len matrix
   - Process in chunks that fit in SRAM
   - FlashAttention-2 approach

4. Sparse Attention: Only compute subset of attention connections
   - Local attention windows
   - Strided patterns (BigBird, Longformer)
   - Reduces both compute and memory to O(seq_len)
""")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Benchmark Attention Implementations')
    parser.add_argument('--compile', action='store_true', 
                       help='Use torch.compile on attention')
    parser.add_argument('--analyze_memory', action='store_true',
                       help='Show detailed memory analysis')
    
    args = parser.parse_args()
    
    # Run benchmarks
    print("\n")
    results = run_benchmarks(use_compile=args.compile)
    
    # Analyze memory if requested
    if args.analyze_memory:
        print("\n")
        # Find smallest OOM configuration
        oom_configs = [r for r in results if r['status'] == 'OOM']
        if oom_configs:
            smallest_oom = min(oom_configs, key=lambda x: x['seq_len'])
            analyze_memory_usage(
                batch_size=8,
                seq_len=smallest_oom['seq_len'],
                d_model=smallest_oom['d_model']
            )
        else:
            # Use largest config
            analyze_memory_usage(batch_size=8, seq_len=16384, d_model=128)
    
    # Summary
    print("\n")
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r['status'] == 'OK']
    failed = [r for r in results if r['status'] == 'OOM']
    
    print(f"Successful configurations: {len(successful)}")
    print(f"Out of memory: {len(failed)}")
    
    if failed:
        print(f"\nSmallest OOM configuration:")
        smallest_oom = min(failed, key=lambda x: (x['seq_len'], x['d_model']))
        print(f"  d_model={smallest_oom['d_model']}, seq_len={smallest_oom['seq_len']}")
    
    print("="*80)


if __name__ == "__main__":
    main()