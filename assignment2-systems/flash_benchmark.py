import torch
import torch.nn as nn
from torch.autograd import Function
import triton
import numpy as np
from typing import List, Tuple
import pandas as pd
from itertools import product
import sys
import os

# Import your FlashAttention implementation
# Adjust the import path based on your project structure
try:
    from flashattention2 import FlashAttention2Triton
    FLASH_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import paths
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from flashattention2 import FlashAttention2Triton
        FLASH_AVAILABLE = True
    except ImportError:
        print("Warning: FlashAttention2Triton not found. Please ensure it's in your Python path.")
        print("Current sys.path:", sys.path[:3])
        FLASH_AVAILABLE = False


def pytorch_attention_forward_backward(Q, K, V, mask=None):
    """
    Standard PyTorch scaled dot-product attention with forward and backward.
    
    Args:
        Q: Query tensor [batch_size, num_heads, seq_len, d_k]
        K: Key tensor [batch_size, num_heads, seq_len, d_k]
        V: Value tensor [batch_size, num_heads, seq_len, d_k]
        mask: Optional causal mask
    
    Returns:
        Output tensor
    """
    d_k = Q.size(-1)
    
    # Enable gradient computation
    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)
    
    # Forward pass
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    
    return output


def create_causal_mask(seq_len: int, device: str) -> torch.Tensor:
    """Create a causal mask for attention."""
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask


def benchmark_forward(fn, *args, **kwargs):
    """Benchmark forward pass only."""
    return triton.testing.do_bench(lambda: fn(*args, **kwargs))


def benchmark_backward(fn, output_shape, *args, **kwargs):
    """Benchmark backward pass only."""
    def backward_fn():
        output = fn(*args, **kwargs)
        grad_output = torch.randn_like(output)
        torch.autograd.backward(output, grad_output)
    
    return triton.testing.do_bench(backward_fn)


def benchmark_forward_backward(fn, output_shape, *args, **kwargs):
    """Benchmark combined forward and backward pass."""
    def forward_backward_fn():
        output = fn(*args, **kwargs)
        grad_output = torch.randn_like(output)
        torch.autograd.backward(output, grad_output)
    
    return triton.testing.do_bench(forward_backward_fn)


def run_benchmarks(
    seq_lengths: List[int],
    embed_dims: List[int],
    dtypes: List[torch.dtype],
    batch_size: int = 1,
    num_heads: int = 8,
    device: str = 'cuda'
):
    """
    Run comprehensive benchmarks comparing FlashAttention-2 with PyTorch attention.
    
    Args:
        seq_lengths: List of sequence lengths to test
        embed_dims: List of embedding dimensions to test
        dtypes: List of data types to test
        batch_size: Batch size (default: 1)
        num_heads: Number of attention heads
        device: Device to run on
    
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    print("Starting benchmarks...")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Num heads: {num_heads}")
    print("-" * 80)
    
    for seq_len, embed_dim, dtype in product(seq_lengths, embed_dims, dtypes):
        d_k = embed_dim // num_heads
        
        print(f"\nBenchmarking: seq_len={seq_len}, embed_dim={embed_dim}, dtype={dtype}")
        
        # Generate random inputs
        Q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=dtype)
        K = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=dtype)
        V = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=dtype)
        
        # Create causal mask
        mask = create_causal_mask(seq_len, device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Clone inputs for fair comparison
        Q_pt = Q.clone().detach()
        K_pt = K.clone().detach()
        V_pt = V.clone().detach()
        
        Q_flash = Q.clone().detach()
        K_flash = K.clone().detach()
        V_flash = V.clone().detach()
        
        try:
            # Benchmark PyTorch implementation
            print("  Benchmarking PyTorch...")
            
            # Forward only
            pt_forward_time = benchmark_forward(
                pytorch_attention_forward_backward,
                Q_pt, K_pt, V_pt, mask
            )
            
            # Forward + Backward
            pt_fwd_bwd_time = benchmark_forward_backward(
                pytorch_attention_forward_backward,
                (batch_size, num_heads, seq_len, d_k),
                Q_pt.clone().detach(), 
                K_pt.clone().detach(), 
                V_pt.clone().detach(), 
                mask
            )
            
            # Backward only (approximation)
            pt_backward_time = pt_fwd_bwd_time - pt_forward_time
            
            # Benchmark FlashAttention-2 if available
            if FLASH_AVAILABLE:
                print("  Benchmarking FlashAttention-2...")
                
                # Forward only
                flash_forward_time = benchmark_forward(
                    FlashAttention2Triton.apply,
                    Q_flash, K_flash, V_flash, True  # causal=True
                )
                
                # Forward + Backward
                flash_fwd_bwd_time = benchmark_forward_backward(
                    FlashAttention2Triton.apply,
                    (batch_size, num_heads, seq_len, d_k),
                    Q_flash.clone().detach(),
                    K_flash.clone().detach(),
                    V_flash.clone().detach(),
                    True  # causal=True
                )
                
                # Backward only (approximation)
                flash_backward_time = flash_fwd_bwd_time - flash_forward_time
                
                # Calculate speedups
                speedup_forward = pt_forward_time / flash_forward_time
                speedup_backward = pt_backward_time / flash_backward_time
                speedup_fwd_bwd = pt_fwd_bwd_time / flash_fwd_bwd_time
            else:
                flash_forward_time = None
                flash_backward_time = None
                flash_fwd_bwd_time = None
                speedup_forward = None
                speedup_backward = None
                speedup_fwd_bwd = None
            
            # Store results
            result = {
                'seq_len': seq_len,
                'embed_dim': embed_dim,
                'dtype': str(dtype).split('.')[-1],
                'pt_forward_ms': pt_forward_time,
                'pt_backward_ms': pt_backward_time,
                'pt_fwd_bwd_ms': pt_fwd_bwd_time,
                'flash_forward_ms': flash_forward_time,
                'flash_backward_ms': flash_backward_time,
                'flash_fwd_bwd_ms': flash_fwd_bwd_time,
                'speedup_forward': speedup_forward,
                'speedup_backward': speedup_backward,
                'speedup_fwd_bwd': speedup_fwd_bwd
            }
            
            results.append(result)
            
            print(f"  PyTorch    - Forward: {pt_forward_time:.3f}ms, Backward: {pt_backward_time:.3f}ms, Fwd+Bwd: {pt_fwd_bwd_time:.3f}ms")
            if FLASH_AVAILABLE:
                print(f"  Flash-2    - Forward: {flash_forward_time:.3f}ms, Backward: {flash_backward_time:.3f}ms, Fwd+Bwd: {flash_fwd_bwd_time:.3f}ms")
                print(f"  Speedup    - Forward: {speedup_forward:.2f}x, Backward: {speedup_backward:.2f}x, Fwd+Bwd: {speedup_fwd_bwd:.2f}x")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df


def format_results_table(df: pd.DataFrame) -> str:
    """Format results as a nice LaTeX-style table."""
    if df.empty:
        return "No results to display"
    
    # Create formatted table
    table = []
    table.append("=" * 150)
    table.append(f"{'Seq Len':<10} {'Embed':<8} {'Dtype':<10} {'PyTorch Fwd':<15} {'PyTorch Bwd':<15} {'PyTorch F+B':<15} {'Flash Fwd':<15} {'Flash Bwd':<15} {'Flash F+B':<15} {'Speedup':<10}")
    table.append("=" * 150)
    
    for _, row in df.iterrows():
        speedup_str = f"{row['speedup_fwd_bwd']:.2f}x" if row['speedup_fwd_bwd'] is not None else "N/A"
        flash_fwd = f"{row['flash_forward_ms']:.3f}ms" if row['flash_forward_ms'] is not None else "N/A"
        flash_bwd = f"{row['flash_backward_ms']:.3f}ms" if row['flash_backward_ms'] is not None else "N/A"
        flash_fb = f"{row['flash_fwd_bwd_ms']:.3f}ms" if row['flash_fwd_bwd_ms'] is not None else "N/A"
        
        table.append(
            f"{row['seq_len']:<10} {row['embed_dim']:<8} {row['dtype']:<10} "
            f"{row['pt_forward_ms']:<15.3f} {row['pt_backward_ms']:<15.3f} {row['pt_fwd_bwd_ms']:<15.3f} "
            f"{flash_fwd:<15} {flash_bwd:<15} {flash_fb:<15} {speedup_str:<10}"
        )
    
    table.append("=" * 150)
    
    return "\n".join(table)


def main():
    """Main benchmarking script."""
    # Configuration
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    embed_dims = [16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("Warning: Running on CPU. Results may not be representative.")
    
    # Run benchmarks
    results_df = run_benchmarks(
        seq_lengths=seq_lengths,
        embed_dims=embed_dims,
        dtypes=dtypes,
        batch_size=1,
        num_heads=8,
        device=device
    )
    
    # Save results
    results_df.to_csv('flash_attention_benchmark_results.csv', index=False)
    print("\n\nResults saved to 'flash_attention_benchmark_results.csv'")
    
    # Print formatted table
    print("\n\nBenchmark Results:")
    print(format_results_table(results_df))
    
    # Print summary statistics
    if not results_df.empty and 'speedup_fwd_bwd' in results_df.columns:
        print("\n\nSummary Statistics:")
        print(f"Average Forward+Backward Speedup: {results_df['speedup_fwd_bwd'].mean():.2f}x")
        print(f"Max Forward+Backward Speedup: {results_df['speedup_fwd_bwd'].max():.2f}x")
        print(f"Min Forward+Backward Speedup: {results_df['speedup_fwd_bwd'].min():.2f}x")


if __name__ == "__main__":
    main()