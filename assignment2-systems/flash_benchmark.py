#!/usr/bin/env python3
"""
FlashAttention-2 vs PyTorch Attention Benchmarking Script
Requires: torch, triton, pandas, numpy
"""

import torch
import torch.nn as nn
import triton
import numpy as np
from typing import List, Optional
import pandas as pd
from itertools import product
import sys
import os

# ============================================================================
# STEP 1: Import your FlashAttention implementation
# ============================================================================
# TODO: Adjust this import to match your project structure
# Option A: If flashattention2.py is in the same directory
try:
    from flashattention2 import FlashAttention2Triton
    FLASH_AVAILABLE = True
    print("✓ FlashAttention2Triton imported successfully")
except ImportError as e:
    print(f"✗ Failed to import FlashAttention2Triton: {e}")
    print(f"  Current directory: {os.getcwd()}")
    print(f"  Python path: {sys.path[:3]}")
    
    # Option B: Try to add parent directory to path
    try:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        from flashattention2 import FlashAttention2Triton
        FLASH_AVAILABLE = True
        print("✓ FlashAttention2Triton imported from parent directory")
    except ImportError:
        FLASH_AVAILABLE = False
        print("✗ FlashAttention2Triton not available. Only PyTorch benchmarks will run.")


# ============================================================================
# PyTorch Baseline Implementation
# ============================================================================

class PyTorchAttention(nn.Module):
    """Standard PyTorch scaled dot-product attention for benchmarking."""
    
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
    
    def forward(self, Q, K, V, causal=True):
        """
        Args:
            Q: [batch, seq_len, embed_dim] (3D format to match FlashAttention)
            K: [batch, seq_len, embed_dim]
            V: [batch, seq_len, embed_dim]
            causal: Whether to apply causal masking
        Returns:
            output: [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = Q.shape
        d_k = embed_dim // self.num_heads
        
        # Reshape to [batch, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        
        # Apply causal mask
        if causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax and weighted sum
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        # Reshape back to [batch, seq_len, embed_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return output


# ============================================================================
# Benchmarking Functions
# ============================================================================

def benchmark_forward_only(fn, *args, warmup=25, rep=100):
    """Benchmark forward pass only using triton.testing.do_bench."""
    return triton.testing.do_bench(lambda: fn(*args), warmup=warmup, rep=rep)


def benchmark_backward_only(fn, output_shape, *args, warmup=25, rep=100):
    """Benchmark backward pass only."""
    def backward_fn():
        # Clone inputs to avoid accumulation
        inputs = [arg.clone().detach().requires_grad_(True) if isinstance(arg, torch.Tensor) and arg.requires_grad is not False else arg for arg in args]
        output = fn(*inputs)
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
    
    return triton.testing.do_bench(backward_fn, warmup=warmup, rep=rep)


def benchmark_forward_backward(fn, *args, warmup=25, rep=100):
    """Benchmark combined forward and backward pass."""
    def fwd_bwd_fn():
        inputs = [arg.clone().detach().requires_grad_(True) if isinstance(arg, torch.Tensor) and arg.requires_grad is not False else arg for arg in args]
        output = fn(*inputs)
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
    
    return triton.testing.do_bench(fwd_bwd_fn, warmup=warmup, rep=rep)


# ============================================================================
# Main Benchmarking Loop
# ============================================================================

def run_single_benchmark(seq_len, embed_dim, dtype, batch_size=1, num_heads=8, device='cuda'):
    """Run benchmark for a single configuration."""
    
    d_k = embed_dim // num_heads
    
    print(f"  Testing: seq_len={seq_len:>6}, embed_dim={embed_dim:>3}, dtype={str(dtype).split('.')[-1]:<10}", end='')
    
    try:
        # Generate inputs in 3D format: [batch, seq, embed_dim]
        Q = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        K = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        V = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        
        # PyTorch baseline
        pytorch_attn = PyTorchAttention(num_heads=num_heads).to(device)
        
        # Benchmark PyTorch
        Q_pt, K_pt, V_pt = Q.clone(), K.clone(), V.clone()
        Q_pt.requires_grad_(True)
        K_pt.requires_grad_(True)
        V_pt.requires_grad_(True)
        
        pt_forward_time = benchmark_forward_only(pytorch_attn, Q_pt, K_pt, V_pt, True)
        pt_fwd_bwd_time = benchmark_forward_backward(pytorch_attn, Q_pt, K_pt, V_pt, True)
        pt_backward_time = pt_fwd_bwd_time - pt_forward_time
        
        result = {
            'seq_len': seq_len,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'd_k': d_k,
            'dtype': str(dtype).split('.')[-1],
            'pt_forward_ms': pt_forward_time,
            'pt_backward_ms': pt_backward_time,
            'pt_fwd_bwd_ms': pt_fwd_bwd_time,
        }
        
        # Benchmark FlashAttention if available
        if FLASH_AVAILABLE:
            Q_flash, K_flash, V_flash = Q.clone(), K.clone(), V.clone()
            Q_flash.requires_grad_(True)
            K_flash.requires_grad_(True)
            V_flash.requires_grad_(True)
            
            # FlashAttention expects 3D: [batch, seq, embed_dim]
            flash_forward_time = benchmark_forward_only(
                FlashAttention2Triton.apply, Q_flash, K_flash, V_flash, True
            )
            flash_fwd_bwd_time = benchmark_forward_backward(
                FlashAttention2Triton.apply, Q_flash, K_flash, V_flash, True
            )
            flash_backward_time = flash_fwd_bwd_time - flash_forward_time
            
            result.update({
                'flash_forward_ms': flash_forward_time,
                'flash_backward_ms': flash_backward_time,
                'flash_fwd_bwd_ms': flash_fwd_bwd_time,
                'speedup_forward': pt_forward_time / flash_forward_time,
                'speedup_backward': pt_backward_time / flash_backward_time,
                'speedup_fwd_bwd': pt_fwd_bwd_time / flash_fwd_bwd_time,
            })
            
            print(f" → Speedup: {result['speedup_fwd_bwd']:.2f}x")
        else:
            result.update({
                'flash_forward_ms': None,
                'flash_backward_ms': None,
                'flash_fwd_bwd_ms': None,
                'speedup_forward': None,
                'speedup_backward': None,
                'speedup_fwd_bwd': None,
            })
            print(" → Flash not available")
        
        return result
        
    except Exception as e:
        print(f" → Error: {e}")
        return None


def run_all_benchmarks(
    seq_lengths=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
    embed_dims=[16, 32, 64, 128],
    dtypes=[torch.bfloat16, torch.float32],
    batch_size=1,
    num_heads=8,
    device='cuda'
):
    """Run benchmarks across all configurations."""
    
    print("=" * 100)
    print("FlashAttention-2 Benchmarking")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Num heads: {num_heads}")
    print(f"FlashAttention available: {FLASH_AVAILABLE}")
    print("=" * 100)
    
    results = []
    total_configs = len(seq_lengths) * len(embed_dims) * len(dtypes)
    current = 0
    
    for seq_len in seq_lengths:
        print(f"\nSequence Length: {seq_len}")
        for embed_dim in embed_dims:
            for dtype in dtypes:
                current += 1
                print(f"[{current}/{total_configs}] ", end='')
                
                result = run_single_benchmark(
                    seq_len=seq_len,
                    embed_dim=embed_dim,
                    dtype=dtype,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    device=device
                )
                
                if result is not None:
                    results.append(result)
    
    return pd.DataFrame(results)


# ============================================================================
# Results Formatting and Display
# ============================================================================

def print_results_table(df):
    """Print formatted results table."""
    if df.empty:
        print("No results to display")
        return
    
    print("\n" + "=" * 160)
    print(f"{'SeqLen':<8} {'Embed':<7} {'Heads':<6} {'d_k':<5} {'Dtype':<10} "
          f"{'PT-Fwd':<10} {'PT-Bwd':<10} {'PT-F+B':<10} "
          f"{'FA-Fwd':<10} {'FA-Bwd':<10} {'FA-F+B':<10} {'Speedup':<8}")
    print("=" * 160)
    
    for _, row in df.iterrows():
        flash_fwd = f"{row['flash_forward_ms']:.2f}" if pd.notna(row['flash_forward_ms']) else "N/A"
        flash_bwd = f"{row['flash_backward_ms']:.2f}" if pd.notna(row['flash_backward_ms']) else "N/A"
        flash_fb = f"{row['flash_fwd_bwd_ms']:.2f}" if pd.notna(row['flash_fwd_bwd_ms']) else "N/A"
        speedup = f"{row['speedup_fwd_bwd']:.2f}x" if pd.notna(row['speedup_fwd_bwd']) else "N/A"
        
        print(f"{row['seq_len']:<8} {row['embed_dim']:<7} {row['num_heads']:<6} {row['d_k']:<5} {row['dtype']:<10} "
              f"{row['pt_forward_ms']:<10.2f} {row['pt_backward_ms']:<10.2f} {row['pt_fwd_bwd_ms']:<10.2f} "
              f"{flash_fwd:<10} {flash_bwd:<10} {flash_fb:<10} {speedup:<8}")
    
    print("=" * 160)


def print_summary_stats(df):
    """Print summary statistics."""
    if df.empty or not FLASH_AVAILABLE:
        return
    
    valid_speedups = df[df['speedup_fwd_bwd'].notna()]
    
    if not valid_speedups.empty:
        print("\n" + "=" * 60)
        print("Summary Statistics (Forward + Backward)")
        print("=" * 60)
        print(f"Average Speedup: {valid_speedups['speedup_fwd_bwd'].mean():.2f}x")
        print(f"Median Speedup:  {valid_speedups['speedup_fwd_bwd'].median():.2f}x")
        print(f"Max Speedup:     {valid_speedups['speedup_fwd_bwd'].max():.2f}x")
        print(f"Min Speedup:     {valid_speedups['speedup_fwd_bwd'].min():.2f}x")
        print("=" * 60)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main benchmarking script."""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)
    
    device = 'cuda'
    
    # Configuration
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    embed_dims = [16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]
    
    # Run benchmarks
    results_df = run_all_benchmarks(
        seq_lengths=seq_lengths,
        embed_dims=embed_dims,
        dtypes=dtypes,
        batch_size=1,
        num_heads=8,
        device=device
    )
    
    # Display results
    print_results_table(results_df)
    print_summary_stats(results_df)
    
    # Save results
    output_file = 'flash_attention_benchmark_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to '{output_file}'")
    
    # Create a simplified table for the report
    if FLASH_AVAILABLE:
        report_df = results_df[['seq_len', 'embed_dim', 'dtype', 
                                'pt_fwd_bwd_ms', 'flash_fwd_bwd_ms', 'speedup_fwd_bwd']]
        report_file = 'flash_attention_report_table.csv'
        report_df.to_csv(report_file, index=False)
        print(f"✓ Simplified report table saved to '{report_file}'")


if __name__ == "__main__":
    main()
