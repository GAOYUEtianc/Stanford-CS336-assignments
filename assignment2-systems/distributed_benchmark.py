"""
Distributed All-Reduce Benchmarking Script
Benchmarks all-reduce operation across different backends, data sizes, and process counts.
"""

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import argparse


def setup(rank: int, world_size: int, backend: str):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Set device for GPU backends
    if backend == "nccl":
        torch.cuda.set_device(rank)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


# Global worker function (must be at module level for pickle)
def _benchmark_worker(rank, world_size, backend, data_size_mb, num_iterations, warmup_iterations, return_dict):
    """Worker function for benchmarking (must be at module level for multiprocessing)."""
    setup(rank, world_size, backend)
    
    # Determine device
    if backend == "nccl":
        device = f"cuda:{rank}"
    else:
        device = "cpu"
    
    # Calculate tensor size
    # float32 = 4 bytes, so for X MB: X * 1024 * 1024 / 4 elements
    num_elements = int(data_size_mb * 1024 * 1024 / 4)
    
    # Create random tensor
    data = torch.randn(num_elements, dtype=torch.float32, device=device)
    
    # Warmup iterations
    for _ in range(warmup_iterations):
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize(device)
    
    # Benchmark iterations
    timings = []
    
    for _ in range(num_iterations):
        # Start timing
        if backend == "nccl":
            torch.cuda.synchronize(device)
            start_time = time.perf_counter()
        else:
            start_time = time.perf_counter()
        
        # All-reduce operation
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
        
        # End timing
        if backend == "nccl":
            torch.cuda.synchronize(device)
            end_time = time.perf_counter()
        else:
            end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000
        timings.append(elapsed_ms)
    
    # Convert timings to tensor for gathering
    local_timings = torch.tensor(timings, dtype=torch.float32, device=device)
    
    # Gather all timings to rank 0
    if rank == 0:
        gathered_timings = [torch.zeros_like(local_timings) for _ in range(world_size)]
    else:
        gathered_timings = None
    
    dist.gather(local_timings, gather_list=gathered_timings, dst=0)
    
    result = None
    if rank == 0:
        # Aggregate statistics across all ranks
        all_timings = torch.stack(gathered_timings).cpu().numpy()
        
        result = {
            'backend': backend,
            'device': 'GPU' if backend == 'nccl' else 'CPU',
            'world_size': world_size,
            'data_size_mb': data_size_mb,
            'mean_time_ms': np.mean(all_timings),
            'std_time_ms': np.std(all_timings),
            'min_time_ms': np.min(all_timings),
            'max_time_ms': np.max(all_timings),
            'median_time_ms': np.median(all_timings),
            'bandwidth_gbps': (data_size_mb * world_size) / (np.mean(all_timings) / 1000) / 1024,
        }
        return_dict['result'] = result
    
    cleanup()


def run_benchmark_config(
    world_size: int,
    backend: str,
    data_size_mb: float,
    num_iterations: int = 100,
    warmup_iterations: int = 5
):
    """
    Run benchmark for a single configuration using multiprocessing.
    
    Args:
        world_size: Number of processes
        backend: Communication backend
        data_size_mb: Data size in MB
        num_iterations: Number of iterations
        warmup_iterations: Number of warmup iterations
    
    Returns:
        Benchmark results dictionary
    """
    # Use a manager to collect results from rank 0
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Spawn processes
    mp.spawn(
        fn=_benchmark_worker,
        args=(world_size, backend, data_size_mb, num_iterations, warmup_iterations, return_dict),
        nprocs=world_size,
        join=True
    )
    
    return return_dict.get('result', None)


def run_all_benchmarks(
    backends_configs: List[Tuple[str, str]],  # [(backend, device)]
    data_sizes_mb: List[float],
    world_sizes: List[int],
    num_iterations: int = 100
) -> pd.DataFrame:
    """
    Run benchmarks for all configurations.
    
    Args:
        backends_configs: List of (backend, device) tuples
        data_sizes_mb: List of data sizes in MB
        world_sizes: List of world sizes to test
        num_iterations: Number of iterations per benchmark
    
    Returns:
        DataFrame with all benchmark results
    """
    results = []
    total_configs = len(backends_configs) * len(data_sizes_mb) * len(world_sizes)
    current = 0
    
    print("=" * 80)
    print("Distributed All-Reduce Benchmarking")
    print("=" * 80)
    print(f"Total configurations: {total_configs}")
    print(f"Iterations per config: {num_iterations}")
    print("=" * 80)
    
    for backend, device in backends_configs:
        print(f"\n{'='*80}")
        print(f"Backend: {backend} ({device})")
        print(f"{'='*80}")
        
        for world_size in world_sizes:
            print(f"\n  World Size: {world_size}")
            
            for data_size_mb in data_sizes_mb:
                current += 1
                print(f"    [{current}/{total_configs}] Data Size: {data_size_mb} MB ... ", end='', flush=True)
                
                try:
                    result = run_benchmark_config(
                        world_size=world_size,
                        backend=backend,
                        data_size_mb=data_size_mb,
                        num_iterations=num_iterations
                    )
                    
                    if result is not None:
                        results.append(result)
                        print(f"✓ {result['mean_time_ms']:.2f} ms (BW: {result['bandwidth_gbps']:.2f} GB/s)")
                    else:
                        print("✗ No result returned")
                
                except Exception as e:
                    print(f"✗ Error: {e}")
    
    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame, output_dir: str = '.'):
    """
    Create visualization plots for benchmark results.
    
    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Latency vs Data Size (grouped by backend and world size)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for backend_device in df['backend'].unique():
        subset = df[df['backend'] == backend_device]
        
        for world_size in sorted(subset['world_size'].unique()):
            data = subset[subset['world_size'] == world_size]
            label = f"{backend_device.upper()} (n={world_size})"
            
            axes[0].plot(
                data['data_size_mb'],
                data['mean_time_ms'],
                marker='o',
                label=label,
                linewidth=2
            )
    
    axes[0].set_xlabel('Data Size (MB)', fontsize=12)
    axes[0].set_ylabel('Latency (ms)', fontsize=12)
    axes[0].set_title('All-Reduce Latency vs Data Size', fontsize=14)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Bandwidth vs Data Size
    for backend_device in df['backend'].unique():
        subset = df[df['backend'] == backend_device]
        
        for world_size in sorted(subset['world_size'].unique()):
            data = subset[subset['world_size'] == world_size]
            label = f"{backend_device.upper()} (n={world_size})"
            
            axes[1].plot(
                data['data_size_mb'],
                data['bandwidth_gbps'],
                marker='s',
                label=label,
                linewidth=2
            )
    
    axes[1].set_xlabel('Data Size (MB)', fontsize=12)
    axes[1].set_ylabel('Effective Bandwidth (GB/s)', fontsize=12)
    axes[1].set_title('All-Reduce Bandwidth vs Data Size', fontsize=14)
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/allreduce_benchmark.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to '{output_dir}/allreduce_benchmark.png'")
    plt.close()
    
    # Plot 3: Scaling behavior (Latency vs World Size for each data size)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for data_size in sorted(df['data_size_mb'].unique()):
        for backend in df['backend'].unique():
            subset = df[(df['data_size_mb'] == data_size) & (df['backend'] == backend)]
            if not subset.empty:
                ax.plot(
                    subset['world_size'],
                    subset['mean_time_ms'],
                    marker='o',
                    label=f"{backend.upper()} - {data_size} MB",
                    linewidth=2
                )
    
    ax.set_xlabel('Number of Processes', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('All-Reduce Scaling: Latency vs Number of Processes', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/allreduce_scaling.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to '{output_dir}/allreduce_scaling.png'")
    plt.close()


def print_summary_table(df: pd.DataFrame):
    """Print formatted summary table of results."""
    print("\n" + "=" * 120)
    print("Summary Table: All-Reduce Benchmark Results")
    print("=" * 120)
    print(f"{'Backend':<10} {'Device':<8} {'Procs':<7} {'Data (MB)':<12} {'Mean (ms)':<12} {'Std (ms)':<12} {'BW (GB/s)':<12}")
    print("=" * 120)
    
    for _, row in df.iterrows():
        print(f"{row['backend']:<10} {row['device']:<8} {row['world_size']:<7} "
              f"{row['data_size_mb']:<12.1f} {row['mean_time_ms']:<12.3f} "
              f"{row['std_time_ms']:<12.3f} {row['bandwidth_gbps']:<12.2f}")
    
    print("=" * 120)


def main():
    """Main benchmarking script."""
    parser = argparse.ArgumentParser(description='Benchmark distributed all-reduce operations')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations per config')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for results')
    args = parser.parse_args()
    
    # Configuration
    # Note: Only use NCCL if CUDA is available
    backends_configs = [('gloo', 'CPU')]
    if torch.cuda.is_available():
        backends_configs.append(('nccl', 'GPU'))
        print(f"✓ CUDA available: {torch.cuda.device_count()} GPUs detected")
    else:
        print("✗ CUDA not available, only testing Gloo backend on CPU")
    
    data_sizes_mb = [1, 10, 100, 1000]  # 1MB, 10MB, 100MB, 1GB
    world_sizes = [2, 4, 6]
    
    # Check GPU availability for NCCL
    if ('nccl', 'GPU') in backends_configs:
        num_gpus = torch.cuda.device_count()
        if num_gpus < max(world_sizes):
            print(f"Warning: Only {num_gpus} GPUs available, adjusting world_sizes")
            world_sizes = [ws for ws in world_sizes if ws <= num_gpus]
    
    # Run benchmarks
    results_df = run_all_benchmarks(
        backends_configs=backends_configs,
        data_sizes_mb=data_sizes_mb,
        world_sizes=world_sizes,
        num_iterations=args.iterations
    )
    
    # Save results
    output_file = f'{args.output_dir}/allreduce_benchmark_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to '{output_file}'")
    
    # Print summary table
    print_summary_table(results_df)
    
    # Create plots
    plot_results(results_df, args.output_dir)
    
    # Print analysis
    print("\n" + "=" * 80)
    print("Analysis:")
    print("=" * 80)
    
    if not results_df.empty:
        # Compare backends
        if len(backends_configs) > 1:
            gloo_avg = results_df[results_df['backend'] == 'gloo']['mean_time_ms'].mean()
            nccl_avg = results_df[results_df['backend'] == 'nccl']['mean_time_ms'].mean()
            print(f"1. Backend Comparison:")
            print(f"   - Gloo (CPU) average latency: {gloo_avg:.2f} ms")
            print(f"   - NCCL (GPU) average latency: {nccl_avg:.2f} ms")
            print(f"   - NCCL speedup: {gloo_avg/nccl_avg:.2f}x")
        
        # Data size impact
        print(f"\n2. Data Size Impact:")
        for backend in results_df['backend'].unique():
            subset = results_df[results_df['backend'] == backend]
            min_size = subset['data_size_mb'].min()
            max_size = subset['data_size_mb'].max()
            min_time = subset[subset['data_size_mb'] == min_size]['mean_time_ms'].mean()
            max_time = subset[subset['data_size_mb'] == max_size]['mean_time_ms'].mean()
            print(f"   - {backend.upper()}: {min_size}MB→{max_size}MB increases latency by {max_time/min_time:.1f}x")
        
        # Scaling behavior
        print(f"\n3. Scaling Behavior (World Size):")
        for backend in results_df['backend'].unique():
            subset = results_df[results_df['backend'] == backend]
            for data_size in sorted(subset['data_size_mb'].unique())[:2]:  # First 2 sizes
                data = subset[subset['data_size_mb'] == data_size].sort_values('world_size')
                if len(data) > 1:
                    scaling_factor = data['mean_time_ms'].iloc[-1] / data['mean_time_ms'].iloc[0]
                    print(f"   - {backend.upper()} @ {data_size}MB: {data['world_size'].iloc[0]}→{data['world_size'].iloc[-1]} procs increases latency by {scaling_factor:.2f}x")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()