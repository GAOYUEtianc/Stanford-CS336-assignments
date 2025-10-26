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

def benchmark_model(
    model,
    optimizer,
    inputs,
    targets,
    num_warmup=10,
    num_iters=100,
    forward_only=False,
    device='cuda'
):
    """Benchmark transformer forward/backward/optimizer."""
    
    model.train()
    
    # Warmup
    for _ in range(num_warmup):
        logits = model(inputs)
        if not forward_only:
            loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()
    
    # Benchmark forward
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_iters):
        logits = model(inputs)
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    forward_time = (end - start) / num_iters
    
    if forward_only:
        return {'forward_ms': forward_time * 1000}
    
    # Benchmark forward + backward + optimizer
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_iters):
        logits = model(inputs)
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    full_step_time = (end - start) / num_iters
    
    return {
        'forward_ms': forward_time * 1000,
        'full_step_ms': full_step_time * 1000,
        'backward_optimizer_ms': (full_step_time - forward_time) * 1000
    }


def run_transformer_benchmark(model_config, use_compile=False):
    """Run benchmark for a specific model configuration."""
    
    device = 'cuda'
    batch_size = 8
    context_length = 256
    
    print(f"\nBenchmarking {'COMPILED' if use_compile else 'VANILLA'} Transformer")
    print(f"Config: {model_config['name']}")
    print("-"*80)
    
    # Initialize model
    model = TransformerLM(
        vocab_size=10000,
        context_length=context_length,
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        device=device,
        dtype=torch.float32
    )
    
    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,}")
    
    # Compile if requested
    if use_compile:
        print("Compiling model...")
        model = torch.compile(model)
        print("Compilation complete.")
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    # Generate data
    torch.manual_seed(42)
    inputs = torch.randint(0, 10000, (batch_size, context_length), device=device)
    targets = torch.randint(0, 10000, (batch_size, context_length), device=device)
    
    # Benchmark forward only
    print("\nBenchmarking forward pass...")
    forward_results = benchmark_model(
        model, optimizer, inputs, targets,
        num_warmup=10, num_iters=100,
        forward_only=True, device=device
    )
    
    # Benchmark full training step
    print("Benchmarking full training step...")
    full_results = benchmark_model(
        model, optimizer, inputs, targets,
        num_warmup=10, num_iters=100,
        forward_only=False, device=device
    )
    
    return {
        'forward_ms': forward_results['forward_ms'],
        'full_step_ms': full_results['full_step_ms'],
        'backward_optimizer_ms': full_results['backward_optimizer_ms']
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark Compiled Transformer')
    parser.add_argument('--model', type=str, default='small',
                       choices=['small', 'medium', 'large'],
                       help='Model size to benchmark')
    
    args = parser.parse_args()
    
    # Model configurations
    models = {
        'small': {
            'name': 'Small (124M)',
            'd_model': 768,
            'num_layers': 12,
            'num_heads': 12,
            'd_ff': 3072
        },
        'medium': {
            'name': 'Medium (350M)',
            'd_model': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'd_ff': 4096
        },
        'large': {
            'name': 'Large (774M)',
            'd_model': 1280,
            'num_layers': 36,
            'num_heads': 20,
            'd_ff': 5120
        }
    }
    
    model_config = models[args.model]
    
    print("="*80)
    print("Transformer Compilation Benchmark")
    print("="*80)
    
    # Benchmark vanilla
    torch.cuda.empty_cache()
    vanilla_results = run_transformer_benchmark(model_config, use_compile=False)
    
    # Benchmark compiled
    torch.cuda.empty_cache()
    compiled_results = run_transformer_benchmark(model_config, use_compile=True)
    
    # Print comparison
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print()
    print(f"{'Metric':<30} {'Vanilla (ms)':<20} {'Compiled (ms)':<20} {'Speedup':<15}")
    print("-"*80)
    
    forward_speedup = vanilla_results['forward_ms'] / compiled_results['forward_ms']
    print(f"{'Forward Pass':<30} {vanilla_results['forward_ms']:<20.2f} "
          f"{compiled_results['forward_ms']:<20.2f} {forward_speedup:<15.2f}x")
    
    full_speedup = vanilla_results['full_step_ms'] / compiled_results['full_step_ms']
    print(f"{'Full Training Step':<30} {vanilla_results['full_step_ms']:<20.2f} "
          f"{compiled_results['full_step_ms']:<20.2f} {full_speedup:<15.2f}x")
    
    back_opt_speedup = vanilla_results['backward_optimizer_ms'] / compiled_results['backward_optimizer_ms']
    print(f"{'Backward + Optimizer':<30} {vanilla_results['backward_optimizer_ms']:<20.2f} "
          f"{compiled_results['backward_optimizer_ms']:<20.2f} {back_opt_speedup:<15.2f}x")
    
    print("="*80)
    
    # Analysis
    print("\nANALYSIS:")
    print(f"- Forward pass speedup: {forward_speedup:.2f}x")
    print(f"- Full training step speedup: {full_speedup:.2f}x")
    
    if forward_speedup > 1.1:
        print("✓ torch.compile provides meaningful speedup!")
    else:
        print("⚠ torch.compile shows minimal benefit (possibly limited by memory bandwidth)")
    
    print("\nReasons for speedup:")
    print("- Kernel fusion (reduces memory traffic)")
    print("- Operator specialization (optimized for specific shapes)")
    print("- Reduced Python overhead")
    print("="*80)


if __name__ == "__main__":
    main()