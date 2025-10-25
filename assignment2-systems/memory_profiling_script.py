import sys
import os

# Add the assignment1 directory to Python path
assignment1_dir = os.path.join(os.path.dirname(__file__), '..', 'assignment1')
sys.path.insert(0, assignment1_dir)

# Now import from transformer
from transformer import *

# Import what we need from train_transformer if available
try:
    from train_transformer import *
except ImportError:
    print("Warning: Could not import from train_transformer, some features may not be available")


import argparse
import timeit
import numpy as np 
import torch 
from contextlib import nullcontext


def profile_memory(
    model,
    inputs,
    targets,
    num_warmup=3,
    num_profile_iters=5,
    forward_only=False,
    output_file="memory_snapshot.pickle",
    use_amp=False,
    amp_dtype=torch.float16
):
    """
    Profile memory usage of model.
    
    Args:
        model: The transformer model to profile
        inputs: Input tensor [batch_size, context_length]
        targets: Target tensor [batch_size, context_length]
        num_warmup: Number of warmup iterations
        num_profile_iters: Number of iterations to profile
        forward_only: If True, only profile forward pass
        output_file: Output pickle file name
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Dtype for mixed precision
    """
    model.train()
    device = next(model.parameters()).device
    
    # Create autocast context if needed
    if use_amp:
        autocast_context = torch.autocast(device_type='cuda', dtype=amp_dtype)
        print(f"Using AMP with {amp_dtype}")
    else:
        from contextlib import nullcontext
        autocast_context = nullcontext()
        print("Using full precision (FP32)")
    
    # Warmup phase
    print(f"Running {num_warmup} warmup iterations...")
    for i in range(num_warmup):
        with autocast_context:
            logits = model(inputs)
            if not forward_only:
                loss = cross_entropy_loss(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
        
        if not forward_only:
            loss.backward()
            model.zero_grad()
        
        torch.cuda.synchronize()
        print(f"  Warmup {i+1}/{num_warmup} complete")
    
    # Clear any cached memory before profiling
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"\nStarting memory profiling for {num_profile_iters} iterations...")
    print(f"Output will be saved to: {output_file}")
    
    # Start recording memory history
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    try:
        for i in range(num_profile_iters):
            print(f"\nProfiling iteration {i+1}/{num_profile_iters}...")
            
            # Forward pass
            with autocast_context:
                logits = model(inputs)
                
                if not forward_only:
                    loss = cross_entropy_loss(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
            
            # Backward pass
            if not forward_only:
                loss.backward()
                model.zero_grad()
            
            torch.cuda.synchronize()
            
            # Print memory stats for this iteration
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Peak: {peak:.2f} GB")
    
    finally:
        # Save snapshot
        print(f"\nSaving memory snapshot to {output_file}...")
        torch.cuda.memory._dump_snapshot(output_file)
        
        # Stop recording
        torch.cuda.memory._record_memory_history(enabled=None)

    # Print final memory statistics
    print("\n" + "="*80)
    print("Memory Profiling Complete")
    print("="*80)
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    allocated_memory = torch.cuda.memory_allocated() / 1e9
    reserved_memory = torch.cuda.memory_reserved() / 1e9
    
    print(f"Peak Memory Allocated: {peak_memory:.3f} GB")
    print(f"Currently Allocated: {allocated_memory:.3f} GB")
    print(f"Reserved by PyTorch: {reserved_memory:.3f} GB")
    print("="*80)
    print(f"\nTo visualize:")
    print(f"1. Go to: https://pytorch.org/memory_viz")
    print(f"2. Drag and drop: {output_file}")
    print(f"3. Look at 'Active Memory Timeline' to see memory usage over time")
    print("="*80)
    
    return {
        'peak_memory_gb': peak_memory,
        'allocated_memory_gb': allocated_memory,
        'reserved_memory_gb': reserved_memory
    }
    

def main():
    parser = argparse.ArgumentParser(description='Memory Profiling for Transformer Models')
    
    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=256, help='Context length')
    parser.add_argument('--d_model', type=int, default=2560, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=32, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=32, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=10240, help='Feed-forward dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    
    # Profiling parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_warmup', type=int, default=3, help='Number of warmup iterations')
    parser.add_argument('--num_profile_iters', type=int, default=5, help='Number of profiling iterations')
    parser.add_argument('--forward_only', action='store_true', help='Only profile forward pass')
    
    # Model configuration
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--amp_dtype', type=str, default='bfloat16',
                       choices=['float16', 'bfloat16'],
                       help='Dtype for mixed precision')
    
    # Output
    parser.add_argument('--output', type=str, default='memory_snapshot.pickle',
                       help='Output pickle file name')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Memory profiling requires NVIDIA GPU.")
        sys.exit(1)
    
    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Set dtype
    amp_dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    amp_dtype = amp_dtype_map[args.amp_dtype] if args.use_amp else None
    
    # Print configuration
    print(f"\n{'='*80}")
    print("Memory Profiling Configuration")
    print(f"{'='*80}")
    print(f"Model: {args.num_layers}L, {args.d_model}D, {args.num_heads}H, {args.d_ff}FF")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    print(f"Mixed Precision: {'Enabled (' + args.amp_dtype + ')' if args.use_amp else 'Disabled'}")
    print(f"Warmup iterations: {args.num_warmup}")
    print(f"Profile iterations: {args.num_profile_iters}")
    print(f"Mode: {'Forward only' if args.forward_only else 'Forward + Backward'}")
    print(f"Output file: {args.output}")
    print(f"{'='*80}\n")
    
    # Initialize model (always in FP32, autocast will handle conversions)
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32  # Always FP32 for parameters
    )
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    print(f"Model size (FP32): {num_params * 4 / 1e9:.3f} GB")
    print(f"Model size (BF16): {num_params * 2 / 1e9:.3f} GB")
    
    # Generate random data
    print("\nGenerating random data...")
    torch.manual_seed(42)
    inputs = torch.randint(
        0, args.vocab_size,
        (args.batch_size, args.context_length),
        device=device
    )
    targets = torch.randint(
        0, args.vocab_size,
        (args.batch_size, args.context_length),
        device=device
    )
    
    # Run memory profiling
    print("\n" + "="*80)
    print("Starting Memory Profiling")
    print("="*80 + "\n")
    
    results = profile_memory(
        model=model,
        inputs=inputs,
        targets=targets,
        num_warmup=args.num_warmup,
        num_profile_iters=args.num_profile_iters,
        forward_only=args.forward_only,
        output_file=args.output,
        use_amp=args.use_amp,
        amp_dtype=amp_dtype
    )
    
    print("\nMemory profiling complete!")
    print(f"\nPeak memory usage: {results['peak_memory_gb']:.3f} GB")


if __name__ == "__main__":
    main()
