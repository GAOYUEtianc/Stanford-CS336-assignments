import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import time
from typing import Dict
import os

def setup_distributed():
    """Initialize distributed training environment."""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()
    

def get_memory_stats() -> Dict[str, float]:
    """
    Get current GPU memory statistics in MB.
    
    Returns:
        Dictionary with allocated, reserved, and peak memory.
    """
    allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    
    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'peak_mb': max_allocated
    }
    
    
def reset_peak_memory():
    """Reset peak memory statistics."""
    torch.cuda.reset_peak_memory_stats()


def print_memory_breakdown(label: str, rank: int, model: nn.Module, 
                          optimizer: torch.optim.Optimizer = None):
    """
    Print detailed memory breakdown.
    
    Args:
        label: Description of current stage
        rank: Current process rank
        model: The model
        optimizer: The optimizer (optional)
    """
    if rank != 0:
        return
    
    stats = get_memory_stats()
    
    # Calculate parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    
    # Calculate gradient memory (if gradients exist)
    grad_memory = sum(
        p.grad.numel() * p.grad.element_size() 
        for p in model.parameters() 
        if p.grad is not None
    ) / 1024**2
    
    # Calculate optimizer state memory
    optimizer_memory = 0
    if optimizer is not None:
        for state in optimizer.state.values():
            for v in state.values():
                if torch.is_tensor(v):
                    optimizer_memory += v.numel() * v.element_size()
        optimizer_memory /= 1024**2
    
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    print(f"Peak Memory:           {stats['peak_mb']:.2f} MB")
    print(f"Currently Allocated:   {stats['allocated_mb']:.2f} MB")
    print(f"Currently Reserved:    {stats['reserved_mb']:.2f} MB")
    print(f"\nMemory Breakdown:")
    print(f"  Parameters:          {param_memory:.2f} MB")
    print(f"  Gradients:           {grad_memory:.2f} MB")
    print(f"  Optimizer State:     {optimizer_memory:.2f} MB")
    print(f"  Other (activations, buffers): {stats['allocated_mb'] - param_memory - grad_memory - optimizer_memory:.2f} MB")
    print(f"{'='*70}\n")


class GPTXLSizeMLP(nn.Module):
    """
    MLP model approximating GPT-XL parameter count.
    
    GPT-XL specs:
    - Parameters: ~1.5B
    - Hidden size: 1600
    - Layers: 48
    - FFN intermediate size: 6400 (4x hidden size)
    
    For simplicity, we create an MLP with similar parameter count.
    """
    def __init__(self, vocab_size: int = 50257, hidden_size: int = 1600, 
                 num_layers: int = 48, ffn_size: int = 6400):
        super().__init__()
        
        # Embedding layer: vocab_size * hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Create MLP layers similar to GPT FFN
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, ffn_size),  # FFN up projection
                nn.GELU(),
                nn.Linear(ffn_size, hidden_size),  # FFN down projection
            ])
        self.layers = nn.Sequential(*layers)
        
        # Output projection
        self.output = nn.Linear(hidden_size, vocab_size)
        
        # Report parameter count
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model created with {total_params:,} parameters ({total_params/1e9:.2f}B)")
    
    
    def forward(self, input_ids):
        """Forward pass."""
        x = self.embedding(input_ids)
        x = x.mean(dim=1)  # Simple pooling
        x = self.layers(x)
        return self.output(x)
    

def create_dummy_batch(batch_size: int = 8, seq_len: int = 512, vocab_size: int = 50257):
    """Create a dummy batch for training."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    return input_ids.cuda()


def benchmark_training(use_sharding: bool, num_iterations: int = 5):
    """
    Benchmark training with or without optimizer state sharding.
    
    Args:
        use_sharding: Whether to use optimizer state sharding
        num_iterations: Number of iterations to run
    """
    rank, world_size = setup_distributed()
    
    if rank == 0:
        print(f"\n{'#'*70}")
        print(f"# Benchmarking with sharding={'ENABLED' if use_sharding else 'DISABLED'}")
        print(f"# Rank: {rank}, World Size: {world_size}")
        print(f"{'#'*70}\n")
    
    # Reset memory stats
    reset_peak_memory()
    # Create model
    model = GPTXLSizeMLP().cuda()
    model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        print_memory_breakdown("AFTER MODEL INITIALIZATION", rank, model.module)
    
    # Create optimizer
    if use_sharding:
        from shard_optimizer import ShardedOptimizer
        optimizer = ShardedOptimizer(
            model.parameters(),
            torch.optim.AdamW,
            lr=1e-4,
            weight_decay=0.01
        )
        
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
    # Warmup iteration (to allocate all memory)
    dummy_batch = create_dummy_batch()
    output = model(dummy_batch)
    loss = output.sum()
    loss.backward()
    
    # Measure memory before optimizer step
    if rank == 0:
        print_memory_breakdown("BEFORE OPTIMIZER STEP (after backward)", rank, model.module, optimizer)
    
    optimizer.step()
    optimizer.zero_grad()
    
    # Measure memory after optimizer step
    if rank == 0:
        print_memory_breakdown("AFTER OPTIMIZER STEP", rank, model.module, optimizer)
    
    # Now benchmark speed
    torch.cuda.synchronize()
    dist.barrier()
    
    if rank == 0:
        print(f"\n{'='*70}")
        print("SPEED BENCHMARK")
        print(f"{'='*70}")
        
    times = []
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Forward
        dummy_batch = create_dummy_batch()
        output = model(dummy_batch)
        loss = output.sum()
        
        # Backward
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        iter_time = end_time - start_time
        times.append(iter_time)
        
        if rank == 0:
            print(f"Iteration {i+1}/{num_iterations}: {iter_time*1000:.2f} ms")
    
    # Report average time (excluding first iteration as warmup)
    avg_time = sum(times[1:]) / len(times[1:])
    
    if rank == 0:
        print(f"\nAverage time per iteration (excluding first): {avg_time*1000:.2f} ms")
        print(f"{'='*70}\n")
    
    cleanup_distributed()
    

def main():
    parser = argparse.ArgumentParser(description='Profile optimizer state sharding')
    parser.add_argument('--sharding', action='store_true', 
                       help='Enable optimizer state sharding')
    parser.add_argument('--no-sharding', dest='sharding', action='store_false',
                       help='Disable optimizer state sharding (default)')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of iterations to benchmark (default: 5)')
    parser.set_defaults(sharding=False)
    
    args = parser.parse_args()
    
    # Set environment variable for better error messages
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    
    benchmark_training(args.sharding, args.iterations)


if __name__ == '__main__':
    main()