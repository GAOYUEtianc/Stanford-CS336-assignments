"""
DDP Implementation with Bucketed Gradients + NVTX Profiling

This version combines the benefits of:
1. Batching communication (fewer calls, less overhead)
2. Overlapping communication with computation (async as buckets are ready)

With detailed NVTX annotations for Nsight Systems profiling.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import List, Optional, Dict, Tuple

# Import NVTX for profiling
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False
    print("Warning: NVTX not available")

class DDPBucketed(nn.Module):
    """ 
    Distributed Data Parallel with gradient bucketing.
    
    Features:
    - Parameters organized into buckets of fixed size
    - Each bucket is all-reduced asynchronously when all its gradients are ready
    - Reduces communication overhead while maintaining overlap with computation
    """
    
    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0):
        """
        Wrap a module for bucketed DDP training.
        
        Args:
            module: The PyTorch module to wrap
            bucket_size_mb: Maximum size of each bucket in megabytes
        """
        super().__init__()
        
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # Convert bucket size to bytes (1 MB = 1024 * 1024 bytes)
        # Assuming float32 (4 bytes per element)
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        
        # Storage for communication handles
        self.comm_handles: List[dist.Work] = []
        
        # Bucket organization
        self.buckets: List[List[torch.nn.Parameter]] = []
        self.bucket_gradients: List[Optional[torch.Tensor]] = [] # Flattened gradients per bucket
        self.bucket_ready_count: List[int] = []  # How many grads ready in each bucket
        self.param_to_bucket: Dict[nn.Parameter, int] = {}  # Param -> bucket index
        self.bucket_locks: List[bool] = []  # Track if bucket is being communicated
        
        # Step 1: Broadcast parameters
        self._broadcast_parameters()
        
        # Step 2: Organize parameters into buckets
        self._create_buckets()
        
        # Step 3: Register hooks
        self._register_hooks()
        

    def _broadcast_parameters(self):
        """Broadcast all parameters from rank 0."""
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
            
            for buffer in self.module.buffers():
                dist.broadcast(buffer.data, src=0)
                
    def _create_buckets(self):
        """
        Organize parameters into buckets.
        
        Uses reverse order of model.parameters() since gradients become
        ready in approximately reverse order during backward pass.
        """
        # Get all parameters that require gradients
        params = [p for p in self.module.parameters() if p.requires_grad]
        
        # Reverse order (gradients computed backward, so reverse = forward order of readiness)
        params = list(reversed(params))
        
        current_bucket = []
        current_bucket_size = 0
        
        for param in params:
            # Calculate parameter size in bytes (assuming float32 = 4 bytes)
            param_size = param.numel() * param.element_size()
            
            # If adding this parameter exceeds bucket size, finalize current bucket
            if current_bucket_size + param_size > self.bucket_size_bytes and len(current_bucket) > 0:
                # Finish current bucket
                self.buckets.append(current_bucket)
                self.bucket_gradients.append(None) # Placeholder
                self.bucket_ready_count.append(0)
                self.bucket_locks.append(False)
                
                # Start new bucket
                current_bucket = [param]
                current_bucket_size = param_size
                
            else:
                # Add parameter to current bucket
                current_bucket.append(param)
                current_bucket_size += param_size
                
        # Last bucket
        if len(current_bucket) > 0:
            self.buckets.append(current_bucket)
            self.bucket_gradients.append(None) # Placeholder
            self.bucket_ready_count.append(0)
            self.bucket_locks.append(False)
            
        # Build param -> bucket mapping
        for bucket_idx, bucket in enumerate(self.buckets):
            for param in bucket:
                self.param_to_bucket[param] = bucket_idx
                
        if self.rank == 0:
            total_params = sum(len(bucket) for bucket in self.buckets)
            total_size_mb = sum(
                sum(p.numel() * p.element_size() for p in bucket) / (1024 * 1024)
                for bucket in self.buckets
            )
            print(f"Created {len(self.buckets)} buckets for {total_params} parameters ({total_size_mb:.2f} MB total)")
            for i, bucket in enumerate(self.buckets):
                bucket_size = sum(p.numel() * p.element_size() for p in bucket) / (1024 * 1024)
                print(f"  Bucket {i}: {len(bucket)} params, {bucket_size:.2f} MB")
                
    def _register_hooks(self):
        """Register backward hooks on parameters to trigger bucket communication."""
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._make_hook(param)
                )
                
    def _make_hook(self, param: nn.Parameter):
        """Create a hook for a specific parameter."""
        def hook(grad: torch.Tensor):
            bucket_idx = self.param_to_bucket[param]
            
            # Increment ready count for this bucket
            self.bucket_ready_count[bucket_idx] += 1
            
            # Check if all gradients in this bucket are ready
            if self.bucket_ready_count[bucket_idx] == len(self.buckets[bucket_idx]):
                # All gradients ready, initiate all-reduce
                self._all_reduce_bucket(bucket_idx)
                
        return hook

    def _all_reduce_bucket(self, bucket_idx: int):
        """ 
        All-reduce a specific bucket asynchronously.
        
        Args:
            bucket_idx: Index of the bucket to all-reduce
        """
        # Prevent double communication
        if self.bucket_locks[bucket_idx]:
            return 
        
        self.bucket_locks[bucket_idx] = True
        
        bucket = self.buckets[bucket_idx]
        
        # NVTX: Mark bucket preparation
        if NVTX_AVAILABLE:
            nvtx.range_push(f"Bucket {bucket_idx} - Prepare")
            
        # Step 1: Flatten gradients into a single tensor
        grad_list = [p.grad.data for p in bucket if p.grad is not None]
        
        if len(grad_list) == 0:
            if NVTX_AVAILABLE:
                nvtx.range_pop()  # End Prepare
            return  # Nothing to communicate
        
        # Flatten graidients
        flattened = torch._utils._flatten_dense_tensors(grad_list)
        
        if NVTX_AVAILABLE:
            nvtx.range_pop() # End Prepare
            bucket_size_mb = sum(p.numel() * p.element_size() for p in bucket) / (1024 * 1024)
            nvtx.range_push(f"Bucket {bucket_idx} - AllReduce ({bucket_size_mb:.1f}MB)")
            
        # Step 2: All-reduce asynchronously
        handle = dist.all_reduce(
            flattened,
            op=dist.ReduceOp.SUM,
            async_op=True
        )
        
        if NVTX_AVAILABLE:
            nvtx.range_pop()  # End AllReduce
            
        # Step 3: Store handle and flattened gradients
        self.comm_handles.append(handle)
        self.bucket_gradients[bucket_idx] = flattened
        
    def forward(self, *inputs, **kwargs):
        """Forward pass through wrapped module."""
        return self.module(*inputs, **kwargs)
        
    def finish_gradient_synchronization(self):
        """
        Wait for all async communication to complete and copy gradients back.
        """
        if NVTX_AVAILABLE:
            nvtx.range_push(f"Wait for All Buckets ({len(self.comm_handles)} handles)")
        
        # Wait for all communication to finish
        for i, handle in enumerate(self.comm_handles):
            if NVTX_AVAILABLE:
                nvtx.range_push(f"Wait Handle {i}")
            handle.wait()
            if NVTX_AVAILABLE:
                nvtx.range_pop()  # End Wait Handle
                
        if NVTX_AVAILABLE:
            nvtx.range_pop()  # End Wait for All Buckets
            nvtx.range_push("Unflatten & Copy Back")
            
        # Copy back flattened gradients to individual parameter grads
        for bucket_idx, bucket in enumerate(self.buckets):
            if self.bucket_gradients[bucket_idx] is not None:
                flattened = self.bucket_gradients[bucket_idx]
                
                # Average the gradients
                flattened /= self.world_size
                
                # Unflatten back to original shapes
                grad_list = [p.grad.data for p in bucket if p.grad is not None]
                unflattened = torch._utils._unflatten_dense_tensors(flattened, grad_list)
                
                # Copy back to parameters
                for param, grad in zip(bucket, unflattened):
                    param.grad.data.copy_(grad)
                    
        if NVTX_AVAILABLE:
            nvtx.range_pop()  # End Unflatten & Copy Back
            
        # Reset for next iteration
        self.comm_handles.clear()
        self.bucket_gradients = [None] * len(self.buckets)
        self.bucket_ready_count = [0] * len(self.buckets)
        self.bucket_locks = [False] * len(self.buckets)
        

# ============================================================================
# Test and Benchmark
# ============================================================================

class ToyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 5)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def _worker_test_correctness(rank: int, world_size: int, bucket_size_mb: float, results):
    """Test correctness of bucketed DDP."""
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29800"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    torch.manual_seed(0)
    model = ToyModel()
    ddp_model = DDPBucketed(model, bucket_size_mb=bucket_size_mb)
    
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    
    torch.manual_seed(42)
    x = torch.randn(16, 10)
    y = torch.randn(16, 5)
    
    # Train for a few steps
    for _ in range(3):
        optimizer.zero_grad()
        outputs = ddp_model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
        
    if rank == 0:
        results['params'] = {
            name: param.data.clone()
            for name, param in ddp_model.module.named_parameters()
        }
        
    dist.destroy_process_group()
    
    
def test_bucketed_correctness():
    """Test that bucketed DDP works correctly."""
    import torch.multiprocessing as mp
    
    print("=" * 80)
    print("Testing Bucketed DDP Correctness")
    print("=" * 80)
    
    manager = mp.Manager()
    results = manager.dict()
    world_size = 2
    
    mp.spawn(_worker_test_correctness, args=(world_size, 25.0, results), nprocs=world_size, join=True)
    
    print("✓ Bucketed DDP correctness test passed!")


def _worker_benchmark_bucketed(rank: int, world_size: int, backend: str, bucket_size_mb: float, results):
    """Benchmark bucketed DDP with NVTX annotations."""
    import os
    import time
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"2980{int(bucket_size_mb)%10}"
    
    if backend == "nccl":
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        device = "cpu"
        
    class BigModel(nn.Module):
        def __init__(self, 
                 num_layers=48,
                 hidden=4096, 
                 ffn_hidden=16384,
                 input_dim=1024,
                 output_dim=1024):
            super().__init__()
            
            self.input_proj = nn.Linear(input_dim, hidden)
            
            self.layers = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(hidden, hidden),      # attention proj (QKVO)
                    nn.Linear(hidden, hidden),
                    nn.Linear(hidden, ffn_hidden),  # FFN 1
                    nn.Linear(ffn_hidden, hidden),  # FFN 2
                ])
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Linear(hidden, output_dim)
            
        def forward(self, x):
            x = self.input_proj(x)

            for attn1, attn2, ffn1, ffn2 in self.layers:
                x = attn1(x)
                x = attn2(x)
                x = ffn2(torch.relu(ffn1(x)))

            x = self.output_proj(x)
            return x
    
    torch.manual_seed(0)
    model = BigModel().to(device)
    
    if NVTX_AVAILABLE:
        nvtx.range_push("DDP Initialization")
    ddp_model = DDPBucketed(model, bucket_size_mb=bucket_size_mb)
    if NVTX_AVAILABLE:
        nvtx.range_pop()  # End DDP Initialization
        
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    
    torch.manual_seed(42)
    x = torch.randn(32, 1024, device=device)
    y = torch.randn(32, 1024, device=device)
    
    # Warm-up
    for _ in range(3):
        optimizer.zero_grad()
        output = ddp_model(x)
        loss = loss_fn(output, y)
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()

    # Benchmark with NVTX annotations
    if backend == "nccl":
        torch.cuda.synchronize()
    
    num_iters = 20
    start = time.perf_counter()
    
    for i in range(num_iters):
        if NVTX_AVAILABLE:
            nvtx.range_push(f"Iteration {i} - Bucket={bucket_size_mb}MB")
        
        optimizer.zero_grad()
        if NVTX_AVAILABLE:
            nvtx.range_push("Forward Pass")
        output = ddp_model(x)
        loss = loss_fn(output, y)
        if NVTX_AVAILABLE:
            nvtx.range_pop()  # End Forward Pass
            nvtx.range_push("Backward Pass (buckets overlap)")
        loss.backward()
        if NVTX_AVAILABLE:
            nvtx.range_pop()  # End Backward Pass
            nvtx.range_push("Finish Gradient Sync")
        ddp_model.finish_gradient_synchronization()
        if NVTX_AVAILABLE:
            nvtx.range_pop()  # End Finish Gradient Sync
            nvtx.range_push("Optimizer Step")
        
        optimizer.step()
        if NVTX_AVAILABLE:
            nvtx.range_pop()  # End Optimizer Step
        
        if backend == "nccl":
            torch.cuda.synchronize()        
            
        if NVTX_AVAILABLE:
            nvtx.range_pop()  # Iteration
            
    total_time = time.perf_counter() - start
    
    if rank == 0:
        results[f'bucket_{bucket_size_mb}'] = total_time / num_iters * 1000
    
    dist.destroy_process_group()
    
    
def benchmark_bucket_sizes():
    """Benchmark different bucket sizes."""
    import torch.multiprocessing as mp
    
    print("\n" + "=" * 80)
    print("Benchmarking Different Bucket Sizes")
    print("=" * 80)
    
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    print(f"Using backend: {backend}")
    
    bucket_sizes = [200, 400, 600]  # MB
    
    manager = mp.Manager()
    results = manager.dict()
    world_size = 2
    
    for bucket_size in bucket_sizes:
        print(f"\nTesting bucket size: {bucket_size} MB...")
        mp.spawn(_worker_benchmark_bucketed, 
                args=(world_size, backend, bucket_size, results),
                nprocs=world_size, join=True)
    
    # Print results
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"{'Bucket Size (MB)':<20} {'Time per Iteration (ms)':<30}")
    print("-" * 80)
    
    for bucket_size in bucket_sizes:
        time_ms = results.get(f'bucket_{bucket_size}', 0)
        print(f"{bucket_size:<20} {time_ms:<30.3f}")
    
    print("=" * 80)

# ============================================================================
# Profiling Functions for Nsight Systems
# ============================================================================

def profile_bucketed_ddp(bucket_size_mb: float):
    """
    Profile bucketed DDP for Nsight Systems.
    Run with: nsys profile -o bucketed_{size}mb python script.py --profile-bucketed {size}
    """
    import torch.multiprocessing as mp
    
    print("=" * 80)
    print(f"Profiling Bucketed DDP (Bucket Size: {bucket_size_mb} MB)")
    print("=" * 80)
    print(f"Run with: nsys profile -o bucketed_{bucket_size_mb}mb --trace=cuda,nvtx python script.py --profile-bucketed {bucket_size_mb}")
    print("=" * 80)
    
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend != "nccl":
        print("Warning: NCCL/GPU not available, profiling on CPU with gloo")
    
    manager = mp.Manager()
    results = manager.dict()
    world_size = 2
    
    mp.spawn(_worker_benchmark_bucketed, args=(world_size, backend, bucket_size_mb, results), nprocs=world_size, join=True)
    
    print(f"\n✓ Bucketed DDP (bucket={bucket_size_mb}MB) profiling completed")
    print(f"  Check bucketed_{bucket_size_mb}mb.nsys-rep with Nsight Systems")


def profile_all_bucket_sizes():
    """Profile all bucket sizes for comparison."""
    bucket_sizes = [200, 400, 600]
    
    print("=" * 80)
    print("Profiling All Bucket Sizes")
    print("=" * 80)
    print("This will generate separate profiles for each bucket size")
    print("=" * 80)
    
    for bucket_size in bucket_sizes:
        print(f"\n{'='*80}")
        print(f"Profiling bucket size: {bucket_size} MB")
        print(f"{'='*80}")
        profile_bucketed_ddp(bucket_size)
        
    
if __name__ == "__main__":
    import torch.multiprocessing as mp
    import sys
    
    mp.set_start_method('spawn', force=True)
    
    # Check for profiling mode
    if "--profile-bucketed" in sys.argv:
        idx = sys.argv.index("--profile-bucketed")
        if idx + 1 < len(sys.argv):
            bucket_size = float(sys.argv[idx + 1])
        else:
            bucket_size = 25.0  # Default
        profile_bucketed_ddp(bucket_size)
    
    elif "--profile-all-buckets" in sys.argv:
        profile_all_bucket_sizes()
    
    else:
        # Normal execution
        print("DDP with Bucketed Gradients")
        print("=" * 80)
        
        # Test correctness
        test_bucketed_correctness()
        
        # Benchmark
        benchmark_bucket_sizes()
