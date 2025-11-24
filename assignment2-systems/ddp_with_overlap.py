"""
DDP Implementation with Overlapping Communication and Computation

This version overlaps backward pass computation with gradient communication by:
1. Using backward hooks to trigger all-reduce as soon as a gradient is ready
2. Using async all-reduce (async_op=True) to avoid blocking
3. Significantly reducing communication overhead
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import List, Optional, Dict, Any


class DDP(nn.Module):
    """
    Distributed Data Parallel wrapper that overlaps communication with computation.
    
    Key features:
    - Broadcasts parameters from rank 0 to all ranks during initialization
    - Registers backward hooks on all parameters
    - When a gradient is ready, immediately starts async all-reduce
    - finish_gradient_synchronization() waits for all async ops to complete
    """
    
    def __init__(self, module: nn.Module):
        """
        Wrap a module for distributed data parallel training.
        
        Args:
            module: The PyTorch module to wrap
        """
        super().__init__()
        
        # Store the wrapped module
        self.module = module
        
        # Get distributed info
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # Storage for async communication handles
        self.grad_handles: List[dist.Work] = []
        
        # Step 1: Broadcast all parameters from rank 0
        self._broadcast_parameters()
        
        # Step 2: Register backward hooks for all parameters
        self._register_hooks()
        
    def _broadcast_parameters(self):
        """Broadcast all parameters and buffers from rank 0 to all other ranks."""
        with torch.no_grad():
            # Broadcast parameters
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
                
            # Broadcast buffers (e.g., BatchNorm running stats)
            for buffer in self.module.buffers():
                dist.broadcast(buffer.data, src=0)
                
    def _register_hooks(self):
        """ 
        Register post-accumulate gradient hooks on all parameters. 
        
        When a parameter's gradient is ready, this hook will:
        1. Start an async all-reduce on that gradient
        2. Store the communication handle for later synchronization
        """
        for param in self.module.parameters():
            if param.requires_grad:
                # Register hook that triggers when gradient is accumulated
                param.register_post_accumulate_grad_hook(
                    self._make_hook(param)
                )
                
    def _make_hook(self, param: nn.Parameter):
        """
        Create a hook function for a specific parameter.
        
        Args:
            param: The parameter to create a hook for
            
        Returns:
            Hook function that will be called when gradient is ready
        """
        def hook(grad: torch.Tensor):
            """
            This function is called when param's gradient is ready.
            
            Args:
                grad: The gradient tensor (same as param.grad)
            """
            # Start async all-reduce (returns immediately)
            # This allows backward pass to continue while communication happens
            handle = dist.all_reduce(
                param.grad.data,
                op=dist.ReduceOp.SUM,
                async_op=True
            )
            
            # Store handle so that we can wait for it later
            self.grad_handles.append(handle)
            
        return hook
    
    def forward(self, *inputs, **kwargs):
        """
        Forward pass through the wrapped module.
        
        Args:
            *inputs: Positional arguments to pass to module
            **kwargs: Keyword arguments to pass to module
            
        Returns:
            Output from the wrapped module
        """
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        """
        Wait for all async all-reduce operations to complete and average gradients.
        
        This should be called after backward() and before optimizer.step().
        """
        # Wait for all async all-reduce operations to complete
        for handle in self.grad_handles:
            handle.wait()
            
        # Clear handles for next iteration
        self.grad_handles.clear()
        
        # Average the gradients by dividing by world_size
        # (all-reduce gives us the sum, we want the average)
        with torch.no_grad():
            for param in self.module.parameters():
                if param.grad is not None:
                    param.grad.data /= self.world_size
                    

# ============================================================================
# Test Model
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
    

# ============================================================================
# Worker Functions for Testing and Benchmarking
# ============================================================================
def _worker_correctness_test(rank: int, world_size: int, results):
    """Test that DDP produces correct results."""
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29700"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    torch.manual_seed(0)
    model = ToyModel()
    ddp_model = DDP(model)
    
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    
    # Same data for all ranks
    torch.manual_seed(42)
    x = torch.randn(16, 10)
    y = torch.randn(16, 5)
    
    # Train for a few steps 
    for _ in range(3):
        optimizer.zero_grad()
        output = ddp_model(x)
        loss = loss_fn(output, y)
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
        
    # Save results from rank 0
    if rank == 0:
        results['params'] = {
            name: param.data.clone()
            for name, param in ddp_model.module.named_parameters()
        }
        
    dist.destroy_process_group()
    
    
def _worker_benchmark_overlap(rank: int, world_size: int, backend: str, results):
    """Benchmark DDP with overlapping communication."""
    import os
    import time
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29701"
    
    if backend == "nccl":
        os.environ['NCCL_ALGO'] = 'Tree'
        os.environ['NCCL_PROTO'] = 'LL' 
        os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'
        os.environ['NCCL_SOCKET_NTHREADS'] = '4'
        os.environ['NCCL_BUFFSIZE'] = '4194304'
        os.environ['NCCL_NTHREADS'] = '512'
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        device = "cpu"
        
    # Create a larger model
    class BigModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(128, 128) for _ in range(100)
            ])
            
        def forward(self, x):
            for layer in self.layers:
                x = torch.relu(layer(x))
            return x
    
    torch.manual_seed(0)
    model = BigModel().to(device)
    ddp_model = DDP(model)
    
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    
    torch.manual_seed(42)
    x = torch.randn(32, 128, device=device)
    y = torch.randn(32, 128, device=device)
    
    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        output = ddp_model(x)
        loss = loss_fn(output, y)
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
        
    # Benchmark
    if backend == "nccl":
        torch.cuda.synchronize()
    
    num_iters = 20
    total_time = 0.0
    total_comm_time = 0.0
    
    for _ in range(num_iters):
        iter_start = time.perf_counter()
        optimizer.zero_grad()
        output = ddp_model(x)
        loss = loss_fn(output, y)
        comm_start = time.perf_counter()
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        comm_end = time.perf_counter()
        optimizer.step()
        
        iter_end = time.perf_counter()
        
        total_time += (iter_end - iter_start)
        total_comm_time += (comm_end - comm_start)
        
        if backend == "nccl":
            torch.cuda.synchronize()
    
    if rank == 0:
        results['overlap_time'] = total_time / num_iters * 1000  # ms
        results['overlap_comm_time'] = total_comm_time / num_iters * 1000  # ms
        results['overlap_comp_time'] = (total_time - total_comm_time) / num_iters * 1000 
    
    dist.destroy_process_group()


def _worker_benchmark_naive(rank: int, world_size: int, backend: str, results):
    """Benchmark naive DDP (no overlap)."""
    import os
    import time
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29702"
    
    if backend == "nccl":
        os.environ['NCCL_ALGO'] = 'Tree'
        os.environ['NCCL_PROTO'] = 'LL' 
        os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'
        os.environ['NCCL_SOCKET_NTHREADS'] = '4'
        os.environ['NCCL_BUFFSIZE'] = '4194304'
        os.environ['NCCL_NTHREADS'] = '512'
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        device = "cpu"
        
    class BigModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(128, 128) for _ in range(100)
            ])
            
        def forward(self, x):
            for layer in self.layers:
                x = torch.relu(layer(x))
            return x
        
        
    torch.manual_seed(0)
    model = BigModel().to(device)
    
    # Broadcast
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
        
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    
    torch.manual_seed(42)
    x = torch.randn(32, 128, device=device)
    y = torch.randn(32, 128, device=device)
    
    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        
        # Naive: wait for all gradients, then all-reduce each
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
        
        optimizer.step()
        
    # Benchmark
    if backend == "nccl":
        torch.cuda.synchronize()
    
    num_iters = 20
    total_time = 0.0
    total_comm_time = 0.0
    
    for _ in range(num_iters):
        iter_start = time.perf_counter()
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        
        comm_start = time.perf_counter()
        
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
        comm_end = time.perf_counter()
        
        optimizer.step()
        
        iter_end = time.perf_counter()
        
        total_time += (iter_end - iter_start)
        total_comm_time += (comm_end - comm_start)
        
        if backend == "nccl":
            torch.cuda.synchronize()
    
    
    if rank == 0:
        results['naive_time'] = total_time / num_iters * 1000  # ms
        results['naive_comm_time'] = total_comm_time / num_iters * 1000  # ms
        results['naive_comp_time'] = (total_time - total_comm_time) / num_iters * 1000
    
    dist.destroy_process_group()
    
    
# ============================================================================
# Test and Benchmark Functions
# ============================================================================

def test_ddp_correctness():
    """Test that DDP produces correct gradient synchronization."""
    import torch.multiprocessing as mp
    
    print("=" * 80)
    print("Testing DDP Correctness")
    print("=" * 80)
    
    manager = mp.Manager()
    results = manager.dict()
    world_size = 2
    
    mp.spawn(_worker_correctness_test, args=(world_size, results), nprocs=world_size, join=True)
    
    return results.get('params')


def benchmark_ddp_overlap():
    """Benchmark DDP with and without overlapping."""
    import torch.multiprocessing as mp
    
    print("\n" + "=" * 80)
    print("Benchmarking: Naive vs Overlap")
    print("=" * 80)
    
    # Use GPU if available
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    print(f"Using backend: {backend}")
    
    manager = mp.Manager()
    results = manager.dict()
    world_size = 2
    
    print("\n1. Running Naive DDP (no overlap)...")
    mp.spawn(_worker_benchmark_naive, args=(world_size, backend, results), nprocs=world_size, join=True)
    
    print("2. Running Overlap DDP (with hooks + async)...")
    mp.spawn(_worker_benchmark_overlap, args=(world_size, backend, results), nprocs=world_size, join=True)
    
    naive_time = results['naive_time']
    overlap_time = results['overlap_time']
    naive_comm_time = results['naive_comm_time']
    overlap_comm_time = results['overlap_comm_time']
    
    speedup = naive_time / overlap_time
    comm_reduction = naive_comm_time / overlap_comm_time
    
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"{'Metric':<25} {'Naive':<15} {'Overlap':<15} {'Improvement':<15}")
    print("-" * 80)
    print(f"{'Total Time (ms)':<25} {naive_time:<15.3f} {overlap_time:<15.3f} {speedup:<15.2f}x")
    print(f"{'Comm Time (ms)':<25} {naive_comm_time:<15.3f} {overlap_comm_time:<15.3f} {comm_reduction:<15.2f}x")
    print(f"{'Compute Time (ms)':<25} {results['naive_comp_time']:<15.3f} {results['overlap_comp_time']:<15.3f} {'-':<15}")
    print("=" * 80)

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    print("DDP with Overlapping Communication and Computation")
    print("=" * 80)
    
    # Test correctness
    test_ddp_correctness()
    
    # Benchmark
    benchmark_ddp_overlap()