import torch
import torch.distributed as dist
import torch.nn as nn
from typing import List, Optional
import time


def get_ddp_flattened(model: nn.Module) -> nn.Module:
    """
    Initialize a model for DDP training with flattened gradients.
    
    This is identical to the naive version - just broadcast parameters from rank 0.
    
    Args:
        model: PyTorch model to be used for DDP training
        
    Returns:
        The same model, with parameters synchronized across all ranks
    """
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=0)
    
    return model


def ddp_flattened_on_after_backward(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> None:
    """
    Synchronize gradients using a SINGLE all-reduce on flattened gradients.
    
    This reduces communication overhead by:
    1. Flattening all gradients into one contiguous tensor
    2. Performing one all-reduce (instead of N all-reduces for N parameters)
    3. Unflattening the results back to original parameter shapes
    
    Args:
        model: The DDP model
        optimizer: The optimizer (not used)
    """
    world_size = dist.get_world_size()
    
    # Step 1: Collect all parameter gradients that need synchronization
    gradients_to_sync = []
    params_with_grad = []
    
    for param in model.parameters():
        if param.grad is not None:
            gradients_to_sync.append(param.grad.data)
            params_with_grad.append(param)
    
    if len(gradients_to_sync) == 0:
        return  # No gradients to sync
    
    # Step 2: Flatten all gradients into a single 1D tensor
    # torch._utils._flatten_dense_tensors concatenates tensors into one contiguous tensor
    flattened_grads = torch._utils._flatten_dense_tensors(gradients_to_sync)
    
    # Step 3: All-reduce the flattened tensor (ONE communication call!)
    dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM)
    
    # Step 4: Average the gradients
    flattened_grads /= world_size
    
    # Step 5: Unflatten back to original shapes and copy to parameters
    # torch._utils._unflatten_dense_tensors splits the flat tensor back
    unflattened_grads = torch._utils._unflatten_dense_tensors(
        flattened_grads, gradients_to_sync
    )
    
    # Step 6: Copy the averaged gradients back to parameters
    for param, unflattened_grad in zip(params_with_grad, unflattened_grads):
        param.grad.data.copy_(unflattened_grad)


# ============================================================================
# Model Definition (must be at module level for pickle)
# ============================================================================

class BigToyModel(nn.Module):
    """Model with many parameters for benchmarking."""
    def __init__(self):
        super().__init__()
        # Create 50 layers to have many parameters
        self.layers = nn.ModuleList([
            nn.Linear(128, 128) for _ in range(50)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class SimpleModel(nn.Module):
    """Simple model for correctness testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ============================================================================
# Worker Functions (must be at module level for pickle)
# ============================================================================

def _worker_naive(rank: int, world_size: int, results):
    """Worker for naive DDP (individual all-reduce per parameter)."""
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29600"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    torch.manual_seed(0)
    model = BigToyModel()
    
    # Broadcast from rank 0
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    
    # Generate data
    torch.manual_seed(42)
    x = torch.randn(32, 128)
    y = torch.randn(32, 128)
    
    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        
        # Naive: all-reduce each parameter individually
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
        
        optimizer.step()
    
    # Benchmark
    num_iters = 20
    start = time.perf_counter()
    comm_time = 0.0
    
    for _ in range(num_iters):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        
        # Time the communication
        comm_start = time.perf_counter()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
        comm_time += time.perf_counter() - comm_start
        
        optimizer.step()
    
    total_time = time.perf_counter() - start
    
    if rank == 0:
        results['naive_total'] = total_time / num_iters * 1000  # ms
        results['naive_comm'] = comm_time / num_iters * 1000  # ms
    
    dist.destroy_process_group()


def _worker_flattened(rank: int, world_size: int, results):
    """Worker for flattened DDP (single all-reduce)."""
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29601"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    torch.manual_seed(0)
    model = BigToyModel()
    model = get_ddp_flattened(model)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    
    # Same data
    torch.manual_seed(42)
    x = torch.randn(32, 128)
    y = torch.randn(32, 128)
    
    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        ddp_flattened_on_after_backward(model, optimizer)
        optimizer.step()
    
    # Benchmark
    num_iters = 20
    start = time.perf_counter()
    comm_time = 0.0
    
    for _ in range(num_iters):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        
        # Time the communication
        comm_start = time.perf_counter()
        ddp_flattened_on_after_backward(model, optimizer)
        comm_time += time.perf_counter() - comm_start
        
        optimizer.step()
    
    total_time = time.perf_counter() - start
    
    if rank == 0:
        results['flattened_total'] = total_time / num_iters * 1000  # ms
        results['flattened_comm'] = comm_time / num_iters * 1000  # ms
    
    dist.destroy_process_group()


def _worker_test(rank: int, world_size: int, use_flattened: bool, results):
    """Worker for correctness testing."""
    import os
    port = "29602" if use_flattened else "29603"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    torch.manual_seed(0)
    model = SimpleModel()
    
    # Broadcast
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    
    torch.manual_seed(42)
    x = torch.randn(16, 10)
    y = torch.randn(16, 5)
    
    # Train for a few steps
    for _ in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        
        if use_flattened:
            ddp_flattened_on_after_backward(model, optimizer)
        else:
            # Naive
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
        
        optimizer.step()
    
    if rank == 0:
        key = 'flattened' if use_flattened else 'naive'
        results[key] = {name: param.data.clone() for name, param in model.named_parameters()}
    
    dist.destroy_process_group()


# ============================================================================
# Comparison Benchmark: Naive vs Flattened
# ============================================================================

def benchmark_ddp_comparison():
    """
    Benchmark comparing naive DDP (individual all-reduce) vs flattened DDP.
    """
    import torch.multiprocessing as mp
    
    # Run benchmarks
    print("=" * 80)
    print("Benchmarking DDP: Naive vs Flattened")
    print("=" * 80)
    
    manager = mp.Manager()
    results = manager.dict()
    world_size = 2
    
    print("\n1. Running Naive DDP (individual all-reduce per parameter)...")
    mp.spawn(_worker_naive, args=(world_size, results), nprocs=world_size, join=True)
    
    print("2. Running Flattened DDP (single all-reduce)...")
    mp.spawn(_worker_flattened, args=(world_size, results), nprocs=world_size, join=True)
    
    # Print results
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"{'Method':<20} {'Total Time (ms)':<20} {'Comm Time (ms)':<20} {'Compute Time (ms)':<20}")
    print("-" * 80)
    
    naive_total = results['naive_total']
    naive_comm = results['naive_comm']
    naive_compute = naive_total - naive_comm
    
    flat_total = results['flattened_total']
    flat_comm = results['flattened_comm']
    flat_compute = flat_total - flat_comm
    
    print(f"{'Naive (Individual)':<20} {naive_total:<20.3f} {naive_comm:<20.3f} {naive_compute:<20.3f}")
    print(f"{'Flattened (Batched)':<20} {flat_total:<20.3f} {flat_comm:<20.3f} {flat_compute:<20.3f}")
    print("=" * 80)
    
    # Calculate speedups
    total_speedup = naive_total / flat_total
    comm_speedup = naive_comm / flat_comm
    
    print(f"\nSpeedup (Flattened vs Naive):")
    print(f"  Total Time: {total_speedup:.2f}x faster")
    print(f"  Communication Time: {comm_speedup:.2f}x faster")
    print(f"  Communication Overhead: Naive {naive_comm/naive_total*100:.1f}% vs Flattened {flat_comm/flat_total*100:.1f}%")



        import os
        port = "29602" if use_flattened else "29603"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 5)
                
            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))
        
        torch.manual_seed(0)
        model = SimpleModel()
        
        # Broadcast
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss()
        
        torch.manual_seed(42)
        x = torch.randn(16, 10)
        y = torch.randn(16, 5)
        
        # Train for a few steps
        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            
            if use_flattened:
                ddp_flattened_on_after_backward(model, optimizer)
            else:
                # Naive
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= world_size
            
            optimizer.step()
        
        if rank == 0:
            key = 'flattened' if use_flattened else 'naive'
            results[key] = {name: param.data.clone() for name, param in model.named_parameters()}
        
        dist.destroy_process_group()
    
    print("\n" + "=" * 80)
    print("Testing Correctness: Naive vs Flattened")
    print("=" * 80)
    
    manager = mp.Manager()
    results = manager.dict()
    world_size = 2
    
    # Run naive
    mp.spawn(worker_test, args=(world_size, False, results), nprocs=world_size, join=True)
    
    # Run flattened
    mp.spawn(worker_test, args=(world_size, True, results), nprocs=world_size, join=True)
    
    # Compare
    naive_params = results['naive']
    flat_params = results['flattened']
    
    print("\nComparing parameters:")
    all_match = True
    for name in naive_params.keys():
        match = torch.allclose(naive_params[name], flat_params[name], atol=1e-6)
        status = "✓ MATCH" if match else "✗ MISMATCH"
        print(f"  {name:<30} {status}")
        if not match:
            diff = (naive_params[name] - flat_params[name]).abs().max().item()
            print(f"    Max difference: {diff:.2e}")
            all_match = False
    
    if all_match:
        print("\n✓ SUCCESS: Flattened DDP produces identical results to Naive DDP!")
    else:
        print("\n✗ FAILURE: Results differ!")
    
    return all_match

# ============================================================================
# Test Correctness
# ============================================================================
def test_flattened_correctness():
    """Verify that flattened DDP produces the same results as naive DDP."""
    import torch.multiprocessing as mp
    from copy import deepcopy
    
    def worker_test(rank: int, world_size: int, use_flattened: bool, results):
        import os
        port = "29602" if use_flattened else "29603"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 5)
                
            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))
            
        torch.manual_seed(0)
        model = SimpleModel()
        
        # Broadcast
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
            
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss()
        
        torch.manual_seed(42)
        x = torch.randn(16, 10)
        y = torch.randn(16, 5)
        
        # Train for a few steps
        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            
            if use_flattened:
                ddp_flattened_on_after_backward(model, optimizer)
            else:
                # Naive all-reduce
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= world_size
                        
            optimizer.step()
            
        if rank == 0:
            key = 'flattened' if use_flattened else 'naive'
            results[key] = {name: param.data.clone() for name, param in model.named_parameters()}
            
        dist.destroy_process_group()
        
    print("\n" + "=" * 80)
    print("Testing Correctness: Naive vs Flattened")
    print("=" * 80)
    
    manager = mp.Manager()
    results = manager.dict()
    world_size = 2
    
    # Run naive
    mp.spawn(worker_test, args=(world_size, False, results), nprocs=world_size, join=True)
    
    # Run flattened
    mp.spawn(worker_test, args=(world_size, True, results), nprocs=world_size, join=True)
    
    # Compare
    naive_params = results['naive']
    flat_params = results['flattened']
    
    print("\nComparing parameters:")
    all_match = True
    for name in naive_params.keys():
        match = torch.allclose(naive_params[name], flat_params[name], atol=1e-6)
        status = "✓ MATCH" if match else "✗ MISMATCH"
        print(f" {name:<30} {status}")
        if not match:
            diff = (naive_params[name] - flat_params[name]).abs().max().item()
            print(f"    Max difference: {diff:.2e}")
            all_match = False 
    
    if all_match:
        print("\n✓ SUCCESS: Flattened DDP matches Naive DDP!")
    else:
        print("\n✗ FAILURE: Flattened DDP does not match Naive DDP.")
        
    return all_match
    
# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    print("DDP with Flattened Gradients")
    print("=" * 80)
    
    # Test correctness first
    test_flattened_correctness()
    
    # Then benchmark
    benchmark_ddp_comparison()
    