
"""
Naive Distributed Data Parallel (DDP) Implementation

This implements a minimal version of DDP by:
1. Broadcasting model parameters from rank 0 to all other ranks
2. All-reducing gradients after backward pass to average them across all processes
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Optional
import torch.multiprocessing as mp


def get_ddp_individual_parameters(model: nn.Module) -> nn.Module:
    """
    Initialize a model for distributed data parallel training.
    
    This function:
    1. Broadcasts all model parameters from rank 0 to all other ranks
    2. Returns the model with synchronized parameters
    
    Args:
        model: PyTorch model to be used for DDP training
        
    Returns:
        The same model, but with parameters synchronized across all ranks
    """
    # Get current rank
    rank = dist.get_rank()
    
    # Broadcast all parameters from rank 0 to all other ranks
    # This ensures all processes start with the same initial model
    for param in model.parameters():
        # broadcast() sends data from src rank to all other ranks
        # src=0 means rank 0 is the source of truth
        dist.broadcast(param.data, src=0)
    
    # Also broadcast buffers (like BatchNorm running stats) if they exist
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=0)
    
    return model


def ddp_individual_parameters_on_after_backward(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> None:
    """
    Synchronize gradients across all processes after the backward pass.
    
    This function should be called after loss.backward() but before optimizer.step().
    It averages the gradients computed on each rank so that all ranks have the same
    gradients (averaged across all examples in the global batch).
    
    Args:
        model: The DDP model
        optimizer: The optimizer (not used in this naive implementation)
    """
    # Get world size for averaging
    world_size = dist.get_world_size()
    
    # All-reduce gradients for each parameter
    # This sums gradients across all ranks, then we divide by world_size to get the average
    for param in model.parameters():
        if param.grad is not None:
            # All-reduce sums the gradients across all processes
            # Each process will end up with the sum of all gradients
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            
            # Divide by world_size to get the average gradient
            # This is equivalent to averaging the loss over the global batch
            param.grad.data /= world_size
            
            
# ============================================================================
# Complete Training Script with Verification
# ============================================================================

def test_naive_ddp():
    """
    Test script to verify naive DDP implementation matches single-process training.
    """
    import torch.multiprocessing as mp
    def worker(rank: int, world_size: int):
        """Worker function for each process."""
        # Setup process group
        import os
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        
        # Simple toy model
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 5)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)
        
        # Set seed for reproducibility (different per rank initially)
        torch.manual_seed(rank)
        
        # Create DDP model
        ddp_model = ToyModel()
        ddp_model = get_ddp_individual_parameters(ddp_model)
        
        # Create optimizer
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss()
        
        # Generate some training data (same across all ranks)
        torch.manual_seed(42)  # Same seed for data
        total_batch_size = 20
        input_size = 10
        output_size = 5
        
        all_x = torch.randn(total_batch_size, input_size)
        all_y = torch.randn(total_batch_size, output_size)
        
        # Each rank gets a subset of the data
        local_batch_size = total_batch_size // world_size
        offset = rank * local_batch_size
        local_x = all_x[offset : offset + local_batch_size]
        local_y = all_y[offset : offset + local_batch_size]
        
        # Training loop
        for epoch in range(3):
            optimizer.zero_grad()
            
            # Forward pass on local data
            outputs = ddp_model(local_x)
            loss = loss_fn(outputs, local_y)
            
            # Backward pass
            loss.backward()
            
            # Synchronize gradients (this is the key DDP step!)
            ddp_individual_parameters_on_after_backward(ddp_model, optimizer)
            
            # Optimizer step
            optimizer.step()
            
            if rank == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Cleanup
        dist.destroy_process_group()
    
    # Run with 2 processes
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
    print("✓ DDP training completed successfully!")


def verify_ddp_equivalence():
    """
    Verify that DDP training produces the same results as single-process training.
    """
    import torch.multiprocessing as mp
    from copy import deepcopy
    
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    def single_process_training():
        """Train on all data in a single process."""
        torch.manual_seed(0)
        model = ToyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss()
        
        # All training data
        torch.manual_seed(42)
        all_x = torch.randn(20, 10)
        all_y = torch.randn(20, 5)
        
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(all_x)
            loss = loss_fn(outputs, all_y)
            loss.backward()
            optimizer.step()
        
        return model.state_dict()

    def ddp_training(rank: int, world_size: int, result_dict):
        """Train with DDP on sharded data."""
        import os
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        
        torch.manual_seed(0)
        model = ToyModel()
        model = get_ddp_individual_parameters(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss()
        
        # Same data as single process
        torch.manual_seed(42)
        all_x = torch.randn(20, 10)
        all_y = torch.randn(20, 5)
        
        # Each rank gets a shard
        local_bs = 20 // world_size
        offset = rank * local_bs
        local_x = all_x[offset : offset + local_bs]
        local_y = all_y[offset : offset + local_bs]
        
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(local_x)
            loss = loss_fn(outputs, local_y)
            loss.backward()
            ddp_individual_parameters_on_after_backward(model, optimizer)
            optimizer.step()
        
        if rank == 0:
            result_dict['ddp_state'] = model.state_dict()
        
        dist.destroy_process_group()
        
    # Train with single process
    print("Training with single process...")
    single_state = single_process_training()
    
    # Train with DDP
    print("Training with DDP (2 processes)...")
    manager = mp.Manager()
    result_dict = manager.dict()
    world_size = 2
    mp.spawn(ddp_training, args=(world_size, result_dict), nprocs=world_size, join=True)
    ddp_state = result_dict['ddp_state']
    
    # Compare results
    print("\nComparing results...")
    all_match = True
    for key in single_state.keys():
        if torch.allclose(single_state[key], ddp_state[key], atol=1e-6):
            print(f"  ✓ {key}: MATCH")
        else:
            print(f"  ✗ {key}: MISMATCH")
            print(f"    Max diff: {(single_state[key] - ddp_state[key]).abs().max().item()}")
            all_match = False
    
    if all_match:
        print("\n✓ SUCCESS: DDP training matches single-process training!")
    else:
        print("\n✗ FAILURE: DDP training does not match single-process training")
    
    return all_match


if __name__ == "__main__":
    print("=" * 80)
    print("Naive DDP Implementation Test")
    print("=" * 80)
    
    # Set spawn method for multiprocessing
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    print("\n1. Running basic DDP training test...")
    test_naive_ddp()
    
    print("\n2. Verifying DDP equivalence with single-process training...")
    verify_ddp_equivalence()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
