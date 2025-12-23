import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Type, Any, Iterable, Optional, List
from collections import defaultdict

class ShardedOptimizer(Optimizer):
    """
    A wrapper optimizer that shards optimizer state across distributed ranks.
    
    Each rank only maintains optimizer state for a subset of parameters
    (approximately 1/world_size of total parameters). After each optimizer
    step, ranks broadcast their updated parameters to synchronize the model.
    """
    
    def __init__(
        self,
        params: Iterable,
        optimizer_cls: Type[Optimizer],
        **kwargs: Any
    ):
        """
        Initialize the sharded optimizer.
        
        Args:
            params: Collection of parameters or parameter groups to optimize
            optimizer_cls: The optimizer class to wrap (e.g., torch.optim.AdamW)
            **kwargs: Additional arguments forwarded to optimizer_cls constructor
        """
        # Store the optimizer class and kwargs for later use
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        
        # Initialize empty parameter groups list
        self.param_groups = []
        
        # Get distributed training info
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            
        else:
            self.rank = 0
            self.world_size = 1
            
        # Storage for the wrapped optimizer (will be created after params are added)
        self.wrapped_optimizer: Optional[Optimizer] = None
        
        # Track all parameters across all ranks (for broadcasting)
        self.all_params: List[torch.nn.Parameter] = []
        
        # Track which parameters this rank owns
        self.owned_params: List[torch.nn.Parameter] = []
        
        # Mapping from parameter to owning rank
        self.param_to_rank: dict[torch.nn.Parameter, int] = {}
        
        # Call parent constructor - this will call add_param_group for each group
        # Convert params to list of parameter groups if needed
        if isinstance(params, torch.Tensor):
            params = [params]
            
        param_groups_list = list(params)
        if len(param_groups_list) == 0:
            raise ValueError("optimizer got an empty parameter list")
        
        if not isinstance(param_groups_list[0], dict):
            param_groups_list = [{'params': param_groups_list}]
        
        # Now we need to call the Optimizer.__init__ but with empty params first
        # Then add the actual params via add_param_group
        super().__init__([], {})
        
        # Add each parameter group
        for param_group in param_groups_list:
            self.add_param_group(param_group)
    
    
    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """
        Add a parameter group to the optimizer.
        
        This method shards the parameters across ranks and creates/updates
        the wrapped optimizer with only the parameters owned by this rank.
        
        Args:
            param_group: Dictionary containing 'params' and optionally other
                        hyperparameters like 'lr', 'weight_decay', etc.
        """
        # Extract parameters from the group
        params = param_group['params']
        if isinstance(params, torch.Tensor):
            params = [params]
        else:
            params = list(params)
            
        # Store all parameters for broadcasting
        new_all_params = [p for p in params if p not in self.all_params]
        self.all_params.extend(new_all_params)
        
        # Assign parameters to ranks using round-robin strategy
        # We need to shard the new parameters
        new_owned_params = []
        for idx, param in enumerate(new_all_params):
            # Determine which rank owns this parameter based on global index
            global_idx = len(self.all_params) - len(new_all_params) + idx
            owning_rank = global_idx % self.world_size
            
            if owning_rank == self.rank:
                new_owned_params.append(param)
                self.owned_params.append(param)
        
        # Create parameter group for wrapped optimizer (only with owned params)
        if new_owned_params:
            wrapped_param_group = param_group.copy()
            wrapped_param_group['params'] = new_owned_params
            
            # If wrapped optimizer doesn't exist yet, create it
            if self.wrapped_optimizer is None:
                self.wrapped_optimizer = self.optimizer_cls(
                    [wrapped_param_group],
                    **self.optimizer_kwargs
                )
            else:
                # Add to existing wrapped optimizer
                self.wrapped_optimizer.add_param_group(wrapped_param_group)
        
        # Add to our param_groups list (with all params, not just owned)
        self.param_groups.append(param_group)
        
        
    def step(self, closure=None, **kwargs) -> Optional[float]:
        """
        Perform an optimization step.
        
        Only updates parameters owned by this rank, then broadcasts
        the updated parameters to all other ranks for synchronization.
        
        Args:
            closure: Optional closure to reevaluate the model
            **kwargs: Additional arguments forwarded to wrapped optimizer
            
        Returns:
            Optional loss value from closure
        """
        loss = None
        
        # Only perform optimizer step if we own any parameters
        if self.wrapped_optimizer is not None:
            loss = self.wrapped_optimizer.step(closure=closure, **kwargs)
        
        # Synchronize parameters across ranks
        if self.world_size > 1:
            self._broadcast_parameters()
        
        return loss
    
    def _broadcast_parameters(self) -> None:
        """
        Broadcast updated parameters from their owning ranks to all other ranks.
        """
        for param in self.all_params:
            owning_rank = self.param_to_rank[param]
            # Broadcast from the owning rank to all others
            dist.broadcast(param.data, src=owning_rank)
            
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero out gradients. Delegates to wrapped optimizer if it exists."""
        if self.wrapped_optimizer is not None:
            self.wrapped_optimizer.zero_grad(set_to_none=set_to_none)
    
    
    def state_dict(self) -> dict:
        """
        Return state dict. Only includes state for owned parameters.
        """
        if self.wrapped_optimizer is not None:
            return self.wrapped_optimizer.state_dict()
        return {'state': {}, 'param_groups': []}
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load state dict. Only loads state for owned parameters.
        """
        if self.wrapped_optimizer is not None:
            self.wrapped_optimizer.load_state_dict(state_dict)


def get_sharded_optimizer(
    params: Iterable,
    optimizer_cls: Type[torch.optim.Optimizer],
    **kwargs
) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.
    
    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor`s
            or :class:`dict`s giving all parameters, which will be sharded
            across ranks.
        optimizer_cls (:class:`torch.optim.Optimizer`): the class of the local
            optimizer.
            
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
        
    Returns:
        Instance of sharded optimizer.
    """
    return ShardedOptimizer(params, optimizer_cls, **kwargs)