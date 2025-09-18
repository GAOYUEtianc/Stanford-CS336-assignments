import torch.nn as nn
import torch

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        # Initialize weight parameter with shape (out_features, in_features)
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # Compute standard deviation for initialization
        std = (2/(in_features + out_features)) ** 0.5
        # Initialize weights with truncated normal distribution
        nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x shape is (..., in_features), self.W shape is (out_features, in_features)
        # Output shape should be (..., out_features)
        return torch.einsum('... i, o i -> ... o', x, self.W)
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        # Create embedding matrix (of dim vocab_size, d_model)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3.0, b=3.0)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids is of shape (batch_size, seq_len), dtype: long
        # Output : (batch_size, seq_len, embedding_dim)
        one_hot = torch.nn.functional.one_hot(token_ids, num_classes = self.weight.shape[0]).to(self.weight.dtype)
        out = torch.einsum('... n, n d -> ... d', one_hot, self.weight)
        return out
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # Learnable gain parameter initialized to ones
        self.gain = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original dtype
        in_dtype = x.dtype
        # Upcast to float32 for numerical stability
        x = x.to(torch.float32)
        
        # Compute RMX over the last dimension
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = x / rms # shape (..., d_model)
        
        # Apply learnable gain
        out = normed * self.gain
        
        # Downcast back to original dtype
        return out.to(in_dtype)
    
