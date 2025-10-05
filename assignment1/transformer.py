import torch.nn as nn
import torch
from typing import Optional

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
    

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, device=None, dtype=None):
        super().__init__()
        
        # Calculate d_ff to be approsimately 8/3 * d_model
        d_ff = int(8 * d_model / 3)
        # Round up to nearrest multiple of 64 for hardware efficiency
        d_ff = ((d_ff + 63) // 64) * 64
        
        # Three linear transformations
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Implements: FFN(x) = W2(SiLU(W1(x)) ⊙ W3(x))
        where SiLU(x) = x * σ(x) = x / (1 + e^(-x))
        
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Output tensor of shape (..., d_model)
        """
        # W1(x): (..., d_ff)
        gate = self.W1(x)
        
        # SiLU(W1(x)): (..., d_ff)
        silu_gate = gate * torch.sigmoid(gate)
        
        # W3(x): (..., d_ff)
        linear = self.W3(x)
        
        # Element-wise multiplication: (..., d_ff)
        gated = silu_gate * linear
        
        # Final output transformation (..., d_model)
        output = self.W2(gated)
        
        return output
    
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta:float, d_k: int, max_seq_len: int, device=None):
        """
        Initialize RoPE module.
        
        Args:
            theta: Base value Θ for computing rotation angles
            d_k: Dimension of query/key vectors (must be even)
            max_seq_len: Maximum sequence length
            device: Device to store buffers on
        """
        super().__init__()
        
        assert d_k % 2 == 0
        
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        # Compute frequency for each dimension pair
        # For dimension pair k (1, 2, 3, ...., d_k /2 - 1), 
        # freq : 1/(Θ^(2*k/d_k))
        k = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)  # (d_k/2,)
        freqs = 1.0 / (theta ** (k/d_k)) # (d_k/2,)
        
        # Precompute cos and sin for all positions 
        # positions: (max_seq_len,)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # (max_seq_len,)
        
        # Outer product to get angles : (max_seq_len, 1) * (1, d_k/2) -> (max_seq_len, d_k/2)
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0) # (max_seq_len, d_k/2)
        
        # Compute cos and sin
        # Repeat each value twice to match the interleaved structure
        # cos: [cos(theta_0), cos(theta_0), cos(theta_1), cos(theta_1), ...]
        # sin: [sin(theta_0), sin(theta_0), sin(theta_1), sin(theta_1), ...]
        cos = torch.cos(angles) # (max_seq_len, d_k/2)
        sin = torch.sin(angles) # (max_seq_len, d_k/2)
        
        # Repeat each value twice to match the interleaved structure
        cos = cos.repeat_interleave(2, dim=-1)  # (max_seq_len, d_k)
        sin = sin.repeat_interleave(2, dim=-1)  # (max_seq_len, d_k)
        
        # Register as no-persistent buffers (not saved in state_dict)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
        
    def forward(self, x: torch.Tensor, token_position: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        
        The rotation formula for each pair [x_{2k}, x_{2k+1}] is:
        [x'_{2k}  ]   [cos(θ)  -sin(θ)] [x_{2k}  ]
        [x'_{2k+1}] = [sin(θ)   cos(θ)] [x_{2k+1}]
        
        Which expands to:
        x'_{2k}   = x_{2k} * cos(θ) - x_{2k+1} * sin(θ)
        x'_{2k+1} = x_{2k} * sin(θ) + x_{2k+1} * cos(θ)
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Token positions of shape (..., seq_len)
        
        Returns:
            Rotated tensor of shape (..., seq_len, d_k)
        """
        # Get cos and sin values for the given positions
        # token_posisions: (..., seq_len)
        # cos/sin buffers : (max_seq_len, d_k)
        
        # Index into precomputed cos and sin using token_positions
        cos = self.cos[token_position]  # (..., seq_len, d_k)
        sin = self.sin[token_position]  # (..., seq_len, d_k)
        
        # Split x into even and odd indices
        # x: (..., seq_len, d_k)
        # x_even: (..., seq_len, d_k/2) elements at 0,2,4,...
        # x_odd:  (..., seq_len, d_k/2) elements at 1,3,5,...
        x_even = x[..., 0::2] 
        x_odd = x[..., 1::2]
        
        # Apply rotation 
        # For the pair [x_{2k}, x_{2k+1}]:
        # x'_{2k}   = x_{2k} * cos - x_{2k+1} * sin
        # x'_{2k+1} = x_{2k} * sin + x_{2k+1} * cos
        
        # Get cos and sin for even / odd positions
        cos_even = cos[..., 0::2] # (..., seq_len, d_k/2)
        cos_odd = cos[..., 1::2] # (..., seq_len, d_k/2)
        sin_even = sin[..., 0::2] # (..., seq_len, d_k/2)
        sin_odd = sin[..., 1::2] # (..., seq_len, d_k/2)
        
        # Compute rortated components
        x_even_rot = x_even * cos_even - x_odd * sin_even
        x_odd_rot = x_even * sin_odd + x_odd * cos_odd
        
        # Interleave the results back
        # Stack and reshape to interleave
        x_out = torch.stack([x_even_rot, x_odd_rot], dim=-1) # (..., seq_len, d_k/2, 2)
        x_out = x_out.flatten(-2) # (..., seq_len, d_k)
        
        return x_out
                

def softmax(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax operation on a tensor along a specified dimension.
    Args:
        tensor: Input tensor of arbitrary shape
        dim: Dimension along which to apply softmax
    Returns:
        Tensor of the same shape with softmax applied along the specified dimension
    """
    # Find the maximum value along the specified dimension
    # keepdim=True to maintain the dimension for broadcasting
    max_vals = tensor.max(dim=dim, keepdim=True)[0]
    
    # Substract max from the original tensor for numerical stability
    shifted = tensor - max_vals
    
    # Compute exponentials
    exp_vals = torch.exp(shifted)
    
    # Sum the exponentials along the dimension
    sum_exp = exp_vals.sum(dim=dim, keepdim=True)
    
    # Normalize by dividing by the sum
    softmax_output = exp_vals / sum_exp 
    
    return softmax_output

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """ 
    Compute scaled dot-product attention using einsum.
    Args : 
        query: Query tensor of shape (batch_size, ..., seq_len_q, d_k)
        key: Key tensor of shape (batch_size, ..., seq_len_q, d_k)
        value: Value tensor of shape (batch_size, ..., seq_len_k, d_v)
        mask: Optional boolean mask of shape (seq_len_q, seq_len_k) 
              where True = attend, False = do not attend
              
    Returns: 
        Output tensor of shape (batch_size, ..., seq_len_q, d_v)
    """
    d_k = query.size(-1)
    
    # Step 1 : Compute attention scores using einsum
    # query : (..., n, d), key: (..., m, d)
    # We want to compute dot product between each query and key
    # Output : (..., n, m)
    scores = torch.einsum('...nd,...md->...nm', query, key)
    
    # Step 2: Scale by 1/sqrt(d_k)
    scores = scores / (d_k ** 0.5)
    
    # Step 3: Apply mask if provided
    if mask is not None:
        # mask shape: (n, m) or broadcastable to (..., n, m)
        scores = scores.masked_fill(~mask, float('-inf'))
        
    # Step 4: Apply softmax to get attention weights
    attention_weights = softmax(scores, dim=-1) # (..., n, m)
    
    # Handle the case where an entire row is -inf (no valid keys to attend to)
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
    
    # Step 5: Multiply by values
    output = torch.einsum('...nm,...mv->...nv', attention_weights, value) # (..., n, d_v)
    
    return output
    
    
