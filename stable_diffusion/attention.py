import torch 
import torch.nn as nn 
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, causal_mask = False):
        # x: (B, Seq_len, d_embed)
        batch_size, seq_len, d_embed = x.shape
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)
        q, k, v = self.in_proj(x).chunk(3, dim = -1)
        q = q.view(interim_shape).transpose(1, 2) 
        k = k.view(interim_shape).transpose(1, 2) 
        v = v.view(interim_shape).transpose(1, 2) 
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            causal_mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal = 1).bool()
            weight = weight.masked_fill(causal_mask, float('-inf'))
        
        weight /= (self.d_head ** 0.5)
        weight = F.softmax(weight, dim = -1)
        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape(batch_size, seq_len, d_embed)
        output = self.out_proj(output)
        return output
        
    
class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias = in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x, y):
        # x (latent) : (B, Seq_len, d_embed)
        # y (context) : (B, Seq_len, d_cross)
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        q = self.q_proj(x).view(interim_shape).transpose(1, 2)
        k = self.k_proj(y).view(interim_shape).transpose(1, 2)
        v = self.v_proj(y).view(interim_shape).transpose(1, 2)
        weight = q @ k.transpose(-1, -2)
        weight /= (self.d_head ** 0.5)
        weight = F.softmax(weight, dim = -1)
        output = weight @ v
        output = output.transpose(1, 2).contiguous().view(input_shape)
        output = self.out_proj(output)
        return output
    