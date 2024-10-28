import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import torch
@dataclass
class VITConfig:
    n_emb: int = 768
    image_size: int = 224
    n_heads: int = 12
    patch_size = 16


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Device: {device}")




class SelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__(config)
        assert config.n_emb % config.n_head == 0
        self.n_head = config.n_heads
        self.n_emb = config.n_emb
        self.n_emb = config.n_emb
        self.proj = nn.Linear(config.n_emb,config.n_emb)
        self.scale = config.n_emb ** -0.5
        self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb)

    def forward(self,x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_emb   )
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_emb, dim=2)
        q = q.view(B,T,self.n_head, C // self.n_head).transpose(1,2)
        k = k.view(B,T,self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head, C // self.n_head).transpose(1,2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        out = out.transpose(1,2).contiguous().view(B,T,C)

        out = self.proj(out)
        return out
    
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__(config)
        self.fc1 = nn.Linear(config.n_emb, 4 * config.n_emb)
        self.gelu = nn.GELU(approximate='tanh') #for our large dataset regularization
        self.fc2 = nn.Linear(4 * config.n_emb, config.n_emb)
        self.dropout = nn.Dropout()
    def forward(self,x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self,config):
        super().__init__(config)
        self.mlp = MLP(config)
        self.msa = SelfAttention(config)
        self.norm1 = nn.LayerNorm(config.n_emb)
        self.norm2 = nn.LayerNorm(config.n_emb)
    def forward(self,x):
        x = x + self.msa(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    


# Input image is (B, C, H, W)
# And patch_size is (P, P)
# B = batch size, C = channels, H = height, W = width, P = patch size
class PatchEmbedding(nn.Module):
    def __init__(self,config):
        super().__init(config)
        #in_channels(n_hidden = n_emb/n_heads)
        self.ln_proj = nn.Conv2d(
            in_channels=config.n_emb//config.n_heads,
            out_channels=config.n_emb,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        self.flatten = nn.Flatten(2)
    def forward(self,x):
        x = self.ln_proj(x)
        x = self.flatten(x)
        x = x.transpose(1,2)
        return x

