import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionAttention(nn.Module):

    def __init__(self,embed_dim:int,num_heads:int,attn_p:float,proj_p:float,flash_attn:bool) -> None:

        super().__init__()

        if embed_dim % num_heads != 0:

            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by n_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.flash_attn = flash_attn
        self.Q = nn.Linear(embed_dim,embed_dim)
        self.K = nn.Linear(embed_dim,embed_dim)
        self.V = nn.Linear(embed_dim,embed_dim)
        self.proj = nn.Linear(embed_dim,embed_dim)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj_drop = nn.Dropout(proj_p)

    def attention(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor) -> torch.Tensor:

        if self.flash_attn:

            y = F.scaled_dot_product_attention(q,k,v,dropout_p = self.attn_drop.p if self.training else 0)

            return y

        else:

            attention_logits = (q @ k.transpose(-2,-1)) * self.scale
            attention_weights = F.softmax(attention_logits,dim = -1)
            attention_weights = self.attn_drop(attention_weights)
            out = attention_weights @ v

            return out

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        batch,seq_len,C = x.shape

        if C != self.embed_dim:

            raise ValueError(f"Expected last dim {self.embed_dim}, got {C}")

        query = self.Q(x).view(batch,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        key = self.K(x).view(batch,seq_len,self.num_heads,self.head_dim).transpose(1,2) 
        value = self.V(x).view(batch,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        output = self.attention(query,key,value)
        output = output.transpose(1,2).contiguous().view(batch,seq_len,C)
        output = self.proj(output)
        output = self.proj_drop(output)

        return output   

                

            