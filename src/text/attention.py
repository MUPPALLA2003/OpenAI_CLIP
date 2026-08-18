import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalAttention(nn.Module):

    def __init__(self,d_model:int,max_seq_len:int,n_heads:int,attn_p:float,proj_p:float,flash_attn:bool) -> None:

        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.flash_attn = flash_attn
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_p)
        self.proj_drop = nn.Dropout(proj_p)
        self.Q = nn.Linear(d_model,d_model)
        self.K = nn.Linear(d_model,d_model)
        self.V = nn.Linear(d_model,d_model)
        self.proj = nn.Linear(d_model,d_model)

        self.register_buffer("causal_mask",torch.triu(torch.ones(max_seq_len,max_seq_len,dtype=torch.bool),diagonal=1),persistent=False)

    def attention(self,query:torch.Tensor,key:torch.Tensor,value:torch.Tensor,causal:bool=True) -> torch.Tensor:

        seq_len = query.size(-2)

        if self.flash_attn:

            out = F.scaled_dot_product_attention(query,key,value,attn_mask=None,dropout_p=self.attn_drop.p if self.training else 0.0,is_causal=causal)

            return out

        attn_scores = query @ key.transpose(-2,-1) * self.scale
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(causal_mask,float("-inf"))
        attn_weights = F.softmax(attn_scores,dim=-1)
        out = self.attn_drop(attn_weights)
        out = attn_weights @ value
        
        return out

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        batch,seq_len,C = x.shape

        assert C == self.d_model

        query = self.Q(x).view(batch,seq_len,self.n_heads,self.head_dim).transpose(1,2)
        key = self.K(x).view(batch,seq_len,self.n_heads,self.head_dim).transpose(1,2)
        value = self.V(x).view(batch,seq_len,self.n_heads,self.head_dim).transpose(1,2)
        out = self.attention(query,key,value)

        out = out.transpose(1,2).view(batch,seq_len,C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out




