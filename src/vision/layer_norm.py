import torch
import torch.nn as nn

class LayerNormalization(nn.Module):

    def __init__(self,embed_dim:int,eps:float = 1e-5,device: torch.device | None = None) -> None:

        super().__init__()

        self.embed_dim = embed_dim 
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embed_dim,device = device,dtype = torch.float32))
        self.beta = nn.Parameter(torch.zeros(embed_dim,device = device,dtype = torch.float32))

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        if x.shape[-1] != self.embed_dim:

            raise ValueError(f"Expected last dim {self.embed_dim}, got {x.shape[-1]}")

        original_dtype = x.dtype
        x = x.float()
        mean = x.mean(dim = -1,keepdim = True)
        var = x.var(dim = -1,keepdim = True,unbiased = False)
        x_hat = (x - mean) * torch.rsqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta
        out = out.to(original_dtype)  

        return out