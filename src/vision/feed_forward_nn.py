import torch
import torch.nn as nn

class VisionFeedForwardNN(nn.Module):

    def __init__(self,in_features:int,hidden_features:int,out_features:int,mlp_p:float = 0.0) -> None:

        super().__init__()

        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(mlp_p)

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        if x.shape[-1] != self.in_features:

            raise ValueError(f"Expected last dim {self.in_features}, got {x.shape[-1]}")

        x = self.act(self.fc1(x))  
        x = self.drop(x)
        x = self.fc2(x)             
        x = self.drop(x)
        
        return x    