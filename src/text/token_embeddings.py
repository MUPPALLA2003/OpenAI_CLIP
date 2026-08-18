import torch
import torch.nn as nn

class Embeddings(nn.Module):

    def __init__(self,d_model:int,vocab_length:int,max_seq_len:int,emb_p:float) -> None:

        super().__init__()

        self.emb_drop = nn.Dropout(emb_p)
        self.embedding = nn.Embedding(vocab_length,d_model)
        self.positional_embed = nn.Embedding(max_seq_len,d_model)
        self.register_buffer("position_ids",torch.arange(max_seq_len).unsqueeze(0),persistent=False)

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        _,T = x.shape
        positions = self.position_ids[:, :T]
        input_embed = self.embedding(x)
        pos_embed = self.positional_embed(positions).unsqueeze(0)
        token_embed = self.emb_drop(input_embed + pos_embed)

        return token_embed
        