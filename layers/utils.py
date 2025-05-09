import torch
from torch import nn
from math import sqrt
from einops import rearrange, repeat


class ReprogrammingLayer(nn.Module):
    # Adapt input embeddings to pretrained knowledge space of LLM (e.g., vocab library or knowledge base)

    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        """
        Cross-Attention with Query from input/target, Key & Value from source embeddings

        Arguments:
            d_model (int): Dimension of input embeddings
            n_heads (int): Number of attention heads
            d_keys (Optional[int]): Dimension of key vectors for each head
            d_llm (Optional[int]): Dimension of source embeddings from LLM
        """

        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)  # Maps output back into d_llm
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape  # B = batch size, L = sequence length of target data
        S, _ = source_embedding.shape  # S = sequence length of source embedding (like LLM vocab size)
        H = self.n_heads

        # Prepare for multi-head attention -> each head operates on vectors of size d_keys
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)  # (B, L, H, d_keys)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)  # (S, H, d_keys)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)  # (S, H, d_keys)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)  # (B, L, H, d_keys) returned

        out = out.reshape(B, L, -1)  # Combine heads (B, L, H * d_keys)

        return self.out_projection(out)  # (B, L, d_llm)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        """
        Compute attention scores & apply them to value embeddings
        """

        B, L, H, E = target_embedding.shape

        scale = 1.0 / sqrt(E)  # scale dot product for stability

        # Compute attention scores
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)  # dot product -> (B, H, L, S)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # Retrieve attention weights across S dimension
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)  # Multiplies A with value embeddings -> reprogrammed embeddings (B, L, H, d_keys)

        return reprogramming_embedding


class FlattenHead(nn.Module):
    # Flattens & projects raw output of Time-LLM into desired output space

    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        """
        Arguments:
            n_vars (int): Number of input variables (channels)
            nf (int): Size of last feature dimension before projection
            target_window (int): Output window size
        """

        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(
            start_dim=-2
        )  # flatten (B, n_vars, d_ff, self.patch_nums) -> (B, n_vars, nf) where nf = d_ff * self.patch_nums
        self.linear = nn.Linear(
            nf, target_window
        )  # project (B, n_vars, nf) -> (B, n_vars, target_window) where target_window = pred_len
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class TwoStageAttention(nn.Module):
    '''
    Modified Crossformer implemented in https://github.com/Thinklab-SJTU/Crossformer/blob/master/cross_models/attn.py

    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Patch_num(L), d_model]
    '''
    def __init__(self, patch_num, d_model, n_heads, dropout, factor=10, d_ff=None, capture_time=False):
        super(TwoStageAttention, self).__init__()
        d_ff = d_ff or 4*d_model

        self.capture_time = capture_time
        if self.capture_time:
            self.time_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )

        self.dim_sender = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dim_receiver = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.router = nn.Parameter(torch.randn(patch_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))

    def forward(self, x):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]

        if self.capture_time:
            # Cross Time Stage: Directly apply MSA to each dimension
            time_in = rearrange(x, 'b n_vars patch_num d_model -> (b n_vars) patch_num d_model')
            time_enc, _ = self.time_attention(
                time_in, time_in, time_in
            )
            dim_in = time_in + self.dropout(time_enc)
            dim_in = self.norm1(dim_in)
            dim_in = dim_in + self.dropout(self.MLP1(dim_in))
            dim_in = self.norm2(dim_in)

            dim_send = rearrange(dim_in, '(b n_vars) patch_num d_model -> (b patch_num) n_vars d_model', b = batch)
        else:
            dim_send = rearrange(x, 'b n_vars patch_num d_model -> (b patch_num) n_vars d_model', b = batch)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        batch_router = repeat(self.router, 'patch_num factor d_model -> (repeat patch_num) factor d_model', repeat = batch)
        dim_buffer, _ = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive, _ = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b patch_num) n_vars d_model -> b n_vars patch_num d_model', b = batch)

        return final_out
