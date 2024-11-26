from functools import wraps

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from torch_cluster import fps

from timm.models.layers import DropPath
from util.neuron_dataset import project_pointcloud, save_im
from decoders.deepset import DeepSet

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))


class PointEmbed(nn.Module):

    def __init__(self, hidden_dim=90, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        pe = self.embed(input, self.basis)
        embed = self.mlp(torch.cat([pe, input], dim=2))  # B x N x C
        return embed


# class DiagonalGaussianDistribution(object):
#     def __init__(self, mean, logvar, deterministic=False):
#         self.mean = mean
#         self.logvar = logvar
#         self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
#         self.deterministic = deterministic
#         self.std = torch.exp(0.5 * self.logvar)
#         self.var = torch.exp(self.logvar)
#         if self.deterministic:
#             self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

#     def sample(self):
#         x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
#         return x

#     def kl(self, other=None):
#         if self.deterministic:
#             return torch.Tensor([0.])
#         else:
#             if other is None:
#                 return 0.5 * torch.mean(torch.pow(self.mean, 2)
#                                        + self.var - 1.0 - self.logvar,
#                                        dim=[1, 2])
#             else:
#                 return 0.5 * torch.mean(
#                     torch.pow(self.mean - other.mean, 2) / other.var
#                     + self.var / other.var - 1.0 - self.logvar + other.logvar,
#                     dim=[1, 2, 3])

#     def nll(self, sample, dims=[1,2,3]):
#         if self.deterministic:
#             return torch.Tensor([0.])
#         logtwopi = np.log(2.0 * np.pi)
#         return 0.5 * torch.sum(
#             logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
#             dim=dims)

#     def mode(self):
#         return self.mean
    
# class BoundedSoftplus(nn.Module):
#     def __init__(self, upper_bound=1.0):
#         super().__init__()
#         self.upper_bound = upper_bound

#     def forward(self, x):
#         return F.softplus(x) - F.softplus(x - self.upper_bound)
    
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),  # Concatenating two node features
            nn.ReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class AutoEncoder(nn.Module):

    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        num_latents=512,
        queries_dim=512,
        # num_inputs=8192,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
    ):
        super().__init__()

        self.depth = depth
        self.num_latents = num_latents

        self.point_embed = PointEmbed(dim=dim)

        self.learnable_query = nn.Parameter(torch.randn(1, num_latents, queries_dim), requires_grad=True)

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.decoder_cross_attn = PreNorm(queries_dim * 2, Attention(queries_dim * 2, dim, heads = 1, dim_head = dim), context_dim = dim)

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.mlp = MLP(queries_dim * 2)

    def m_layer_attention(self, x,):
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x
        return x
    
    def encode(self, pc_embeddings, mask = None):
        B, _, _ = pc_embeddings.shape
        cross_attn, cross_ff = self.cross_attend_blocks
        learnable_q = self.learnable_query.expand(B, -1, -1)
        x = cross_attn(learnable_q, context = pc_embeddings, mask = mask) + learnable_q
        x = cross_ff(x) + x
        return x

    def forward(self, pc, mask, pairs):
        B, N, D = pc.shape

        # encode 
        pc_embeddings = self.point_embed(pc)
        x = self.encode(pc_embeddings, mask)
        x = self.m_layer_attention(x) # multi-layer attention

        # decode
        node_features_pairs = pair(pc_embeddings, pairs)
        query_features = self.decoder_cross_attn(node_features_pairs, context = x)
        affinity = self.mlp(query_features).squeeze(-1)

        return {'affinity': affinity, "embeddings": query_features, "latents": x}
    

def pair(data: torch.tensor, pairs: torch.tensor): 
    idx = pairs[..., 0].unsqueeze(-1).expand(-1, -1, data.shape[-1])
    node_features_x = torch.gather(data, dim=1, index=idx)
    idx = pairs[..., 1].unsqueeze(-1).expand(-1, -1, data.shape[-1])
    node_features_y = torch.gather(data, dim=1, index=idx)
    pairs = torch.cat([node_features_x, node_features_y], dim=-1)
    return pairs



def create_autoencoder(dim=512, num_latents=256, N=8192, depth=24):
    model = AutoEncoder(
        depth=depth,
        dim=dim,
        num_latents=num_latents,
        queries_dim=dim,
        # num_inputs = N,
        heads = 8,
        dim_head = 64,
    )

    return model



def ae_d2048_m64(N=8192, depth=24):
    return create_autoencoder(dim=2048, num_latents=64, N=N, depth=depth)

def ae_d1024_m64(N=8192, depth=24):
    return create_autoencoder(dim=1024, num_latents=64, N=N, depth=depth)

def ae_d512_m64(N=8192, depth=24):
    return create_autoencoder(dim=512, num_latents=64, N=N, depth=depth)

def ae_d256_m64(N=8192, depth=24):
    return create_autoencoder(dim=256, num_latents=64, N=N, depth=depth)

def ae_d128_m64(N=8192, depth=24):
    return create_autoencoder(dim=128, num_latents=64, N=N, depth=depth)

def ae_d64_m64(N=8192, depth=24):
    return create_autoencoder(dim=64, num_latents=64, N=N, depth=depth)



###
def ae_d2048_m256(N=8192, depth=24):
    return create_autoencoder(dim=2048, num_latents=256, N=N, depth=depth)

def ae_d1024_m256(N=8192, depth=24):
    return create_autoencoder(dim=1024, num_latents=256, N=N, depth=depth)

def ae_d512_m256(N=8192, depth=24):
    return create_autoencoder(dim=512, num_latents=256, N=N, depth=depth)

def ae_d256_m256(N=8192, depth=24):
    return create_autoencoder(dim=256, num_latents=256, N=N, depth=depth)

def ae_d128_m256(N=8192, depth=24):
    return create_autoencoder(dim=128, num_latents=256, N=N, depth=depth)

def ae_d64_m256(N=8192, depth=24):
    return create_autoencoder(dim=64, num_latents=256, N=N, depth=depth)


### 512
def ae_d2048_m512(N=8192, depth=24):
    return create_autoencoder(dim=2048, num_latents=512, N=N, depth=depth)

def ae_d1024_m512(N=8192, depth=24):
    return create_autoencoder(dim=1024, num_latents=512, N=N, depth=depth)

def ae_d512_m512(N=8192, depth=24):
    return create_autoencoder(dim=512, num_latents=512, N=N, depth=depth)

def ae_d256_m512(N=8192, depth=24):
    return create_autoencoder(dim=256, num_latents=512, N=N, depth=depth)

def ae_d128_m512(N=8192, depth=24):
    return create_autoencoder(dim=128, num_latents=512, N=N, depth=depth)

def ae_d64_m512(N=8192, depth=24):
    return create_autoencoder(dim=64, num_latents=512, N=N, depth=depth)


### 1024
def ae_d2048_m1024(N=8192, depth=24):
    return create_autoencoder(dim=2048, num_latents=1024, N=N, depth=depth)

def ae_d1024_m1024(N=8192, depth=24):
    return create_autoencoder(dim=1024, num_latents=1024, N=N, depth=depth)

def ae_d512_m1024(N=8192, depth=24):
    return create_autoencoder(dim=512, num_latents=1024, N=N, depth=depth)

def ae_d256_m1024(N=8192, depth=24):
    return create_autoencoder(dim=256, num_latents=1024, N=N, depth=depth)

def ae_d128_m1024(N=8192, depth=24):
    return create_autoencoder(dim=128, num_latents=1024, N=N, depth=depth)

def ae_d64_m1024(N=8192, depth=24):
    return create_autoencoder(dim=64, num_latents=1024, N=N, depth=depth)