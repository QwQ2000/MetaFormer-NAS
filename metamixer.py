import math
import torch
from torch import nn, einsum
from einops import rearrange

class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

class AbsPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.pos_emb = AbsPosEmb(fmap_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb(q)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        
        return out

class SpatialMLP(nn.Module):
    def __init__(self, fmap_size, hidden_rate=0.5):
        super().__init__()

        H, W = fmap_size
        hidden_dim = int(H * W * hidden_rate)
        
        self.fc1 = nn.Linear(H * W, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, H * W)
    
    def forward(self, x):
        N, C, H, W = x.shape

        x = x.view(N, C, H * W)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.view(N, C, H ,W)
        
        return x

class MetaMixerEntrance(nn.Module):
    def __init__(self, dim, fmap_size):
        super().__init__()

        self.mlp1 = SpatialMLP(fmap_size=fmap_size)
        self.mlp2 = SpatialMLP(fmap_size=fmap_size)
        self.attn = Attention(dim=dim, fmap_size=fmap_size)
    
    def forward(self, x):
        return self.mlp1(self.attn(x)) + self.mlp2(x)

class MetaMixer(nn.Module):
    def __init__(self, dim, fmap_size, num_nodes):
        super().__init__()
        
        self.num_nodes = num_nodes

        self.pooling = Pooling()
        self.identity = nn.Identity()
        self.entrance = MetaMixerEntrance(dim=dim, fmap_size=fmap_size)

        self.shared_mlps = nn.ModuleList([SpatialMLP(fmap_size=fmap_size) for _ in range(num_nodes - 1)])
    
    def __forward_edge(self, x, node_id, op_str):
        if op_str != 'avg':
            x = getattr(self, op_str)(x)
            x = self.shared_mlps[node_id - 1](x)
        return x
    
    def forward(self, x, arch):
        feature_queue = [(x, 0)]
        out_features = []
        while feature_queue:
            if feature_queue[0][1] == self.num_nodes:
                out_features.append(feature_queue[0][0])
                feature_queue = feature_queue[1:]
                continue

            for node in arch[feature_queue[0][1]]:
                feature_queue.append((self.__forward_edge(x, node.id, node.name), node.id))
            feature_queue = feature_queue[1:]
        return torch.stack(out_features).sum(0) / len(out_features)
