import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import RMSNorm
 
 
def init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
 
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )
 
def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)
 
class MultiheadDiffAttn(nn.Module):
    def __init__(
            self,
            embed_dim,
            depth,
            num_heads,
            model_parallel_size=1,
            decoder_kv_attention_heads=None,
            vis=False,
            return_2=False,
    ):
        # print(f"MultiheadDiffAttn with num_heads={num_heads}, embed_dim={embed_dim}, depth={depth}")
        super().__init__()
        self.vis = vis
        self.return_2 = return_2
        # self.args = args
        self.embed_dim = embed_dim
        # num_heads set to half of Transformer's #heads
        self.num_heads = num_heads // model_parallel_size
        self.num_kv_heads = decoder_kv_attention_heads // model_parallel_size if decoder_kv_attention_heads is not None else num_heads // model_parallel_size
        self.n_rep = self.num_heads // self.num_kv_heads
 
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
 
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
 
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
 
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)
 
    def forward(
            self,
            x,
            rel_pos=None,
            attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len
 
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
 
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
 
        # # commented out by me
        # q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        # k = apply_rotary_emb(k, *rel_pos, interleaved=True)
 
        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        # # commented out by me
        # attn_weights += attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
 
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
 
        # added by me
        attn_weights += rel_pos if rel_pos is not None else 0
 
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
 
        attn = self.out_proj(attn)
        if self.return_2:
            return attn, attn_weights if self.vis else None
        return attn
 
 
class MultiheadDiffAttnCrossV1(nn.Module):
    """
    Cross attention with MultiheadDiffAttn
    In this version, the query is from decoder.
    """
 
    def __init__(
            self,
            embed_dim,
            depth,
            num_heads,
            model_parallel_size=1,
            decoder_kv_attention_heads=None,
            vis=False,
            return_2=False,
            H=None,
            W=None,
    ):
        # print(f"MultiheadDiffAttn with num_heads={num_heads}, embed_dim={embed_dim}, depth={depth}")
        super().__init__()
        self.h = H
        self.w = W
 
        self.vis = vis
        self.return_2 = return_2
        # self.args = args
        self.embed_dim = embed_dim
        # num_heads set to half of Transformer's #heads
        self.num_heads = num_heads // model_parallel_size
        self.num_kv_heads = decoder_kv_attention_heads // model_parallel_size if decoder_kv_attention_heads is not None else num_heads // model_parallel_size
        self.n_rep = self.num_heads // self.num_kv_heads
 
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
 
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
 
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
 
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)
 
    def forward(
            self,
            x,
            q_dec,
            rel_pos=None,
            attn_mask=None,
    ):
        x_ = x
        if self.h is not None and self.w is not None:
            x = x.view(x.size(0), -1, x.size(1))
            q_dec = q_dec.view(q_dec.size(0), -1, q_dec.size(1))
        #             print("x is:",x.shape)
        #             print("q-dec is",q_dec.shape)
 
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len
 
        q = self.q_proj(q_dec)
        k = self.k_proj(x)
        v = self.v_proj(x)
 
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
 
        # # commented out by me
        # q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        # k = apply_rotary_emb(k, *rel_pos, interleaved=True)
 
        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        # # commented out by me
        # attn_weights += attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
 
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
 
        # added by me
        attn_weights += rel_pos if rel_pos is not None else 0
 
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
 
        attn = self.out_proj(attn)
        if self.h is not None and self.w is not None:
            attn = attn.view(attn.size(0), attn.size(2), attn.size(1) // self.h, attn.size(1) // self.w)
            attn = attn + x_
        if self.return_2:
            return attn, attn_weights if self.vis else None
        return attn
 
 
class MultiheadDiffAttnCrossV2(nn.Module):
    """
    Cross attention with MultiheadDiffAttn
    In this version, the query is from encoder + skip connection is added.
    """
 
    def __init__(
            self,
            embed_dim,
            depth,
            num_heads,
            model_parallel_size=1,
            decoder_kv_attention_heads=None,
            vis=False,
            return_2=False,
            H=None,
            W=None,
    ):
        # print(f"MultiheadDiffAttn with num_heads={num_heads}, embed_dim={embed_dim}, depth={depth}")
        super().__init__()
        self.h = H
        self.w = W
 
        self.vis = vis
        self.return_2 = return_2
        # self.args = args
        self.embed_dim = embed_dim
        # num_heads set to half of Transformer's #heads
        self.num_heads = num_heads // model_parallel_size
        self.num_kv_heads = decoder_kv_attention_heads // model_parallel_size if decoder_kv_attention_heads is not None else num_heads // model_parallel_size
        self.n_rep = self.num_heads // self.num_kv_heads
 
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
 
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
 
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
 
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)
 
    def forward(
            self,
            x,
            q_enc,
            rel_pos=None,
            attn_mask=None,
    ):
        x_ = q_enc
        if self.h is not None and self.w is not None:
            x = x.view(x.size(0), -1, x.size(1))
            q_enc = q_enc.view(q_enc.size(0), -1, q_enc.size(1))
 
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len
 
        q = self.q_proj(q_enc)
        k = self.k_proj(x)
        v = self.v_proj(x)
 
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
 
        # # commented out by me
        # q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        # k = apply_rotary_emb(k, *rel_pos, interleaved=True)
 
        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        # # commented out by me
        # attn_weights += attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
 
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
 
        # added by me
        attn_weights += rel_pos if rel_pos is not None else 0
 
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
 
        attn = self.out_proj(attn)
        if self.h is not None and self.w is not None:
            attn = attn.view(attn.size(0), -1, attn.size(1) // self.h, attn.size(1) // self.w)
            # attn = attn + x_
        if self.return_2:
            return attn, attn_weights if self.vis else None
        return attn
 
 
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
 
class FEA(nn.Module):
    def __init__(self, dim: int, scale_factors: list, label="", writer=None) -> None:
        super().__init__()
        self.writer = writer
        self.label = label
        self.scale_factors = scale_factors
        self.n = n = len(scale_factors)
        self.m = m = n * (n - 1) // 2
        # self.ew = nn.Parameter(torch.randn(m, 1, 1, 1, 1))
        self.w = nn.Parameter(torch.randn(1, dim, 1, 1) + 0.5)
        # Get the indices of the upper triangle (j > i)
        self.indices = torch.triu_indices(row=self.n, col=self.n, offset=1)
        self.global_step = 0
 
    def compute_weighted_edges(self, edges_stack):
        # edges_stack: (m, B, C, H, W)
 
        # Compute the pairwise absolute differences. diff_matrix: (m, m, B, C, H, W)
        diff_matrix = torch.abs(edges_stack[:, None, ...] - edges_stack[None, :, ...])
 
        w_edge = 0
        for idx, (i, j) in enumerate(zip(self.indices[0], self.indices[1])):
            # w_edge += F.sigmoid(1e1*self.ew[idx]) * diff_matrix[i, j, ...]
            w_edge += 1. / self.m * diff_matrix[i, j, ...]
        return w_edge
 
    def write_info(self):
        # write the weights of the edges to tensorboard
        for idx, (i, j) in enumerate(zip(self.indices[0], self.indices[1])):
            title = f"EdgeWeights/L:{self.label}_({self.scale_factors[i]}-{self.scale_factors[j]})"
            self.writer.add_scalar(title, F.sigmoid(1e1 * self.ew[i]).mean().item(), self.global_step)
            # title = f"EdgeWeights/L:{self.label},Std({self.scale_factors[i]}-{self.scale_factors[j]})"
            # self.writer.add_scalar(title, F.sigmoid(1e1*self.ew[i]).std().item(), self.global_step)
            # title = f"EdgeWeights/L:{self.label},Hist({self.scale_factors[j]}-{self.scale_factors[i]})"
            # self.writer.add_histogram(title, F.sigmoid(1e1*self.ew[i]).flatten(), self.global_step)
        self.global_step += 1
 
    def forward(self, x):
        _, C, H, W = x.shape
        edges = []
        for scale in self.scale_factors:
            x_1 = F.interpolate(x, scale_factor=scale, mode="bilinear")
            x_1 = F.interpolate(x_1, size=(H, W), mode='bilinear')
            edges.append(torch.abs(x - x_1))
        edge = self.compute_weighted_edges(torch.stack(edges))
 
        # write the weights of the edges to tensorboard if it's train mode and writer is not None
        # if self.writer is not None and self.train:
        #     self.write_info()
 
        return x + self.w * edge
 
 
class DSEBFusion(nn.Module):
    '''
    if use_command is 'no', the block will not be used
    if use_command is 'dat', the block will only use DiffAttn
    if use_command is 'fea', the block will only use Feature Edge Amplification
    if use_command is 'dat-fea', the block will use both DiffAttn and Feature Edge Amplification
    if use_command is 'dog', the block will use difference of Gaussian (with learnable sigma) for feature sharpening
    if ues_command is 'dat-dog', the block will use both DiffAttn and difference of Gaussian
    if use_command includes 'seq', the block will use sequential attention (last is DiffAttn)
    '''
 
    def __init__(self, dim, input_size=32,scale_factors=[0.5, 0.75, 1.25], num_heads=8,  mode='add', use_command='dat-fea', depth=1, label="",
                 writer=None):
        super().__init__()
        self.use_command = use_command
        self.not_use_this = 'no' in self.use_command.lower()
        if self.not_use_this:
            return
        self.input_size = input_size
        self.mode = mode.lower()
        _dim = dim * 2 if self.mode == 'cat' else dim
 
        self.use_diffattn = 'dat' in self.use_command.lower()
        self.use_fea = 'fea' in self.use_command.lower()
        self.use_dog = 'dog' in self.use_command.lower()
        self.do_seq = 'seq' in self.use_command.lower()
 
        if self.use_fea:
            self.boundary = FEA(dim=_dim, scale_factors=scale_factors, label=label, writer=writer)
        if self.use_diffattn:
            self.diffattn = MultiheadDiffAttn(embed_dim=_dim, depth=depth, num_heads=num_heads)
        if self.use_dog:
            self.sigma_raw_1 = nn.Parameter(torch.randn(1, _dim, 1, 1) - .4, requires_grad=True)
            self.sigma_raw_2 = nn.Parameter(torch.randn(1, _dim, 1, 1) - .0, requires_grad=True)
        self.mixer = nn.Conv2d(in_channels=_dim, out_channels=dim, kernel_size=1, stride=1, bias=False)
 
    def apply_diffattn(self, x):
        y_token = x.view(x.shape[0], -1, x.shape[1])
        diff = self.diffattn(y_token)  # * y_token
        diff = diff.view(diff.shape[0], diff.shape[2], diff.shape[1] // self.input_size,
                         diff.shape[1] // self.input_size)
        return diff * x
 
    def get_sigma(self, sigma):
        # Apply sigmoid to keep values in (0, 1), then scale to (0, 2)
        sigma = 2 * torch.sigmoid(sigma)
        return sigma  # Now sigma is in range [0, 2]
 
    def gaussian_kernel_3x3(self, sigma):
        device = sigma.device
        C = sigma.shape[1]  # Number of channels
        # 1D coordinate grid
        coords = torch.tensor([-1.0, 0.0, 1.0], device=device)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")  # Shape: [3, 3]
        # Compute 3x3 Gaussian kernel
        sigma_sq = self.get_sigma(sigma) ** 2  # Shape: [1, C, 1, 1]
        kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma_sq))  # Broadcasted
        # Normalize
        kernel = kernel / kernel.sum(dim=[0, 1], keepdim=True)
        return kernel.view(C, 1, 3, 3)
 
    def smooth_with_gaussian(self, x, sigma):
        B, C, H, W = x.shape
        # Ensure sigma has the right shape
        if sigma.shape != (1, C, 1, 1):
            sigma = sigma.view(1, C, 1, 1)
        kernel = self.gaussian_kernel_3x3(sigma)  # [C, 1, 3, 3]
        padding = 1  # To maintain spatial dimensions
        smoothed_x = F.conv2d(x, kernel, groups=C, padding=padding)
        return smoothed_x
 
    def apply_dog(self, x):
        x_smoother = self.smooth_with_gaussian(x, self.sigma_raw_1)
        x_smoothest = self.smooth_with_gaussian(x, self.sigma_raw_2)
        return x_smoother - x_smoothest
 
    def forward(self, data):
        skip, dec = data
        if self.not_use_this:
            return skip
        y = dec + skip if self.mode == 'add' else torch.cat([dec, skip], dim=1)
        x_fea = self.boundary(y) + y if self.use_fea else 0
        x_dog = self.apply_dog(y) + y if self.use_dog else 0
        if self.do_seq:
            y = x_fea + x_dog if self.use_fea or self.use_dog else y
            x_fea = x_dog = 0
        x_dat = self.apply_diffattn(y) if self.use_diffattn else 0
        z = x_fea + x_dog + x_dat if self.use_fea or self.use_dog or self.use_diffattn else y
        z = self.mixer(z)
        return z + skip
 