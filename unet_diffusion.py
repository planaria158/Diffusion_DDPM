"""
Mark Thompson.  UNet 
https://arxiv.org/pdf/1807.10165.pdf
https://arxiv.org/pdf/1912.05074v2.pdf

This UNet is modified for use as a diffusion model.  It contains the time embeddings layers.
"""

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum


def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, numgroups=8):
        super().__init__()
        self.in_block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(numgroups, in_channels),
                    nn.SiLU(),
            )
        self.time_block = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, in_channels)
            )
        self.out_block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(numgroups, out_channels),
                )
        
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()        

    def forward(self, x, t_emb):
        out = self.in_block(x)
        out = out + self.time_block(t_emb)[:, :, None, None]
        out = self.out_block(out)
        out = out + self.res_conv(x)
        return F.silu(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, numgroups=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.norm = nn.GroupNorm(numgroups, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, numgroups=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))
        self.norm = nn.GroupNorm(numgroups, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
    

# class AttentionBlock(nn.Module):
#     def __init__(self, out_channels, num_heads=4, numgroups=8, dropout=0.):
#         super().__init__()
#         self.attention_norms = nn.GroupNorm(numgroups, out_channels)
#         self.attentions = nn.MultiheadAttention(out_channels, num_heads, dropout=dropout, batch_first=True)

#     def forward(self, x):
#         out = x
#         # Attention block of Unet
#         batch_size, channels, h, w = out.shape
#         in_attn = out.reshape(batch_size, channels, h * w)
#         in_attn = self.attention_norms(in_attn)
#         in_attn = in_attn.transpose(1, 2)    #So, I guess: [N, (h*w), C] where (h*w) is the target "sequence length", and C is the embedding dimension
#         out_attn, _ = self.attentions(in_attn, in_attn, in_attn)
#         out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
#         return out_attn

class AttentionBlock(nn.Module):
    def __init__(self, attention_type, dim, heads, dim_head):
        super().__init__()
        if attention_type == 'linear':
            self.attention_block = LinearAttention(dim, heads=heads, dim_head=dim_head)
        elif attention_type == 'full':
            self.attention_block = Attention(dim, heads=heads, dim_head=dim_head)
        else:
            print(f"Attention type {attention_type} not recognized.  Using Identity.")
            self.attention_block = nn.Identity()

    def forward(self, x):
        return self.attention_block(x)
        

class DownBlock(nn.Module):
    """
    Down conv block with attention.
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, attention_type, num_heads=4, dim_head=32, dropout=0, attn_dropout=0):
        super().__init__()
        self.residual_block_1 = ResidualBlock(in_channels, in_channels, t_emb_dim)
        self.attention_block = AttentionBlock(attention_type, in_channels, heads=num_heads, dim_head=dim_head)
        self.residual_block_2 = ResidualBlock(in_channels, in_channels, t_emb_dim)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # Strided convolution to downsize
        self.down_sample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t_emb):
        out = self.residual_block_1(x, t_emb)
        out_attn = self.attention_block(out)
        out = out + out_attn  
        out = self.residual_block_2(out, t_emb)
        skip_out = self.conv(out)
        out = self.down_sample_conv(skip_out)
        return out, skip_out


class MidBlock(nn.Module):
    """
    Mid conv block with attention.
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, attention_type, num_heads=4, dim_head=32, dropout=0, attn_dropout=0):
        super().__init__()
        self.residual_block_1 = ResidualBlock(in_channels, in_channels, t_emb_dim)
        self.attention_block = AttentionBlock(attention_type, in_channels, heads=num_heads, dim_head=dim_head)
        self.residual_block_2 = ResidualBlock(in_channels, in_channels, t_emb_dim)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t_emb):
        out = self.residual_block_1(x, t_emb) 
        out_attn = self.attention_block(out)
        out = out + out_attn  
        out = self.residual_block_2(out, t_emb)        
        out = self.conv(out)
        return out


class UpBlock(nn.Module):
    """
    Up conv block with attention.
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, attention_type, num_heads=4, dim_head=32, dropout=0, attn_dropout=0):
        super().__init__()
        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.residual_block_1 = ResidualBlock((in_channels + out_channels), (in_channels + out_channels), t_emb_dim)
        self.attention_block = AttentionBlock(attention_type, (in_channels + out_channels), heads=num_heads, dim_head=dim_head)
        self.residual_block_2 = ResidualBlock((in_channels + out_channels), out_channels, t_emb_dim)  
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        out = torch.cat([x, out_down], dim=1)  # add in the skip connection from corresponding DownBlock
        out = self.residual_block_1(out, t_emb)        
        out_attn = self.attention_block(out)
        out = out + out_attn  
        out = self.residual_block_2(out, t_emb)        
        out = self.conv(out)
        return F.silu(out)

#--------------------------------------------------------------------
# The full model
# Assumes input images are rgb, 3 channel
#--------------------------------------------------------------------
class UNet_Diffusion(nn.Module):
    def __init__(self, config):
        super(UNet_Diffusion, self).__init__()

        self.t_emb_dim = config['time_emb_dim']
        num_heads = config['num_heads']
        channels = config['channels']
        dim_head = config['dim_head']
        dropout = config['dropout']
        attn_dropout = config['attn_dropout']
        self.config = config
                
        assert(channels == [32, 64, 128, 256, 512]) # temp debug code for now
   
        down_attn_type = config['down_attn_type']
        down_channel_indices = config['down_channel_indices']
        assert(len(down_channel_indices) == len(down_attn_type))

        mid_attn_type = config['mid_attn_type']
        mid_channel_indices = config['mid_channel_indices']
        assert(len(mid_channel_indices) == len(mid_attn_type))

        up_attn_type = config['up_attn_type']
        up_channel_indices = config['up_channel_indices']
        assert(len(up_channel_indices) == len(up_attn_type))

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        # first conv increase filter count of the input image.
        self.conv_in = nn.Conv2d(3, channels[0], kernel_size=3, padding=1)

        # last layers
        self.norm_out   = nn.GroupNorm(8, channels[0])
        self.conv_out_1 = nn.Conv2d(channels[0], channels[0]//2, kernel_size=3, padding=1)
        self.norm_out_1 = nn.GroupNorm(8, channels[0]//2)
        self.conv_out_2 = nn.Conv2d(channels[0]//2, 3, kernel_size=1, padding=0)

        #------------------------------------------------------------
        # The Encoding down blocks. Input image = (size, size)
        #------------------------------------------------------------
        self.down_blocks = nn.ModuleList()
        for (in_idx, out_idx), attn in zip(down_channel_indices, down_attn_type):
            self.down_blocks.append(DownBlock(channels[in_idx], channels[out_idx], self.t_emb_dim, attention_type=attn, 
                                              num_heads=num_heads, dim_head=dim_head, dropout=dropout, attn_dropout=attn_dropout))

        #------------------------------------------------------------
        # The Middle blocks
        #------------------------------------------------------------
        self.mid_blocks = nn.ModuleList()
        for (in_idx, out_idx), attn in zip(mid_channel_indices, mid_attn_type):
            self.mid_blocks.append(MidBlock(channels[in_idx], channels[out_idx], self.t_emb_dim, attention_type=attn, 
                                            num_heads=num_heads, dim_head=dim_head, dropout=dropout, attn_dropout=attn_dropout))
            
        #------------------------------------------------------------
        # The Decoding Up blocks
        #------------------------------------------------------------
        self.up_blocks = nn.ModuleList()
        for (in_idx, out_idx), attn in zip(up_channel_indices, up_attn_type):
            self.up_blocks.append(UpBlock(channels[in_idx], channels[out_idx], self.t_emb_dim, attention_type=attn, 
                                          num_heads=num_heads, dim_head=dim_head, dropout=dropout, attn_dropout=attn_dropout))
        

    def forward(self, x, t):
        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        # Initial input image convolution
        out = self.conv_in(x)

        #------------------------------------------------------------
        # Encoder
        #
        # enc_0 = self.down_0(out, t_emb)
        # enc_1 = self.down_1(enc_0, t_emb)
        # enc_2 = self.down_2(enc_1, t_emb)
        # enc_3 = self.down_3(enc_2, t_emb)
        #------------------------------------------------------------
        tensor_in = out
        skip_connections = []
        encodings = []
        for down_block in self.down_blocks:
            out, skip_out = down_block(tensor_in, t_emb)
            encodings.append(out)
            skip_connections.append(skip_out)
            tensor_in = out

        #------------------------------------------------------------
        # the Center
        #
        # mid_out_1 = self.mid_1(enc_3, t_emb)
        #------------------------------------------------------------
        tensor_in = encodings[-1]
        mids = []
        for mid_block in self.mid_blocks:
            out = mid_block(tensor_in, t_emb)
            mids.append(out)
            tensor_in = out


        #------------------------------------------------------------
        # Decoder
        #
        # dec_2 = self.up_2(mid_out_1, enc_2, t_emb)
        # dec_1 = self.up_1(dec_2, enc_1, t_emb)
        # dec_0 = self.up_0(dec_1, enc_0, t_emb)
        #------------------------------------------------------------
        tensor_in = mids[-1]
        skip_idx = -1
        decodings = []
        for idx, up_block in enumerate(self.up_blocks):
            out = up_block(tensor_in, skip_connections[skip_idx], t_emb)
            decodings.append(out)
            tensor_in = out
            skip_idx -= 1

        # Last output layers
        out = self.norm_out(decodings[-1])  
        out = F.silu(out)
        out = self.conv_out_1(out)
        out = self.norm_out_1(out)
        out = F.silu(out)
        out = self.conv_out_2(out)
        return out 