"""
Mark Thompson.  UNet 
https://arxiv.org/pdf/1807.10165.pdf
https://arxiv.org/pdf/1912.05074v2.pdf

This UNet is modified for use as a diffusion model.  It contains the time embeddings layers.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_channels, out_channels, t_emb_dim, residual=True, numgroups=8):
        super().__init__()
        self.residual = residual
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

    def forward(self, x, t_emb):
        out = self.in_block(x)
        out = out + self.time_block(t_emb)[:, :, None, None]
        out = self.out_block(out)
        if self.residual:
            out = out + x

        return F.silu(out)


class AttentionBlock(nn.Module):
    """
    Attention block
    """
    def __init__(self, out_channels, num_heads=4, numgroups=8):
        super().__init__()
        self.attention_norms = nn.GroupNorm(numgroups, out_channels)
        self.attentions = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

    def forward(self, x):
        out = x
        # Attention block of Unet
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norms(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attentions(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        return out_attn

class DownBlock(nn.Module):
    """
    Down conv block with attention.
    Sequence of following block
    1. Residual block with time embedding x 2
    2. Attention block
    3. Downsample strided convolution
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, attention=True, num_heads=4):
        super().__init__()
        self.attention = attention
        self.residual_block_1 = ResidualBlock(in_channels, in_channels, t_emb_dim)
        self.attention_block = AttentionBlock(in_channels, num_heads=num_heads)
        self.residual_block_2 = ResidualBlock(in_channels, in_channels, t_emb_dim)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # Strided convolution to downsize
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t_emb):
        out = self.residual_block_1(x, t_emb)

        if self.attention:
            out_attn = self.attention_block(out)
            out = out + out_attn  

        out = self.residual_block_2(x, t_emb)
        out = self.conv(out)
        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    """
    Mid conv block with attention.
    Sequence of following blocks
    1. Residual block with time embedding
    2. Attention block
    3. Residual block with time embedding
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, attention=True, num_heads=4):
        super().__init__()
        self.attention = attention
        self.residual_block_1 = ResidualBlock(in_channels, in_channels, t_emb_dim)
        self.attention_block = AttentionBlock(in_channels, num_heads=num_heads)
        self.residual_block_2 = ResidualBlock(in_channels, in_channels, t_emb_dim)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t_emb):
        out = self.residual_block_1(x, t_emb) 

        if self.attention:    
            out_attn = self.attention_block(out)
            out = out + out_attn  

        out = self.residual_block_2(out, t_emb)        
        out = self.conv(out)
        return out


class UpBlock(nn.Module):
    """
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Residual block with time embedding
    3. Attention Block
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, attention=True, num_heads=4):
        super().__init__()
        self.attention = attention
        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.residual_block_1 = ResidualBlock((in_channels + out_channels), (in_channels + out_channels), t_emb_dim)
        self.attention_block = AttentionBlock((in_channels + out_channels), num_heads=num_heads)
        self.residual_block_2 = ResidualBlock((in_channels + out_channels), out_channels, t_emb_dim, residual=False)  
        # self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        out = torch.cat([x, out_down], dim=1)  # add in the skip connection from corresponding DownBlock
        out = self.residual_block_1(out, t_emb)        

        if self.attention:    
            out_attn = self.attention_block(out)
            out = out + out_attn  

        out = self.residual_block_2(out, t_emb)        
        # out = self.conv(out)
        return out

#--------------------------------------------------------------------
# The full model
#--------------------------------------------------------------------
class UNet_Diffusion(nn.Module):
    def __init__(self, t_emb_dim, attention=True):
        super(UNet_Diffusion, self).__init__()

        self.t_emb_dim = t_emb_dim
        # nb_filter = [32, 64, 128, 256, 512, 1024]
        nb_filter = [64, 128, 256, 512, 1024]
   
        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        # first conv increase filter count of the input image.
        self.conv_in = nn.Conv2d(3, nb_filter[0], kernel_size=3, padding=1)

        # last layers
        self.norm_out = nn.GroupNorm(8, nb_filter[0])
        self.up_conv_out = nn.ConvTranspose2d(nb_filter[0], nb_filter[0], kernel_size=4, stride=2, padding=1)
        self.conv_out_1 = nn.Conv2d(nb_filter[0], nb_filter[0], kernel_size=3, padding=1)
        self.conv_out_2 = nn.Conv2d(nb_filter[0], 3, kernel_size=1, padding=0)

        #------------------------------------------------------------
        # The Encoding down blocks. Input image = (size, size)
        #------------------------------------------------------------
        self.down_0 = DownBlock(nb_filter[0], nb_filter[0], self.t_emb_dim, attention=True)   # (size/2,  size/2)
        self.down_1 = DownBlock(nb_filter[0], nb_filter[1], self.t_emb_dim, attention=True)   # (size/4,  size/4)
        self.down_2 = DownBlock(nb_filter[1], nb_filter[2], self.t_emb_dim, attention=True)   # (size/8,  size/8)
        self.down_3 = DownBlock(nb_filter[2], nb_filter[3], self.t_emb_dim, attention=False)  # (size/16, size/16)

        #------------------------------------------------------------
        # The Middle blocks
        #------------------------------------------------------------
        self.mid_1 = MidBlock(nb_filter[3], nb_filter[4], self.t_emb_dim, attention=False)
        self.mid_2 = MidBlock(nb_filter[4], nb_filter[4], self.t_emb_dim, attention=False)
        self.mid_3 = MidBlock(nb_filter[4], nb_filter[3], self.t_emb_dim, attention=False)
            
        #------------------------------------------------------------
        # The Decoding Up blocks
        #------------------------------------------------------------
        self.up_2 = UpBlock(nb_filter[3], nb_filter[2], self.t_emb_dim, attention=True) 
        self.up_1 = UpBlock(nb_filter[2], nb_filter[1], self.t_emb_dim, attention=True) 
        self.up_0 = UpBlock(nb_filter[1], nb_filter[0], self.t_emb_dim, attention=True) 

        
    def forward(self, x, t):
        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        # Initial input image convolution
        out = self.conv_in(x)

        #------------------------------------------------------------
        # Encoder
        #------------------------------------------------------------
        enc_0 = self.down_0(out, t_emb)
        enc_1 = self.down_1(enc_0, t_emb)
        enc_2 = self.down_2(enc_1, t_emb)
        enc_3 = self.down_3(enc_2, t_emb)

        #------------------------------------------------------------
        # the "Center"        
        #------------------------------------------------------------
        mid_out_1 = self.mid_1(enc_3, t_emb)
        mid_out_2 = self.mid_2(mid_out_1, t_emb)
        mid_out_3 = self.mid_3(mid_out_2, t_emb)

        #------------------------------------------------------------
        # Decoder
        #------------------------------------------------------------
        dec_2 = self.up_2(mid_out_3, enc_2, t_emb)
        dec_1 = self.up_1(dec_2, enc_1, t_emb)
        dec_0 = self.up_0(dec_1, enc_0, t_emb)

        # Last output layer
        out = self.norm_out(dec_0)
        out = F.silu(out)
        out = self.up_conv_out(out)
        out = self.conv_out_1(out)
        out = self.conv_out_2(out)

        return out 

