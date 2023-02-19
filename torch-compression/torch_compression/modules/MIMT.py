import cv2
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from torch_compression.modules.layers import Mlp, PatchEmbed, PatchUnEmbed, window_partition, window_reverse, WindowAttention, SwinTransformerBlock
from timm.models.layers import to_2tuple, DropPath, trunc_normal_


class CrossWindowAttention(WindowAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        delattr(self, 'qkv')
        self.q = nn.Linear(kwargs['dim'], kwargs['dim'], bias=kwargs['qkv_bias'])
        self.kv = nn.Linear(kwargs['dim'], kwargs['dim']*2, bias=kwargs['qkv_bias'])

    def forward(self, x, ref, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            ref: reference features with shape of (num_windows*B, cN, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        num_ref = len(ref)
        rB_, rN, rC = ref[0].shape
        assert B_ == rB_ and N == rN and C == rC, ValueError

        
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = [self.kv(r).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) for r in ref]
        
        q = q[0]
        k = [_kv[0] for _kv in kv]
        v = [_kv[1] for _kv in kv]

        q = q * self.scale
        
        attns = [(q @ _k.transpose(-2, -1)) for _k in k]

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attns = [attn + relative_position_bias.unsqueeze(0) for attn in attns]

        if mask is not None:
            nW = mask.shape[0]
            attns = [attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0) for attn in attns]
            attn = torch.cat([attn.view(-1, self.num_heads, N, N) for attn in attns], dim=3)
            attn = self.softmax(attn)
        else:
            attn = torch.cat(attns, dim=3)
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        
        v = torch.cat(v, dim=2)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossSwinTransformerBlock(SwinTransformerBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        delattr(self, 'attn')
        self.attn = CrossWindowAttention(
            dim=kwargs['dim'], window_size=to_2tuple(self.window_size), num_heads=kwargs['num_heads'],
            qkv_bias=kwargs['qkv_bias'], qk_scale=kwargs['qk_scale'], attn_drop=kwargs['attn_drop'], proj_drop=kwargs['drop'])


    def forward(self, x, ref, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        assert isinstance(ref, tuple)
        num_ref = len(ref)

        shortcut = x
        x = self.norm1(x)

        x = x.view(B, H, W, C)
        ref = [r.view(B, H, W, C) for r in ref]

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_ref = [torch.roll(r, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) for r in ref]
        else:
            shifted_x = x
            shifted_ref = ref

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        ref_windows = [window_partition(r, self.window_size) for r in shifted_ref]

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        ref_windows = [ref_window.view(-1, self.window_size * self.window_size, C) for ref_window in ref_windows]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, ref_windows, mask=self.cal_attn_mask(x_size, x.device))  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinTransformerDecoderBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.s_attn = SwinTransformerBlock(*args, **kwargs)
        self.c_attn = CrossSwinTransformerBlock(*args, **kwargs)
    
    def forward(self, x, ref, x_size):
        x = self.s_attn(x, x_size)
        x = self.c_attn(x, ref, x_size)
        return x


#class CrossWindowAttention(nn.Module):
#    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
#
#        super().__init__()
#        self.dim = dim
#        self.window_size = window_size  # Wh, Ww
#        self.num_heads = num_heads
#        head_dim = dim // num_heads
#        self.scale = qk_scale or head_dim ** -0.5
#
#        # define a parameter table of relative position bias
#        self.relative_position_bias_table = nn.Parameter(
#            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
#
#        # get pair-wise relative position index for each token inside the window
#        coords_h = torch.arange(self.window_size[0])
#        coords_w = torch.arange(self.window_size[1])
#        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#        relative_coords[:, :, 1] += self.window_size[1] - 1
#        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#        self.register_buffer("relative_position_index", relative_position_index)
#
#        self.q = nn.Linear(dim, dim, bias=qkv_bias)
#        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
#        self.attn_drop = nn.Dropout(attn_drop)
#        self.proj = nn.Linear(dim, dim)
#
#        self.proj_drop = nn.Dropout(proj_drop)
#
#        trunc_normal_(self.relative_position_bias_table, std=.02)
#        self.softmax = nn.Softmax(dim=-1)
#
#    def forward(self, current_term, cross_term, mask=None):
#        """
#        Args:
#            x: input features with shape of (num_windows*B, N, C)
#            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#        """
#        B_, N, C = current_term.shape
#        cB_, cN, cC = cross_term.shape
#        cross_scale = cN // N
#        
#        q = self.q(current_term).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#        kv = self.kv(cross_term).reshape(cB_, cN, 2, self.num_heads, cC // self.num_heads).permute(2, 0, 3, 1, 4)
#        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
#
#        q = q * self.scale
#        
#        attn = (q @ k.transpose(-2, -1))    #   B_, heads, N, cN
#
#        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#        attn = attn + torch.cat([relative_position_bias for _ in range(cross_scale)], dim=2).unsqueeze(0)
#
#        if mask is not None:
#            nW = mask.shape[0]
#            attn = attn.view(B_ // nW, nW, self.num_heads, N, cN)
#            for i in range(cross_scale):
#                attn[:, :, :, :, i*N:(i+1)*N] += mask.unsqueeze(1).unsqueeze(0)
#            attn = attn.view(-1, self.num_heads, N, cN)
#            attn = self.softmax(attn)
#        else:
#            attn = self.softmax(attn)
#        
#        attn = self.attn_drop(attn)
#
#        current_term = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#        current_term = self.proj(current_term)
#        current_term = self.proj_drop(current_term)
#        return current_term
#
#    def extra_repr(self) -> str:
#        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
#
#    def flops(self, N):
#        # calculate flops for 1 window with token length of N
#        flops = 0
#        # qkv = self.qkv(x)
#        flops += N * self.dim * 3 * self.dim
#        # attn = (q @ k.transpose(-2, -1))
#        flops += self.num_heads * N * (self.dim // self.num_heads) * N
#        #  x = (attn @ v)
#        flops += self.num_heads * N * N * (self.dim // self.num_heads)
#        # x = self.proj(x)
#        flops += N * self.dim * self.dim
#        return flops
#        
#    
#class CrossSwinTransformerBlock(nn.Module):
#    r""" Swin Transformer Block.
#    Args:
#        dim (int): Number of input channels.
#        input_resolution (tuple[int]): Input resulotion.
#        num_heads (int): Number of attention heads.
#        window_size (int): Window size.
#        shift_size (int): Shift size for SW-MSA.
#        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#        drop (float, optional): Dropout rate. Default: 0.0
#        attn_drop (float, optional): Attention dropout rate. Default: 0.0
#        drop_path (float, optional): Stochastic depth rate. Default: 0.0
#        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#    """
#
#    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
#                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
#                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#        super().__init__()
#        self.dim = dim
#        self.input_resolution = input_resolution
#        self.num_heads = num_heads
#        self.window_size = window_size
#        self.shift_size = shift_size
#        self.mlp_ratio = mlp_ratio
#        if min(self.input_resolution) <= self.window_size:
#            # if window size is larger than input resolution, we don't partition windows
#            self.shift_size = 0
#            self.window_size = min(self.input_resolution)
#        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
#
#        self.norm1 = norm_layer(dim) if not isinstance(norm_layer, nn.Identity) else norm_layer()
#        self.s_attn = WindowAttention(
#            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
#            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
#        )
#        self.c_attn = CrossWindowAttention(
#            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
#            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
#        )
#
#        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#        self.norm2 = norm_layer(dim) if not isinstance(norm_layer, nn.Identity) else norm_layer()
#        self.norm3 = norm_layer(dim) if not isinstance(norm_layer, nn.Identity) else norm_layer()
#        mlp_hidden_dim = int(dim * mlp_ratio)
#        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
#
#        if self.shift_size > 0:
#            attn_mask = self.calculate_mask(self.input_resolution)
#        else:
#            attn_mask = None
#
#        self.register_buffer("attn_mask", attn_mask)
#
#    def calculate_mask(self, x_size):
#        # calculate attention mask for SW-MSA
#        H, W = x_size
#        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
#        h_slices = (slice(0, -self.window_size),
#                    slice(-self.window_size, -self.shift_size),
#                    slice(-self.shift_size, None))
#        w_slices = (slice(0, -self.window_size),
#                    slice(-self.window_size, -self.shift_size),
#                    slice(-self.shift_size, None))
#        cnt = 0
#        for h in h_slices:
#            for w in w_slices:
#                img_mask[:, h, w, :] = cnt
#                cnt += 1
#
#        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
#        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
#        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#
#        return attn_mask
#
#    def forward(self, current, cross, cur_size, cross_size):
#        H, W = cur_size
#        c_H, c_W = cross_size
#        B, L, C = current.shape
#        cross_scale = c_W // W
#        # assert L == H * W, "input feature has wrong size"
#
#        shortcut = current
#        current = self.norm1(current)
#        current = current.view(B, H, W, C)
#
#        # cyclic shift
#        if self.shift_size > 0:
#            shifted_x = torch.roll(current, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#        else:
#            shifted_x = current
#
#        # partition windows
#        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
#        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
#        
#        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size        
#        if self.shift_size > 0:
#            if self.input_resolution == cur_size:
#                # print("OLD MASK")
#                mask = self.attn_mask  
#            else:
#                # print("NEW MASK")
#                self.input_resolution = cur_size
#                mask = self.calculate_mask(cur_size).to(current.device)
#                self.attn_mask = mask
#        else:
#            mask = None
#        attn_windows = self.s_attn(x_windows, mask=mask)
#
#        # merge windows
#        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
#        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
#
#        # reverse cyclic shift
#        if self.shift_size > 0:
#            current = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#        else:
#            current = shifted_x
#        current = current.view(B, H * W, C)
#
#        # FFN
#        current = shortcut + self.drop_path(current)
#        shortcut = current
#        current = self.norm2(current)
#        current = current.view(B, H, W, C)
#        cross = cross.permute(2, 1, 0).reshape(-1, H*W, B).permute(2, 1, 0)
#        cross = cross.view(B, H, W, -1)
#        
#        # cyclic shift
#        if self.shift_size > 0:
#            shifted_current = torch.roll(current, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#            shifted_cross = torch.roll(cross, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#        else:
#            shifted_current = current
#            shifted_cross = cross
#
#        # partition windows
#        current_windows = window_partition(shifted_current, self.window_size)  # nW*B, window_size, window_size, C
#        current_windows = current_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
#        cross_windows = window_partition(shifted_cross, self.window_size)  # nW*B, window_size, window_size, C
#        cross_windows = cross_windows.view(-1, self.window_size * self.window_size, cross_scale*C)  # nW*B, window_size*window_size, C
#        cross_windows = cross_windows.permute(2, 1, 0).reshape(C, cross_scale*self.window_size*self.window_size, -1).permute(2, 1, 0)
#        
#        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
#        if self.shift_size > 0:
#            if self.input_resolution == cross_size:
#                # print("OLD MASK")
#                mask = self.attn_mask  
#            else:
#                # print("NEW MASK")
#                self.input_resolution = cross_size
#                mask = self.calculate_mask(cur_size).to(current.device)
#                self.attn_mask = mask
#        else:
#            mask = None
#        attn_windows = self.c_attn(current_windows, cross_windows, mask=mask)
#        
#        # merge windows
#        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
#        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
#
#        # reverse cyclic shift
#        if self.shift_size > 0:
#            current = torch.roll(shifted_current, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#        else:
#            current = shifted_current
#        current = current.view(B, H * W, C)
#        
#        current = shortcut + self.drop_path(current)
#        current = current + self.drop_path(self.mlp(self.norm3(current)))
#        
#        return current
#
#    def extra_repr(self) -> str:
#        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
#               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
#
#    def flops(self):
#        flops = 0
#        H, W = self.input_resolution
#        # norm1
#        flops += self.dim * H * W
#        # W-MSA/SW-MSA
#        nW = H * W / self.window_size / self.window_size
#        flops += nW * self.s_attn.flops(self.window_size * self.window_size)
#        # mlp
#        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
#        # norm2
#        flops += self.dim * H * W
#        return flops    
    

class MIMTBasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False, shift_first=False, l_dim=None,
                 Decoder=False):

        super().__init__()
        self.dim = dim
        self.l_dim = l_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.decoder = Decoder
        self.window_size = window_size

        remainder = 1 if shift_first else 0

        self.pos_embed = nn.Parameter(torch.zeros(1, window_size, window_size, dim))
        trunc_normal_(self.pos_embed, std=.02)

        if l_dim is not None:
            self.linear = nn.Linear(in_features=dim, out_features=l_dim, bias=True)
            _dim = l_dim
        else:
            _dim = dim

        # build blocks
        self.blocks = []
        if Decoder:
            building_block = SwinTransformerDecoderBlock
        else:
            building_block = SwinTransformerBlock

        for i in range(depth):
            self.blocks.append(building_block(
                                  dim=_dim,
                                  num_heads=num_heads, window_size=window_size,
                                  shift_size=0 if (i % 2 == remainder) else window_size // 2,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop, attn_drop=attn_drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                  norm_layer=norm_layer)
                              )

        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x, x_size, y_m=None):
        H, W = x_size
        B, L, C = x.shape

        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size) + self.pos_embed
        x = window_reverse(x, self.window_size, H, W).view(B, L, C)

        if self.l_dim is not None:
            x = self.linear(x)
        
        if self.decoder:
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x, y_m)
                else:
                    x = blk(x, y_m, x_size)
            return x
        else:
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x, x_size) 
            return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops
    
    
class MIMTEncoder(nn.Module):
    def __init__(self, input_dim, l_dim, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, 
                 use_checkpoint=False, shift_first=False, num_sep=2, num_joint=4):
        r'''  MIMT Entropy Model
        Args:
            dim (int): Number of input channels.
            depth (int): Number of blocks.
            num_heads (int): Number of attention heads.
            window_size (int): Local window size.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
            use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
            shift_first (bool): Whether to use shift transformer block at first(odd) time. Default: False
            num_sep (int): Number of transformer blocks for separate encoder. Default: 2
            num_joint (int): Number of transformer blocks for joint encoder. Default: 4
        '''
        super().__init__()
        self.Sep_Encoder = MIMTBasicLayer(dim=input_dim,
                                      l_dim=l_dim,
                                      depth=num_sep,
                                      num_heads=num_heads,
                                      window_size=window_size,
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path,
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint, 
                                      shift_first=shift_first
                                     )

        self.Joint_Encoder = MIMTBasicLayer(dim=l_dim,
                                            depth=num_joint,
                                            num_heads=num_heads,
                                            window_size=window_size,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop, attn_drop=attn_drop,
                                            drop_path=drop_path,
                                            norm_layer=norm_layer,
                                            use_checkpoint=use_checkpoint, 
                                            shift_first=shift_first,
                                           )
        
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed()

    def forward(self, priors):
        prior_size = priors[0].shape[2:]
        num_priors = len(priors)
        
        priors = [self.patch_embed(p) for p in priors]
        joint = [self.Sep_Encoder(p, prior_size) for p in priors]
        
        joint = torch.cat(joint, dim=1)
        joint_size = (prior_size[0], prior_size[1]*2)

        joint = self.Joint_Encoder(joint, joint_size)

        joint = joint.chunk(num_priors, dim=1)
        
        return joint
    

class MIMTDecoder(nn.Module):
    def __init__(self, input_dim, l_dim, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 use_checkpoint=False, shift_first=False, num_dec=2):
        r'''  MIMT Entropy Model
        Args:
            dim (int): Number of input channels.
            depth (int): Number of blocks.
            num_heads (int): Number of attention heads.
            window_size (int): Local window size.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
            use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
            shift_first (bool): Whether to use shift transformer block at first(odd) time. Default: False
            num_dec (int): Number of transformer blocks for decoder. Default: 4
        '''
        super().__init__()
        self.Decoder = MIMTBasicLayer(dim=input_dim,
                                  l_dim=l_dim,
                                  depth=num_dec,
                                  num_heads=num_heads,
                                  window_size=window_size,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop, attn_drop=attn_drop,
                                  drop_path=drop_path,
                                  norm_layer=norm_layer,
                                  use_checkpoint=use_checkpoint, 
                                  shift_first=shift_first, 
                                  Decoder=True,
                                 )
        self.conv = nn.Conv2d(in_channels=l_dim, out_channels=input_dim*2, kernel_size=1, stride=1, padding=0)
        
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed()

        self.latent_embed = nn.Parameter(torch.zeros(1, 1, input_dim))
        trunc_normal_(self.latent_embed, std=.02)

        self.patch_unembed = PatchUnEmbed()
        
    def forward(self, joint, y, mask):
        y_size = y.shape[2:]
        mask = self.patch_embed(mask)
        y = self.patch_embed(y)

        y_m = y + self.latent_embed * (1. - mask) # Replace to-be-predicted token with latent_embed
        
        result = self.Decoder(y_m, y_size, joint)
        result = self.conv(self.patch_unembed(result, y_size))
        
        return result
