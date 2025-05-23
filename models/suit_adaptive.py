from functools import partial
from typing import Optional

import torch
import os
import gdown

import torch.nn as nn
from torch.nn.functional import interpolate, scaled_dot_product_attention
from torch.jit import Final

from timm.models.vision_transformer import VisionTransformer, _cfg, LayerScale
from timm.models import register_model
from timm.layers import Mlp, DropPath, get_act_layer, get_norm_layer, use_fused_attn

from einops import repeat, rearrange
from torch_scatter import scatter_max, scatter_mean, scatter_sum, scatter_softmax, scatter_min
from torch_scatter.composite import scatter_std

from models.parameter_predictor import ParameterPredictor
from utils.superpixels import generate_superpixels

__all__ = [
    'suit_tiny_224_adaptive', 'suit_small_224_adaptive', 'suit_base_224_adaptive', 'suit_base_dino_adaptive'
]

# Poisitional Encoding proposed in Vaswani et al., https://arxiv.org/abs/1706.03762
class PositionalEncoding(nn.Module):
    def __init__(self, pos_dim, ch, denominator=10000.0):
        super(PositionalEncoding, self).__init__()
        assert ch % (pos_dim * 2) == 0, 'dimension of positional encoding must be equal to dim * 2.'
        enc_dim = int(ch / 2)
        div_term = torch.exp(torch.arange(0., enc_dim, 2) * -(torch.log(denominator) / enc_dim))
        freqs = torch.zeros([pos_dim, enc_dim])
        for i in range(pos_dim):
            freqs[i, : enc_dim // 2] = div_term
            freqs[i, enc_dim // 2:] = div_term
        self.freqs = freqs

    def forward(self, pos):
        # pos: (B L C), (B H W C), (B H W T C)
        pos_enc = torch.matmul(pos.float(), self.freqs.to(pos.device))
        pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)
        return pos_enc


# Fourier Features as Positional Encoding proposed in Tancik et al., https://arxiv.org/abs/2006.10739
class FourierFeatures(nn.Module):
    def __init__(self, pos_dim, ch, sigma=10, train=False):
        super(FourierFeatures, self).__init__()
        assert ch % 2 == 0, 'number of channels must be divisible by 2.'
        enc_dim = int(ch / 2)
        B = torch.randn([pos_dim, enc_dim]) * sigma
        if train:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)

    def forward(self, pos):
        # pos: (B L C), (B H W C), (B H W T C)
        pos_enc = torch.matmul(pos.float(), self.B)
        pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)
        return pos_enc


# modified from Self-attention block of ViT: timm.models.vision_transformer.Block
class EmptyMaskingBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EmptyMaskingAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_attention: bool = False) -> torch.Tensor:
        if return_attention:
            x_, attn = self.attn(self.norm1(x), mask, return_attention=return_attention)
        else:
            x_ = self.attn(self.norm1(x), mask)
        x = x + self.drop_path1(self.ls1(x_))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        if return_attention:
            return x, attn
        else:
            return x
    

# modified from original attention layer of transformers: timm.models.vision_transformer.Attention
class EmptyMaskingAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_attention: bool = False) -> torch.Tensor:
        if return_attention:
            self.fused_attn = False

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:  # no masking
            x = scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            # mask padded tokens (empty superpixel clusters)
            if mask is not None:
                # attn shape: [B, heads, N, N]
                # mask shape: [B, N, 1]
                mask = torch.where(mask, torch.tensor(0.0), torch.tensor(-float('inf')))
                attn = attn + mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x


class SuitAdaptive(VisionTransformer):
    def __init__(self, *args, **kwargs):
        self.pe_type = kwargs.get('pe_type', 'ff')
        self.pe_injection = kwargs.get('pe_injection', 'concat')
        self.downsample = kwargs.get('downsample', 2)
        self.aggregate = kwargs.get('aggregate', ['max', 'avg'])
        self.base_dim = kwargs.get('base_dim', 96)
        self.embed_dim = kwargs.get('embed_dim', 384)
        self.use_proj = kwargs.get('use_proj', True)
        # Instantiate ParameterPredictor
        self.parameter_predictor = ParameterPredictor()
        # filter keywords
        # Remove n_spix_segments, compactness if they were part of kwargs, as they are now predicted.
        # Assuming they were not explicit args to SuperpixelVisionTransformer's __init__ but rather general **kwargs
        # or handled by a config object not directly visible here.
        # If they were specific named arguments, they would need to be explicitly removed from the method signature
        # and from being passed to super().__init__ if VisionTransformer doesn't expect them.
        suit_only_keywords = ['pe_type', 'pe_injection', 'downsample', 'aggregate', 'base_dim', 'use_proj']
        # Add any superpixel specific args that are now redundant to keywords_to_filter
        # e.g., 'n_spix_segments', 'compactness' if they were passed in kwargs
        old_timm_keywords = ['pretrained_cfg', 'pretrained_cfg_overlay', 'cache_dir']
        keywords_to_filter = suit_only_keywords + old_timm_keywords
        for k in keywords_to_filter:
            if k in kwargs:
                kwargs.pop(k)
        super().__init__(*args, **kwargs)
        
        self.img_size = kwargs.get('img_size', 224)
        self.make_coords(self.img_size, self.downsample)

        token_dim = self.base_dim * len(self.aggregate)
        if self.pe_injection == 'concat':
            token_dim = token_dim * 2

        if self.use_proj:
            assert self.embed_dim % len(self.aggregate) == 0, 'embed dim must be divisible by number of aggregation methods.'
            if self.pe_injection == 'concat':
                self.projection = nn.Conv2d(self.base_dim * 2, int(self.embed_dim / len(self.aggregate)), 1)
            else:
                self.projection = nn.Conv2d(self.base_dim, int(self.embed_dim / len(self.aggregate)), 1)

        self.get_feats = nn.Sequential(
            nn.Conv2d(3, self.base_dim, 7, self.downsample, padding=3, padding_mode='replicate'),
            nn.BatchNorm2d(self.base_dim),
            nn.GELU(),
        )
        if self.pe_type == 'ff':
            self.pe = FourierFeatures(2, self.base_dim, train=True)
        else:
            self.pe = PositionalEncoding(2, self.base_dim)

        # re-init transformer blocks capable of masking padded tokens (empty superpixel clusters)
        norm_layer = get_norm_layer(kwargs.get('norm_layer', None)) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(kwargs.get('act_layer', None)) or nn.GELU
        mlp_layer = kwargs.get('mlp_layer', Mlp)
        dpr = [x.item() for x in torch.linspace(0, kwargs.get('drop_path_rate', 0.), kwargs.get('depth', 12))]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            EmptyMaskingBlock(
                dim=kwargs.get('embed_dim', 384),# embed_dim,
                num_heads=kwargs.get('num_heads', 6), # num_heads,
                mlp_ratio=kwargs.get('mlp_ratio', 4.), # mlp_ratio,
                qkv_bias=kwargs.get('qkv_bias', True), # qkv_bias,
                qk_norm=kwargs.get('qk_norm', False), # qk_norm,
                init_values=kwargs.get('init_values', None), # init_values,
                proj_drop=kwargs.get('proj_drop_rate', 0.), # proj_drop_rate,
                attn_drop=kwargs.get('attn_drop_rate', 0.), # attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(kwargs.get('depth', 12))])
        
        if kwargs.get('weight_init', '') != 'skip':
            self.init_weights(kwargs.get('weight_init', ''))
        if kwargs.get('fix_init', False):
            self.fix_init_weight()

        # remove redundant components from original ViT
        del self.patch_embed, self.pos_embed
        
    def tokenization(self, x, spix_label=None):
        b, c, h, w = x.shape
        spix_label = spix_label.long()
        
        # NOTE: Superpixel label starts from 0.
        n_tokens = int(spix_label.max()) + 1 # spix_label: [b,1,h,w] 
        spix_label = spix_label.view(b, -1)  # Flatten the labels to shape (b, h * w)
        x = x.view(b, c, -1)  # Flatten the features to shape (b, c, h * w)
        # Expand labels to shape (b, c, h * w) to match features' shape
        labels_expanded = spix_label.unsqueeze(1).expand(-1, c, -1) # Shape: (b, c, h * w)
        # Flatten the batch and channel dimensions to shape (b * c, h * w)
        labels_expanded = labels_expanded.reshape(-1, h * w).long()
        x_flat  = x.reshape(-1, h * w)
        x_flat = x_flat.float()

        out_list = []
        # NOTE: MAX
        if 'max' in self.aggregate:
            # Perform scatter_max
            max_out, _ = scatter_max(x_flat, labels_expanded, dim=1, dim_size=n_tokens)
            max_out = max_out.view(b, c, n_tokens).permute(0, 2, 1)  # Shape: (b, n_tokens, c)
            out_list.append(max_out)

        # NOTE: MIN
        if 'min' in self.aggregate:
            # Perform scatter_min
            min_out, _ = scatter_min(x_flat, labels_expanded, dim=1, dim_size=n_tokens)
            min_out = min_out.view(b, c, n_tokens).permute(0, 2, 1)  # Shape: (b, n_tokens, c)
            out_list.append(min_out)
            
        # NOTE: AVG
        if 'avg' in self.aggregate:
            # Perform scatter_mean
            mean_out = scatter_mean(x_flat, labels_expanded, dim=1, dim_size=n_tokens)
            mean_out = mean_out.view(b, c, n_tokens).permute(0, 2, 1)  # Shape: (b, n_tokens, c)
            out_list.append(mean_out)

        # NOTE: STD
        if 'std' in self.aggregate:
            # Perform scatter_max
            std_out = scatter_std(x_flat, labels_expanded, dim=1, dim_size=n_tokens)
            std_out = std_out.view(b, c, n_tokens).permute(0, 2, 1)  # Shape: (b, n_tokens, c)
            out_list.append(std_out)

        # NOTE: SOFTMAX
        if 'softmax' in self.aggregate:
            # Perform scatter_softmax
            softmax_weights = scatter_softmax(x_flat, labels_expanded, dim=1, dim_size=n_tokens)
            # Weighted features
            weighted_softmax_features = x_flat * softmax_weights
            # Perform scatter_sum to aggregate the weighted features
            softmax_out = scatter_sum(weighted_softmax_features, labels_expanded, dim=1, dim_size=n_tokens)
            softmax_out = softmax_out.view(b, c, n_tokens).permute(0, 2, 1)  # Shape: (b, n_tokens, c)
            out_list.append(softmax_out)
        
        # NOTE: mask padded tokens (empty superpixel clusters)
        ones = torch.ones(b, 1, h, w, device=x.device, requires_grad=False).reshape(-1, h * w).int()
        spix_labels_flat = spix_label.reshape(-1, h * w).long()
        cluster_counter = scatter_sum(ones, spix_labels_flat, dim=1, dim_size=n_tokens).view(b, 1, n_tokens).permute(0, 2, 1)
        mask = (cluster_counter != 0)  # mask empty tokens (B, N, 1)
        mask = rearrange(mask, 'b n 1 -> b 1 1 n')

        # Concatenate all selected pooled features
        tokens = torch.cat(out_list, dim=2)  # Concatenate along the channel dimension

        return tokens, mask

    def prepare_tokens(self, x: torch.Tensor, spix_label: torch.Tensor) -> torch.Tensor:
        x = self.get_feats(x) # x: [b,3,h,w] -> x: [b,embed_dim,h,w]
        b, _, _, _ = x.shape

        pe = self.pe(self.coords).permute(0, 3, 2, 1) # [1,w,h,embed_dim] -> [1,embed_dim,h,w]
        pe = repeat(pe, '1 c h w -> b c h w', b=x.shape[0])
        x = torch.cat([x, pe], dim=1) if self.pe_injection == 'concat' else x + pe
        if self.use_proj:
            x = self.projection(x)
        
        # tokenization with superpixel labels
        spix_label = interpolate(spix_label.to(torch.float), size=x.shape[-2:], mode='nearest')
        tokens, mask = self.tokenization(x, spix_label)
        cls_tokens = repeat(self.cls_token, '1 1 c -> b 1 c', b=b) # cls_tokens: [b,1,embed_dim] 
        x = torch.cat((cls_tokens, tokens), dim=1)

        cls_mask = torch.ones(b, 1, 1, 1, dtype=torch.bool, device=x.device, requires_grad=False)
        mask = torch.cat((cls_mask, mask), dim=-1)
        x = self.pos_drop(x)
        return x, mask

    def forward_features(self, x: torch.Tensor, spix_label: torch.Tensor) -> torch.Tensor:
        x, mask = self.prepare_tokens(x, spix_label) # patch_embed and _pos_embed in once
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict superpixel parameters
        params = self.parameter_predictor(x)
        K_pred = params["num_superpixels"]
        m_pred = params["compactness"]

        # Normalize and clip parameters
        # K should be integer for slic, generate_superpixels handles .item() and int()
        K = torch.clamp(K_pred, min=50, max=1000) 
        m = torch.clamp(m_pred, min=1, max=40)

        # Generate superpixels
        # generate_superpixels expects K and m to be tensors of shape [B]
        # If K_pred, m_pred are [B, 1], squeeze them. If they are already [B], it's fine.
        # Assuming ParameterPredictor outputs [B, 1] or [B] for each.
        if K.dim() > 1 and K.shape[1] == 1:
            K = K.squeeze(1)
        if m.dim() > 1 and m.shape[1] == 1:
            m = m.squeeze(1)
            
        spix_label = generate_superpixels(x, K, m)
        # Add channel dimension to spix_label to match expected shape [B, 1, H, W] in forward_features
        spix_label = spix_label.unsqueeze(1)

        x = self.forward_features(x, spix_label)
        x = self.forward_head(x)
        return x
    
    def make_coords(self, img_size=None, downsample=None):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if img_size is None:
            img_size = self.img_size
        if downsample is None:
            downsample = self.downsample

        if isinstance(img_size, tuple):
            # Coordinates for Positional Encoding
            x_coord, y_coord = torch.arange(0, int(img_size[1]), device=device), torch.arange(0, int(img_size[0]), device=device) # x_coord: [w], y_coord: [h]
            x_coord, y_coord = x_coord / (img_size[1]-1), y_coord / (img_size[0]-1)
            self.coords = torch.cartesian_prod(x_coord, y_coord).reshape(1, img_size[1], img_size[0], 2) # [w*h,2] -> [1,w,h,2]
        else:
            img_size = int(img_size / downsample)
            # Coordinates for Positional Encoding
            x_coord, y_coord = torch.arange(0, img_size, device=device), torch.arange(0, img_size, device=device) # x_coord: [w], y_coord: [h]
            x_coord, y_coord = x_coord / (img_size-1), y_coord / (img_size-1)
            self.coords = torch.cartesian_prod(x_coord, y_coord).reshape(1, img_size, img_size, 2) # [w*h,2] -> [1,w,h,2]

    def reset_stride(self, new_stride, reset_coords=True):
        self.get_feats[0].stride = new_stride
        self.downsample = new_stride
        if reset_coords:
            self.make_coords(downsample=new_stride)

    def reset_img_size(self, img_size, reset_coords=True):
        self.img_size = img_size
        if reset_coords:
            self.make_coords(img_size=img_size)
    
    def get_last_selfattention(self, x: torch.Tensor) -> torch.Tensor:
        # Predict superpixel parameters
        params = self.parameter_predictor(x)
        K_pred = params["num_superpixels"]
        m_pred = params["compactness"]
        K = torch.clamp(K_pred, min=50, max=1000)
        m = torch.clamp(m_pred, min=1, max=40)
        if K.dim() > 1 and K.shape[1] == 1: K = K.squeeze(1)
        if m.dim() > 1 and m.shape[1] == 1: m = m.squeeze(1)
        spix_label = generate_superpixels(x, K, m).unsqueeze(1)

        x, mask = self.prepare_tokens(x, spix_label) # patch_embed and _pos_embed in once
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, mask)
            else:
                _, attn = blk(x, mask, return_attention=True)
        
        return attn

    def get_selfattentions(self, x: torch.Tensor) -> torch.Tensor:
        # Predict superpixel parameters
        params = self.parameter_predictor(x)
        K_pred = params["num_superpixels"]
        m_pred = params["compactness"]
        K = torch.clamp(K_pred, min=50, max=1000)
        m = torch.clamp(m_pred, min=1, max=40)
        if K.dim() > 1 and K.shape[1] == 1: K = K.squeeze(1)
        if m.dim() > 1 and m.shape[1] == 1: m = m.squeeze(1)
        spix_label = generate_superpixels(x, K, m).unsqueeze(1)

        x, mask = self.prepare_tokens(x, spix_label) # patch_embed and _pos_embed in once
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        attns = []
        for blk in self.blocks:
            x, attn = blk(x, mask, return_attention=True)
            attns.append(attn)

        attns = torch.stack(attns)
        return attns
    
    def get_intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
        # Predict superpixel parameters
        params = self.parameter_predictor(x)
        K_pred = params["num_superpixels"]
        m_pred = params["compactness"]
        K = torch.clamp(K_pred, min=50, max=1000)
        m = torch.clamp(m_pred, min=1, max=40)
        if K.dim() > 1 and K.shape[1] == 1: K = K.squeeze(1)
        if m.dim() > 1 and m.shape[1] == 1: m = m.squeeze(1)
        spix_label = generate_superpixels(x, K, m).unsqueeze(1)
        
        features = []
        x, mask = self.prepare_tokens(x, spix_label) # patch_embed and _pos_embed in once
        features.append(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        for blk in self.blocks:
            x = blk(x, mask)
            features.append(x)
                
        features = torch.stack(features)
        return features


@register_model
def suit_tiny_224_adaptive(pretrained=False, **kwargs):
    model = SuitAdaptive(
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, base_dim=48,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # Pretrained weights are not available for adaptive models yet.
    if pretrained:
        print("Warning: Pretrained weights are not available for suit_tiny_224_adaptive. Initializing from scratch.")
    return model

@register_model
def suit_small_224_adaptive(pretrained=False, **kwargs):
    model = SuitAdaptive(
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, base_dim=96,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        print("Warning: Pretrained weights are not available for suit_small_224_adaptive. Initializing from scratch.")
    return model

@register_model
def suit_base_224_adaptive(pretrained=False, **kwargs):
    model = SuitAdaptive(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, base_dim=192,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        print("Warning: Pretrained weights are not available for suit_base_224_adaptive. Initializing from scratch.")
    return model

@register_model
def suit_base_dino_adaptive(pretrained=False, **kwargs):
    model = SuitAdaptive(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, base_dim=192,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        print("Warning: Pretrained weights are not available for suit_base_dino_adaptive. Initializing from scratch.")
    return model