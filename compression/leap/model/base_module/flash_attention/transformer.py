'''Copy from https://github.com/zhaoyue-zephyrus/avion/blob/main/avion/models/transformer.py'''

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torch.cuda.amp import autocast

from timm.models.layers import trunc_normal_
from timm.models.layers import to_2tuple
from timm.models.layers import DropPath
from timm.models.vision_transformer import Attention

from flash_attn.modules.mha import MHA as FlashMHA
from flash_attn.modules.mlp import Mlp as FlashMlp



class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerScaleFP32(nn.Module):
    '''Scale layer handling fp16'''
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim).half())

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class FlashSelfAttnLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNormFp32,
            norm_first: bool = False,
    ):
        super().__init__()

        self.norm_first = norm_first

        self.ln_1 = norm_layer(d_model)
        self.attn = FlashMHA(d_model, n_head, cross_attn=False, fused_bias_fc=False, dropout=attn_drop, use_flash_attn=True)
        self.ls_1 = LayerScaleFP32(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = FlashMlp(d_model, hidden_features=mlp_width, activation=act_layer())
        self.ls_2 = LayerScaleFP32(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor):
        with autocast():
            if self.norm_first:
                x = x + self.drop_path(self.ls_1(self.attn(self.ln_1(x))))
                x = x + self.drop_path(self.ls_2(self.mlp(self.ln_2(x))))
            else:
                x = self.ln_1(x + self.drop_path(self.ls_1(self.attn(x))))
                x = self.ln_2(x + self.drop_path(self.ls_2(self.mlp(x))))
            return x
    

class FlashCrossAttnLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNormFp32,
            norm_first: bool = False,
    ):
        super().__init__()
        self.norm_first = norm_first

        self.ln_1 = norm_layer(d_model)
        self.attn = FlashMHA(d_model, n_head, cross_attn=True, fused_bias_fc=False, dropout=attn_drop, use_flash_attn=True)
        self.ls_1 = LayerScaleFP32(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = FlashMlp(d_model, hidden_features=mlp_width, activation=act_layer())
        self.ls_2 = LayerScaleFP32(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, memory: torch.Tensor):
        with autocast():
            if self.norm_first:
                x = x + self.drop_path(self.ls_1(self.attn(self.ln_1(x), x_kv=memory)))
                x = x + self.drop_path(self.ls_2(self.mlp(self.ln_2(x))))
            else:
                x = self.ln_1(x + self.drop_path(self.ls_1(self.attn(x, x_kv=memory))))
                x = self.ln_2(x + self.drop_path(self.ls_2(self.mlp(x))))
            return x
    

class FlashTxDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNormFp32,
            norm_first: bool = False,
    ):
        super().__init__()
        self.norm_first = norm_first

        self.ln_1 = norm_layer(d_model)
        self.self_attn = FlashMHA(d_model, n_head, cross_attn=False, fused_bias_fc=False, dropout=attn_drop, use_flash_attn=True)
        self.ls_1 = LayerScaleFP32(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        self.cross_attn = FlashMHA(d_model, n_head, cross_attn=True, fused_bias_fc=False, dropout=attn_drop, use_flash_attn=True)
        self.ls_2 = LayerScaleFP32(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_3 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = FlashMlp(d_model, hidden_features=mlp_width, activation=act_layer())
        self.ls_3 = LayerScaleFP32(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, memory: torch.Tensor):
        with autocast():
            if self.norm_first:
                x = x + self.drop_path(self.ls_1(self.self_attn(self.ln_1(x))))
                x = x + self.drop_path(self.ls_2(self.cross_attn(self.ln_2(x), x_kv=memory)))
                x = x + self.drop_path(self.ls_3(self.mlp(self.ln_3(x))))
            else:
                x = self.ln_1(x + self.drop_path(self.ls_1(self.self_attn(x))))
                x = self.ln_2(x + self.drop_path(self.ls_2(self.cross_attn(x, x_kv=memory))))
                x = self.ln_3(x + self.drop_path(self.ls_3(self.mlp(x))))
            return x