from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed

from models.st_mem_reference import encoder

class ImageEncoder(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768, 
                 depth=12, 
                 num_heads=16,
                 mlp_ratio=4., 
                 norm_layer=nn.LayerNorm,
                 num_classes=5):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 기존 코드 patch+1 -> patch (cls token 사용 X)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_norm=False, norm_layer=norm_layer)
                                     for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        if num_classes == 0:
            self.head = nn.Identity()
        else:
            self.head = nn.Linear(embed_dim, num_classes)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x, mask_ratio=0.5):
        
        x = self.patch_embed(x)
        
        x = x + self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
            
        x = x.mean(dim=1)
        outcome = self.norm(x)
        outcome = self.head(outcome)
            
        return outcome

class FusionConcat_con(nn.Module):
    def __init__(self, 
                 signal_path, 
                 image_path, 
                 dropout=0.5, 
                 num_classes=5):
        super().__init__()
        
        self.sig_encoder = encoder.__dict__['st_mem_vit_base'](num_leads=12,
                                                               seq_len=2250,
                                                               patch_size=75,
                                                               num_classes=768)
        signal_ckpt = torch.load(signal_path, map_location='cpu')
        signal_state_dict = signal_ckpt['model'] if 'model' in signal_ckpt else signal_ckpt
    
        missing, unexpected = self.sig_encoder.load_state_dict(signal_state_dict, strict=False)
        print(f"[Signal] Missing keys: {missing}")
        print(f"[Signal] Unexpected keys: {unexpected}")
    
        self.img_encoder = ImageEncoder(img_size=224,
                                        patch_size=16,
                                        in_chans=3,
                                        embed_dim=768,
                                        depth=12,
                                        num_heads=12,
                                        mlp_ratio=4,
                                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                        num_classes=768)
        image_ckpt = torch.load(image_path, map_location='cpu')
        image_state_dict = image_ckpt['model'] if 'model' in image_ckpt else image_ckpt
        
        missing, unexpected = self.img_encoder.load_state_dict(image_state_dict, strict=False)
        print(f"[Image] Missing keys: {missing}")
        print(f"[Image] Unexpected keys: {unexpected}")

        self.dropout = nn.Dropout(dropout)
        self.head_for_cls = nn.Linear(768*2, num_classes)

    def forward(self, img, sig):

        latent_img = self.img_encoder(img)
        latent_sig = self.sig_encoder(sig)

        outcome = torch.cat([latent_sig, latent_img], dim=1)
        outcome = self.dropout(outcome)
        outcome = self.head_for_cls(outcome)

        return outcome