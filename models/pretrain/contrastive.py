import timm
import numpy as np
from functools import partial

import torch
import torch.nn as nn

from models.st_mem_reference import encoder
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed

class ImageEncoder(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768, 
                 depth=12, 
                 num_heads=16,
                 mlp_ratio=4., 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 기존 코드 patch+1 -> patch (cls token 사용 X)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_norm=False, norm_layer=norm_layer)
                                     for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        # self.head = nn.Linear(embed_dim, num_classes)
        
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
            
    def forward(self, x):
        
        x = self.patch_embed(x)
        
        x = x + self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
            
        x = x.mean(dim=1)
        outcome = self.norm(x)
        # outcome = self.head(outcome)
            
        return outcome

class ECGCLIP(nn.Module):
    def __init__(self, 
                 image_model, 
                 signal_model):
        super().__init__()

        # image encoder
        if image_model == 'vit_timm':
            self.image_encoder = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0, drop_path_rate=0.5)
            self.proj_image = nn.Linear(768, 512)
        elif image_model == 'vit_recon':
            self.image_encoder = ImageEncoder(img_size=224,
                                              patch_size=16,
                                              in_chans=3,
                                              embed_dim=768,
                                              depth=12,
                                              num_heads=12,
                                              mlp_ratio=4,
                                              norm_layer=partial(nn.LayerNorm, eps=1e-6))
            pretrain_state_dict_image = torch.load('your_path_image_encoder_for_init')['model']
            self.image_encoder.load_state_dict(pretrain_state_dict_image, strict=True)
            self.proj_image = nn.Linear(768, 512)
        elif image_model == 'convnext_timm':
            self.image_encoder = timm.create_model('convnext_base', pretrained=True, num_classes=0, drop_path_rate=0.5)
            self.proj_image = nn.Linear(1024, 512)

        # signal encoder
        if signal_model == 'st_mem':
            self.signal_encoder = encoder.__dict__['st_mem_vit_base'](num_leads=12,
                                                                      seq_len=2250,
                                                                      patch_size=75,
                                                                      num_classes=None)
            pretrain_state_dict_signal = torch.load('your_path_signal_encoder_for_init')['model']
            self.signal_encoder.load_state_dict(pretrain_state_dict_signal, strict=True)
            self.proj_signal = nn.Linear(768, 512)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        image_emb = self.image_encoder(image)
        proj_image_emb = self.proj_image(image_emb)
        return proj_image_emb

    def encode_signal(self, signal):
        signal_emb = self.signal_encoder(signal)
        proj_signal_emb = self.proj_signal(signal_emb)
        return proj_signal_emb

    def forward(self, image, signal):
        image_features = self.encode_image(image)
        signal_features = self.encode_signal(signal)
        
        # normalize
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        signal_features = signal_features / signal_features.norm(dim=1, keepdim=True)
        
        # cosine similarity
        logit_scale = self.logit_scale.clamp(0, np.log(100)).exp()
        logits_per_image = logit_scale * image_features @ signal_features.t()
        logits_per_signal = logits_per_image.t()
        
        return logits_per_image, logits_per_signal