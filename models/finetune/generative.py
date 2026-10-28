from einops import rearrange
from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from models.st_mem_reference.st_mem_vit import ST_MEM_ViT, TransformerBlock
from models.pretrain.cross_attention import CrossAttentionBlock

from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed

class ImageEncoder(nn.Module):
    '''
    이미지 인코더
    '''
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
            
        return x
    
class SignalEncoder(nn.Module):
    '''
    신호 인코더
    '''
    def __init__(self,
                 seq_len=2250,
                 patch_size=75,
                 num_leads=12,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.num_leads = num_leads
        
        self.encoder = ST_MEM_ViT(seq_len=seq_len,
                                  patch_size=patch_size,
                                  num_leads=num_leads,
                                  width=embed_dim,
                                  depth=depth,
                                  mlp_dim=mlp_ratio * embed_dim,
                                  heads=num_heads,
                                  qkv_bias=qkv_bias)
        # self.to_patch_embedding = self.encoder.to_patch_embedding
        self.encoder.to_patch_embedding = self.encoder.to_patch_embedding
        
        self.initialize_weights()
        
    def initialize_weights(self):
        
        pos_embed = get_1d_sincos_pos_embed(self.encoder.pos_embedding.shape[-1],
                                            self.num_patches,
                                            sep_embed=True)
        self.encoder.pos_embedding.data.copy_(pos_embed.float().unsqueeze(0))
        self.encoder.pos_embedding.requires_grad = False
        
        torch.nn.init.normal_(self.encoder.sep_embedding, std=.02)
        for i in range(self.num_leads):
            torch.nn.init.normal_(self.encoder.lead_embeddings[i], std=.02)
            
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
        
        x = self.encoder.to_patch_embedding(x)
        b, _, n, _ = x.shape
        
        x = x + self.encoder.pos_embedding[:, 1:n + 1, :].unsqueeze(1)
    
        sep_embedding = self.encoder.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, self.num_leads, -1, -1) + self.encoder.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, self.num_leads, -1, -1) + self.encoder.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)
        
        n_masked_with_sep = x.shape[2]
        lead_embeddings = torch.stack([self.encoder.lead_embeddings[i] for i in range(self.num_leads)]).unsqueeze(0)
        lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, n_masked_with_sep, -1)
        x = x + lead_embeddings
        
        x = rearrange(x, 'b c n p -> b (c n) p')
        for i in range(self.encoder.depth):
            x = getattr(self.encoder, f'block{i}')(x)
            
        return x

class FusionConcat_gen(nn.Module):
    def __init__(self,
                img_size=224,
                img_patch_size=16,
                img_in_chans=3,
                
                sig_seq_len=2250,
                sig_patch_size=75,
                sig_num_leads=12,
                
                embed_dim=768,
                num_heads=12,
                depth=12,
                
                decoder_embed_dim=512,
                img_decoder_num_heads=16,
                sig_decoder_num_heads=4,
                decoder_depth=4,
                
                cross_attention_depth=1,
                
                mlp_ratio=4,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                num_classes=5):
        super().__init__()

        self.img_encoder = ImageEncoder(img_size=img_size,
                                        patch_size=img_patch_size,
                                        in_chans=img_in_chans,
                                        embed_dim=embed_dim,
                                        depth=depth,
                                        num_heads=num_heads,
                                        mlp_ratio=mlp_ratio,
                                        norm_layer=norm_layer)
        
        self.sig_encoder = SignalEncoder(seq_len=sig_seq_len,
                                        patch_size=sig_patch_size,
                                        num_leads=sig_num_leads,
                                        embed_dim=embed_dim,
                                        depth=depth,
                                        num_heads=num_heads,
                                        mlp_ratio=mlp_ratio,
                                        norm_layer=norm_layer)
        
        self.cross_attn_sig = nn.ModuleList([CrossAttentionBlock(query_dim=embed_dim,
                                                                    kv_dim=embed_dim,
                                                                    output_dim=embed_dim,
                                                                    hidden_dim=int(mlp_ratio * embed_dim),
                                                                    heads=num_heads,
                                                                    dim_head=64) for i in range(cross_attention_depth)])

        
        self.cross_attn_img = nn.ModuleList([CrossAttentionBlock(query_dim=embed_dim,
                                                                kv_dim=embed_dim,
                                                                output_dim=embed_dim,
                                                                hidden_dim=int(mlp_ratio * embed_dim),
                                                                heads=num_heads,
                                                                dim_head=64) for i in range(cross_attention_depth)])

        self.sig_norm = norm_layer(embed_dim)
        self.img_norm = norm_layer(embed_dim)
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(embed_dim*2, num_classes)

    def forward(self, img, sig):

        latent_img = self.img_encoder(img)
        latent_sig = self.sig_encoder(sig)

        for blk_sig, blk_img in zip(self.cross_attn_sig, self.cross_attn_img):
            updated_sig = blk_sig(latent_sig, latent_img)
            updated_img = blk_img(latent_img, latent_sig)
            
            latent_sig = updated_sig
            latent_img = updated_img

        # global pooling (signal)
        latent_sig = rearrange(latent_sig, 'b (c n) p -> b c n p', c=12)
        latent_sig = latent_sig[:,:,1:-1,:]
        latent_sig = torch.mean(latent_sig, dim=(1,2))

        # global pooling (image)
        latent_img = latent_img.mean(dim=1)

        outcome_sig = self.sig_norm(latent_sig)
        outcome_img = self.img_norm(latent_img)

        outcome = torch.cat([outcome_sig, outcome_img], dim=1)
        outcome = self.dropout(outcome)
        outcome = self.head(outcome)

        return outcome