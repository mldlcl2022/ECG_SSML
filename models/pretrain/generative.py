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
            
    def random_masking(self, x, mask_ratio):

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
            
    def forward(self, x, mask_ratio=0.5):
        
        x = self.patch_embed(x)
        
        x = x + self.pos_embed
        
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        for blk in self.blocks:
            x = blk(x)
            
        return x, mask, ids_restore

class ImageDecoder(nn.Module):
    '''
    이미지 디코더
    '''
    def __init__(self,
                 num_patches=196,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False):
        super().__init__()
        
        self.num_patches = num_patches
        self.patch_size = patch_size
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # 기존 코드 patch+1 -> patch (cls token 사용 X)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False) 
        
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_norm=False, norm_layer=norm_layer)
                                             for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        
        self.norm_pix_loss = norm_pix_loss
        
        self.initialize_weights()
        
    def initialize_weights(self):
        
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        torch.nn.init.normal_(self.mask_token, std=.02)
        
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
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        
        return x
        
    def forward_decoder(self, x, ids_restore):
        
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        x = x + self.decoder_pos_embed
        
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        x = self.decoder_pred(x)
        
        return x
    
    def forward_loss(self, imgs, pred, mask):
        
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def forward(self, x, latent, ids_restore, mask):
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        
        return loss, pred

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
            
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: (batch_size, num_leads, n, embed_dim)
        """
        b, num_leads, n, d = x.shape
        len_keep = int(n * (1 - mask_ratio))

        noise = torch.rand(b, num_leads, n, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=2)

        # keep the first subset
        ids_keep = ids_shuffle[:, :, :len_keep]
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, d))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([b, num_leads, n], device=x.device)
        mask[:, :, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=2, index=ids_restore)

        return x_masked, mask, ids_restore
            
    def forward(self, x, mask_ratio=0.5):
        
        x = self.encoder.to_patch_embedding(x)
        b, _, n, _ = x.shape
        
        x = x + self.encoder.pos_embedding[:, 1:n + 1, :].unsqueeze(1)
        
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
    
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
        
        return x, mask, ids_restore

class SignalDecoder(nn.Module):
    '''
    신호 디코더
    '''
    def __init__(self,
                 seq_len=2250,
                 patch_size=75,
                 num_leads=12,
                 embed_dim=768,
                 decoder_embed_dim=512,
                 decoder_depth=4,
                 decoder_num_heads=4,
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.num_leads = num_leads
        
        self.to_decoder_embedding = nn.Linear(embed_dim, decoder_embed_dim)
        
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 2, decoder_embed_dim),
                                              requires_grad=False)
        self.decoder_blocks = nn.ModuleList([TransformerBlock(input_dim=decoder_embed_dim,
                                                              output_dim=decoder_embed_dim,
                                                              hidden_dim=decoder_embed_dim * mlp_ratio,
                                                              heads=decoder_num_heads,
                                                              dim_head=64,
                                                              qkv_bias=qkv_bias)
                                             for _ in range(decoder_depth)])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, patch_size)
        
        self.norm_pix_loss = norm_pix_loss
        
        self.initialize_weights()
        
    def initialize_weights(self):
        
        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    self.num_patches,
                                                    sep_embed=True)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float().unsqueeze(0))
        
        torch.nn.init.normal_(self.mask_embedding, std=.02)
        
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
            
    def patchify(self, series):
        """
        series: (batch_size, num_leads, seq_len)
        x: (batch_size, num_leads, n, patch_size)
        """
        p = self.patch_size
        assert series.shape[2] % p == 0
        x = rearrange(series, 'b c (n p) -> b c n p', p=p)
        
        return x
            
    def forward_decoder(self, x, ids_restore):
        x = self.to_decoder_embedding(x)
        
        x = rearrange(x, 'b (c n) p -> b c n p', c=self.num_leads)
        b, _, n_masked_with_sep, d = x.shape
        n = ids_restore.shape[2]
        mask_embeddings = self.mask_embedding.unsqueeze(1)
        mask_embeddings = mask_embeddings.repeat(b, self.num_leads, n + 2 - n_masked_with_sep, 1)
        
        x_wo_sep = torch.cat([x[:, :, 1:-1, :], mask_embeddings], dim=2)
        x_wo_sep = torch.gather(x_wo_sep, dim=2, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, d))
        
        x_wo_sep = x_wo_sep + self.decoder_pos_embed[:, 1:n + 1, :].unsqueeze(1)
        left_sep = x[:, :, :1, :] + self.decoder_pos_embed[:, :1, :].unsqueeze(1)
        right_sep = x[:, :, -1:, :] + self.decoder_pos_embed[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x_wo_sep, right_sep], dim=2)
        
        x_decoded = []
        for i in range(self.num_leads):
            x_lead = x[:, i, :, :]
            for block in self.decoder_blocks:
                x_lead = block(x_lead)
            x_lead = self.decoder_norm(x_lead)
            x_lead = self.decoder_head(x_lead)
            x_decoded.append(x_lead[:, 1:-1, :])
        x = torch.stack(x_decoded, dim=1)
        
        return x
    
    def forward_loss(self, series, pred, mask):
        
        target = self.patchify(series)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (batch_size, num_leads, n), mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        
        return loss
    
    def forward(self, x, latent, ids_restore, mask):
        
        pred = self.forward_decoder(latent, ids_restore)
        recon_loss = self.forward_loss(x, pred, mask)
        
        return recon_loss, pred

class CrossAttentionModel(nn.Module):
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
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        
        self.img_encoder = ImageEncoder(img_size=img_size,
                                        patch_size=img_patch_size,
                                        in_chans=img_in_chans,
                                        embed_dim=embed_dim,
                                        depth=depth,
                                        num_heads=num_heads,
                                        mlp_ratio=mlp_ratio,
                                        norm_layer=norm_layer)
        
        # 이미지 인코더 사전 학습된 가중치로 초기화
        state_dict = torch.load('your_path_image_encoder_for_init', map_location='cpu')['model']
        load_result = self.img_encoder.load_state_dict(state_dict, strict=False)

        print("❗ Image Missing keys:")
        for key in load_result.missing_keys:
            print(f"  - {key}")
        print("❗ Image Unexpected keys:")
        for key in load_result.unexpected_keys:
            print(f"  - {key}")
        
        self.img_decoder = ImageDecoder(num_patches=self.img_encoder.patch_embed.num_patches,
                                        patch_size=img_patch_size,
                                        in_chans=img_in_chans,
                                        embed_dim=embed_dim,
                                        decoder_embed_dim=decoder_embed_dim,
                                        decoder_depth=decoder_depth,
                                        decoder_num_heads=img_decoder_num_heads,
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
        
        # 신호 인코더 사전 학습된 가중치로 초기화
        state_dict = torch.load('your_path_signal_encoder_for_init', map_location='cpu')['model']

        renamed = {}
        for k, v in state_dict.items():
            renamed[f'encoder.{k}'] = v

        load_result = self.sig_encoder.load_state_dict(renamed, strict=False)

        print("❗ Signal Missing keys:")
        for key in load_result.missing_keys:
            print(f"  - {key}")
        print("❗ Signal Unexpected keys:")
        for key in load_result.unexpected_keys:
            print(f"  - {key}")
        
        self.sig_decoder = SignalDecoder(seq_len=sig_seq_len,
                                         patch_size=sig_patch_size,
                                         num_leads=sig_num_leads,
                                         embed_dim=embed_dim,
                                         decoder_embed_dim=decoder_embed_dim,
                                         decoder_depth=decoder_depth,
                                         decoder_num_heads=sig_decoder_num_heads,
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
        
    def forward(self, img, sig):       
        
        latent_img, mask_img, ids_restore_img = self.img_encoder(img)
        latent_sig, mask_sig, ids_restore_sig = self.sig_encoder(sig)
        
        
        for blk_sig, blk_img in zip(self.cross_attn_sig, self.cross_attn_img):
            updated_sig = blk_sig(latent_sig, latent_img)
            updated_img = blk_img(latent_img, latent_sig)
            
            latent_sig = updated_sig
            latent_img = updated_img
            
        latent_sig = self.sig_norm(latent_sig)
        latent_img = self.img_norm(latent_img)
        
        loss_img, pred_img = self.img_decoder(img, latent_img, ids_restore_img, mask_img)
        loss_sig, pred_sig = self.sig_decoder(sig, latent_sig, ids_restore_sig, mask_sig)
        
        return loss_img + loss_sig, loss_img, loss_sig, (pred_img, pred_sig)