import logging

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from smart.vit import vit_base_patch16_128
from smart.bert import BertConfig, BertModel

from transformers import BertTokenizer
from sklearn.metrics import accuracy_score,roc_auc_score


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Autoencoder(nn.Module):
    """AutoEncoder module"""

    def __init__(self,
                 input_dim=768,
                 feature_dim=512,
                 n_clusters=10,
                 ):
        """Constructor.
        """
        super(Autoencoder, self).__init__()
        # self._activation = activation
        # self._batchnorm = batchnorm
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, feature_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
        self.mlp = Mlp(in_features=feature_dim, out_features=n_clusters)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        """Pass through model.
        """
        bs,len_ = x.shape[:2]
        z = self.encoder(x.flatten(0,1))
        x_hat = self.decoder(z)
        q = self.softmax(self.mlp(z))
        return x_hat.reshape(bs,len_,-1), z.reshape(bs,len_,-1), q.reshape(bs,len_,-1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,xw):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1]+xw.unsqueeze(1).unsqueeze(-1), qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # import pdb;pdb.set_trace()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y, yw):
        B, N, C = x.shape
        B, M, C = y.shape
        q = self.q(x).reshape(B,N,self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k(y).reshape(B,M,self.num_heads, C // self.num_heads).permute(0,2,1,3)+yw.unsqueeze(1).unsqueeze(-1)
        v = self.v(y).reshape(B,M,self.num_heads, C // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.cross_attn1 = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn2 = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.norm5 = norm_layer(dim)
        self.norm6 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x,xw,y,yw):
        x = x + self.drop_path(self.cross_attn1(self.norm1(x),self.norm2(y), yw))
        y = y + self.drop_path(self.cross_attn2(self.norm3(y),self.norm4(x), xw))
        x = x + self.drop_path(self.mlp1(self.norm5(x)))
        y = y + self.drop_path(self.mlp2(self.norm6(y)))
        return x, y
    
class BlockSelf(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x,xw):
        x = x + self.drop_path(self.attn(self.norm1(x), xw))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SMARTModel(nn.Module):
    def __init__(
        self,
        max_txt_len=40,
        hidden_size=768,
        feature_dim=512,
        n_clusters=10,
        n_modals=2,
        v_tokens=512,
        t_tokens=128,
        n_classes=3,
    ):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained("/data-pool/data/data2/qiuhui/tokenizer", truncation_side='right')
        config = BertConfig.from_pretrained("/data-pool/data/data2/qiuhui/tokenizer")
        self.text_transformer = BertModel(config=config)
        self.vision_transformer = vit_base_patch16_128()
        self.vision_proj = nn.Linear(self.vision_transformer.dim, hidden_size)
        self.text_proj = nn.Linear(self.text_transformer.config.hidden_size, hidden_size)
        self.vision_ae = Autoencoder(input_dim=hidden_size,feature_dim=feature_dim,n_clusters=n_clusters)
        self.text_ae = Autoencoder(input_dim=hidden_size,feature_dim=feature_dim,n_clusters=n_clusters)
        self.interfunc = Mlp(in_features=feature_dim, out_features=n_clusters)
        self.vw = nn.Parameter(torch.ones(1, v_tokens, 1))
        self.tw = nn.Parameter(torch.ones(1, t_tokens, 1))
        self.w = nn.Parameter(torch.ones(1, n_modals, 1))
        self.res_loss = nn.MSELoss()
        self.cos_sim = nn.CosineSimilarity()
        self.criterion = nn.CrossEntropyLoss()
        self.mrat_cross = Block(hidden_size, 12)
        self.mrat_self = BlockSelf(hidden_size, 12)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size*2, 256),  # 70+51
            nn.ReLU(),
            nn.Linear(256, n_classes)  # Only one output neuron
        )
        self.max_txt_len = max_txt_len
        

    def text2sentences(self, text, num=128):
        text = list(text)
        bs = len(text)
        for b in range(bs):
            # import pdb;pdb.set_trace()
            text[b] = text[b].strip().split('.')
            pri_len = len(text[b])
            text[b] += (num-pri_len)*['']
        return text
    
    def cos(self, cbar,cs):
        n = cs.shape[1]
        sims = []
        losses = []
        for ni in range(n):
            sim = self.cos_sim(cbar,cs[:,ni])
            loss = 1 - sim
            sims.append(sim)
            losses.append(loss.mean()) # batch mean
        return torch.stack(sims,dim=1), sum(losses)/len(losses) # token mean

    
    
       
    def forward(self, samples):

        # forward image encoder and text encoder to get embedding
        image,text,label = samples
        image = image.unsqueeze(1).cuda() # bs c 128 128 128
        sentences = self.text2sentences(text)
        label = label.cuda()
        bs = image.shape[0]

        image_embeds = self.vision_transformer(img = image) # bs img_len 768
        image_embeds = self.vision_proj(image_embeds) # bs img_len 768

        text_embeds = []
        for bi in range(bs):
            sentence_tokens = self.tokenizer(sentences[bi], padding="max_length", truncation=True, max_length=self.max_txt_len, return_tensors="pt",).to(image.device)
            text_embed = self.text_transformer(sentence_tokens.input_ids,attention_mask=sentence_tokens.attention_mask,output_attentions=True)[0][-1][:,0,:]
            text_embeds.append(text_embed)
        text_embeds = torch.stack(text_embeds)
        text_embeds = self.text_proj(text_embeds) # bs txt_len 768
        
        # image and text auto encoder
        ix, iz, ics = self.vision_ae(image_embeds)
        tx, tz, tcs = self.text_ae(text_embeds)

        # restore loss
        i_res_l = self.res_loss(ix,image_embeds)
        t_res_l = self.res_loss(tx,text_embeds)


        vw = torch.exp(self.vw) / torch.sum(torch.exp(self.vw))
        tw = torch.exp(self.tw) / torch.sum(torch.exp(self.tw))
        izbar = (vw*iz).sum(1)
        tzbar = (tw*tz).sum(1)

        icbar = self.vision_ae.mlp(izbar)
        tcbar = self.text_ae.mlp(tzbar)

        icbarbar = self.interfunc(izbar)
        tcbarbar = self.interfunc(tzbar)
        w = torch.exp(self.w) / torch.sum(torch.exp(self.w))
        
        cbarbar = (w*(torch.stack((icbarbar,tcbarbar),dim=1))).sum(1)
        
        
        intra_i_w, intra_i_l = self.cos(icbar,ics)
        intra_t_w, intra_t_l = self.cos(tcbar, tcs)
        inter_w, inter_l = self.cos(cbarbar,torch.stack((icbarbar,tcbarbar),dim=1))

        intra_l = (i_res_l + intra_i_l)+(t_res_l + intra_t_l)

        ssl_l = intra_l + inter_l

        icross,tcross = self.mrat_cross(image_embeds,intra_i_w,text_embeds,intra_t_w)
        it_f = torch.stack([icross.mean(1), tcross.mean(1)], dim=1)
        itself = self.mrat_self(it_f,inter_w)
        y_hat = self.mlp(itself.flatten(-2,-1))
        # import pdb;pdb.set_trace()
        cls_l = self.criterion(y_hat, label)
        loss = ssl_l + cls_l

        return {"loss": loss}
