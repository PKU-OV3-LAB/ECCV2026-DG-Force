import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
from model.registry import MODELS

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.float()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W



class UpsampleConcatConvSegformer(nn.Module):
    def __init__(self):
        super(UpsampleConcatConvSegformer, self).__init__()
        # 192到96的上采样，单次上采样
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        # 384到96的上采样，两次上采样，逐步降低通道数
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )

        # 768到96的上采样，三次上采样，逐步降低通道数
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(512, 320, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )


    def forward(self, inputs):
        # 上采样
        x1,x2,x3,x4 = inputs
        up2 = self.upsample1(x2)
        up3 = self.upsample2(x3)
        up4 = self.upsample3(x4)
        
        x = torch.cat([x1, up2, up3, up4], dim=1)
        return x

class TokenMLP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int = None,
                 hidden_dim: int = None,
                 is_up_channel: bool = None,
                 act: str = "relu",
                 dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        if is_up_channel is not None:
            # 输入的in_dim为对应层的embed_dim
            if is_up_channel:
                in_dim = in_dim//16
                hidden_dim = in_dim*4
                out_dim = in_dim*16
            else: # down
                hidden_dim = in_dim//4
                out_dim = in_dim//16

            # 根据base_dim作为输入
            self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
            self.fc2 = nn.Linear(hidden_dim, out_dim , bias=bias)
        else:
            # 根据base_dim作为输入
            self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
            self.fc2 = nn.Linear(hidden_dim, out_dim , bias=bias)

        if act == "gelu":
            self.act = lambda x: F.gelu(x, approximate="tanh")
        elif act == "silu":
            self.act = F.silu
        else: 
            self.act = F.relu # default relu

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 1x1卷积
class LabelTokenDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim//2
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        if x.ndim == 3:  # (B, h*w, C)
            B, HW, C = x.shape
            h = w = int(math.sqrt(HW))
            x = x.view(B, h, w, C).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

# class EdgeTokenDecoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         pass
#     def forward(self, x):
#         return x

# 对整个特征图做卷积
class EdgeFeatureConv(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int = None,
                 hidden_dim: int = None,
                 is_up_channel: bool = None,
                 dropout: float = 0.0):
        super().__init__()
        if is_up_channel is not None:
            # 输入的in_dim为对应层的embed_dim
            if is_up_channel:
                in_dim = in_dim//16
                hidden_dim = in_dim*4
                out_dim = in_dim*16
            else: # down
                hidden_dim = in_dim//4
                out_dim = in_dim//16
        
        # TODO: 卷积后添加BatchNorm2d，稳定数值分布
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1)
        self.act = F.relu # default relu
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


# '''
# 将各尺度的特征图输出512x512
# 初始卷积：通道变为 base_ch，分辨率不变
# 每个上采样层：通道数减半,分辨率 ×2
# 输出卷积：通道变为 1
# 最后统一上采样：所有尺度都到 (1,512,512)
# '''
class EdgeFeatureDecoder(nn.Module):
    """使用转置卷积的简单 decode"""
    def __init__(self, in_ch, num_layers=1, base_ch=64):
        super().__init__()
        layers = []
        # 初始卷积
        layers.append(nn.Conv2d(in_ch, base_ch, 3, padding=1))
        layers.append(nn.ReLU())
        
        # 上采样层
        ch = base_ch
        for i in range(num_layers):
            layers.append(nn.ConvTranspose2d(ch, ch//2, kernel_size=2, stride=2))
            layers.append(nn.BatchNorm2d(ch//2))
            layers.append(nn.ReLU())
            ch = ch // 2
        
        # 输出 1 通道
        layers.append(nn.Conv2d(ch, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1,C,H,W)
        x = self.model(x)
        # 上采样到 512x512
        x = F.interpolate(x, size=(512,512), mode='bilinear', align_corners=False)
        return x.squeeze(0)  # (1,512,512)

    # debug：打印转置卷积输出的范围值
    # def forward(self, x):
    #     if x.dim() == 3:
    #         x = x.unsqueeze(0)  # (1,C,H,W)
    #     print('Decoder input (min,max,isnan):', x.min().item(), x.max().item(), torch.isnan(x).any().item())
    #     for i, layer in enumerate(self.model):
    #         x = layer(x)
    #         if isinstance(layer, nn.ConvTranspose2d):
    #             print(f'After ConvTranspose2d {i}: min={x.min().item():.2e}, max={x.max().item():.2e}, nan={torch.isnan(x).any().item()}')
    #     x = F.interpolate(x, size=(512,512), mode='bilinear', align_corners=False)
    #     return x


# 插值上采样 + 卷积
# class EdgeFeatureDecoder(nn.Module):
#     """
#     稳定版 Decoder: Conv + Upsample
#     - 初始卷积：通道 -> base_ch
#     - 每层：Conv + ReLU + Upsample(×2)
#     - 输出卷积：通道 -> 1
#     - 最后统一上采样到 (512,512)
#     """
#     def __init__(self, in_ch, num_layers=1, base_ch=64, out_size=(512, 512)):
#         super().__init__()
#         self.out_size = out_size
#         layers = []

#         # 初始卷积
#         layers.append(nn.Conv2d(in_ch, base_ch, 3, padding=1))
#         layers.append(nn.ReLU(inplace=True))

#         ch = base_ch
#         # for i in range(num_layers):
#         #     layers.append(nn.Conv2d(ch, ch // 2, 3, padding=1))
#         #     layers.append(nn.ReLU(inplace=True))
#         #     layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
#         #     ch = ch // 2
        
#         for i in range(num_layers):
#             layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
#             layers.append(nn.Conv2d(ch, max(ch // 2, 16), 3, padding=1))  # 保证通道不少于16
#             layers.append(nn.ReLU(inplace=True))
#             ch = max(ch // 2, 16)

#         layers.append(nn.Conv2d(ch, 1, kernel_size=1))
#         self.model = nn.Sequential(*layers)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)

#     def forward(self, x):
#         if x.dim() == 3:
#             x = x.unsqueeze(0)
#         x = self.model(x)
#         x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)
#         return x.squeeze(0)

# 各尺度特征下采样+通道变换
class DownAlign(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.down_layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        # x: [B, C_in, H, W]
        return self.down_layer(x)  # [B, C_out, H/2, W/2]


class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim_q % num_heads == 0 and dim_kv % num_heads == 0

        self.num_heads = num_heads
        self.scale = (dim_q // num_heads) ** -0.5

        # 不同输入的线性层
        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.k = nn.Linear(dim_kv, dim_q, bias=qkv_bias)  # 输出与 q 相同维度
        self.v = nn.Linear(dim_kv, dim_q, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.q_norm = nn.LayerNorm(dim_q)
        self.k_norm = nn.LayerNorm(dim_q)

    def forward(self, x_q, x_kv):
        """
        x_q: [B, Nq, Cq]
        x_kv: [B, Nk, Ck
        """
        B, Nq, Cq = x_q.shape
        _, Nk, Ck = x_kv.shape

        q = self.q(x_q)
        k = self.k(x_kv)
        v = self.v(x_kv)
        
        # 防止q@k溢出float16
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = q.reshape(B, Nq, self.num_heads, Cq // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, Nk, self.num_heads, Cq // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, Nk, self.num_heads, Cq // self.num_heads).permute(0, 2, 1, 3)
        
        # 无参数norm，兼容无nrom结构的模型权重
        # q = F.layer_norm(q, q.shape[-1:])
        # k = F.layer_norm(k, k.shape[-1:])

        # DEBUG：稳定版的softmax，防止attn溢出float16
        # with torch.cuda.amp.autocast(enabled=False):
        #     attn = (q @ k.transpose(-2, -1)) * self.scale
        #     attn = attn - attn.amax(dim=-1, keepdim=True)  # 稳定 softmax
        #     attn = attn.softmax(dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, Cq)
        x = self.proj(x)
        x = self.proj_drop(x) 
        return x


class CrossFuseBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=1, use_cross_attn=True):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.align = nn.Conv2d(dim_kv, dim_q, kernel_size=3, stride=2, padding=1)  # 下采样
        
        # 对齐网络，避免出现网格状错误
        # self.align = nn.Sequential(
        #     nn.Conv2d(dim_kv,dim_q, kernel_size=1, stride=1, bias=False),
        #     nn.AvgPool2d(kernel_size=2, stride=2)
        # )
        
        # nn.Conv2d(3,1,1)+unsample(0.5) toke预测依然出现网格状

        if self.use_cross_attn:
            # cross_attn fuse
            self.cross_attn = CrossAttention(dim_q, dim_q, num_heads=num_heads)
            self.norm_q = nn.LayerNorm(dim_q)
            self.norm_kv = nn.LayerNorm(dim_q)
            
        else:
            # concat+conv fuse
            self.conv_fuse =  nn.Sequential(
                nn.Conv2d(dim_q*2,dim_q,kernel_size=3,padding=1),
                nn.BatchNorm2d(dim_q),
                nn.GELU()
                )
            self.norm_q = nn.BatchNorm2d(dim_q)
            self.norm_kv = nn.BatchNorm2d(dim_q)
        
        # # 门控残差
        # self.gate = nn.Sequential(
        #     nn.Linear(dim_q,out_features=dim_q),
        #     nn.Sigmoid()
        # )
        # # 初始化0.1
        # nn.init.xavier_uniform_(self.gate[0].weight, gain=0.1)
        # nn.init.constant_(self.gate[0].bias, -2.0)
        
        # 更好的门控结构
        self.gate = nn.Sequential(
            nn.Linear(dim_q, dim_q // 4),
            nn.GELU(),
            nn.Linear(dim_q // 4, dim_q),
            nn.Sigmoid()
        )
        # 初始化
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.gate[-2].bias, -2.0)

    def forward(self, x_q, x_kv):
        if self.use_cross_attn: # Cross-Attn 融合
            # 将邻接层token变换到当前尺度
            B,N,C = x_kv.shape
            x_kv_2d = x_kv.transpose(1, 2).reshape(B,C,int(math.sqrt(N)),-1) # [B, C, H, W]
            x_kv_aligned = self.align(x_kv_2d) # [B, C_, H//2, W//2]
            _,C_,_,_ = x_kv_aligned.shape
            x_kv_aligned = x_kv_aligned.reshape(B,C_,-1).transpose(1,2) # [B, N, C_]
            
            fused = self.cross_attn(self.norm_q(x_q), self.norm_kv(x_kv_aligned))
            gate = self.gate(x_q)
            return x_q + gate*fused
        else: # concat_conv
            B_,N_,C_ = x_q.shape
            x_q_2d = x_q.transpose(1,2).reshape(B_,C_,int(math.sqrt(N_)),-1)
            B,N,C = x_kv.shape
            x_kv_2d = x_kv.transpose(1, 2).reshape(B,C,int(math.sqrt(N)),-1) # [B, C, H, W]
            x_kv_aligned = self.align(x_kv_2d) # [B, C_, H//2, W//2]
            
            fused = torch.concat([self.norm_q(x_q_2d),self.norm_kv(x_kv_aligned)], dim=1)
            fused = self.conv_fuse(fused).reshape(B_,C_,-1).transpose(1,2)
            gate = self.gate(x_q)
            return x_q + gate*fused

# 尺度内部各层特征融合
class InterFuseBlock(nn.Module):
    def __init__(self, dim_q):
        super().__init__()
        self.conv_proj = nn.Conv2d(dim_q, dim_q, kernel_size=1)  # 1x1 conv 投影

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(dim_q*2, dim_q, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_q),
            nn.GELU()
        )
        self.norm_q = nn.BatchNorm2d(dim_q)
        self.norm_kv = nn.BatchNorm2d(dim_q)

        self.gate = nn.Sequential(
            nn.Linear(dim_q, dim_q // 4),
            nn.GELU(),
            nn.Linear(dim_q // 4, dim_q),
            nn.Sigmoid()
        )
        # 初始化门控
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.gate[-2].bias, -2.0)

    def forward(self, x_q, x_kv):
        B, N, C = x_q.shape
        H = W = int(math.sqrt(N))
        x_q_2d = x_q.transpose(1,2).reshape(B,C,H,W)
        x_kv_2d = x_kv.transpose(1,2).reshape(B,C,H,W)

        # 1x1 conv 投影
        x_kv_2d = self.conv_proj(x_kv_2d)
        fused = torch.concat([self.norm_q(x_q_2d), self.norm_kv(x_kv_2d)], dim=1)
        fused = self.conv_fuse(fused).reshape(B,C,-1).transpose(1,2)

        gate = self.gate(x_q)
        return x_q + gate * fused

class SegFormer(nn.Module):
    def __init__(self,pretrain_path=None, img_size=512, patch_size=4, in_chans=3,embed_dims=[64, 128, 320, 512],num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.n_fuse_layers = 6
        self.n_cross_scale_fuse_layers = 2
        self.is_tail2head_cross_fuse = True
        assert self.n_fuse_layers>=self.n_cross_scale_fuse_layers
        
        # debug
        print(f"n_fuse_layers = {self.n_fuse_layers}")
        print(f"n_cross_scale_fuse_layers = {self.n_cross_scale_fuse_layers}")
        print(f"is_tail2head_cross_fuse = {self.is_tail2head_cross_fuse}")
        
        self.depths = depths
        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        self.down_convs_edge_emb1 = nn.ModuleList([
            EdgeFeatureConv(in_dim=embed_dims[0], is_up_channel=False)
            for i in range(depths[0])])
        self.up_convs_edge_emb1 = nn.ModuleList([
            EdgeFeatureConv(in_dim=embed_dims[0], is_up_channel=True)
            for i in range(depths[0])])
        self.down_convs_edge_emb2 = nn.ModuleList([
            EdgeFeatureConv(in_dim=embed_dims[1], is_up_channel=False)
            for i in range(depths[1])])
        self.up_convs_edge_emb2 = nn.ModuleList([
            EdgeFeatureConv(in_dim=embed_dims[1], is_up_channel=True)
            for i in range(depths[1])])
        self.down_convs_edge_emb3 = nn.ModuleList([
            EdgeFeatureConv(in_dim=embed_dims[2], is_up_channel=False)
            for i in range(depths[2])])
        self.up_convs_edge_emb3 = nn.ModuleList([
            EdgeFeatureConv(in_dim=embed_dims[2], is_up_channel=True)
            for i in range(depths[2])])
        self.down_convs_edge_emb4 = nn.ModuleList([
            EdgeFeatureConv(in_dim=embed_dims[3], is_up_channel=False)
            for i in range(depths[3])])
        self.up_convs_edge_emb4 = nn.ModuleList([
            EdgeFeatureConv(in_dim=embed_dims[3], is_up_channel=True)
            for i in range(depths[3])])
        self.down_mlps_label_emb1 = nn.ModuleList([
            TokenMLP(in_dim=embed_dims[0], is_up_channel=False)
            for i in range(depths[0])])
        self.up_mlps_label_emb1 = nn.ModuleList([
            TokenMLP(in_dim=embed_dims[0], is_up_channel=True)
            for i in range(depths[0])])
        self.down_mlps_label_emb2 = nn.ModuleList([
            TokenMLP(in_dim=embed_dims[1], is_up_channel=False)
            for i in range(depths[1])])
        self.up_mlps_label_emb2 = nn.ModuleList([
            TokenMLP(in_dim=embed_dims[1], is_up_channel=True)
            for i in range(depths[1])])
        self.down_mlps_label_emb3 = nn.ModuleList([
            TokenMLP(in_dim=embed_dims[2], is_up_channel=False)
            for i in range(depths[2])])
        self.up_mlps_label_emb3 = nn.ModuleList([
            TokenMLP(in_dim=embed_dims[2], is_up_channel=True)
            for i in range(depths[2])])
        self.down_mlps_label_emb4 = nn.ModuleList([
            TokenMLP(in_dim=embed_dims[3], is_up_channel=False)
            for i in range(depths[3])])
        self.up_mlps_label_emb4 = nn.ModuleList([
            TokenMLP(in_dim=embed_dims[3], is_up_channel=True)
            for i in range(depths[3])])

        # patch token decoder
        self.label_decoders_emb1 = LabelTokenDecoder(in_dim=embed_dims[0]//16, out_dim=1)
        self.label_decoders_emb2 = LabelTokenDecoder(in_dim=embed_dims[1]//16, out_dim=1)
        self.label_decoders_emb3 = LabelTokenDecoder(in_dim=embed_dims[2]//16, out_dim=1)
        self.label_decoders_emb4 = LabelTokenDecoder(in_dim=embed_dims[3]//16, out_dim=1)
        
        # edge token decoder
        self.edge_decoder_emb1 = EdgeFeatureDecoder(in_ch=embed_dims[0]//16, num_layers=1)
        self.edge_decoder_emb2 = EdgeFeatureDecoder(in_ch=embed_dims[1]//16, num_layers=2)
        self.edge_decoder_emb3 = EdgeFeatureDecoder(in_ch=embed_dims[2]//16, num_layers=3)
        self.edge_decoder_emb4 = EdgeFeatureDecoder(in_ch=embed_dims[3]//16, num_layers=4)
        
        self.weight_net_emb1 = nn.ModuleList([nn.Linear(embed_dims[0], 2) for i in range(depths[0])])
        self.weight_net_emb2 = nn.ModuleList([nn.Linear(embed_dims[1], 2) for i in range(depths[1])])
        self.weight_net_emb3 = nn.ModuleList([nn.Linear(embed_dims[2], 2) for i in range(depths[2])])
        self.weight_net_emb4 = nn.ModuleList([nn.Linear(embed_dims[3], 2) for i in range(depths[3])])
        
        # cross-scale fuse
        self.patch_token_fuse_block2 = nn.ModuleList([CrossFuseBlock(128,64,num_heads=1) for i in range(3)])
        self.patch_token_fuse_block3 = nn.ModuleList([CrossFuseBlock(320,128,num_heads=4) for i in range(4)])
        self.patch_token_fuse_block4 = nn.ModuleList([CrossFuseBlock(512,320,num_heads=8) for i in range(3)])
        self.edge_token_fuse_block2 = nn.ModuleList([CrossFuseBlock(128,64,num_heads=1) for i in range(3)])
        self.edge_token_fuse_block3 = nn.ModuleList([CrossFuseBlock(320,128,num_heads=4) for i in range(4)])
        self.edge_token_fuse_block4 = nn.ModuleList([CrossFuseBlock(512,320,num_heads=8) for i in range(3)])

        # inter-scale fuse
        # emb1: 0->2
        # emb2: 0->3
        # emb3: 0,1->4,5
        # emb4: 0->2
        self.start_fuse_idx = {
            "emb1": [0],
            "emb2": [0],
            "emb3": [0,1],
            "emb4": [0]
        }
        self.end_fuse_idx = {
            "emb1": [2],
            "emb2": [3],
            "emb3": [4,5],
            "emb4": [2]
        }
        
        self.patch_inter_fuse_block1 = nn.ModuleList([InterFuseBlock(64) for i in range(len(self.start_fuse_idx['emb1']))])
        self.patch_inter_fuse_block2 = nn.ModuleList([InterFuseBlock(128) for i in range(len(self.start_fuse_idx['emb2']))])
        self.patch_inter_fuse_block3 = nn.ModuleList([InterFuseBlock(320) for i in range(len(self.start_fuse_idx['emb3']))])
        self.patch_inter_fuse_block4 = nn.ModuleList([InterFuseBlock(512) for i in range(len(self.start_fuse_idx['emb4']))])
        self.edge_inter_fuse_block1 = nn.ModuleList([InterFuseBlock(64) for i in range(len(self.start_fuse_idx['emb1']))])
        self.edge_inter_fuse_block2 = nn.ModuleList([InterFuseBlock(128) for i in range(len(self.start_fuse_idx['emb2']))])
        self.edge_inter_fuse_block3 = nn.ModuleList([InterFuseBlock(320) for i in range(len(self.start_fuse_idx['emb3']))])
        self.edge_inter_fuse_block4 = nn.ModuleList([InterFuseBlock(512) for i in range(len(self.start_fuse_idx['emb4']))])
        
        # debug
        print(f"start_fuse_idx: {self.start_fuse_idx}")
        print(f"end_fuse_idx: {self.end_fuse_idx}")
        
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        if pretrain_path is not None:
            print("Load segformer pretrain pth.")
            self.load_state_dict(torch.load(pretrain_path),
                                strict=False)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        labels_logits_list = {
            "emb1": [],
            "emb2": [],
            "emb3": [],
            "emb4": [],
        }
        edges_logits_list = {
            "emb1": [],
            "emb2": [],
            "emb3": [],
            "emb4": [],
        }
        patch_tokens_list= {
            "emb1": [],
            "emb2": [],
            "emb3": [],
            "emb4": [],
        }
        edge_tokens_list= {
            "emb1": [],
            "emb2": [],
            "emb3": [],
            "emb4": [],
        }

        act_layers = range(self.n_fuse_layers) # DFDG layers
        act_cross_layers = range(self.n_cross_scale_fuse_layers) # MFT layers
        
        # stage 1
        x, H, W = self.patch_embed1(x)
        B, _, C = x.shape
        for i, blk in enumerate(self.block1):
            if i in act_layers:
                # token label
                f_label = self.down_mlps_label_emb1[i](x)
                x_label = self.up_mlps_label_emb1[i](f_label)
                # label decoder
                logit_label = self.label_decoders_emb1(f_label) # B,1,h,w
                labels_logits_list["emb1"].append(logit_label)
                
                # feature edge
                f_edge = x.permute(0, 2, 1).reshape(B, C, H, W)
                f_edge = self.down_convs_edge_emb1[i](f_edge)
                x_edge = self.up_convs_edge_emb1[i](f_edge)
                x_edge = x_edge.flatten(2).transpose(1, 2)
                # edge decoder
                logit_edge = self.edge_decoder_emb1(f_edge) # B,1,512,512
                edges_logits_list["emb1"].append(logit_edge)
                
                if i in self.end_fuse_idx['emb1']:
                    idx = self.end_fuse_idx['emb1'].index(i)
                    fused_idx = self.start_fuse_idx['emb1'][idx]
                    x_label = self.patch_inter_fuse_block1[idx](x_label, patch_tokens_list["emb1"][fused_idx])
                    x_edge = self.edge_inter_fuse_block1[idx](x_edge, edge_tokens_list["emb1"][fused_idx])
                    
                patch_tokens_list["emb1"].append(x_label)
                edge_tokens_list["emb1"].append(x_edge)

                w = self.weight_net_emb1[i](x) # B, H*W, 2
                w = torch.softmax(w, dim=-1)
                x = x + w[...,0:1]*x_label + w[...,1:2]*x_edge
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        B, _, C = x.shape
        for i, blk in enumerate(self.block2):
            if i in act_layers:      
                # token label
                f_label = self.down_mlps_label_emb2[i](x)
                x_label = self.up_mlps_label_emb2[i](f_label)
                # label decoder
                logit_label = self.label_decoders_emb2(f_label) # B,1,h,w
                labels_logits_list["emb2"].append(logit_label)
                
                # feature edge
                f_edge = x.permute(0, 2, 1).reshape(B, C, H, W)
                f_edge = self.down_convs_edge_emb2[i](f_edge)
                x_edge = self.up_convs_edge_emb2[i](f_edge)
                x_edge = x_edge.flatten(2).transpose(1, 2)
                # edge decoder
                logit_edge = self.edge_decoder_emb2(f_edge) # B,1,512,512
                edges_logits_list["emb2"].append(logit_edge)
                
                if i in self.end_fuse_idx['emb2']:
                    idx = self.end_fuse_idx['emb2'].index(i)
                    fused_idx = self.start_fuse_idx['emb2'][idx]
                    x_label = self.patch_inter_fuse_block2[idx](x_label, patch_tokens_list["emb2"][fused_idx])
                    x_edge = self.edge_inter_fuse_block2[idx](x_edge, edge_tokens_list["emb2"][fused_idx])
                
                if i < len(self.patch_token_fuse_block2) and i in act_cross_layers:
                    if self.is_tail2head_cross_fuse:
                        max_valid_layer = 3 if self.n_fuse_layers>3 else self.n_fuse_layers
                        offset = max_valid_layer-self.n_cross_scale_fuse_layers
                        offset = 0 if offset<0 else offset
                    else:
                        offset=0
                    x_label = self.patch_token_fuse_block2[i](x_label, patch_tokens_list["emb1"][i+offset])
                    x_edge = self.edge_token_fuse_block2[i](x_edge, edge_tokens_list["emb1"][i+offset])
                    
                patch_tokens_list["emb2"].append(x_label)
                edge_tokens_list["emb2"].append(x_edge)
                w = self.weight_net_emb2[i](x) # B, H*W, 2
                w = torch.softmax(w, dim=-1)
                x = x + w[...,0:1]*x_label + w[...,1:2]*x_edge
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        B, _, C = x.shape
        for i, blk in enumerate(self.block3):
            if i in act_layers:
                # token label
                f_label = self.down_mlps_label_emb3[i](x)
                x_label = self.up_mlps_label_emb3[i](f_label)
                # label decoder
                logit_label = self.label_decoders_emb3(f_label) # B,1,h,w
                labels_logits_list["emb3"].append(logit_label)
                
                # feature edge
                f_edge = x.permute(0, 2, 1).reshape(B, C, H, W)
                f_edge = self.down_convs_edge_emb3[i](f_edge)
                x_edge = self.up_convs_edge_emb3[i](f_edge)
                x_edge = x_edge.flatten(2).transpose(1, 2)
                # edge decoder
                logit_edge = self.edge_decoder_emb3(f_edge) # B,1,512,512
                edges_logits_list["emb3"].append(logit_edge)
                
                if i in self.end_fuse_idx['emb3']:
                    idx = self.end_fuse_idx['emb3'].index(i)
                    fused_idx = self.start_fuse_idx['emb3'][idx]
                    x_label = self.patch_inter_fuse_block3[idx](x_label, patch_tokens_list["emb3"][fused_idx])
                    x_edge = self.edge_inter_fuse_block3[idx](x_edge, edge_tokens_list["emb3"][fused_idx])
                
                if i < len(self.patch_token_fuse_block3) and i in act_cross_layers:
                    if self.is_tail2head_cross_fuse:
                        max_valid_layer = 4 if self.n_fuse_layers>4 else self.n_fuse_layers
                        offset = max_valid_layer-self.n_cross_scale_fuse_layers
                        offset = 0 if offset<0 else offset
                    else:
                        offset=0
                    x_label = self.patch_token_fuse_block3[i](x_label, patch_tokens_list["emb2"][i+offset])
                    x_edge = self.edge_token_fuse_block3[i](x_edge, edge_tokens_list["emb2"][i+offset])
                    
                patch_tokens_list["emb3"].append(x_label)
                edge_tokens_list["emb3"].append(x_edge)

                w = self.weight_net_emb3[i](x) # B, H*W, 2
                w = torch.softmax(w, dim=-1)
                x = x + w[...,0:1]*x_label + w[...,1:2]*x_edge
            
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        B, _, C = x.shape
        for i, blk in enumerate(self.block4):
            if i in act_layers:
                # token label
                f_label = self.down_mlps_label_emb4[i](x)
                x_label = self.up_mlps_label_emb4[i](f_label)
                # label decoder
                logit_label = self.label_decoders_emb4(f_label) # B,1,h,w
                labels_logits_list["emb4"].append(logit_label)
                
                # feature edge
                f_edge = x.permute(0, 2, 1).reshape(B, C, H, W)
                f_edge = self.down_convs_edge_emb4[i](f_edge)
                x_edge = self.up_convs_edge_emb4[i](f_edge)
                x_edge = x_edge.flatten(2).transpose(1, 2)
                # edge decoder
                logit_edge = self.edge_decoder_emb4(f_edge)
                edges_logits_list["emb4"].append(logit_edge)
                
                if i in self.end_fuse_idx['emb4']:
                    idx = self.end_fuse_idx['emb4'].index(i)
                    fused_idx = self.start_fuse_idx['emb4'][idx]
                    x_label = self.patch_inter_fuse_block4[idx](x_label, patch_tokens_list["emb4"][fused_idx])
                    x_edge = self.edge_inter_fuse_block4[idx](x_edge, edge_tokens_list["emb4"][fused_idx])
                
                if i < len(self.patch_token_fuse_block4) and i in act_cross_layers:
                    if self.is_tail2head_cross_fuse:
                        max_valid_layer = 18 if self.n_fuse_layers>18 else self.n_fuse_layers
                        offset = max_valid_layer-self.n_cross_scale_fuse_layers
                        offset = 0 if offset<0 else offset
                    else:
                        offset=0
                    x_label = self.patch_token_fuse_block4[i](x_label, patch_tokens_list["emb3"][i+offset])
                    x_edge = self.edge_token_fuse_block4[i](x_edge, edge_tokens_list["emb3"][i+offset])
                    
                patch_tokens_list["emb4"].append(x_label)
                edge_tokens_list["emb4"].append(x_edge)

                w = self.weight_net_emb4[i](x) # B, H*W, 2
                w = torch.softmax(w, dim=-1)
                x = x + w[...,0:1]*x_label + w[...,1:2]*x_edge   
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return x,outs,edges_logits_list,labels_logits_list

    def forward(self, x):
        return self.forward_features(x)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
    
class UpsampleConcatConv_Transformer_Only(nn.Module):
    def __init__(self):
        super(UpsampleConcatConv_Transformer_Only, self).__init__()
        self.upsamples2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upsamples3 = nn.Sequential(
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )
        self.upsamples4 = nn.Sequential(
            nn.ConvTranspose2d(512, 320, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, inputs):
        s1,s2,s3,s4 = inputs
        s2 = self.upsamples2(s2)
        s3 = self.upsamples3(s3)
        s4 = self.upsamples4(s4)
        x = torch.cat([s1,s2,s3,s4], dim=1)
        features = [s1,s2,s3,s4]
        return x, features

class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

@MODELS.register_module(name="dg_force")
class dg_force(nn.Module):
    def __init__(self, seg_pretrain_path=None):
        super().__init__()
        self.segformer = SegFormer(seg_pretrain_path)
        self.upsample_for_transformer = UpsampleConcatConv_Transformer_Only()
        self.inverse_for_transformer = nn.ModuleList([nn.Conv2d(64, 1, 1) for _ in range(4)])
        self.resize = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.use_dice_loss = False
    
    def model_forward(self, image, *args, **kwargs):
        _,outs2,edges_logits_list,labels_logits_list = self.segformer(image) # Transformer
        features = outs2
        x, features = self.upsample_for_transformer(features)
        reduced = torch.cat([self.inverse_for_transformer[i](features[i]) for i in range(4)], dim=1)
        features = torch.sum(reduced, dim=1, keepdim=True)
        return features,edges_logits_list,labels_logits_list
    
    def forward(self, image, mask, edge_mask, *args, **kwargs):
        features,edges_logits_list,labels_logits_list = self.model_forward(image)
        
        # patch loss
        patch_label = []
        for size in [128, 64, 32, 16]:
            pl = self.mask_to_patch_label(mask, size, is_soft_label=True)  # (B, size, size)
            pl = pl.unsqueeze(1).float()  # (B, 1, size, size)
            patch_label.append(pl)

        patch_losses = self.compute_multi_scale_bce(labels_logits_list, patch_label)
        avg_patch_loss = torch.stack(patch_losses).mean()

        # edge loss
        edge_losses = self.compute_edge_loss(edges_logits_list, edge_mask)
        avg_edge_loss = torch.stack(edge_losses).mean()
         
        # mask loss
        pred_mask = self.resize(features) #logits
        mask_loss = self.loss_fn(pred_mask, mask.float())
        
        pred_mask = pred_mask.float()
        pred_mask = torch.sigmoid(pred_mask) # prob
        
        # Dice loss
        if self.use_dice_loss:
            smooth = 1e-6
            intersection = torch.sum(pred_mask * mask)
            dice_coeff = (2. * intersection + smooth) / (torch.sum(pred_mask) + torch.sum(mask) + smooth)
            dice_loss = 1 - dice_coeff
            mask_loss_sum = mask_loss + dice_loss # add dice_loss
        else:
            mask_loss_sum = mask_loss
            
        loss = mask_loss_sum*2.0 + avg_patch_loss*1.0 + avg_edge_loss*1.0
       
        output_dict = {
            # loss for backward
            "backward_loss": loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": pred_mask,
            # DEBUG: other loss
            "mask_loss": mask_loss,
            # "dice_loss": dice_loss,
            "patch_loss": avg_patch_loss,
            "edge_loss": avg_edge_loss,

            "visual_loss": {
                "predict_loss": loss,
                'predict_mask_loss': mask_loss,
                'patch_loss': avg_patch_loss,
                'edge_loss': avg_edge_loss,
            },

            "visual_image": {
                "pred_mask": pred_mask,
            }
            # -----------------------------------------
        }
        return output_dict

    def mask_to_patch_label(self, mask, size, is_soft_label=False):
        B, C, H, W = mask.shape
        assert C == 1, "mask should have 1 channel"
        mask = mask.squeeze(1)  # (B, H, W)
        patch_h = H // size
        patch_w = W // size
        # reshape to patch
        mask_reshaped = mask.reshape(B, size, patch_h, size, patch_w)
        
        if not is_soft_label:
            patch_label = mask_reshaped.max(dim=4)[0].max(dim=2)[0]  # (B, size, size)
        else:
            mask_reshaped = mask_reshaped.float()
            patch_label = mask_reshaped.mean(dim=4).mean(dim=2)  # (B, size, size)
        return patch_label.float()
    

    def compute_multi_scale_bce(self, labels_logits_list, patch_label):
        bce_loss_fn = nn.BCEWithLogitsLoss()
        loss_list = []
        scale_keys = ["emb1", "emb2", "emb3", "emb4"]

        for i, key in enumerate(scale_keys):
            logits_list = labels_logits_list[key]  # (B,1,H_patch,W_patch)
            if len(logits_list)==0:
                continue
            labels = patch_label[i]                # (B,1,H_patch,W_patch)
            
            logits_tensor = torch.stack(logits_list, dim=0)
            if logits_tensor.dim() == 4:  
                logits_tensor = logits_tensor.unsqueeze(2)
            labels_tensor = labels.unsqueeze(0).expand_as(logits_tensor)
            loss = bce_loss_fn(logits_tensor, labels_tensor)
            loss_list.append(loss)
        
        return loss_list  # list of 4 loss values

    def compute_edge_loss(self, edges_logits_list, edge_mask):
        bce_loss_fn = nn.BCEWithLogitsLoss()
        loss_list = []
        scale_keys = ["emb1", "emb2", "emb3", "emb4"]
        
        for key in scale_keys:
            logits_list = edges_logits_list[key]  # list of tensors (B,1,H,W)
            if len(logits_list) == 0:
                continue

            logits_tensor = torch.stack(logits_list, dim=0)
            if logits_tensor.dim() == 4:  
                logits_tensor = logits_tensor.unsqueeze(2)
            labels_tensor = edge_mask.unsqueeze(0).expand_as(logits_tensor)

            loss = bce_loss_fn(logits_tensor, labels_tensor)
            loss_list.append(loss)
        
        return loss_list  # list of 4 loss values
    
    def compute_edge_loss_check_nan(self, edges_logits_list, edge_mask):
        bce_loss_fn = nn.BCEWithLogitsLoss()    
        loss_list = []
        scale_keys = ["emb1", "emb2", "emb3", "emb4"]

        for key in scale_keys:
            logits_list = edges_logits_list[key]  # list of tensors (B,1,H,W)
            if len(logits_list) == 0:
                loss_list.append(torch.tensor(0., device=edge_mask.device))
                continue

            # check if nan
            for i, t in enumerate(logits_list):
                if not torch.isfinite(t).all():
                    n_nan = torch.isnan(t).sum().item()
                    n_inf = torch.isinf(t).sum().item()
                    print(f"[WARNING] non-finite in {key} layer {i}: nan={n_nan}, inf={n_inf}, min={t.min().item()}, max={t.max().item()}")
                    # warning: replace nan/inf with finite values
                    logits_list[i] = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)

            # stack -> shape (num_layers, B, 1, H, W)
            logits_tensor = torch.stack(logits_list, dim=0)

            if logits_tensor.dtype != edge_mask.dtype:
                logits_tensor = logits_tensor.to(dtype=edge_mask.dtype)
            labels_tensor = edge_mask.unsqueeze(0).expand_as(logits_tensor)
            L, B, C, H, W = logits_tensor.shape
            logits_flat = logits_tensor.reshape(L * B, C, H, W)
            labels_flat = labels_tensor.reshape(L * B, C, H, W)
            logits_flat = logits_flat.clamp(min=-50.0, max=50.0)

            loss = bce_loss_fn(logits_flat, labels_flat)
            loss_list.append(loss)

        return loss_list

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """denormalize image with mean and std
    """
    image = image.clone().detach().cpu()
    image = image * torch.tensor(std).view(3, 1, 1)
    image = image + torch.tensor(mean).view(3, 1, 1)
    return image

if __name__ == "__main__":
    print(MODELS)
