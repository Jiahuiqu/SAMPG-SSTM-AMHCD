import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# 1. ConvBlock（支持自定义kernel/stride/padding，适配下采样）
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),  # 控制尺寸的核心层
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),  # 内层保持尺寸
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# 2. Spatial Mamba（仅保留Mamba标准参数）
class SpatialMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim,
            expand=2,
            d_state=16,
            dt_rank="auto"
        )

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W

        def run(seq):
            return self.mamba(seq)

        x_flat = x.view(B, C, L).transpose(1, 2)  # [B, L, C]
        seqs = [
            x_flat,
            torch.flip(x_flat, dims=[1]),
            x_flat.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B, L, C),
            torch.flip(
                x_flat.view(B, H, W, C)
                    .permute(0, 2, 1, 3)
                    .reshape(B, L, C),
                dims=[1],
            ),
        ]

        outs = [run(s) for s in seqs]
        out = sum(outs) / 4.0
        out = out.transpose(1, 2).view(B, C, H, W)
        return out


# 3. Spectral Mamba（仅保留Mamba标准参数）
class SpectralMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim,
            expand=2,
            d_state=16,
            dt_rank="auto"
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(-1, C)  # [(BHW), C]
        x_seq = x_seq.unsqueeze(1)  # [(BHW), 1, C]

        y1 = self.mamba(x_seq)
        y2 = self.mamba(torch.flip(x_seq, dims=[1]))

        y = (y1 + y2) / 2.0
        y = y.squeeze(1).view(B, H, W, C).permute(0, 3, 1, 2)
        return y


# 4. SSCM Block
class SSCMBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = ConvBlock(dim, dim)
        self.spa_mamba = SpatialMamba(dim)
        self.spe_mamba = SpectralMamba(dim)

        self.spa_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Softmax(dim=-1),
        )

        self.spe_att = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        floc = self.conv(x)

        # spatial
        fspa = self.spa_mamba(x)
        spa_pool = torch.cat(
            [floc.mean(1, keepdim=True), floc.max(1, keepdim=True)[0]], dim=1
        )
        aspa = self.spa_att(spa_pool)
        fspa = (fspa + floc) * aspa

        # spectral
        fspe = self.spe_mamba(x)
        spe_pool = torch.cat(
            [
                floc.mean([2, 3], keepdim=True).expand_as(floc),
                torch.amax(floc, dim=[2, 3], keepdim=True).expand_as(floc)
            ],
            dim=1,
        )
        aspe = self.spe_att(spe_pool)
        fspe = (fspe + floc) * aspe

        return fspa + fspe


# 5. SST Encoder
class SSTEncoder(nn.Module):
    def __init__(self, in_ch, dim, depth):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, 1)  # 保持9×9尺寸
        self.blocks = nn.ModuleList([SSCMBlock(dim) for _ in range(depth)])

    def forward(self, x):
        x = self.proj(x)  # [B, dim, 9, 9]
        for blk in self.blocks:
            x = blk(x)  # 保持9×9尺寸
        return x


# 6. Temporal Mamba（核心修复：输出维度改为in_dim，匹配fd）
class TemporalMamba(nn.Module):
    def __init__(self, in_dim):  # 移除out_dim参数，仅输出in_dim
        super().__init__()
        self.mamba = Mamba(
            d_model=in_dim * 2,  # 输入最后一维是in_dim*2，必须匹配
            expand=2,
            d_state=16,
            dt_rank="auto"
        )
        # proj层：输入in_dim*2 → 输出in_dim（和fd维度一致）
        self.proj = nn.Linear(in_dim * 2, in_dim)

    def forward(self, f1, f2):
        B, C, H, W = f1.shape  # C = in_dim
        s = torch.stack([f1, f2], dim=2)  # [B, C, 2, H, W]
        # 维度重排：确保最后一维是C*2
        s = s.permute(0, 3, 4, 1, 2).reshape(B * H * W, C * 2)  # [B*H*W, C*2]
        s = s.unsqueeze(1)  # [B*H*W, 1, C*2] → 满足Mamba输入格式

        # Mamba前向：维度匹配
        out = self.mamba(s)  # [B*H*W, 1, C*2]
        out = out.squeeze(1)  # [B*H*W, C*2]

        # 维度映射到in_dim（和fd一致）
        out = self.proj(out)  # [B*H*W, in_dim]

        # 恢复空间维度：匹配原H/W
        return out.view(B, H, W, -1).permute(0, 3, 1, 2)  # [B, in_dim, H, W]


# 7. Center Diff Attention
class CenterDiffAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5

    def forward(self, diff):
        B, C, H, W = diff.shape
        center = diff[:, :, H // 2, W // 2].unsqueeze(-1).unsqueeze(-1)
        att = (diff * center).sum(1, keepdim=True) * self.scale
        att = F.softmax(att.flatten(2), dim=-1).view(B, 1, H, W)
        return diff * att


# 8. Decoder Block (CTMB) - 核心修复：先匹配维度再相加，再下采样
class CTMB(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, s=1, p=1):
        super().__init__()
        # 下采样卷积：in_dim → out_dim，尺寸由k/s/p控制
        self.conv_f1 = ConvBlock(in_dim, out_dim, k=k, s=s, p=p)
        self.conv_f2 = ConvBlock(in_dim, out_dim, k=k, s=s, p=p)
        self.conv_fd = ConvBlock(in_dim, out_dim, k=k, s=s, p=p)
        # TemporalMamba：仅输入in_dim，输出in_dim（和fd一致）
        self.temporal = TemporalMamba(in_dim)
        self.att = CenterDiffAttention(out_dim)

    def forward(self, f1, f2, fd):
        # 1. 计算时间特征ft：维度in_dim（和fd一致），尺寸未下采样
        ft = self.temporal(f1, f2)  # [B, in_dim, H, W]

        # 2. fd + ft：维度一致（均为in_dim），可直接相加
        fd = fd + ft  # [B, in_dim, H, W]

        # 3. 下采样并更新f1/f2/fd：尺寸9→5/5→3，维度in_dim→out_dim
        f1 = self.conv_f1(f1)  # [B, out_dim, H_new, W_new]
        f2 = self.conv_f2(f2)  # [B, out_dim, H_new, W_new]
        fd = self.conv_fd(fd)  # [B, out_dim, H_new, W_new]

        # 4. 注意力机制（尺寸已下采样，维度out_dim）
        fd = self.att(fd)

        return f1, f2, fd


# 9. SST-MCDNet - 9→5→3（padding+stride=2）
class SST_MCDNet(nn.Module):
    def __init__(self, in_ch, dim=32, enc_depth=3, dec_depth=3):
        super().__init__()
        # Encoder：输出[B, dim, 9, 9]
        self.encoder = SSTEncoder(in_ch, dim, enc_depth)

        # Decoder配置：9→5→3→3（padding+stride=2）
        dec_config = [
            # 第1层：9→5 (k=3, s=2, p=1)，dim→dim*2
            {"in_dim": dim, "out_dim": dim * 2, "k": 3, "s": 2, "p": 1},
            # 第2层：5→3 (k=3, s=2, p=1)，dim*2→dim*4
            {"in_dim": dim * 2, "out_dim": dim * 2, "k": 3, "s": 2, "p": 1},
            # 第3层：3→3 (k=3, s=1, p=1)，dim*4→dim*8
            {"in_dim": dim * 2, "out_dim": dim * 2, "k": 3, "s": 1, "p": 1},
        ]
        self.decoder = nn.ModuleList([
            CTMB(**cfg) for cfg in dec_config
        ])

        # Classifier：打平3×3特征 → 全连接输出[B,2]
        # 最终特征尺寸：[B, dim*8, 3, 3] → 打平维度：dim*8*3*3
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 打平任意C×H×W为一维
            nn.Linear(dim * 2 * 3 * 3, 128),  # 适配3×3特征维度
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # 最终输出[B,2]
        )

    def forward(self, x1, x2):
        # Encoder：输出[B, dim, 9, 9]
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        fd = torch.abs(f1 - f2)  # [B, dim, 9, 9]

        # Decoder：循环更新f1/f2/fd
        for i, blk in enumerate(self.decoder):
            f1, f2, fd = blk(f1, f2, fd)

        # Classifier：打平 + 全连接输出
        out = self.classifier(fd)
        return out


# 使用示例（验证9→5→3路径）
if __name__ == "__main__":
    B, C, H, W = 8, 224, 9, 9  # 输入尺寸：8,224,9,9
    x1 = torch.randn(B, C, H, W).cuda()
    x2 = torch.randn(B, C, H, W).cuda()
    # 模型（可正常初始化）
    model = SST_MCDNet(in_ch=C, dim=32, enc_depth=3, dec_depth=3).cuda()
    y = model(x1, x2)
    print("\n最终输出尺寸：", y.shape)  # 输出 torch.Size([8, 2])