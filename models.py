import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNETR, SwinUNETR


# =========================================================
# Modality-Aware Token Interaction (MATI)
# =========================================================
class ModalityAwareTokenInteraction(nn.Module):
    def __init__(self, embed_dim, num_modalities=4, num_heads=4):
        super().__init__()
        assert embed_dim % num_modalities == 0

        self.num_modalities = num_modalities
        self.sub_dim = embed_dim // num_modalities

        self.norms = nn.ModuleList([
            nn.LayerNorm(self.sub_dim)
            for _ in range(num_modalities)
        ])

        self.attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.sub_dim,
                num_heads=num_heads,
                batch_first=True,
            )
            for _ in range(num_modalities)
        ])

        self.mix_proj = nn.Sequential(
            nn.Linear(self.sub_dim * num_modalities, num_modalities),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, N, C]
        chunks = torch.chunk(x, self.num_modalities, dim=-1)
        updated = []

        # Intra-subspace refinement
        for i in range(self.num_modalities):
            xi = chunks[i]
            xi_norm = self.norms[i](xi)
            attn_out, _ = self.attns[i](xi_norm, xi_norm, xi_norm)
            updated.append(xi + attn_out)

        # Inter-subspace gated mixing
        summaries = [u.mean(dim=1) for u in updated]          # M x [B, C/M]
        summary_cat = torch.cat(summaries, dim=-1)            # [B, C]
        mix_weights = self.mix_proj(summary_cat)              # [B, M]
        mix_weights = mix_weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, M, 1]

        stacked = torch.stack(updated, dim=2)                 # [B, N, M, C/M]
        mixed = (stacked * mix_weights).sum(dim=2)            # [B, N, C/M]

        refined = [u + mixed for u in updated]
        return torch.cat(refined, dim=-1)                     # [B, N, C]


# =========================================================
# ModaUNETR^S
# =========================================================
class UNETR_ModalityAware_S(nn.Module):
    def __init__(
        self,
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        feature_size=16,
        hidden_size=768,
        num_heads=12,
        mlp_dim=3072,
    ):
        super().__init__()

        self.unetr = UNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
        )

        self.mati_z6 = ModalityAwareTokenInteraction(hidden_size)
        self.mati_z9 = ModalityAwareTokenInteraction(hidden_size)

    def forward(self, x):
        vit_out, hidden_states = self.unetr.vit(x)

        # According to UNETR/MONAI: hidden_states = [Z3, Z6, Z9, Z12]
        Z3, Z6, Z9, _Z12 = hidden_states

        Z6 = self.mati_z6(Z6)
        Z9 = self.mati_z9(Z9)

        enc1 = self.unetr.encoder1(x)
        enc2 = self.unetr.encoder2(self.unetr.proj_feat(Z3))
        enc3 = self.unetr.encoder3(self.unetr.proj_feat(Z6))
        enc4 = self.unetr.encoder4(self.unetr.proj_feat(Z9))

        vit_feat = self.unetr.proj_feat(vit_out)

        dec4 = self.unetr.decoder5(vit_feat, enc4)
        dec3 = self.unetr.decoder4(dec4, enc3)
        dec2 = self.unetr.decoder3(dec3, enc2)
        dec1 = self.unetr.decoder2(dec2, enc1)

        return self.unetr.out(dec1)


# =========================================================
# ModaUNETR^E
# =========================================================
class UNETR_ModalityAware_E(nn.Module):
    def __init__(
        self,
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        feature_size=16,
        hidden_size=768,
        num_heads=12,
        mlp_dim=3072,
        mati_block_indices=(2, 5),  # 0-indexed: after blocks 3 and 6
    ):
        super().__init__()

        self.unetr = UNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
        )

        self.mati_block_indices = mati_block_indices
        self.mati_blocks = nn.ModuleDict({
            str(i): ModalityAwareTokenInteraction(hidden_size)
            for i in mati_block_indices
        })

    def forward(self, x_in):
        vit = self.unetr.vit

        x = vit.patch_embedding(x_in)
        hidden_states = []

        for i, blk in enumerate(vit.blocks):
            x = blk(x)

            if i in self.mati_block_indices:
                x = self.mati_blocks[str(i)](x)

            if i in {2, 5, 8, 11}:
                hidden_states.append(x)

        x = vit.norm(x)

        Z3, Z6, Z9, _Z12 = hidden_states

        enc1 = self.unetr.encoder1(x_in)
        enc2 = self.unetr.encoder2(self.unetr.proj_feat(Z3))
        enc3 = self.unetr.encoder3(self.unetr.proj_feat(Z6))
        enc4 = self.unetr.encoder4(self.unetr.proj_feat(Z9))

        vit_feat = self.unetr.proj_feat(x)

        dec4 = self.unetr.decoder5(vit_feat, enc4)
        dec3 = self.unetr.decoder4(dec4, enc3)
        dec2 = self.unetr.decoder3(dec3, enc2)
        dec1 = self.unetr.decoder2(dec2, enc1)

        return self.unetr.out(dec1)


# =========================================================
# SwinUNETR
# =========================================================
def build_swin_unetr():
    return SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True,
    )


# =========================================================
# UNet++ (3D)
# =========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetPP(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_channels=32):
        super().__init__()

        self.conv00 = ConvBlock(in_channels, base_channels)
        self.pool0 = nn.MaxPool3d(2)

        self.conv10 = ConvBlock(base_channels, base_channels * 2)
        self.pool1 = nn.MaxPool3d(2)

        self.conv20 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool2 = nn.MaxPool3d(2)

        self.conv30 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool3 = nn.MaxPool3d(2)

        self.conv40 = ConvBlock(base_channels * 8, base_channels * 16)

        self.up31 = ConvBlock(base_channels * 8 + base_channels * 16, base_channels * 8)
        self.up21 = ConvBlock(base_channels * 4 + base_channels * 8, base_channels * 4)
        self.up11 = ConvBlock(base_channels * 2 + base_channels * 4, base_channels * 2)
        self.up01 = ConvBlock(base_channels + base_channels * 2, base_channels)

        self.final = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool0(x00))
        x20 = self.conv20(self.pool1(x10))
        x30 = self.conv30(self.pool2(x20))
        x40 = self.conv40(self.pool3(x30))

        x31 = self.up31(torch.cat([
            F.interpolate(x40, scale_factor=2, mode="trilinear", align_corners=True),
            x30,
        ], dim=1))

        x21 = self.up21(torch.cat([
            F.interpolate(x31, scale_factor=2, mode="trilinear", align_corners=True),
            x20,
        ], dim=1))

        x11 = self.up11(torch.cat([
            F.interpolate(x21, scale_factor=2, mode="trilinear", align_corners=True),
            x10,
        ], dim=1))

        x01 = self.up01(torch.cat([
            F.interpolate(x11, scale_factor=2, mode="trilinear", align_corners=True),
            x00,
        ], dim=1))

        return self.final(x01)


# =========================================================
# Model getter
# =========================================================
def get_model(model_name: str):
    name = model_name.lower()

    if name == "unetr":
        return UNETR(
            in_channels=4,
            out_channels=3,
            img_size=(128, 128, 128),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            norm_name="instance",
            res_block=True,
        )

    if name == "unetpp":
        return UNetPP()

    if name == "swin_unetr":
        return build_swin_unetr()

    if name == "unetr_modality_aware_s":
        return UNETR_ModalityAware_S()

    if name == "unetr_modality_aware_e":
        return UNETR_ModalityAware_E()

    raise ValueError(f"Unknown model '{model_name}'")
