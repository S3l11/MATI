import torch
import torch.nn as nn
import torch.nn.functional as F

from mati import ModalityAwareTokenInteraction
from monai.networks.nets import UNETR, SwinUNETR


# =========================================================
# UNETR baseline 
# =========================================================
class UNETR_Baseline(UNETR):
    def forward(self, x_in):
        vit_out, hidden_states = self.vit(x_in)

        Z3 = hidden_states[2]
        Z6 = hidden_states[5]
        Z9 = hidden_states[8]

        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(self.proj_feat(Z3))
        enc3 = self.encoder3(self.proj_feat(Z6))
        enc4 = self.encoder4(self.proj_feat(Z9))

        vit_feat = self.proj_feat(vit_out)

        dec4 = self.decoder5(vit_feat, enc4)
        dec3 = self.decoder4(dec4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)

        return self.out(dec1)


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
            proj_type="perceptron",
            norm_name="instance",
            res_block=True,
        )

        self.mati_z6 = ModalityAwareTokenInteraction(hidden_size)
        self.mati_z9 = ModalityAwareTokenInteraction(hidden_size)

    def forward(self, x):
        vit_out, hidden_states = self.unetr.vit(x)

        Z3 = hidden_states[2]
        Z6 = hidden_states[5]
        Z9 = hidden_states[8]

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
            proj_type="perceptron",
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
# Model getter
# =========================================================
def get_model(model_name: str):
    name = model_name.lower()

    if name == "unetr":
        return UNETR_Baseline(
            in_channels=4,
            out_channels=3,
            img_size=(128, 128, 128),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="perceptron",
            norm_name="instance",
            res_block=True,
        )

    if name == "unetr_modality_aware_s":
        return UNETR_ModalityAware_S()

    if name == "unetr_modality_aware_e":
        return UNETR_ModalityAware_E()

    raise ValueError(f"Unknown model '{model_name}'")
