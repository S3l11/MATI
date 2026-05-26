import torch
import torch.nn as nn


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
