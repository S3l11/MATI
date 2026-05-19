# Structured Modality-Aware Token Interaction for Multimodal Medical Imaging

This repository contains the PyTorch implementation of **Modality-Aware Token Interaction (MATI)** and its UNETR-based instantiations, **ModaUNETR<sup>S</sup>** and **ModaUNETR<sup>E</sup>** (where <sup>S</sup> and <sup>E</sup> denote skip-level and in-encoder MATI interaction, respectively), as described in:

> Structured Modality-Aware Token Interaction for Multimodal Medical Imaging  
> IJCAI-ECAI 2026 

---

## Overview

MATI introduces an explicit modality-aware inductive bias inside the token embedding space while preserving a single-stream transformer architecture. It operates on token sequences of shape `[B, N, C]` by:

1. splitting the embedding dimension into modality-aligned subspaces;
2. applying independent intra-subspace self-attention;
3. computing global modality summaries by token-wise average pooling;
4. predicting modality mixing weights with a lightweight gating network;
5. broadcasting the mixed representation back to all modality subspaces through residual fusion.

This repository includes:

- **MATI** module;
- **UNETR** backbone;
- **ModaUNETR<sup>S</sup>**, where MATI refines the UNETR skip-token representations `Z6` and `Z9`;
- **ModaUNETR<sup>E</sup>**, where MATI is injected inside the ViT encoder after transformer blocks 3 and 6;
- **SwinUNETR** transformer-based baseline;
- **3D UNet++** convolutional-based baseline.

---

## Repository structure

```text
.
├── mati.py          
├── models.py        
├── requirements.txt 
└── README.md
```

- Use `mati.py` if you want to integrate MATI into your own architecture(s).
- Use `models.py` if you want to directly use our proposed models.

---

## Method-code alignment

### MATI

Implemented by:

```python
class ModalityAwareTokenInteraction(nn.Module)
```

### ModaUNETR<sup>S</sup>

Implemented by:

```python
class UNETR_ModalityAware_S(nn.Module)
```

The implementation assumes the UNETR ViT encoder returns:

```python
hidden_states = [Z3, Z6, Z9, Z12]
```

MATI is applied only to:

```python
Z6 = MATI(Z6)
Z9 = MATI(Z9)
```

The ViT encoder itself is not modified. The refined skip tokens are then projected and passed to the standard UNETR convolutional decoder.

### ModaUNETR<sup>E</sup>

Implemented by:

```python
class UNETR_ModalityAware_E(nn.Module)
```

MATI is injected into the ViT token stream after transformer blocks 3 and 6, corresponding to 0-indexed block indices, and skip tokens are extracted at:

```python
{2, 5, 8, 11} # Z3, Z6, Z9, Z12
```

---

## Installation

Create and activate a local or virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The code is intended for GPU execution. CUDA-enabled PyTorch should be installed according to the CUDA version available on your system.

---

## Reproducibility notes

For full reproducibility, use the dependency versions in `requirements.txt` and keep the MONAI version fixed, since the implementation relies on the UNETR ViT hidden-state convention:

```python
hidden_states = [Z3, Z6, Z9, Z12]
```

---

## Important notes

- MATI assumes a fixed number of input modalities, defaulting to `num_modalities=4`.
- The embedding dimension must be divisible by the number of modalities.
- ModaUNETR<sup>S</sup> preserves the UNETR ViT encoder and applies MATI only to skip-token representations.
- ModaUNETR<sup>E</sup> modifies the ViT forward pass by injecting MATI directly into the running token stream.
- No modality-specific encoders, additional token streams, or cross-attention streams are introduced.

---

## Citation

If you use MATI or its instantiations (ModaUNETR<sup>S</sup> and ModaUNETR<sup>E</sup>) in your work, please cite the following paper:

```bibtex
@inproceedings{tomassini2026mati,
  title     = {Structured Modality-Aware Token Interaction for Multimodal Medical Imaging},
  author    = {Tomassini, Selene and Chaudhry, Hafiza Ayesha Hoor and Galdelli, Alessandro and Giorgini, Paolo},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence and the Twenty-Eighth European Conference on Artificial Intelligence},
  year      = {2026}
}
```
