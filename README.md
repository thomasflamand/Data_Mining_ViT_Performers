# Improving Efficient Vision Transformers with Relative Positional Encodings (RPEs)

**IEOR E4540: Data Mining — Project 3**  
Thomas Flamand (UNI: tf2636) — Columbia University, Fall 2025  
Instructor: Dr. Krzysztof Choromanski

---

## Overview

This project investigates whether **Relative Positional Encodings (RPEs)** can recover the accuracy lost when replacing standard softmax attention with efficient linear attention in Vision Transformers.

We build a lightweight **Performer-ViT** and augment it with three RPE mechanisms, evaluating all variants on **MNIST** and **CIFAR-10** across accuracy, training stability, and inference time.

---

## Architecture

- **Base model**: lightweight ViT (depth=4, embed_dim=64, 4 heads, MLP ratio=4×)
- **Patch size**: 4×4 → 64 patch tokens + 1 [CLS] token = sequence length 65
- **Attention variants**:
  - Full softmax attention (baseline)
  - Performer FAVOR+ (positive random features, m=64)
  - Performer-ReLU (deterministic ReLU feature map)

---

## Relative Positional Encodings

| RPE | Method | Key property |
|---|---|---|
| **Classic** (Luo et al., 2021) | Learned additive bias with Toeplitz structure | Applied via FFT convolution, O(N log N) |
| **Circulant-STRING** (Schenck et al., 2025) | Skew-symmetric circulant rotation matrices | 2D-aware, FFT-diagonalizable |
| **RoPE** (Su et al., 2024) | Multiplicative sinusoidal rotation of Q and K | No extra parameters, norm-preserving |

---

## Results

### Final Test Accuracy

| Model | MNIST | CIFAR-10 |
|---|---|---|
| ViT (Full Attention) | 87.56% | 56.93% |
| Performer-FAVOR+ (no RPE) | 86.62% | 55.24% |
| Performer-FAVOR+ + Classic RPE | 86.71% | 54.39% |
| Performer-FAVOR+ + RoPE | **97.44%** | 62.70% |
| Performer-FAVOR+ + circulant-STRING | 95.76% | **67.30%** |
| Performer-ReLU (no RPE) | 88.99% | 54.49% |
| Performer-ReLU + Classic RPE | 75.66% | 30.60% |
| Performer-ReLU + RoPE | **96.97%** | 62.02% |
| Performer-ReLU + circulant-STRING | 95.85% | **69.43%** |

> MNIST: 5 training epochs. CIFAR-10: 20 training epochs. All models trained from scratch with Adam (lr=1e-3, batch size=128), no weight decay.

### Key Findings

- **RoPE and circulant-STRING consistently outperform the full-attention ViT** on both datasets when combined with Performer attention.
- **Classic RPE underperforms** — and can even hurt performance (especially Performer-ReLU + Classic on CIFAR-10: 30.60%).
- **RPEs are critical for stabilizing ReLU-based Performer** on CIFAR-10, where training without positional information leads to severe instability.
- **Inference time overhead is modest**: RPE-equipped Performers remain competitive with the full-attention ViT in wall-clock time at this sequence length (65 tokens).

---

## Repository Structure

```
.
├── performer_vit_RPE.ipynb       # Main notebook: all Performer + RPE variants
├── performer_vit_noRPE.ipynb     # Baseline: Performer without RPE
├── Plot_Results.ipynb            # Figures from experimental results
├── Results/                      # CSV logs of training runs
└── Flamand_Thomas_F25_DataMining.pdf  # Full project report
```

---

## Setup

The notebooks were developed on **Google Colab** with GPU acceleration. File paths reference Google Drive mounts and may need to be updated for local execution.

**Dependencies** (standard Colab environment):
```
torch
torchvision
pandas
matplotlib
```

**Datasets** are downloaded automatically via `torchvision.datasets` (MNIST, CIFAR-10).

---

## Reproducing the Main Experiments

Open `performer_vit_RPE.ipynb` and run all cells. The training loop iterates over:

```python
datasets     = ["mnist", "cifar10"]
attn_types   = ["favor+", "relu"]
rpe_types    = ["none", "rope", "classic", "string"]
```

Results are logged to a timestamped CSV in `Results/`.

To reproduce the extended training run (60 epochs, AdamW + cosine annealing + warmup) used for the accuracy-maximization experiment, run the final section of the notebook labeled *"Special case used to maximize accuracy"*.

---

## References

1. Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
2. Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words.* arXiv:2010.11929.
3. Choromanski et al. (2021). *Rethinking Attention with Performers.* ICLR.
4. Luo et al. (2021). *Stable, Fast and Accurate: Kernelized Attention with Relative Positional Encoding.* NeurIPS.
5. Schenck et al. (2025). *Learning the RoPEs: Better 2D and 3D Position Encodings with STRING.* ICML. arXiv:2502.02562.
6. Su et al. (2024). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing, 568:127063.
