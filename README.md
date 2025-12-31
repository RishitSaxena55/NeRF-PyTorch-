<div align="center">

# 🌐 NeRF: Neural Radiance Fields from Scratch

**A from-scratch PyTorch implementation of Neural Radiance Fields for novel view synthesis**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)](LICENSE)

[Paper](https://www.matthewtancik.com/nerf) • [Architecture](#-architecture) • [Components](#-core-components) • [Results](#-results)

</div>

---

> 🎯 **A modular, from-scratch implementation** of Neural Radiance Fields designed for research experimentation and transparent understanding of differentiable volume rendering.

<div align="center">

![NeRF Pipeline](imgs/pipeline.jpg)

</div>

---

## 📌 About This Project

This repository contains a **modular from-scratch implementation** of Neural Radiance Fields (NeRF). Each component is implemented independently to enable research experimentation and easy modification:

- ✅ **Positional Encoding** — Fourier feature mapping for high-frequency detail
- ✅ **MLP Architecture** — The neural radiance field network
- ✅ **Ray Generation** — Camera ray casting and sampling
- ✅ **Hierarchical Sampling** — Coarse-to-fine importance sampling
- ✅ **Volume Rendering** — Differentiable alpha compositing

The `nerf-pytorch/` folder contains a reference implementation for comparison and demo purposes.

---

## 🏗️ Architecture

```
                        ┌─────────────────────────────────────┐
                        │         Input: Ray (o, d)           │
                        └─────────────────┬───────────────────┘
                                          │
                        ┌─────────────────▼───────────────────┐
                        │     Sample Points Along Ray         │
                        │         t₁, t₂, ..., tₙ             │
                        └─────────────────┬───────────────────┘
                                          │
         ┌────────────────────────────────┼────────────────────────────────┐
         │                                │                                │
         ▼                                ▼                                ▼
┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
│  Positional     │              │  Positional     │              │  Positional     │
│  Encoding γ(x)  │              │  Encoding γ(x)  │              │  Encoding γ(x)  │
└────────┬────────┘              └────────┬────────┘              └────────┬────────┘
         │                                │                                │
         ▼                                ▼                                ▼
┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
│   MLP Network   │              │   MLP Network   │              │   MLP Network   │
│  F: (x,d)→(c,σ) │              │  F: (x,d)→(c,σ) │              │  F: (x,d)→(c,σ) │
└────────┬────────┘              └────────┬────────┘              └────────┬────────┘
         │                                │                                │
         └────────────────────────────────┼────────────────────────────────┘
                                          │
                        ┌─────────────────▼───────────────────┐
                        │       Volume Rendering              │
                        │   C(r) = Σ Tᵢ(1-exp(-σᵢδᵢ))cᵢ      │
                        └─────────────────┬───────────────────┘
                                          │
                        ┌─────────────────▼───────────────────┐
                        │        Output: RGB Pixel            │
                        └─────────────────────────────────────┘
```

---

## 🧩 Core Components

### 1. Positional Encoding (`positional_encoding.py`)

Maps low-dimensional inputs to high-dimensional space using Fourier features, enabling the network to learn high-frequency scene details.

```python
γ(p) = [sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]
```

**Key Implementation Details:**
- Position encoding with L=10 frequencies (60D output)
- Direction encoding with L=4 frequencies (24D output)
- Enables learning of fine geometric structures and textures

---

### 2. NeRF MLP Network (`model.py`)

The core neural network that maps 3D position + viewing direction to RGB color and volume density.

```python
class NeRF(nn.Module):
    """
    Architecture:
    - 8 fully-connected layers (256 units each) with ReLU
    - Skip connection at layer 5 (concatenates input)
    - Density σ output after layer 8
    - Additional layer for view-dependent color
    """
```

**Network Design:**
- Input: Encoded position (60D) + Encoded direction (24D)
- 8 dense layers with skip connection at layer 5
- Output: RGB (3D) + Density σ (1D)

---

### 3. Ray Helpers (`ray_helpers.py`)

Utilities for generating camera rays from pixel coordinates.

```python
def get_rays(H, W, focal, c2w):
    """
    Generate rays for each pixel in the image.
    
    Returns:
        rays_o: Ray origins (H, W, 3)
        rays_d: Ray directions (H, W, 3)
    """
```

**Implemented Functions:**
- Camera intrinsic matrix handling
- World-to-camera transformations
- Ray origin and direction computation

---

### 4. Hierarchical Sampling (`hierarchical_sampling.py`)

Two-stage sampling strategy for efficient rendering:

```python
# Coarse sampling: Uniform samples along ray
t_coarse = stratified_sampling(t_near, t_far, N_coarse)

# Fine sampling: Importance sampling based on coarse weights
t_fine = importance_sampling(t_coarse, weights, N_fine)
```

**Benefits:**
- Concentrates samples in regions that contribute most to the final color
- Reduces computational waste in empty space
- Improves rendering quality with same sample budget

---

### 5. Volume Rendering (`volume_rendering.py`)

The differentiable rendering equation that converts density and color to pixel values.

```python
def volume_render(rgb, density, t_vals, rays_d):
    """
    Classic volume rendering with alpha compositing.
    
    C(r) = Σᵢ Tᵢ · (1 - exp(-σᵢ · δᵢ)) · cᵢ
    
    where:
        Tᵢ = exp(-Σⱼ₌₁ⁱ⁻¹ σⱼ · δⱼ)  (transmittance)
        δᵢ = tᵢ₊₁ - tᵢ              (distance between samples)
    """
```

**Key Features:**
- Fully differentiable for end-to-end training
- Computes depth maps as byproduct
- Handles view-dependent effects (reflections, specularity)

---

## 📁 Project Structure

```
NeRF-PyTorch-/
├── model.py                 # NeRF MLP architecture
├── positional_encoding.py   # Fourier feature encoding
├── ray_helpers.py           # Camera ray generation
├── hierarchical_sampling.py # Coarse-to-fine sampling
├── volume_rendering.py      # Differentiable rendering
├── requirements.txt         # Dependencies
├── nerf-pytorch/            # Reference implementation (for demo)
└── imgs/                    # Sample results
```

---

## 🎨 Capabilities

**Supported Datasets:**
| Dataset | Scenes | Resolution |
|---------|--------|------------|
| **NeRF Synthetic** | Lego, Chair, Drums, Ficus, Hotdog, Materials, Mic, Ship | 800×800 |
| **LLFF** | Fern, Flower, Fortress, Horns, Leaves, Orchids, Room, Trex | 1008×756 |

**Rendered Outputs:**
- 🖼️ Novel view RGB images
- 📏 Depth maps  
- 🎥 360° video synthesis

---

## 🔬 Technical Insights

### Why Positional Encoding?
Neural networks are biased towards learning low-frequency functions. By mapping inputs to a higher-dimensional space using Fourier features, we enable the network to capture high-frequency variations in geometry and appearance.

### Why Hierarchical Sampling?
Naive uniform sampling wastes computation in empty regions. By first doing coarse sampling to identify important regions, then focusing samples there, we achieve better quality with fewer total samples.

### Why View-Dependent Color?
Real materials exhibit view-dependent appearance (specular highlights, reflections). By conditioning the color output on viewing direction, NeRF can represent these non-Lambertian effects.

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/RishitSaxena55/NeRF-PyTorch-.git
cd NeRF-PyTorch-

# Install dependencies
pip install -r requirements.txt

# For demo with reference implementation
cd nerf-pytorch
python run_nerf.py --config configs/lego.txt
```

---

## 📚 References

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf) (Mildenhall et al., ECCV 2020)
- [Fourier Features Let Networks Learn High Frequency Functions](https://bmild.github.io/fourfeat/) (Tancik et al., NeurIPS 2020)

---

## 🤝 Connect

Built with 💜 by [Rishit Saxena](https://github.com/RishitSaxena55)

[![Portfolio](https://img.shields.io/badge/Portfolio-rishitsaxena55.github.io-8B5CF6?style=flat-square)](https://rishitsaxena55.github.io)
[![Email](https://img.shields.io/badge/Email-rishitsaxena55@gmail.com-EA4335?style=flat-square)](mailto:rishitsaxena55@gmail.com)

---

## 📝 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
