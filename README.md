<div align="center">

# VLM-DR: A Domain-Generalizable Vision-Language Model for Diabetic Retinopathy Grading

**Doaa Dalaq, Hamzah Luqman**

[![Paper](https://img.shields.io/badge/IEEE%20Access-under%20review-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-orange)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

</div>

---

## Overview

VLM-DR is a CLIP-based framework for **ordinal diabetic retinopathy (DR) grading under domain shift**.
Each DR grade is represented by a learnable text prompt, and a fundus image is graded by its similarity to these grade representations.
The model is trained with three losses: **cross-modal correlation alignment (CORAL)** between image and text embeddings, a **text-to-image KL** loss, and an **ordinal rank** loss.

<p align="center">
  <img src="framework.png" width="90%">
</p>

## Installation

```bash
git clone https://github.com/DoaaDalaq/VLM-DR.git
cd VLM-DR
pip install -r requirements.txt
```

## Datasets

We use five datasets from **GDRBench**: APTOS, DeepDR, IDRiD, Messidor-2 and RLDR.
The images are **not included** in this repository. Please download them by following the GDRBench instructions:
👉 https://github.com/chehx/DGDR

## Training and evaluation

*Instructions will be added soon.*

## Acknowledgements

This code builds on [CLIP-DR](https://github.com/Qinkaiyu/CLIP-DR), [OrdinalCLIP](https://github.com/xk-huang/OrdinalCLIP) and [CLIP](https://github.com/openai/CLIP).
The datasets and splits follow [GDRBench](https://github.com/chehx/DGDR). We thank the authors for making their work available.

## Citation

*Will be added after publication.*
