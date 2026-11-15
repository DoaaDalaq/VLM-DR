<div align="center">

# 👁️ VLM-DR

### A Domain-Generalizable Vision-Language Model for Diabetic Retinopathy Grading

**Doaa Dalaq · Hamzah Luqman**

King Fahd University of Petroleum and Minerals (KFUPM)

[![Paper](https://img.shields.io/badge/IEEE%20Access-Under%20Review-1f6feb?style=for-the-badge)](#)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Overview

Diabetic retinopathy (DR) grading models often fail when tested on images from **new hospitals, cameras or populations**.
**VLM-DR** tackles this with a vision–language approach: each DR grade is represented by a **learnable text prompt**, and a fundus image is graded by its **similarity** to these grade representations in the CLIP embedding space.

<p align="center">
  <img src="framework.png" width="95%">
</p>

## ✨ Highlights

- 🧠 **Vision–language grading:** fundus images are matched with learnable prompts for the five DR grades.
- 🔗 **Cross-modal correlation alignment (CORAL):** aligns the structure of the image and text embeddings.
- 📈 **Ordinal-aware training:** a rank loss keeps the grades in their natural order (No DR → PDR).
- 🌍 **Domain generalization:** trained on source datasets and tested on an **unseen** dataset (leave-one-domain-out).
- 🏥 **Five public benchmarks:** APTOS, DeepDR, IDRiD, Messidor-2 and RLDR.

## 🧩 Method at a glance

| Component | Role |
|---|---|
| **Image encoder** (CLIP ResNet-50, fine-tuned) | Extracts the fundus image features |
| **Perceptual adaptation** | Projects the image features into the shared space |
| **Learnable prompts** (10 context tokens + 1 grade token) | Represent the five DR grades |
| **Text encoder** (CLIP, frozen) | Encodes the grade prompts |
| **CORAL + KL + Rank losses** | Align image and text and enforce the grade order |

## 🗂️ Datasets

We use five datasets from the **GDRBench** benchmark. The images are **not included** in this repository.

| Dataset | Download |
|---|---|
| APTOS | via [GDRBench](https://github.com/chehx/DGDR) |
| DeepDR | via [GDRBench](https://github.com/chehx/DGDR) |
| IDRiD | via [GDRBench](https://github.com/chehx/DGDR) |
| Messidor-2 | via [GDRBench](https://github.com/chehx/DGDR) |
| RLDR | via [GDRBench](https://github.com/chehx/DGDR) |

## ⚙️ Installation

```bash
git clone https://github.com/DoaaDalaq/VLM-DR.git
cd VLM-DR
pip install -r requirements.txt
```

## 🚀 Training and evaluation

> 🛠️ Code and instructions will be released soon.

## 🙏 Acknowledgements

This code builds on [CLIP-DR](https://github.com/Qinkaiyu/CLIP-DR), [OrdinalCLIP](https://github.com/xk-huang/OrdinalCLIP) and [CLIP](https://github.com/openai/CLIP).
The datasets follow [GDRBench](https://github.com/chehx/DGDR). We thank the authors for making their work available.

## 📖 Citation

```bibtex
@article{dalaq2026vlmdr,
  title   = {VLM-DR: A Domain-Generalizable Vision-Language Model for Diabetic Retinopathy Grading},
  author  = {Dalaq, Doaa and Luqman, Hamzah},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

## 📬 Contact

For questions, please open an [issue](https://github.com/DoaaDalaq/VLM-DR/issues).
