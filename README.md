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
  <img src="assets/framework.png" width="90%">
</p>

## Repository structure
