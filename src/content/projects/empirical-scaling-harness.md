---
title: "Empirical Scaling Harness"
tagline: "Scaling law research"
date: 2025-01-13
description: "A controlled experimentation framework for training families of transformer models across multiple sizes and fitting scaling curves to quantify performance vs compute. It's designed to test hypotheses like 'SwiGLU changes scaling behavior vs GeLU' using parameter-matched, apples-to-apples runs and validation on held-out sizes. Net effect: evidence-driven architecture and budget decisions instead of vibes."
repoUrl: "https://github.com/mmcmanus1/empirical-scaling-harness"
tags: ["PyTorch", "scaling laws", "transformers", "research"]
---

Research harness that trains transformer models at multiple scales (3M-85M parameters) with different activation functions (GeLU vs SwiGLU) to fit power-law relationships to empirical loss curves. Self-contained Jupyter notebooks optimized for Google Colab with TensorBoard logging and holdout validation for testing prediction accuracy.
