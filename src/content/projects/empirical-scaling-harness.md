---
title: "Empirical Scaling Harness"
date: 2025-01-13
description: "A research toolkit for investigating power-law scaling behavior in transformer language models, automating experiments comparing activation functions across multiple parameter scales."
repoUrl: "https://github.com/mmcmanus1/empirical-scaling-harness"
tags: ["PyTorch", "scaling laws", "transformers", "research"]
---

Research harness that trains transformer models at multiple scales (3M-85M parameters) with different activation functions (GeLU vs SwiGLU) to fit power-law relationships to empirical loss curves. Self-contained Jupyter notebooks optimized for Google Colab with TensorBoard logging and holdout validation for testing prediction accuracy.
