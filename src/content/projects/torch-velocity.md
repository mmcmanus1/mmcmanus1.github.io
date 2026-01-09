---
title: "Torch-Velocity"
date: 2025-01-01
description: "An implementation of speculative decoding with adaptive lookahead mechanisms for LLM inference optimization, achieving 1.5-2.5x speedups on transformer-based models."
repoUrl: "https://github.com/mmcmanus1/Torch-Velocity"
tags: ["PyTorch", "LLM", "speculative decoding", "transformers"]
---

Speculative decoding implementation that uses a smaller draft model (distilgpt2) to generate candidate tokens verified by a larger target model (gpt2-medium) in parallel. Features adaptive lookahead that dynamically adjusts based on token acceptance rates and a pre-allocated KV cache with O(1) rollback.
