---
title: "Torch-Velocity"
tagline: "Adaptive speculative decoding"
date: 2025-01-01
description: "An inference engine that speeds up LLM generation by using a small draft model to propose multiple tokens and a larger target model to verify them in parallel, preserving correctness with rejection sampling. It adapts the lookahead length based on real-time acceptance/entropy and manages KV cache rollback efficiently. Net effect: faster throughput and lower serving cost without changing model weights."
repoUrl: "https://github.com/mmcmanus1/Torch-Velocity"
tags: ["PyTorch", "LLM", "speculative decoding", "transformers"]
---

Speculative decoding implementation that uses a smaller draft model (distilgpt2) to generate candidate tokens verified by a larger target model (gpt2-medium) in parallel. Features adaptive lookahead that dynamically adjusts based on token acceptance rates and a pre-allocated KV cache with O(1) rollback.
