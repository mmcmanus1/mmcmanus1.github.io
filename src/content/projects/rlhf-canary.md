---
title: "RLHF Canary"
tagline: "Regression detection for training pipelines"
date: 2025-01-01
description: "An automated test suite for SFT/DPO/PPO-style training pipelines that flags performance slowdowns, instability (NaNs/divergence), and correctness issues when code changes. It's meant to run in CI with tiers (smoke/perf/nightly) so regressions get caught before they burn days of GPU time. Net effect: safer iteration speed and fewer 'we broke training' surprises."
repoUrl: "https://github.com/mmcmanus1/rlhf-canary"
tags: ["Python", "PyTorch", "RLHF", "ML Ops", "Hugging Face"]
---

Regression detection tool for ML training pipelines that monitors tokens/second, memory usage, and numerical stability. Features automated root cause analysis, GitHub Actions integration, and YAML-based configuration with smoke, performance, and nightly test tiers.
