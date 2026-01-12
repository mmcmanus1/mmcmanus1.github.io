---
title: "Elastic Training Harness"
date: 2025-01-12
description: "A fault-tolerant distributed training system for LLM development that automatically manages node failures and workload rebalancing, achieving sub-30 second recovery times."
repoUrl: "https://github.com/mmcmanus1/elastic-training-harness"
tags: ["Python", "PyTorch", "distributed systems", "LLM training", "fault tolerance"]
---

Distributed training framework enabling resilient LLM training across multiple nodes through automatic failure detection, dynamic re-sharding, and multi-tier checkpointing (in-memory, NVMe, S3). Features learning rate scaling that automatically adjusts when batch size changes due to cluster size variations.
