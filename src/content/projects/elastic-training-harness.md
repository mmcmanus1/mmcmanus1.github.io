---
title: "Elastic Training Harness"
tagline: "Fault-tolerant distributed training"
date: 2025-01-12
description: "A distributed PyTorch training harness built to survive node failures and cluster resizing without losing training progress or corrupting data order. It combines fast, layered checkpointing with deterministic data/token sharding so runs can resume cleanly even when the world size changes. Net effect: fewer dead runs, less wasted GPU spend, and faster recovery when hardware or networking flakes."
repoUrl: "https://github.com/mmcmanus1/elastic-training-harness"
tags: ["Python", "PyTorch", "distributed systems", "LLM training", "fault tolerance"]
---

Distributed training framework enabling resilient LLM training across multiple nodes through automatic failure detection, dynamic re-sharding, and multi-tier checkpointing (in-memory, NVMe, S3). Features learning rate scaling that automatically adjusts when batch size changes due to cluster size variations.
