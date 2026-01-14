---
title: "Resy Bot"
date: 2024-01-01
description: "An async Rust bot for automating restaurant reservations on Resy. Uses tokio for concurrent requests, interacting with the Resy API to find available slots and book reservations automatically."
tags: ["Rust", "async", "tokio", "automation"]
---

A reservation bot built in Rust that automates the process of booking hard-to-get restaurant reservations on Resy. The bot queries available time slots, retrieves booking tokens, and submits reservations through the Resy API. Built with tokio for async HTTP requests, reqwest for API communication, and chrono for timezone-aware scheduling.
