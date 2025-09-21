---
title: 'Welcome to My Blog'
date: 2025-01-24
permalink: /posts/2025/01/welcome-to-my-blog/
tags:
  - introduction
  - personal
  - technology
---

Welcome to my blog! This is my first post where I'll share insights about technology, finance, and my experiences in the Engineering field.

## Why I Started This Blog

After working at Bridgewater Associates and completing my studies at MIT, I've accumulated experiences and insights that I believe are worth sharing. This blog will serve as a platform to discuss:

- Engineering techniques and methodologies
- Technology trends in quantitative finance
- Personal projects and research
- Lessons learned from my academic and professional journey

## What to Expect

### Technical Deep Dives

I'll be writing about various technical topics including:

1. **Quantitative Analysis**: Exploring statistical methods and their applications in finance
2. **Software Engineering**: Best practices for building robust financial systems
3. **Data Science**: Techniques for analyzing large financial datasets
4. **Machine Learning**: Applications of ML in investment strategies

### Code Examples

Here's a simple Python example of calculating moving averages, a fundamental concept in technical analysis:

```python
import pandas as pd
import numpy as np

def calculate_moving_average(prices, window):
    """
    Calculate simple moving average for a given window size
    
    Args:
        prices: List or array of price values
        window: Number of periods for the moving average
    
    Returns:
        Array of moving average values
    """
    return pd.Series(prices).rolling(window=window).mean()

# Example usage
stock_prices = [100, 102, 101, 105, 107, 110, 108, 112, 115, 113]
ma_5 = calculate_moving_average(stock_prices, 5)
print(f"5-day Moving Average: {ma_5.dropna().values}")
```

## Topics I Plan to Cover

- **Portfolio Optimization**: Modern portfolio theory and beyond
- **Risk Management**: Techniques for measuring and mitigating financial risk
- **System Design**: Building scalable financial applications
- **Career Insights**: Navigating the intersection of technology and finance
- **Academic Research**: Summaries and discussions of interesting papers

## Connect With Me

Feel free to reach out if you have questions or topics you'd like me to cover. You can find me on:

- GitHub: [@mmcmanus1](https://github.com/mmcmanus1)
- LinkedIn: [Matt McManus](https://www.linkedin.com/in/mattmcm)

## Final Thoughts

This blog represents the beginning of my journey in sharing knowledge and experiences with the broader community. Whether you're a student, professional, or simply curious about the intersection of technology and finance, I hope you'll find value in these posts.

Stay tuned for more content, and thank you for reading!

---

*Note: Views expressed here are my own and do not necessarily reflect those of my employer.*