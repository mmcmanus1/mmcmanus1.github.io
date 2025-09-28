---
title: 'PPO Deep Dive for LLM Post-Training (Revised)'
date: 2024-12-27
permalink: /posts/drafts/ppo-deep-dive-revised/
tags:
  - machine-learning
  - reinforcement-learning
  - llm
  - ppo
published: false
---

## Proximal Policy Optimization (PPO) — A Deep Dive

PPO is the workhorse policy-gradient algorithm used in post-training for LLMs. It's a first-order, sample-based relaxation of TRPO's KL-constrained objective. TRPO solves:

$$
\max_\theta\ \mathbb{E}_{\pi_{\theta_{\text{old}}}}\!\Big[\tfrac{1}{T}\!\sum_t r_t(\theta)\,A_t\Big]
\quad \text{s.t.}\quad
\mathbb{E}_{\pi_{\theta_{\text{old}}}}\!\big[\mathrm{KL}(\pi_{\theta_{\text{old}}}\|\pi_\theta)\big]\le \delta
$$

PPO replaces the hard constraint with clipping and/or a Lagrangian KL penalty. We'll derive the estimator we can compute with samples, then show how PPO turns that into a conservative, trainable surrogate.

### Notation
Define state $s_t = (q, o_{<t})$ (prompt + tokens so far) and action $a_t = o_t$ (next token). The autoregressive policy factorizes as $\pi_\theta(o_{1:T}\mid q) = \prod_{t=1}^T \pi_\theta(a_t\mid s_t)$. We use $\pi_{\text{old}}$ for the policy that generated the batch, and $\pi_{\text{ref}}$ for the fixed reference (e.g., SFT) used for KL regularization.

### The true objective

The quantity we care about is the expected reward of completions drawn from $\pi_\theta$:

$$
J(\theta)=\mathbb{E}_{q\sim P(Q),\,o_{1:T}\sim \pi_\theta(\cdot\mid q)}\big[R(q,o_{1:T})\big]
$$

*Why this matters.* Discrete tokens → can't backprop through sampling. Everything below turns this into a gradient we can estimate from rollouts.

### The gradient estimator (REINFORCE)

The log-derivative trick[^1] converts "gradient of an expectation" into an "expectation of a score":

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t\mid s_t)\,R(q,o_{1:T})\right]
$$

Subtracting a baseline $b(s_t)$ that doesn't depend on $a_t$ reduces variance without bias. Define the **advantage**:

$$
A_t = R(q,o_{1:T}) - b(s_t)
$$

With sequence-level rewards, we either broadcast $A_t \equiv R - b(q)$ to all tokens, or use GAE with a value head $V_\phi(s_t)$, or use GRPO-style group baselines.

### Importance sampling for off-policy reuse

We reuse rollouts from $\pi_{\theta_{\text{old}}}$ for multiple epochs. To correct for the distribution mismatch, multiply by per-token importance ratios:

$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
$$

This re-expresses the **on-policy** gradient under $\pi_{\theta_{\text{old}}}$ via importance weighting so we can reuse the batch across epochs:

$$
\begin{aligned}
\nabla J(\theta) &= \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A_t\right] \\
&= \mathbb{E}_{\pi_{\theta_{\text{old}}}}\left[\sum_t \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A_t\right] \\
&= \mathbb{E}_{\pi_{\theta_{\text{old}}}}\left[\sum_t r_t(\theta)\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A_t\right]
\end{aligned}
$$

*The problem:* As $\pi_\theta$ drifts from $\pi_{\theta_{\text{old}}}$, ratios explode → variance and bias explode; updates become unstable.

Rather than optimize the IS gradient $\mathbb{E}_{\pi_{\text{old}}}[r_t\,\nabla\log\pi_\theta\,A_t]$ directly, PPO defines a **scalar surrogate** $L^{\text{CLIP}}(\theta)=\mathbb{E}_{\pi_{\text{old}}}[\min(r_t A_t,\text{clip}(r_t) A_t)]$ whose gradient matches the policy-gradient *locally* but is stable under large $r_t$.

### PPO's clipping mechanism (trust region, cheaply)

PPO replaces $r_t A_t$ with $\min(r_t A_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) A_t)$ to block destructive steps:

$$
\mathcal{J}_{\mathrm{PPO}}(\theta)=
\mathbb{E}_{q\sim P(Q),\,o\sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid q)}
\left[
\frac{1}{|o|}\sum_{t=1}^{|o|}
\min\Big(r_t(\theta)A_t,\;
\operatorname{clip}\big(r_t(\theta),1-\varepsilon,1+\varepsilon\big)\,A_t\Big)
\right]
$$

* **Length normalization:** $\frac{1}{|o|}\sum_t$ prevents long sequences from dominating
  - Alternative: mask-sum without dividing by $|o|$; choose one and align your reward/KL so you don't double penalize long outputs
* **Clip band:** $[1-\varepsilon, 1+\varepsilon]$ (typically $\varepsilon=0.2$) caps how much credit/penalty each token gets
* **Pessimistic min:** Takes the worse of {unclipped, clipped} to stay conservative

The per-token behavior:
* If $A_t > 0$ and $r_t > 1+\varepsilon$: saturate positive updates (ceiling hit)
* If $A_t < 0$ and $r_t < 1-\varepsilon$: saturate negative updates (floor hit)
* Inside $[1-\varepsilon, 1+\varepsilon]$: normal REINFORCE gradient scaled by $r_t$

### Two guardrails: Clip vs KL

**Clip** constrains ratios at *sampled actions only*—it bounds how much you can change the probability of tokens that actually appeared in your batch. **KL penalty** tethers the *full distribution* over all possible tokens:

$$
\beta \cdot \mathbb{E}\big[\mathrm{KL}(\pi_\theta(·\mid s_t)\,\|\,\pi_{\text{ref}}(·\mid s_t))\big]
$$

The key difference: clipping acts locally (only on observed tokens), while KL acts globally (penalizing drift across the entire vocabulary). Together they implement complementary trust-region constraints.

Adaptive $\beta$: Update $\beta \leftarrow \beta \cdot \text{clip}(\mathrm{KL}/\text{target}, 0.5, 2.0)$ to maintain target KL/token.

### The composite loss

$$
\mathcal{L}(\theta,\phi)=
-\mathcal{J}_{\mathrm{PPO}}(\theta)
+ c_v\,\mathbb{E}\!\left[\underbrace{\min\!\big((V_\phi-\hat V)^2,\ (V_{\text{old}}+\text{clip}(V_\phi\!-\!V_{\text{old}},\pm\varepsilon_v)-\hat V)^2\big)}_{\text{clipped value loss}}\right]
- c_{\text{ent}}\,\mathbb{E}[H(\pi_\theta)]
+ \beta\,\mathbb{E}\big[\mathrm{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})\big]
$$

Actor (clipped surrogate) + critic MSE (often with value clipping to limit $|V_\phi-V_{\text{old}}|$ by $\varepsilon_v$) + entropy bonus + KL penalty.

### Practical algorithm

1. **Collect batch** under $\pi_{\theta_{\text{old}}}$. Cache $\log\pi_{\theta_{\text{old}}}(a_t|s_t)$, masks, EOS positions.

2. **Compute rewards** $R(q, o_{1:T})$. If sequence-level, broadcast to tokens or use GRPO group baselines.

3. **Compute advantages** $A_t$:
   - Broadcast: $A_t = R - b(q)$
   - GAE: Use $V_\phi(s_t)$ with $\gamma, \lambda$
   - GRPO: Group-relative (no critic needed)

4. **Normalize advantages** per batch: $A_t \leftarrow (A_t - \mu)/\sigma$
   - Optional: clip $\hat A_t$ to e.g. $[-5, 5]$ after normalization to tame outliers

5. **For K epochs** (typically 1-4) over minibatches:
   - Compute $\log\pi_\theta(a_t|s_t)$, then $r_t = \exp(\log\pi_\theta - \log\pi_{\theta_{\text{old}}})$
   - Actor loss: $-\frac{1}{|o|}\sum_t \min(r_t A_t, \text{clip}(r_t) A_t)$
   - Critic loss: $(V_\phi(s_t) - \hat V_t)^2$ (if using)
   - Add entropy bonus, KL penalty
   - Gradient clip (global norm), optimizer step
   - **Early stop** if empirical KL > 1.5-2× target or if fraction of samples with $|r_t-1|>\varepsilon$ exceeds threshold


*Where this is headed.* The critic is the fragile component with sparse rewards. GRPO keeps the trust-region machinery (clipping, KL) but replaces $A_t$ with group-relative comparisons—no value learning needed.

---

## Group Relative Policy Optimization (GRPO)

Now that we have laid the foundations of PPO, we can move onto GRPO. So, what is so different about GRPO? 

---

## Appendix: The Log-Derivative Trick

[^1]: The "log-derivative trick" (aka likelihood-ratio or score-function trick) converts gradients of expectations over discrete distributions into expectations we can estimate with samples.

### The problem

You want $\nabla_\theta \mathbb{E}_{x\sim p_\theta}[f(x)]$ but $x$ is discrete (can't reparameterize).

### The identity

For any differentiable $p_\theta(x)$:
$$\nabla_\theta p_\theta(x) = p_\theta(x)\,\nabla_\theta \log p_\theta(x)$$

### Apply inside expectation

$$
\begin{aligned}
\nabla_\theta \mathbb{E}[f(x)]
&= \sum_x \nabla_\theta p_\theta(x)\,f(x) \\
&= \sum_x p_\theta(x)\,\nabla_\theta\log p_\theta(x)\,f(x) \\
&= \mathbb{E}_{x\sim p_\theta}\big[f(x)\,\nabla_\theta\log p_\theta(x)\big]
\end{aligned}
$$

Now you can sample $x \sim p_\theta$ and compute the gradient as a Monte Carlo average.

### For autoregressive LLMs

With $\log p_\theta(o_{1:T}\mid q) = \sum_t \log \pi_\theta(o_t\mid q, o_{<t})$:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t f(o_{1:T})\,\nabla_\theta\log \pi_\theta(o_t\mid q,o_{<t})\right]
$$

### Variance reduction

Since $\mathbb{E}[\nabla\log p] = 0$, subtracting any baseline $b$ independent of the sampled action leaves the expectation unchanged but reduces variance. In RL, $f - b$ becomes the advantage $A_t$.

[1]: https://huggingface.co/blog/deep-rl-ppo "Proximal Policy Optimization (PPO)"