---
title: 'PPO Deep Dive for LLM Post-Training'
date: 2024-12-27
permalink: /posts/drafts/ppo-deep-dive/
tags:
  - machine-learning
  - reinforcement-learning
  - llm
published: false
---

## Proximal Policy Optimization (PPO) — A Deep Dive

PPO is the workhorse policy-gradient algorithm used in post-training for LLMs. We'll be explicit about the target we actually care about, derive the estimator we can compute with samples, then show how PPO turns that into a conservative, trainable surrogate.

### The true objective

Let $q \sim P(Q)$ be a prompt. The autoregressive policy factorizes as

$$
\pi_\theta(o_{1:T}\mid q)=\prod_{t=1}^T \pi_\theta(o_t\mid q,o_{<t})
$$

The quantity we care about is the expected reward of completions drawn from $\pi_\theta$:

$$
J(\theta)=\mathbb{E}_{q\sim P(Q),\,o_{1:T}\sim \pi_\theta(\cdot\mid q)}\big[R(q,o_{1:T})\big]
$$

*Why this matters.* This is the "real" objective: push the model distribution toward sequences that score higher under your reward (preference score, pass/fail correctness, etc.). Everything below is about getting a **usable gradient** of $J$ from samples.

### The gradient estimator (REINFORCE, made explicit)

*Why we need a gradient estimator.* $J(\theta)=\mathbb{E}_{o\sim\pi_\theta}[R(q,o_{1:T})]$ is an expectation over **discrete** samples from $\pi_\theta$. You can't backpropagate through the sampling step (a categorical draw is piecewise-constant in $\theta$), and there's no exact reparameterization for tokens. The likelihood-ratio (log-derivative) trick[^1] converts "gradient of an expectation" into an **expectation of a score term** you *can* compute with rollouts.

Start from the definition and apply the log-derivative trick ($\nabla_\theta p(x)=p(x)\nabla_\theta\log p(x)$):

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{\pi_\theta}\left[R(q,o_{1:T})\,\nabla_\theta\log \pi_\theta(o_{1:T}\mid q)\right]
$$

Because the policy is autoregressive,

$$
\log \pi_\theta(o_{1:T}\mid q)=\sum_{t=1}^T \log \pi_\theta(o_t\mid q,o_{<t})
$$

so the gradient decomposes token-wise:

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_{t=1}^T \nabla_\theta \log \pi_\theta(o_t\mid q,o_{<t})\,R(q,o_{1:T})\right]
$$

Insert a **baseline** $b(q,o_{<t})$ that does not depend on the sampled token $o_t$; this leaves the expectation unchanged and reduces variance. Define the **advantage**

$$
A_t \triangleq R(q,o_{1:T})-b(q,o_{<t})
\quad\text{(or use GAE if you attach a value head)}
$$

to get the on-policy policy-gradient:

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_{t=1}^T \nabla_\theta \log \pi_\theta(o_t\mid q,o_{<t})\,A_t\right]
$$

*Why we're doing this.* Sampling is discrete; you can't backprop through token draws. The likelihood-ratio form gives an **unbiased Monte-Carlo gradient** using only rewards and log-probs. Broadcasting $A_t\equiv R-b(q)$ is common with outcome-only rewards; GAE reduces variance when a stable value head exists.

### Importance sampling and where PPO starts to differ

*Why we need importance sampling.* We don't regenerate fresh rollouts after every micro-update—the model is huge, and sampling is expensive—so we **reuse a batch** collected under a **frozen** policy $\pi_{\theta_{\text{old}}}$ for several optimization epochs. If we ignored that mismatch and treated those samples as if they came from $\pi_\theta$, we'd bias the gradient. **Importance sampling** fixes this: multiply each token's contribution by the ratio so the expectation is *as if* it were taken under $\pi_\theta$. We use **per-token ratios** (not a product over the whole sequence) to avoid explosive variance and to match the TRPO/PPO **first-order** surrogate that keeps the **state distribution** fixed while adjusting the **action probabilities** at those states.

To point the gradient at $\pi_\theta$ while using data from $\pi_{\theta_{\text{old}}}$, introduce the per-token importance ratio

$$
r_t(\theta)=\frac{\pi_\theta(o_t\mid q,o_{<t})}{\pi_{\theta_{\text{old}}}(o_t\mid q,o_{<t})}
$$

*Where we're going.* The naïve surrogate $\mathbb{E}_{\pi_{\theta_{\text{old}}}}[\sum_t r_t A_t]$ has the right direction near $\theta_{\text{old}}$, but $r_t$ can explode as the policy moves. PPO's entire "proximal" idea is to keep those steps **small but productive**.

### PPO's clipping mechanism (the soft trust region)

PPO replaces the unstable surrogate with a **pessimistic, clipped** one:

$$
\mathcal{J}_{\mathrm{PPO}}(\theta)=
\mathbb{E}_{q\sim P(Q),\,o\sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid q)}
\left[
\frac{1}{|o|}\sum_{t=1}^{|o|}
\min\Big(r_t(\theta)A_t,\;
\operatorname{clip}\big(r_t(\theta),1-\varepsilon,1+\varepsilon\big)\,A_t\Big)
\right]
$$

Read it literally. The outer expectation averages over prompts and trajectories **from the old policy** (the batch you have). The inner mean $\tfrac{1}{|o|}\sum_t$ neutralizes length bias so long completions don't dominate by having more tokens. The ratio says "how much more/less likely is the sampled token under the new policy," and the clip $[1-\varepsilon,\,1+\varepsilon]$ (commonly $\varepsilon=0.2$, i.e., $[0.8,1.2]$) enforces a soft trust region. ([Hugging Face][1])

It's helpful to isolate the per-token piece

$$
L_t(\theta)=\min\Big(r_t A_t,\;\text{clip}(r_t,1-\varepsilon,1+\varepsilon)A_t\Big)
$$

If $A_t>0$, $L_t=\min(r_t,1+\varepsilon)A_t$: once $r_t>1+\varepsilon$, you **stop getting more credit** for increasing that token's prob—ceiling hit. If $A_t<0$, $L_t=\max(r_t,1-\varepsilon)A_t$: once $r_t<1-\varepsilon$, further suppression **doesn't pay**—floor hit. Inside the band, the gradient is the usual REINFORCE direction scaled by $r_tA_t$; outside, it saturates. This is exactly the "six cases" operational picture popularized in tutorials. ([Hugging Face][1])

*Why this matters.* The min-with-clip biases you **toward conservative steps** on purpose. You give up some local optimality in exchange for lower variance and fewer catastrophic jumps—what makes PPO tractable for big LLMs.

### KL regularization and the trust-region view

Clipping constrains ratios at sampled actions. You typically also tether the **full distribution** to a reference $\pi_{\text{ref}}$ (often the SFT checkpoint) with a KL guardrail. You can fold it into the reward,

$$
R \leftarrow R_{\text{raw}}-\beta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid s_t)\,\|\,\pi_{\text{ref}}(\cdot\mid s_t)\big)
$$

or add an explicit penalty $+\beta\,\mathbb{E}[\mathrm{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})]$ and adapt $\beta$ to hit a target KL/token. Think of this as a Lagrangian relaxation of a TRPO-style constraint; clipping is the **local** proxy, KL is the **global** guardrail.

### The composite loss you actually optimize

In practice you minimize

$$
\mathcal{L}(\theta,\phi)=
-\mathcal{J}_{\mathrm{PPO}}(\theta)
+ c_v\,\mathbb{E}\big[(V_\phi-\hat V)^2\big]
- c_{\text{ent}}\,\mathbb{E}[H(\pi_\theta)]
+ \beta\,\mathbb{E}\big[\mathrm{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})\big]
$$

i.e., clipped surrogate (actor) + value loss (critic) − entropy bonus + KL regularizer—the same actor-critic-entropy recipe shown in most PPO intros. ([Hugging Face][1])

*A quick roadmap.* Collect a batch under $\pi_{\theta_{\text{old}}}$; compute rewards; form $A_t$ (broadcast or GAE) and normalize them; compute $r_t$ from frozen vs. current logits; take a few epochs of updates while monitoring empirical KL/token; stop early or adapt $\beta$ if KL overshoots.

*Where this is headed.* The only fragile component here is the **critic** when rewards are sparse. GRPO keeps everything you see above—the surrogate, clipping, KL—but **removes the critic** and builds $A_t$ from **group-relative** comparisons of multiple completions of the same prompt. That swap preserves PPO's safety rails while deleting the variance/instability vector that often bites in LLM RL.



---

## Appendix: The Log-Derivative Trick

[^1]: The "log-derivative trick" (aka likelihood-ratio or score-function trick) is a way to differentiate an **expectation over a distribution that depends on parameters**—even when the random variable is **discrete** (so you can't reparameterize).

### The problem

You want the gradient of

$$
J(\theta)=\mathbb{E}_{x\sim p_\theta(x)}\big[f(x)\big]
= \sum_x p_\theta(x)\,f(x)\quad\text{(discrete)}
\text{ or }
\int p_\theta(x)\,f(x)\,dx\quad\text{(continuous)}
$$

You can sample $x\sim p_\theta$, and you can evaluate $f(x)$, but you **can't** backprop through the sampling step (e.g., categorical tokens).

### The identity

For any positive, differentiable density/mass function $p_\theta(x)$,

$$
\nabla_\theta p_\theta(x) = p_\theta(x)\,\nabla_\theta \log p_\theta(x)
$$

This is just the chain rule: $\nabla \log p = (\nabla p)/p\Rightarrow \nabla p = p\,\nabla \log p$.

### Apply it inside the expectation

Discrete case (integral is analogous):

$$
\begin{aligned}
\nabla_\theta J(\theta)
&= \sum_x \nabla_\theta p_\theta(x)\,f(x) \\
&= \sum_x p_\theta(x)\,\nabla_\theta\log p_\theta(x)\,f(x) \\
&= \mathbb{E}_{x\sim p_\theta}\big[f(x)\,\nabla_\theta\log p_\theta(x)\big]
\end{aligned}
$$

That is the **REINFORCE / score-function estimator**.

This step is valid under mild conditions: $p_\theta$ is differentiable in $\theta$, and its support doesn't change with $\theta$ (so you can swap $\nabla$ and $\sum/\int$).

### Why it helps

You've turned "gradient of an expectation" into "expectation of something computable":

* You can **sample** $x\sim p_\theta$
* You can **compute** $f(x)$ and $\nabla_\theta\log p_\theta(x)$ (the **score**)
* Monte-Carlo estimate:

$$
\nabla_\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^N f(x^{(i)})\,\nabla_\theta\log p_\theta(x^{(i)}),\quad x^{(i)}\sim p_\theta
$$

For LLMs, $x$ is a **sequence** $o_{1:T}$. Using factorization

$$
\log p_\theta(o_{1:T}\mid q)=\sum_{t=1}^T \log \pi_\theta(o_t\mid q,o_{<t})
$$

you get the token-wise form used in policy gradients:

$$
\nabla_\theta J(\theta)
=\mathbb{E}\Big[\sum_{t=1}^T f(o_{1:T})\,\nabla_\theta\log \pi_\theta(o_t\mid q,o_{<t})\Big]
$$

### Baselines (variance reduction) and "advantage"

Key facts:

* $\mathbb{E}_{x\sim p_\theta}[\nabla_\theta\log p_\theta(x)] = \nabla_\theta \mathbb{E}[1]=0$
* Therefore, for any baseline $b$ **independent of the sampled action**,

$$
\mathbb{E}\big[(f(x)-b)\,\nabla_\theta\log p_\theta(x)\big]
= \mathbb{E}\big[f(x)\,\nabla_\theta\log p_\theta(x)\big]
$$

So subtracting $b$ **does not change the expectation** but can dramatically reduce variance. In RL we set $f=R$ and $b\approx V$, so $f-b$ is the **advantage**.

### Intuition (covariance view)

Because $\mathbb{E}[\nabla\log p]=0$,

$$
\nabla_\theta J(\theta)=\operatorname{Cov}\big(f(x),\,\nabla_\theta\log p_\theta(x)\big)
$$

* If a sample has **above-baseline** $f(x)$, we increase its log-prob (positive correlation)
* If **below-baseline**, we decrease it

That's exactly the RL update: push up probabilities of actions that yielded higher return.

### Why not reparameterization?

For **continuous** $x$, you often write $x=g_\theta(\varepsilon)$ (pathwise trick) and differentiate through $g$. For **discrete** tokens there is no exact smooth $g_\theta$. The score-function estimator is **unbiased** and general; that's why we use it for categorical LLM outputs. (Gumbel-Softmax etc. are biased/relaxed alternatives.)

### Tiny categorical example

Let $x\in\{1,\dots,K\}$ with softmax $\pi_\theta(x)=\frac{e^{z_x}}{\sum_j e^{z_j}}$. Then

$$
\nabla_\theta \log \pi_\theta(x)=\nabla_\theta z_x - \sum_j \pi_\theta(j)\,\nabla_\theta z_j
$$

which any deep net can compute via backprop. The gradient estimator becomes a Monte-Carlo average of $f(x)$ times that score.

### In PPO specifically

Start from the score-function form (above), then:

* Replace on-policy expectation by **importance sampling** with the ratio $r=\pi_\theta/\pi_{\text{old}}$
* **Clip** $r$ to a band $[1-\varepsilon,1+\varepsilon]$ to form a **stable surrogate**
* Optionally add a **KL penalty** to keep the full distribution near a reference

All of PPO sits **on top of** this log-derivative foundation.
