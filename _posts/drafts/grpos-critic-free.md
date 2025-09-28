


# GRPO vs. PPO for LLM RL: the critic free trick that works 

It’s become standard to finish an LLM with some form of RL: you define what “good” looks like via a reward model or verifiable signal, and you push the policy toward higher-reward behavior. PPO has been the default for this stage because it’s simple and stable enough to use at scale. The catch is the critic. In LLM RL, rewards are often sparse and delayed; value learning can oscillate, inject bias into the advantages, and waste a lot of capacity. GRPO attacks that bottleneck directly: keep the PPO-style trust region, drop the critic, and estimate advantages by comparing a group of completions for the same prompt. This post focuses on post-training and, specifically, why GRPO’s critic-free baseline often behaves better than PPO in practice.

Since 2023, “post-training” has gotten more granular: SFT to get the model in the right neighborhood, then RL to push on what actually matters under your reward(s). We’ll stay in that second stage. I’ll start from the math of PPO, spell out where instability creeps in for LLMs, and then show how GRPO swaps the learned value function for a group-relative estimator without giving up the trust-region safeguards.


## Proximal Policy Optimization (PPO) - A Deep Dive

PPO is the workhorse policy-gradient algorithm used in post-training for LLMs. Let's start by being precise about what we're trying to optimize.

### The True Objective

$J(\theta)$ is the **true objective** you're trying to maximize with RL: the **expected reward** of sequences the model generates under policy $\pi_\theta$. Formally, with prompts $q \sim P(Q)$ and an autoregressive policy $\pi_\theta(o_{1:T}\mid q) = \prod_{t=1}^T \pi_\theta(o_t\mid q, o_{<t})$:

$$
J(\theta) = \mathbb{E}_{q\sim P(Q), o_{1:T}\sim \pi_\theta(\cdot\mid q)}\big[R(q,o_{1:T})\big]
$$

Breaking this down:
- $R(q, o_{1:T})$ is whatever scalar reward you define for the completion: a preference-model score, pass/fail correctness, BLEU/ROUGE, etc. If you have token-level shaping, you can think of $R = \sum_{t=1}^T r_t^{\text{env}}$
- The "state" at step $t$ is $s_t = (q, o_{<t})$ (the prompt plus tokens generated so far)
- Actions are tokens $o_t$
- The policy factorizes autoregressively: each token depends only on the prompt and previous tokens

**Important:** PPO does **not** optimize $J(\theta)$ directly from fresh on-policy samples every step. Instead, it maximizes a **clipped surrogate** $\mathcal{J}_{\text{PPO}}(\theta)$ computed on data from $\pi_{\theta_{\text{old}}}$ using importance ratios and clipping. That surrogate is a *biased but stable* proxy for improving $J(\theta)$. We'll see exactly how this works below.

### The Gradient Estimator

Next is the Gradient Estimator. As explained above we have $J(\theta)$ and we want to find the gradient of $J(\theta)$ with respect to $\theta$. All the Gradient Estimator is, is the 

The basic gradient estimator comes from the score-function trick (REINFORCE). Because $\pi_\theta$ factorizes over tokens, $\pi_\theta(o_{1:T}\mid q)=\prod_{t=1}^T \pi_\theta(o_t\mid q,o_{<t})$, and:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=1}^T \nabla_\theta \log \pi_\theta(o_t\mid q,o_{<t}) A_t\right]
$$

Here $A_t$ is an **advantage** signal with zero mean under the sampling policy at step $t$. In LLM RL it is common to have only an end-of-sequence reward; then one simply broadcasts a constant advantage to every token, $A_t\equiv R-b(q)$, where $b(q)$ is a baseline that reduces variance without biasing the gradient.

If you train a value head $V_\phi$, you can construct lower-variance advantages with generalized advantage estimation (GAE):

$$
\delta_t = u_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t), \qquad
A_t = \sum_{l\ge 0}(\gamma\lambda)^l\delta_{t+l}
$$

where $u_t$ is a per-step shaping reward (often zero for text until the end). The key condition that keeps the gradient unbiased is that the baseline used inside $A_t$ must not depend on the sampled action $o_t$; for any such baseline $b$:

$$
\mathbb{E}_{o_t\sim \pi_\theta}\big[\nabla_\theta \log \pi_\theta(o_t\mid\cdot) b(\cdot)\big] = b \nabla_\theta \sum_{o_t}\pi_\theta(o_t\mid\cdot) = b \nabla_\theta 1 = 0
$$

### Importance Sampling and Off-Policy Correction

The expression above still assumes you draw fresh samples from $\pi_\theta$ at every update. In practice you collect data with a **frozen** behavior policy $\pi_{\theta_{\text{old}}}$ and reuse those samples for several optimization epochs. To evaluate expectations under $\pi_\theta$ using data from $\pi_{\theta_{\text{old}}}$, you introduce a per-token **importance ratio**:

$$
r_t(\theta) = \frac{\pi_\theta(o_t\mid q,o_{<t})}{\pi_{\theta_{\text{old}}}(o_t\mid q,o_{<t})} = \exp\Big(\log\pi_\theta-\log\pi_{\theta_{\text{old}}}\Big)
$$

If you simply maximized the **unclipped** surrogate $\mathbb{E}_{\pi_{\theta_{\text{old}}}}\big[\sum_t r_t(\theta)A_t\big]$, you would recover the correct direction near $\theta_{\text{old}}$, but the ratios can explode as the policy drifts, leading to high variance and brittle updates.

### The PPO Clipping Mechanism

PPO fixes this by imposing a **soft trust region** through ratio clipping. The token-level, length-normalized PPO surrogate is:

$$
\mathcal{J}_{\mathrm{PPO}}(\theta) = \mathbb{E}_{q \sim P(Q), o \sim \pi_{\theta_{\mathrm{old}}}(\cdot \mid q)}
\left[
\frac{1}{|o|}\sum_{t=1}^{|o|}
\min\Big(
r_t(\theta)A_t,
\operatorname{clip}\big(r_t(\theta),1-\varepsilon,1+\varepsilon\big) A_t
\Big)
\right]
$$

Every symbol here is doing specific work:
- The outer expectation averages over prompts and sequences sampled from the **old** policy; in code this is your minibatch, reused for a few epochs
- The inner average $\frac{1}{|o|}\sum_{t=1}^{|o|}$ neutralizes **length bias** so longer completions don't dominate updates purely by having more tokens
- The ratio $r_t(\theta)$ reweights each token so the gradient points in the direction that would improve reward **under the new policy**, even though your data came from the old one
- The clipping operator $\operatorname{clip}(x,1-\varepsilon,1+\varepsilon)=\min(\max(x,1-\varepsilon),1+\varepsilon)$ with a small $\varepsilon$ (typically 0.1–0.2 for text) enforces the soft trust region

### Understanding the Per-Token Surrogate

The interplay of the ratio and the clip is captured by the per-token surrogate:

$$
L_t(\theta) = \min\Big(r_t(\theta)A_t, \text{clip}(r_t(\theta),1-\varepsilon,1+\varepsilon) A_t\Big)
$$

Condition on the sign of $A_t$:
- If $A_t>0$, then $L_t(\theta)=\min(r_t, 1+\varepsilon) A_t$. As soon as $r_t>1+\varepsilon$, trying to increase the probability of $o_t$ further does not increase the objective; you have hit the ceiling for that token
- If $A_t<0$, then $L_t(\theta)=\max(r_t, 1-\varepsilon) A_t$. As soon as $r_t<1-\varepsilon$, decreasing the probability further no longer reduces the objective; you have reached the floor

This **pessimistic** construction biases you toward conservative steps on purpose: you sacrifice exactness in exchange for update stability.

The gradient behavior mirrors that intuition. Inside the clip band $r_t\in[1-\varepsilon,1+\varepsilon]$, the active branch is the unclipped term and:

$$
\nabla_\theta\big[r_t(\theta)A_t\big] = r_t(\theta)A_t \nabla_\theta \log \pi_\theta(o_t\mid q,o_{<t})
$$

i.e., the REINFORCE direction scaled by the ratio. Outside the band, the active branch is constant with respect to $r_t$ (equal to $(1\pm\varepsilon)A_t$), so the gradient **stops encouraging** further movement that would push the ratio beyond the trust-region edge.

### Advantages and Normalization

With outcome-only rewards, a simple and effective choice is $A_t=R-b(q)$ broadcast across tokens. The baseline recenters the signal and, because it does not depend on $o_t$, does not bias the gradient; $\mathbb{E}[\nabla\log\pi \cdot b]=0$. Many implementations also **normalize** advantages per minibatch to zero mean and unit variance; this just rescales the effective step without changing the expected direction and helps tame gradient scale.

If you do use a value head $V_\phi$, you form $A_t$ via GAE. This typically reduces variance further when you have dense or well-shaped rewards, but in LLM RL with sparse end-of-sequence rewards it is also where instability appears: a misfit critic contaminates $A_t$ and can make the whole update drift.

### KL Regularization and Trust Regions

Clipping acts at the sampled actions; you also want to constrain the **full distribution** to remain near a reference. In practice you add a **KL safeguard** against a reference policy $\pi_{\text{ref}}$ (often the SFT checkpoint). One way is **reward shaping**, replacing $R$ by $R_{\text{raw}}-\beta \cdot \mathrm{KL}(\pi_\theta|\pi_{\text{ref}})$ so the KL penalty is folded into $A_t$. Another is an **explicit penalty** added to the loss, $+\beta \cdot \mathbb{E}[\mathrm{KL}(\pi_\theta|\pi_{\text{ref}})]$, with $\beta$ adapted during training to hit a target KL per token.

This has a clean trust-region interpretation: TRPO would solve a constrained problem that maximizes a first-order surrogate subject to a KL-ball constraint; PPO approximates that with clipping, and the explicit KL penalty plays the role of a Lagrange multiplier that keeps the policy close in function space, complementing clipping's control at sampled points.

### The Complete PPO Objective

What you actually minimize in code is a composite objective:

$$
\mathcal{L}(\theta,\phi) = -\mathcal{J}_{\mathrm{PPO}}(\theta)
+ c_v \mathbb{E}\big[(V_\phi-\hat V)^2\big]
- c_{\text{ent}} \mathbb{E}[H(\pi_\theta)]
+ \beta \mathbb{E}\big[\mathrm{KL}(\pi_\theta|\pi_{\text{ref}})\big]
$$

The components are:
- The first term is the policy surrogate we just unpacked
- The value term trains the critic if you use one; in a critic-free setup it vanishes
- The entropy bonus discourages premature peaking of token distributions and can help maintain diversity early in training
- The KL penalty implements the global trust region against the reference model

### Implementation and Training Loop

Training typically proceeds by:
1. Collecting a batch under $\pi_{\theta_{\text{old}}}$
2. Computing rewards (sequence-level for text)
3. Forming $A_t$ (broadcast or GAE)
4. Normalizing advantages
5. Computing $r_t$ from the frozen and current logits
6. Taking a few epochs of updates while monitoring empirical KL/token

You either stop early when KL exceeds a target or adapt $\beta$ to keep it in range.

### True Objective vs Surrogate

It's important to understand that $J(\theta)$ is the **true objective** you're trying to maximize with RL: the **expected reward** of sequences the model generates under policy $\pi_\theta$. PPO does **not** optimize $J(\theta)$ directly from fresh on-policy samples every step. Instead it maximizes the **clipped surrogate** $\mathcal{J}_{\text{PPO}}(\theta)$ computed on data from $\pi_{\theta_{\text{old}}}$ using importance ratios and clipping. That surrogate is a *biased but stable* proxy for improving $J(\theta)$.

If you also include a KL term to a reference policy, the effective target becomes a **regularized objective**:

$$
J_\beta(\theta) = J(\theta) - \beta \mathbb{E}\big[\mathrm{KL}(\pi_\theta|\pi_{\text{ref}})\big]
$$

which is the "keep me near SFT while increasing reward" version of the same thing.

So: $J(\theta)$ = the expected reward you actually care about; $\mathcal{J}_{\text{PPO}}(\theta)$ = the conservative, trainable stand-in you use to move $J(\theta)$ up without blowing up the policy.

### The Critic Problem in LLM RL

That is the full mathematical plumbing behind the PPO equation. The success of PPO in LLM RL comes from exactly these choices: an on-policy surrogate with importance correction, a conservative **min-with-clip** that caps per-token gains and losses, sequence-length normalization to avoid bias, baseline-centered (or value-based) advantages, and a KL tether to a reference.

The main pain point in practice is not the surrogate; it is the **critic** when rewards are sparse. GRPO keeps the same trust-region story but **removes the critic** and replaces $A_t$ with a per-prompt, **group-relative** estimator—eliminating the value-learning failure mode while retaining PPO's conservative updates.




## Group Relative Policy Optimization (GRPO)




<!-- ## Proximal Policy Optimization (PPO)

Start with the object we actually care about: maximize expected reward of completions sampled from your model. For a prompt $q$ from your training data and a token sequence $o_{1:T}$ sampled from a policy $\pi_\theta$, the goal is:

$$
J(\theta) = \mathbb{E}_{q \sim P(Q),\, o \sim \pi_\theta(\cdot \mid q)} \left[ R(q, o_{1:T}) \right]
$$

Then using the score-function (REINFORCE) trick the gradient of this objective is:

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_{t=1}^T \nabla_\theta \log \pi_\theta\left(o_t \mid q, o_{<t}\right) A_t\right],
$$

where $A_t$ is the \textbf{advantage} term with zero mean under the sampling policy at step $t$. 


It could be beneficial to walk through the math and pros/cons of GRPO & PPO and why GRPO is "better" mainly b/c of its group based average estimation


1. Intro 
- Why RL is centrla to fine-tuning LLMs 
- Standard PPO relies on a critic, which is unstable and costly at scale. 
- GRPO simplies RL for LLMs by replacing the critic with group-based average estimation 


Before we start understanding grpo and ppo its important to take a step and to really understand the math behind grpo and ppo 


Since 2023, a lot has evolved in language models—we've gone from using just pre-training and post-training to something a bit more granular, as shown in the image below.

![LLM Recipe History](../assets/images/llm-recipe-history.jpeg)

This blog is meant to purely focus on Post-Training and, even more specifically, I will focus on the critic-free trick that works in GRPO vs. PPO.


The most widely used algorithm in reinforcement learning is PPO or Proximal Policy Optimization which was originally introduced by Schulman et al.
(2017) and in recent work on math reasoning, Group Relative Policy Optimization (GRPO) was proposed as a PPO varient tailored for LLMs. Now, like 
PPO, GRPO optimzies a policy to maximize rewards by comparing groups of sample responses. 
In this post I will derive the core equtions of PPO and GRPO and compare their properties 
(sample efficinecy, variance, complexity and scaling) as well as their trust-region safeguarding methods like ratio clipping and KL divergence penalty.  -->
