


# GRPO VS. PPO for LLM RL: the critic free trick that works 

It could be beneficial to walk through the math and pros/cons of GRPO & PPO and why GRPO is "better" mainly b/c of its group based average estimation


1. Intro 
- Why RL is centrla to fine-tuning LLMs 
- Standard PPO relies on a critic, which is unstable and costly at scale. 
- GRPO simplies RL for LLMs by replacing the critic with group-based average estimation 


Before we start understanding grpo and ppo its important to take a step and to really understand the math behind grpo and ppo 