# **RLHF with GRPO** (Reinforcement Learning from Human Feedback using Group Relative Policy Optimization) 👇

## 🚀 Extremly Important Terms for ( RL + LLMs )

| Term | Definition |
|------|-------------|
| **State (sₜ)** | The current text context — prompt plus tokens generated so far. |
| **Action (aₜ)** | The next token the model chooses to emit. |
| **State–action (sₜ, aₜ)** | A specific context and the specific next token chosen in it. |
| **Policy (πθ)** | The LLM’s probability distribution over next tokens given the context. |
| **Value (V(sₜ))** | Predicted total future preference/reward from this context if we keep sampling from πθ. |
| **Reward (rₜ)** | A scalar score from a reward model or rule (only at the end of the answer). |
| **Returns (Gₜ)** | Discounted sum of future rewards for the rest of the generation. |
| **Q-value (Q(sₜ, aₜ))** | Expected return if we emit that token now and then continue with πθ. |
| **Advantage (A = Q – V)** | How much better that token is than the model’s average continuation at this context. |
| **KL divergence (Dₖₗ(πθ ∥ π_ref))** | Penalty measuring how far the current token distribution drifts from a frozen SFT/reference model at the same context. |
| **Policy vs Ref** | **Policy:** the RL-tuned, updating LLM. **Ref:** the frozen SFT model used for safety and KL regularization. |

## 🧠 What is GRPO?

GRPO stands for **Group Relative Policy Optimization**.
It’s a variant of policy-optimization used in RLHF for large language models (LLMs), proposed to replace or augment standard methods like Proximal Policy Optimization (PPO).

**Key characteristics:**

* It **eliminates the need for a learned “value function” (critic)**, reducing complexity and memory requirements.
* Instead, it uses **group-based sampling**: for each prompt/state, multiple answers (actions) are generated, each scored by a reward model. The *average reward* across the group becomes a baseline, and each answer’s advantage is its reward minus that baseline.
* The policy update then favors answers that are *better than the group average* and disfavours those that are worse.

## 🔍 Why GRPO? (How it improves over standard PPO in RLHF)

| Challenge in PPO for LLMs                                                    | How GRPO addresses it                                                                                                                |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Need to learn a separate value network (critic) → high memory & compute cost | GRPO removes the critic and uses the group average as a baseline instead.                                              |
| High variance in reward signals, unstable advantage estimates                | By generating multiple outputs per prompt and using the group mean, GRPO stabilizes the baseline and reduces variance. |
| Large models + long outputs → huge cost for RL fine‐tuning                   | GRPO cuts resource needs, making RLHF more accessible even for smaller setups.                                       |

## 🛠️ How GRPO Works — Step by Step

1. **Prompt sampling:** Choose a batch of prompts (states) from your dataset.
2. **Generate a group of responses per prompt:** For each prompt, the policy model generates multiple candidate answers (e.g., 8, 16, 64).
3. **Reward scoring:** A reward model (or feedback function) assigns a reward to each response.
4. **Compute baseline:** For each prompt, compute the *average reward* of its group of responses:
    Rˉ = 1/k * ∑​(1→N) Rκ
5. **Compute advantage for each response:**
   Aκ ​= Rκ ​− Rˉ
   So responses above average get positive advantage; below average get negative.
6. **Policy update:** Use these advantages in a surrogate objective (similar to PPO) to update the policy. Often also a KL‐penalty or regularization term ensures the updated policy doesn’t drift too far from a reference policy.
7. **Repeat:** Continue with new batches of prompts/responses.

## 🎯 GRPO in the RLHF Pipeline

Here is how it fits into RLHF:

* **Pretraining:** The base language model learns from large text corpora.
* **Supervised Fine-Tuning (SFT):** The model is finetuned on prompt-response pairs so it follows instructions.
* **Reward Modeling (RM):** A separate model learns to score responses based on human preferences or other criteria.
* **GRPO (instead of PPO):** The SFT model becomes the *policy*, uses the RM for rewards, and is finetuned via the GRPO algorithm to produce higher-reward (more human-aligned) outputs.

## 🧾 Summary Table

| Component                     | Role in GRPO                                         | Why it matters                                     |
| ----------------------------- | ---------------------------------------------------- | -------------------------------------------------- |
| Policy model ( πθ​ )   | The LLM being optimized                              | We want it to produce better answers.              |
| Reward model                  | Scores each generated response                       | Provides the “good vs bad” signal.                 |
| Group of responses per prompt | Multiple answers generated per prompt                | Enables using group baseline instead of value‐net. |
| Baseline = group mean reward  | Used to compute advantage                            | Simplifies advantage estimation.                   |
| Advantage ( A = R − Rˉ ) | Drives updates: positive→increase, negative→decrease | Focuses on relative improvement.                   |
| KL / regularization           | Keeps policy from diverging too far                  | Ensures stability and safety.                      |

## 📌 In One Sentence

> **GRPO is a policy-optimization algorithm for RLHF that generates multiple responses per prompt, uses the group average as a baseline instead of a learned critic, and updates the policy to favour responses that score above that baseline — making RLHF more efficient and stable for large language models.**
