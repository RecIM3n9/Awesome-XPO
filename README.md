# Awesome XPO

A structured and comprehensive reading list & resource tracker for Reinforcement Learning and Preference Optimization (PPO, DPO, GRPO, etc.) in Large Language Models.

---

## 📚 Core Literature
#### 0. [Proximal policy optimization algorithms](http://arxiv.org/abs/1707.06347) & [Training language models to follow instructions with human feedback](http://arxiv.org/abs/2203.02155)

* **Algorithm:** PPO
* **Venue/Year:** NeurIPS 2017 & NeurIPS 2022
* **Reference Code:** [stable-baseline3](https://github.com/DLR-RM/stable-baselines3) | [Hugging Face TRL (PPOTrainer)](https://github.com/huggingface/trl) | [OpenRLHF (PPO)](https://github.com/OpenRLHF/OpenRLHF)
* **Local PDF:**  [Schulman et al. - 2017 - Proximal policy optimization algorithms.pdf](<papers/Schulman et al. - 2017 - Proximal policy optimization algorithms.pdf>) & [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback.pdf](<papers/Ouyang et al. - 2022 - Training language models to follow instructions with human feedback.pdf>) 
* **Rating:** 
> **Description:** 
> * **Core Intuition:** The foundational Actor-Critic framework for RLHF. It optimizes a policy against a trained Reward Model (RM) while anchoring updates to a Reference Model to prevent reward hacking.
> * **Engineering Traits:** Offers a high performance ceiling and enables online exploration. However, it is highly sensitive to hyperparameters and extremely memory-intensive, requiring four concurrent models (Actor, Critic, Reward, Reference).
> * **Mathematical Formulation:** Utilizes a clipped surrogate objective to constrain policy updates:
> 
> $$L^{\text{CLIP}}(\phi) = \hat{\mathbb{E}}_t \left[ \min(r_t(\phi)\hat{A}_t, \text{clip}(r_t(\phi), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$

#### 1. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
* **Algorithm:** DPO
* **Venue/Year:** NeurIPS 2023
* **Reference Code:** [Official Repo](https://github.com/eric-mitchell/direct-preference-optimization) | [Hugging Face TRL (DPOTrainer)](https://huggingface.co/docs/trl/main/en/dpo_trainer)
* **Local PDF:**  [Rafailov et al. - 2024 - Direct preference optimization your language model is secretly a reward model.pdf](<papers/Rafailov et al. - 2024 - Direct preference optimization your language model is secretly a reward model.pdf>) 
* **Rating:** 
> **Description:** 
> * **Core Intuition:** Leverages the Bradley-Terry model to mathematically demonstrate that the LLM implicitly acts as a reward model. It bypasses explicit reward modeling and RL entirely, framing alignment as a binary classification problem.
> * **Engineering Traits:** Highly stable and memory-efficient (requires only the Policy and Reference models). It serves as the standard baseline for aligning Small Language Models (SLMs).
> * **Mathematical Formulation:** Optimizes the negative log-likelihood of preference pairs:
> 
> $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma \left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

#### 2. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) & [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
* **Algorithm:** GRPO
* **Venue/Year:** ArXiv 2024 & ArXiv 2025
* **Reference Code:** [OpenRLHF (GRPO Implementation)](https://github.com/OpenRLHF/OpenRLHF) | [Unsloth (GRPO Acceleration)](https://github.com/unslothai/unsloth)
* **Local PDF:**  [Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf](<papers/Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf>) & [DeepSeek-AI et al. - 2025 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.pdf](<papers/DeepSeek-AI et al. - 2025 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.pdf>) 
* **Rating:** 
> **Description:** 
> * **Core Intuition:** Eliminates the Critic (Value Network) by sampling a group of outputs for a single prompt. It estimates advantages via group-relative Z-score normalization based on rule-based rewards.
> * **Engineering Traits:** Significantly reduces memory and communication overhead in distributed setups, serving as the infrastructural foundation for Reinforcement Learning for Verifiable Reasoning (RLVR) in reasoning models.
> * **Mathematical Formulation:** Group-relative advantage estimation:
> 
> $$\hat{A}_{i,j} = \frac{r_{i,j} - \text{mean}(r_i)}{\text{std}(r_i)}$$

#### 3. [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
* **Algorithm:** KTO
* **Venue/Year:** ICML 2024
* **Reference Code:** [ContextualAI/HALOs (Official)](https://github.com/ContextualAI/HALOs) | [Hugging Face TRL (KTOTrainer)](https://huggingface.co/docs/trl/main/en/kto_trainer)
* **Local PDF:**  [Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization.pdf](<papers/Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization.pdf>) 
* **Rating:** 
> **Description:**
> * **Core Intuition:** Grounded in Kahneman & Tversky's Prospect Theory, formulating alignment without requiring paired preference data. It optimizes the policy using purely independent "thumbs-up/down" feedback.
> * **Engineering Traits:** Drastically lowers data annotation costs, enabling the utilization of highly imbalanced, real-world implicit feedback datasets.

#### 4. [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)
* **Algorithm:** ORPO
* **Venue/Year:** ArXiv 2024
* **Reference Code:** [xfact/ORPO (Official)](https://github.com/xfact/ORPO) | [Hugging Face TRL (ORPOTrainer)](https://huggingface.co/docs/trl/main/en/orpo_trainer)
* **Local PDF:**  [Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model.pdf](<papers/Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model.pdf>) 
* **Rating:** 
> **Description:** 
> * **Core Intuition:** A monolithic approach unifying SFT and alignment. It discards both the Reward and Reference models by appending an odds-ratio-based relative penalty directly to the standard SFT loss.
> * **Engineering Traits:** Extremely lightweight. Optimal for resource-constrained environments, allowing rapid iterative fine-tuning on SLMs using methods like LoRA.
> * **Mathematical Formulation:**  
>
> $$\mathcal{L}_{\text{ORPO}} = \mathbb{E}_{(x, y_w, y_l)} \left[ \mathcal{L}_{\text{SFT}}(y_w) + \lambda \cdot \log \sigma \left( \log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)} \right) \right]$$

#### 5. [A general theoretical paradigm to understand learning from human preferences](http://arxiv.org/abs/2310.12036)
* **Algorithm:** $\Psi $PO / IPO
* **Venue/Year:** ArXiv 2024
* **Reference Code:** [Hugging Face TRL (Supported via DPOTrainer `loss_type="ipo"`)](https://huggingface.co/docs/trl/main/en/dpo_trainer)
* **Local PDF:**  [Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model.pdf](<papers/Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model.pdf>) 
* **Rating:** 
> **Description:** 
> * **Core Intuition:** Addresses DPO's theoretical vulnerability to overfitting. IPO frames preference learning as a root-finding problem rather than maximum likelihood, applying stronger regularization.
> * **Engineering Traits:** Prevents policy degradation and out-of-distribution generation issues inherent in unregularized Bradley-Terry models.
> * **Mathematical Formulation:** Replaces logistic loss with a squared error formulation:
> 
> $$\mathcal{L}_{\text{IPO}} = \mathbb{E} \left[ \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \frac{\tau}{2} \right)^2 \right]$$

#### 6. [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
* **Algorithm:** SimPO
* **Venue/Year:** ArXiv 2024
* **Reference Code:** [princeton-nlp/SimPO (Official)](https://github.com/princeton-nlp/SimPO) | [LLaMA-Factory (Supported natively)](https://github.com/hiyouga/LLaMA-Factory)
* **Local PDF:**  [Meng et al. - 2024 - SimPO simple preference optimization with a reference-free reward.pdf](<papers/Meng et al. - 2024 - SimPO simple preference optimization with a reference-free reward.pdf>) 
* **Rating:** 
> **Description:** 
> * **Core Intuition:** Utilizes length-normalized log probabilities as implicit rewards and introduces a target reward margin, entirely discarding the Reference model.
> * **Engineering Traits:** Halves memory requirements compared to DPO while effectively mitigating the verbosity bias (the tendency for models to generate excessively long outputs to game the reward).
> * **Mathematical Formulation:** Implicit reward defined as $r_{\text{SimPO}}(x,y) = \frac{\beta}{|y|} \log \pi_\theta(y|x)$.

#### 7. [Preference Ranking Optimization for Human Alignment](https://arxiv.org/abs/2306.17492)
* **Algorithm:** PRO
* **Venue/Year:** AAAI 2024
* **Reference Code:** [sail-sg/PRO (Official)](https://github.com/sail-sg/PRO)
* **Local PDF:**  [Song et al. - 2024 - Preference Ranking Optimization for Human Alignment.pdf](<papers/Song et al. - 2024 - Preference Ranking Optimization for Human Alignment.pdf>) 
* **Rating:** 
> **Description:** 
> * **Core Intuition:** Extends binary preference learning to listwise ranking using contrastive InfoNCE objectives, enhancing the model's sensitivity to fine-grained quality differences across multiple candidates.

#### 8. [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](https://arxiv.org/abs/2401.08417)
* **Algorithm:** CPO
* **Venue/Year:** ICML 2024
* **Reference Code:** [AILab-CVC/CPO (Official)](https://github.com/AILab-CVC/CPO) | [Hugging Face TRL (CPOTrainer)](https://huggingface.co/docs/trl/main/en/cpo_trainer)
* **Local PDF:**  [Xu et al. - 2024 - Contrastive preference optimization pushing the boundaries of LLM performance in machine translatio_1.pdf](<papers/Xu et al. - 2024 - Contrastive preference optimization pushing the boundaries of LLM performance in machine translatio_1.pdf>) 
* **Rating:** 
> **Description:** 
> * **Core Intuition:** Designed for tasks with objective quality boundaries (e.g., Machine Translation). It penalizes sub-optimal outputs while heavily regularizing against deviation from the SFT baseline to prevent catastrophic forgetting.

#### 9. [Token-level Direct Preference Optimization](https://arxiv.org/abs/2404.10719)

* **Algorithm:** TDPO
* **Venue/Year:** ICML 2024
* **Reference Code:** [Vchitect/TDPO (Official)](https://github.com/Vchitect/TDPO)
* **Local PDF:** [Zeng et al. - 2024 - Token-level Direct Preference Optimization.pdf](<papers/Zeng et al. - 2024 - Token-level Direct Preference Optimization.pdf>) 
* **Rating:**
> **Description:**
> * **Core Intuition:** Identifies that standard DPO treats the entire response as a single action, which leads to sparse rewards and poor credit assignment for long outputs. TDPO formulates alignment as a token-level Markov Decision Process (MDP), aligning the policy at the granularity of individual tokens by optimizing forward KL divergence.
> * **Engineering Traits:** Significantly improves the quality of long-form generation and reasoning tasks without increasing the memory footprint compared to sequence-level DPO.
> * **Mathematical Formulation:** Optimizes the token-level Q-value differences:
>
> $$\mathcal{L}_{\text{TDPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \sum_{t} \log \sigma \left( \hat{Q}_\theta(y_{w,t}) - \hat{Q}_\theta(y_{l,t}) \right) \right]$$

#### 10. [Self-Play Preference Optimization for Language Model Alignment](https://arxiv.org/abs/2405.00675)
* **Algorithm:** SPPO
* **Venue/Year:** ArXiv 2024
* **Reference Code:** [uclaml/SPPO (Official)](https://github.com/uclaml/SPPO)
* **Local PDF:** [Wu et al. - 2024 - Self-Play Preference Optimization for Language Model Alignment.pdf](<papers/Wu et al. - 2024 - Self-Play Preference Optimization for Language Model Alignment.pdf>) 
* **Rating:** 
> **Description:**
> * **Core Intuition:** Abandons the reliance on static human preference datasets. SPPO frames LLM alignment as a two-player zero-sum game where the model continuously plays against its own previous iterations to construct a Nash equilibrium, effectively bootstrapping its own alignment capabilities.
> * **Engineering Traits:** Highly scalable and self-improving. It drastically reduces the need for external, massive human-annotated preference pairs, making it ideal for iterative data flywheels.
> * **Mathematical Formulation:** Formulated as finding the policy $\pi$ that minimizes the maximum preference advantage of a challenger $\mu$:
>
> $$\min_{\pi} \max_{\mu} \mathbb{E}_{y_1 \sim \pi, y_2 \sim \mu} [ P(y_1 \succ y_2 | x) - P(y_2 \succ y_1 | x) ]$$

#### 11. [Bottom-up Policy Optimization: Your Language Model Policy Secretly Contains Internal Policies](http://arxiv.org/abs/2512.19673) 
* **Algorithm:** BUPO
* **Venue/Year:** ArXiv 2025
* **Reference Code:** [Trae1ounG/BuPO (Official)](https://github.com/Trae1ounG/BuPO)
* **Local PDF:** [Tan et al. - 2026 - Bottom-up Policy Optimization Your Language Model Policy Secretly Contains Internal Policies.pdf](<papers\Tan et al. - 2026 - Bottom-up Policy Optimization Your Language Model Policy Secretly Contains Internal Policies.pdf>) 
* **Rating:**
> **Description:**
> * **Core Intuition:** Challenges the standard view of an LLM as a monolithic policy. BUPO reveals that LLMs harbor multiple latent "internal policies" (e.g., tone, formatting, factual accuracy). It aligns the model by optimizing these disentangled sub-policies from the bottom up.
> * **Engineering Traits:** Provides incredibly fine-grained control over specific generation attributes. It prevents the catastrophic forgetting of stylistic traits that often occurs during monolithic reward maximization.

#### 12. [AT$^2$PO: Agentic Turn-based Policy Optimization via Tree Search](http://arxiv.org/abs/2601.04767)

* **Algorithm:** AT$^2$PO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** [zzfoutofspace/ATPO (Official)](https://github.com/zzfoutofspace/ATPO)
* **Local PDF:** [Zong et al. - 2026 - AT$^2$PO Agentic Turn-based Policy Optimization via Tree Search.pdf](<papers/Zong et al. - 2026 - AT$^2$PO Agentic Turn-based Policy Optimization via Tree Search.pdf>) 
* **Rating:** 
> **Description:** 
> * **Core Intuition:** Tailored specifically for complex, multi-turn agentic workflows (e.g., tool use, web browsing). It integrates look-ahead tree search (similar to MCTS) with policy optimization to evaluate the long-term viability of an action, rather than relying on sparse terminal rewards.
> * **Engineering Traits:** Computationally heavier during the rollout phase due to tree exploration, but vastly more sample-efficient for tasks requiring long-horizon planning and sequential decision-making.

#### 13. [REINFORCE++: An Efficient RLHF Algorithm with Robustness to Both Prompt and Reward Models](http://arxiv.org/abs/2501.03262)
* **Algorithm:** REINFORCE++
* **Venue/Year:** ArXiv 2025
* **Reference Code:** [OpenRLHF (REINFORCE++)](https://github.com/OpenRLHF/OpenRLHF)
* **Local PDF:** [Hu et al. - 2025 - REINFORCE++ an efficient RLHF algorithm with robustness to both prompt and reward models.pdf](<papers/Hu et al. - 2025 - REINFORCE++ an efficient RLHF algorithm with robustness to both prompt and reward models.pdf>) 
* **Rating:** 
> **Description:** 
> * **Core Intuition:** Modernizes the classic REINFORCE algorithm for LLMs. It achieves PPO-level stability without a Critic network by introducing robust reward baseline estimation and PPO-style clipping mechanisms to handle high variance in prompt difficulties and RM scales.
> * **Engineering Traits:** A highly practical middle-ground. By stripping away the Value Network (Critic), it saves massive VRAM, making online RL viable on hardware setups where PPO would OOM (Out of Memory).
> * **Mathematical Formulation:** Utilizes a normalized advantage with a running baseline:
>
> $$\nabla J(\theta) = \mathbb{E}_{x, y \sim \pi_\theta} \left[ \left( \frac{r(x,y) - b(x)}{\sigma_R} - \beta \mathbb{D}_{\text{KL}} \right) \nabla_\theta \log \pi_\theta(y|x) \right]$$

#### 14. [Single-stream Policy Optimization](http://arxiv.org/abs/2509.13232)
* **Algorithm:** SSPO
* **Venue/Year:** ArXiv 2025
* **Reference Code:** ...
* **Local PDF:** [Xu and Ding - 2025 - Single-stream Policy Optimization.pdf](<papers/Xu and Ding - 2025 - Single-stream Policy Optimization.pdf>) 
* **Rating:** 
> **Description:** 


#### 15. [F-GRPO: Don't Let Your Policy Learn the Obvious and Forget the Rare](http://arxiv.org/abs/2602.06717) 
* **Algorithm:** F-GRPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** ...
* **Local PDF:** [Plyusov et al. - 2026 - F-GRPO Don't Let Your Policy Learn the Obvious and Forget the Rare.pdf](<papers\Plyusov et al. - 2026 - F-GRPO Don't Let Your Policy Learn the Obvious and Forget the Rare.pdf>) 
* **Rating:** 
> **Description:** 


---

## 🌟 Useful Resources (Surveys & Codebases)

### 📖 Comprehensive Surveys
* **[A Comprehensive Survey of Direct Preference Optimization: Datasets, Theories, Variants, and Applications](https://arxiv.org/abs/2410.15595)**
  * **Brief**: A must-read survey! Exhaustively categorizes the evolution, mathematical essence, and common datasets of DPO and its dozens of variants (IPO, KTO, CPO, etc.).
* **[A Survey of Reinforcement Learning for Large Reasoning Models](http://arxiv.org/abs/2509.08827) **
  * **Brief**: Focuses on the frontier transition from "preference alignment" to "Verifiable Reasoning (RLVR)" within the reinforcement learning paradigm.
* **[RLHF Algorithms Ranked: An Extensive Evaluation](https://aclanthology.org/2025.emnlp-industry.35/) **
  * **Brief**: A hardcore engineering benchmark evaluating over a dozen mainstream XPO algorithms horizontally, providing empirical recommendations for hyperparameter setups.

### 💻 Open-Source Codebases & Frameworks

* **[Hugging Face TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)**
  * **Brief**: The most authoritative RL fine-tuning library in the industry. Features out-of-the-box APIs like `DPOTrainer`, `PPOTrainer`, and `ORPOTrainer`. Essential for all XPO researchers.
* **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**
  * **Brief**: A highly mature and user-friendly fine-tuning framework supporting native DPO, ORPO, and KTO. It is exceptionally friendly to low-compute environments and highly recommended for efficient alignment training on Small Language Models (SLMs) using methods like LoRA.
* **[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)**
  * **Brief**: Tailored for large-scale, distributed RLHF. It natively integrates Ray, making it the best choice for reproducing the highly complex pipeline parallelism of GRPO or standard PPO in multi-GPU environments.
* **[Unsloth](https://github.com/unslothai/unsloth)**
  * **Brief**: Focuses on "extreme acceleration." It rewrites the underlying operators for DPO and PPO, doubling training speed and significantly saving memory, perfect for rapid idea verification.

---

**⏳ To be continued... (We are continuously updating this repository!)**
