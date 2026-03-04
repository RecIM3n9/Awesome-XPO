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
>   $L^{\text{CLIP}}(\phi) = \hat{\mathbb{E}}_t \left[ \min(r_t(\phi)\hat{A}_t, \text{clip}(r_t(\phi), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$

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
>   $\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma \left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$

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
>   $\hat{A}_{i,j} = \frac{r_{i,j} - \text{mean}(r_i)}{\text{std}(r_i)}$

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
>
> * **Engineering Traits:** Extremely lightweight. Optimal for resource-constrained environments, allowing rapid iterative fine-tuning on SLMs using methods like LoRA.
>
> * **Mathematical Formulation:**  
>
>   $\mathcal{L}_{\text{ORPO}} = \mathbb{E}_{(x, y_w, y_l)} \left[ \mathcal{L}_{\text{SFT}}(y_w) + \lambda \cdot \log \sigma \left( \log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)} \right) \right]$

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
>   $\mathcal{L}_{\text{IPO}} = \mathbb{E} \left[ \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \frac{\tau}{2} \right)^2 \right]$

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
