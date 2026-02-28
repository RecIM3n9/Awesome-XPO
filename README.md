# Awesome XPO

A structured and comprehensive reading list & resource tracker for Reinforcement Learning and Preference Optimization (PPO, DPO, GRPO, etc.) in Large Language Models.

---

## 📚 Core Literature
#### 0. [Proximal policy optimization algorithms](http://arxiv.org/abs/1707.06347)
* **Algorithm:** PPO
* **Venue/Year:** NeurIPS 2022
* **Local PDF:**  [Schulman et al. - 2017 - Proximal policy optimization algorithms.pdf](papers\Schulman et al. - 2017 - Proximal policy optimization algorithms.pdf) 
* **Rating:** 
> **Description:** *Pending review. To be summarized by the team.*

#### 1. [Training language models to follow instructions with human feedback](http://arxiv.org/abs/2203.02155)
* **Algorithm:** PPO
* **Venue/Year:** NeurIPS 2022
* **Local PDF:**  [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback.pdf](papers\Ouyang et al. - 2022 - Training language models to follow instructions with human feedback.pdf) 
* **Rating:** 
> **Description:** *Pending review. To be summarized by the team.*

#### 2. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
* **Algorithm:** DPO
* **Venue/Year:** NeurIPS 2023
* **Local PDF:**  [Rafailov et al. - 2024 - Direct preference optimization your language model is secretly a reward model.pdf](papers\Rafailov et al. - 2024 - Direct preference optimization your language model is secretly a reward model.pdf) 
* **Rating:** 
> **Description:** *Pending review. To be summarized by the team.*

#### 3. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
* **Algorithm:** GRPO
* **Venue/Year:** ArXiv 2024
* **Local PDF:**  [Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf](papers\Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf) 
* **Rating:** 
> **Description:** *Pending review. To be summarized by the team.*

#### 4. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
* **Algorithm:** GRPO / RLVR
* **Venue/Year:** ArXiv 2025
* **Local PDF:**  [DeepSeek-AI et al. - 2025 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.pdf](papers\DeepSeek-AI et al. - 2025 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.pdf) 
* **Rating:** 
> **Description:** *Pending review. To be summarized by the team.*

#### 5. [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
* **Algorithm:** KTO
* **Venue/Year:** ICML 2024
* **Local PDF:**  [Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization.pdf](papers\Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization.pdf) 
* **Rating:** 
> **Description:** *Pending review. To be summarized by the team.*

#### 6. [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)
* **Algorithm:** ORPO
* **Venue/Year:** ArXiv 2024
* **Local PDF:**  [Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model.pdf](papers\Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model.pdf) 
* **Rating:** 
> **Description:** *Pending review. To be summarized by the team.*

#### 7. [A general theoretical paradigm to understand learning from human preferences](http://arxiv.org/abs/2310.12036)
* **Algorithm:** $\Psi $PO / IPO
* **Venue/Year:** ArXiv 2024
* **Local PDF:**  [Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model.pdf](papers\Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model.pdf) 
* **Rating:** 
> **Description:** *Pending review. To be summarized by the team.*

#### 8. [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
* **Algorithm:** SimPO
* **Venue/Year:** ArXiv 2024
* **Local PDF:**  [Meng et al. - 2024 - SimPO simple preference optimization with a reference-free reward.pdf](papers\Meng et al. - 2024 - SimPO simple preference optimization with a reference-free reward.pdf) 
* **Rating:** 
> **Description:** *Pending review. To be summarized by the team.*

#### 9. [Preference Ranking Optimization for Human Alignment](https://arxiv.org/abs/2306.17492)
* **Algorithm:** PRO
* **Venue/Year:** AAAI 2024
* **Local PDF:**  [Song et al. - 2024 - Preference Ranking Optimization for Human Alignment.pdf](papers\Song et al. - 2024 - Preference Ranking Optimization for Human Alignment.pdf) 
* **Rating:** 
> **Description:** *Pending review. To be summarized by the team.*

#### 10. [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](https://arxiv.org/abs/2401.08417)
* **Algorithm:** CPO
* **Venue/Year:** ICML 2024
* **Local PDF:**  [Xu et al. - 2024 - Contrastive preference optimization pushing the boundaries of LLM performance in machine translatio_1.pdf](papers\Xu et al. - 2024 - Contrastive preference optimization pushing the boundaries of LLM performance in machine translatio_1.pdf) 
* **Rating:** 
> **Description:** *Pending review. To be summarized by the team.*
---

## 🌟 Useful Resources (Surveys & Codebases)

### 📖 Comprehensive Surveys
* **[A Comprehensive Survey of Direct Preference Optimization: Datasets, Theories, Variants, and Applications](https://arxiv.org/abs/2406.18341)**
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