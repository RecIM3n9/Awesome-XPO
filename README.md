# Awesome XPO

A structured and project-oriented reading list for preference optimization and RL post-training in language models, with a special focus on Small Language Models (SLMs) under capacity constraints.

---

## 📚 Core Literature

**Rating Note:** Each `Rating` is a provisional `0-10` SLM-fit score derived from the GITA framework in `0000.md`, jointly considering `knowledge retention`, `parameterization compatibility`, and `feedback resolvability`.

### Online RLHF Foundations and RLVR for Reasoning

#### 0. [Proximal Policy Optimization Algorithms](http://arxiv.org/abs/1707.06347) & [Training language models to follow instructions with human feedback](http://arxiv.org/abs/2203.02155)
* **Algorithm:** PPO
* **Venue/Year:** NeurIPS 2017 & NeurIPS 2022
* **Reference Code:** [stable-baseline3](https://github.com/DLR-RM/stable-baselines3) | [Hugging Face TRL (PPOTrainer)](https://github.com/huggingface/trl) | [OpenRLHF (PPO)](https://github.com/OpenRLHF/OpenRLHF)
* **Local PDF:** [Schulman et al. - 2017 - Proximal policy optimization algorithms.pdf](<papers/Schulman et al. - 2017 - Proximal policy optimization algorithms.pdf>) & [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback.pdf](<papers/Ouyang et al. - 2022 - Training language models to follow instructions with human feedback.pdf>)
* **Rating:** `5.8/10`
> **Description:**
> * **Core Intuition:** The classical actor-critic RLHF pipeline optimizes a policy against a learned reward model while constraining deviation from a reference policy.
> * **Engineering Traits:** Strong but heavy. PPO remains the canonical online RL baseline, yet its critic, reward model, reference model, and hyperparameter sensitivity make it poorly matched to small-model, low-budget settings.
> * **Mathematical Formulation:** The standard clipped surrogate is
>
> $$L_{PPO}(\theta)=\hat{\mathbb{E}}[\min(r_t(\theta)\hat A_t,\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t)],\quad r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}$$
>
> * **Policy Update:** In RLHF, one typically optimizes
>
> $$\max_\theta \; L_{PPO}(\theta)-\beta\,\mathbb{E}_{x,y\sim\pi_\theta}[D_{KL}(\pi_\theta(\cdot\mid x)\|\pi_{ref}(\cdot\mid x))]$$
>
> with gradient ascent on the actor and a separate value-loss update for the critic.

### Offline Preference Optimization

#### 1. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
* **Algorithm:** DPO
* **Venue/Year:** NeurIPS 2023
* **Reference Code:** [Official Repo](https://github.com/eric-mitchell/direct-preference-optimization) | [Hugging Face TRL (DPOTrainer)](https://huggingface.co/docs/trl/main/en/dpo_trainer)
* **Local PDF:** [Rafailov et al. - 2024 - Direct preference optimization your language model is secretly a reward model.pdf](<papers/Rafailov et al. - 2024 - Direct preference optimization your language model is secretly a reward model.pdf>)
* **Rating:** `9.4/10`
> **Description:**
> * **Core Intuition:** Rewrites preference alignment as a closed-form classification problem, bypassing explicit reward modeling and RL.
> * **Engineering Traits:** The strongest default baseline for SLM studies: stable, efficient, and highly interpretable under the GITA lens.
> * **Mathematical Formulation:** For a preference pair $(x,y_w,y_l)$, DPO minimizes
>
> $$\mathcal L_{DPO}=-\log\sigma(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)})$$
>
> * **Policy Update:** Gradient descent on this logistic loss implicitly moves the policy toward the preferred sample and away from the rejected one without training a separate reward model.

#### 2. [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
* **Algorithm:** KTO
* **Venue/Year:** ICML 2024
* **Reference Code:** [ContextualAI/HALOs (Official)](https://github.com/ContextualAI/HALOs) | [Hugging Face TRL (KTOTrainer)](https://huggingface.co/docs/trl/main/en/kto_trainer)
* **Local PDF:** [Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization.pdf](<papers/Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization.pdf>)
* **Rating:** `8.6/10`
> **Description:**
> * **Core Intuition:** Replaces pairwise labels with independent desirable/undesirable signals inspired by prospect theory.
> * **Engineering Traits:** Attractive for SLMs when pairwise data is scarce or noisy, though its pointwise supervision can be less expressive than strong pairwise preference data.
> * **Mathematical Formulation:** A representative KTO loss can be written as
>
> $$\mathcal L_{KTO}=-\mathbb E_{(x,y)\sim\mathcal D^+}[\log\sigma(\beta z_\theta(x,y)-\tau_+)]-\mathbb E_{(x,y)\sim\mathcal D^-}[\log\sigma(\tau_- - \beta z_\theta(x,y))]$$
>
> where $z_\theta(x,y)=\log \pi_\theta(y\mid x)-\log \pi_{ref}(y\mid x)$.
> * **Policy Update:** Positive examples increase relative log-likelihood, while undesirable examples push it down with asymmetric prospect-style treatment.

#### 3. [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)
* **Algorithm:** ORPO
* **Venue/Year:** ArXiv 2024
* **Reference Code:** [xfact/ORPO (Official)](https://github.com/xfact/ORPO) | [Hugging Face TRL (ORPOTrainer)](https://huggingface.co/docs/trl/main/en/orpo_trainer)
* **Local PDF:** [Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model.pdf](<papers/Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model.pdf>)
* **Rating:** `8.9/10`
> **Description:**
> * **Core Intuition:** Folds preference learning into the supervised objective via an odds-ratio penalty and eliminates the reference model.
> * **Engineering Traits:** Very SLM-friendly. Low memory, simple training, and good compatibility with LoRA make it one of the best resource-constrained baselines.
> * **Mathematical Formulation:** The ORPO objective is
>
> $$\mathcal L_{ORPO}=\mathcal L_{SFT}(x,y_w)+\lambda\,\mathcal L_{OR}(x,y_w,y_l)$$
>
> with
>
> $$\mathcal L_{OR}=-\log\sigma(\log\frac{\mathrm{odds}_\theta(y_w\mid x)}{\mathrm{odds}_\theta(y_l\mid x)}),\quad \mathrm{odds}_\theta(y\mid x)=\frac{\pi_\theta(y\mid x)}{1-\pi_\theta(y\mid x)}$$
>
> * **Policy Update:** Each step jointly preserves next-token imitation on the chosen response and increases the chosen-vs-rejected odds ratio.

#### 4. [A General Theoretical Paradigm to Understand Learning from Human Preferences](http://arxiv.org/abs/2310.12036)
* **Algorithm:** IPO
* **Venue/Year:** ArXiv 2024
* **Reference Code:** [Hugging Face TRL (via DPOTrainer `loss_type="ipo"`)](https://huggingface.co/docs/trl/main/en/dpo_trainer)
* **Local PDF:** [Azar et al. - 2023 - A general theoretical paradigm to understand learning from human preferences.pdf](<papers/Azar et al. - 2023 - A general theoretical paradigm to understand learning from human preferences.pdf>)
* **Rating:** `8.7/10`
> **Description:**
> * **Core Intuition:** Reinterprets preference optimization as solving a regularized fixed-point style objective rather than pure maximum likelihood.
> * **Engineering Traits:** More conservative than DPO and often more retention-friendly, making it especially relevant for capacity-limited models.
> * **Mathematical Formulation:** IPO replaces the DPO logistic objective with a squared error:
>
> $$\mathcal L_{IPO}=(\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}-\frac{1}{2\beta})^2$$
>
> * **Policy Update:** The update pushes the pairwise log-ratio toward a target margin instead of unboundedly sharpening preferences.

#### 5. [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
* **Algorithm:** SimPO
* **Venue/Year:** ArXiv 2024
* **Reference Code:** [princeton-nlp/SimPO (Official)](https://github.com/princeton-nlp/SimPO) | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
* **Local PDF:** [Meng et al. - 2024 - SimPO simple preference optimization with a reference-free reward.pdf](<papers/Meng et al. - 2024 - SimPO simple preference optimization with a reference-free reward.pdf>)
* **Rating:** `8.8/10`
> **Description:**
> * **Core Intuition:** Uses length-normalized policy scores as implicit rewards and removes the reference model.
> * **Engineering Traits:** Strong practical candidate for SLMs because it is light, reference-free, and explicitly engages with verbosity bias.
> * **Mathematical Formulation:** Define the implicit reward
>
> $$r_{SimPO}(x,y)=\frac{\beta}{|y|}\log\pi_\theta(y\mid x)$$
>
> and optimize
>
> $$\mathcal L_{SimPO}=-\log\sigma(r_{SimPO}(x,y_w)-r_{SimPO}(x,y_l)-\gamma)$$
>
> * **Policy Update:** Preferred outputs are pushed above rejected ones by a target margin $\gamma$, while length normalization suppresses verbosity-driven reward inflation.

#### 6. [SLiC-HF: Sequence Likelihood Calibration with Human Feedback](https://arxiv.org/abs/2305.10425)
* **Algorithm:** SLiC-HF
* **Venue/Year:** ArXiv 2023
* **Reference Code:** [Google Research](https://github.com/google-research/google-research/tree/master/slic)
* **Local PDF:** [Zhao et al. - 2023 - SLiC-HF sequence likelihood calibration with human feedback.pdf](<papers/Zhao et al. - 2023 - SLiC-HF sequence likelihood calibration with human feedback.pdf>)
* **Rating:** `8.0/10`
> **Description:**
> * **Core Intuition:** Aligns sequence likelihoods directly using human feedback data, offering an efficient alternative to PPO-style RLHF.
> * **Engineering Traits:** Historically important because it shows that preference alignment can be effective without the full RLHF stack, but it is now more of a reference point than a primary frontier method.
> * **Mathematical Formulation:** A representative calibration loss is a pairwise ranking objective over sequence scores $s_\theta(x,y)$:
>
> $$\mathcal L_{SLiC-HF}=-\log\sigma(s_\theta(x,y_w)-s_\theta(x,y_l))$$
>
> with $s_\theta$ usually derived from normalized sequence log-likelihood.
> * **Policy Update:** Gradient descent calibrates preferred sequences upward and dispreferred ones downward without a separate online RL phase.

#### 7. [Towards Efficient Exact Optimization of Language Model Alignment](https://arxiv.org/abs/2402.00856)
* **Algorithm:** EXO
* **Venue/Year:** ICML 2024
* **Reference Code:** [haozheji/exact-optimization](https://github.com/haozheji/exact-optimization)
* **Local PDF:** [Ji et al. - 2024 - Towards Efficient Exact Optimization of Language Model Alignment.pdf](<papers/Ji et al. - 2024 - Towards Efficient Exact Optimization of Language Model Alignment.pdf>)
* **Rating:** `7.8/10`
> **Description:**
> * **Core Intuition:** Seeks a closer approximation to the exact alignment optimum than DPO's common approximation.
> * **Engineering Traits:** Theoretically valuable and relevant for related work, though less standardized than DPO-family baselines in current SLM workflows.
> * **Mathematical Formulation:** EXO targets the exact KL-regularized optimum rather than DPO's approximation, yielding an objective that can be optimized with unbiased Monte Carlo estimators of the exact alignment gradient.
> * **Policy Update:** In practice, EXO performs direct policy updates toward the exact mode-seeking optimum, avoiding an explicit reward-model-plus-PPO decomposition.

#### 8. [Token-level Direct Preference Optimization](https://arxiv.org/abs/2404.10719)
* **Algorithm:** TDPO
* **Venue/Year:** ICML 2024
* **Reference Code:** [Vchitect/TDPO](https://github.com/Vchitect/TDPO)
* **Local PDF:** [Zeng et al. - 2024 - Token-level direct preference optimization.pdf](<papers/Zeng et al. - 2024 - Token-level direct preference optimization.pdf>)
* **Rating:** `8.3/10`
> **Description:**
> * **Core Intuition:** Breaks sequence-level preference learning into token-level credit assignment.
> * **Engineering Traits:** Particularly relevant to the project's Dimension 3 because it directly attacks sparse, coarse credit assignment in long outputs.
> * **Mathematical Formulation:** TDPO replaces one global comparison with token-level comparisons:
>
> $$\mathcal L_{TDPO}=-\sum_t \log\sigma(\hat Q_\theta(x,y_w,t)-\hat Q_\theta(x,y_l,t))$$
>
> where $\hat Q_\theta$ is a token-level preference score derived from policy logits and reference calibration.
> * **Policy Update:** Tokens that causally support the preferred trajectory get stronger positive gradients than irrelevant filler tokens.

#### 9. [Preference Ranking Optimization for Human Alignment](https://arxiv.org/abs/2306.17492)
* **Algorithm:** PRO
* **Venue/Year:** AAAI 2024
* **Reference Code:** [sail-sg/PRO](https://github.com/sail-sg/PRO)
* **Local PDF:** [Song et al. - 2024 - Preference Ranking Optimization for Human Alignment.pdf](<papers/Song et al. - 2024 - Preference Ranking Optimization for Human Alignment.pdf>)
* **Rating:** `7.2/10`
> **Description:**
> * **Core Intuition:** Generalizes binary preference learning to ranking over multiple candidates.
> * **Engineering Traits:** Useful as a broader preference-learning reference, but less central than DPO/KTO/ORPO/SimPO for the project's main SLM matrix.
> * **Mathematical Formulation:** PRO typically optimizes a listwise contrastive ranking loss such as
>
> $$\mathcal L_{PRO}=-\log\frac{\exp(s_\theta(y^+))}{\sum_{j=1}^K \exp(s_\theta(y_j))}$$
>
> * **Policy Update:** Better-ranked completions are assigned larger likelihood mass relative to the full candidate list.

#### 10. [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](https://arxiv.org/abs/2401.08417)
* **Algorithm:** CPO
* **Venue/Year:** ICML 2024
* **Reference Code:** [AILab-CVC/CPO](https://github.com/AILab-CVC/CPO) | [Hugging Face TRL (CPOTrainer)](https://huggingface.co/docs/trl/main/en/cpo_trainer)
* **Local PDF:** [Xu et al. - 2024 - Contrastive preference optimization pushing the boundaries of LLM performance in machine translatio_1.pdf](<papers/Xu et al. - 2024 - Contrastive preference optimization pushing the boundaries of LLM performance in machine translatio_1.pdf>)
* **Rating:** `7.4/10`
> **Description:**
> * **Core Intuition:** Uses contrastive alignment with stronger regularization to preserve the SFT model's base competence.
> * **Engineering Traits:** Valuable for discussing retention-sensitive objectives, though its main empirical success comes from MT rather than generic SLM alignment.
> * **Mathematical Formulation:** CPO augments preference optimization with an explicit regularizer to stay near the supervised model:
>
> $$\mathcal L_{CPO}=\mathcal L_{contrastive}+\lambda\,D_{KL}(\pi_\theta\|\pi_{SFT})$$
>
> * **Policy Update:** The policy is updated to separate good and bad outputs while explicitly limiting drift from the base model.

#### 11. [Self-Play Preference Optimization for Language Model Alignment](https://arxiv.org/abs/2405.00675)
* **Algorithm:** SPPO
* **Venue/Year:** ArXiv 2024
* **Reference Code:** [uclaml/SPPO](https://github.com/uclaml/SPPO)
* **Local PDF:** [Wu et al. - 2024 - Self-play preference optimization for language model alignment.pdf](<papers/Wu et al. - 2024 - Self-play preference optimization for language model alignment.pdf>)
* **Rating:** `7.0/10`
> **Description:**
> * **Core Intuition:** Replaces static preference datasets with iterative self-play style preference generation.
> * **Engineering Traits:** Interesting for scalable data generation, but harder to keep controlled in a mechanism-focused SLM study.
> * **Mathematical Formulation:** SPPO can be viewed as solving a min-max preference game:
>
> $$\min_\pi\max_\mu\; \mathbb E_{y_1\sim\pi, y_2\sim\mu}[P(y_1\succ y_2\mid x)-P(y_2\succ y_1\mid x)]$$
>
> * **Policy Update:** The current policy is repeatedly updated against a challenger policy induced by previous policy snapshots or self-play opponents.

### Online RLHF and RLVR for Reasoning

#### 12. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) & [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
* **Algorithm:** GRPO
* **Venue/Year:** ArXiv 2024 & ArXiv 2025
* **Reference Code:** [OpenRLHF (GRPO)](https://github.com/OpenRLHF/OpenRLHF) | [Unsloth (GRPO)](https://github.com/unslothai/unsloth)
* **Local PDF:** [Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf](<papers/Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf>) & [DeepSeek-AI et al. - 2025 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.pdf](<papers/DeepSeek-AI et al. - 2025 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.pdf>)
* **Rating:** `7.4/10`
> **Description:**
> * **Core Intuition:** Removes the critic and estimates advantages from grouped sampled responses via relative normalization over verifiable rewards.
> * **Engineering Traits:** Essential for the project. GRPO is the core RLVR reference, but on SLMs it is prone to feedback-resolution and update-instability pathologies that the project is designed to measure.
> * **Mathematical Formulation:** For a prompt $x_i$ with group rewards $r_{i,1},\dots,r_{i,G}$,
>
> $$\hat A_{i,j}=\frac{r_{i,j}-\mathrm{mean}(r_i)}{\mathrm{std}(r_i)+\epsilon}$$
>
> and the clipped policy objective is
>
> $$L_{GRPO}(\theta)=\hat{\mathbb E}_{i,j,t}[\min(\rho_{i,j,t}(\theta)\hat A_{i,j},\mathrm{clip}(\rho_{i,j,t}(\theta),1-\epsilon,1+\epsilon)\hat A_{i,j})]$$
>
> where $\rho_{i,j,t}(\theta)=\pi_\theta(o_{i,j,t}\mid x_i,o_{i,j,<t})/\pi_{\theta_{old}}(o_{i,j,t}\mid x_i,o_{i,j,<t})$.
> * **Policy Update:** Update by gradient ascent on $L_{GRPO}-\beta D_{KL}(\pi_\theta\|\pi_{ref})$.

#### 13. [REINFORCE++: An Efficient RLHF Algorithm with Robustness to Both Prompt and Reward Models](http://arxiv.org/abs/2501.03262)
* **Algorithm:** REINFORCE++
* **Venue/Year:** ArXiv 2025
* **Reference Code:** [OpenRLHF (REINFORCE++)](https://github.com/OpenRLHF/OpenRLHF)
* **Local PDF:** [Hu et al. - 2025 - REINFORCE++ an efficient RLHF algorithm with robustness to both prompt and reward models.pdf](<papers/Hu et al. - 2025 - REINFORCE++ an efficient RLHF algorithm with robustness to both prompt and reward models.pdf>)
* **Rating:** `8.2/10`
> **Description:**
> * **Core Intuition:** Modernizes vanilla REINFORCE with better baselines and clipping-style stabilizers.
> * **Engineering Traits:** One of the most important online-RL methods for this project because it offers a lighter alternative to PPO and a cleaner comparison against GRPO on SLM hardware.
> * **Mathematical Formulation:** A representative REINFORCE++ gradient is
>
> $$\nabla J(\theta)=\mathbb E_{x,y\sim\pi_\theta}[(\frac{r(x,y)-b(x)}{\sigma_R}-\beta D_{\mathrm{KL}})\nabla_\theta\log\pi_\theta(y\mid x)]$$
>
> with robust prompt-wise baselines and reward normalization.
> * **Policy Update:** REINFORCE++ performs policy-gradient ascent directly on sampled trajectories, but stabilizes the update magnitude with normalized rewards and clipping-inspired controls.

#### 14. [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
* **Algorithm:** GSPO
* **Venue/Year:** ArXiv 2025
* **Reference Code:** Pending release / ecosystem implementations
* **Local PDF:** [Zheng et al. - 2025 - Group sequence policy optimization.pdf](<papers/Zheng et al. - 2025 - Group sequence policy optimization.pdf>)
* **Rating:** `7.9/10`
> **Description:**
> * **Core Intuition:** Replaces GRPO's token-level importance ratio with a sequence-level ratio to better match sequence-level rewards.
> * **Engineering Traits:** A key extension for the project because it changes credit-assignment granularity and may trade lower variance for coarser supervision.
> * **Mathematical Formulation:** Define the sequence-level ratio by
>
> $$r_i^{seq}=\frac{\pi_\theta(y_i\mid x_i)}{\pi_{old}(y_i\mid x_i)}$$
>
> and optimize
>
> $$L_{GSPO}=\hat{\mathbb E}_i[\min(r_i^{seq}\hat A_i,\mathrm{clip}(r_i^{seq},1-\epsilon,1+\epsilon)\hat A_i)]$$
>
> * **Policy Update:** All tokens in a sequence share one clipped importance ratio, which reduces variance but coarsens token-level credit assignment.

#### 15. [BandPO: Bridging Trust Regions and Ratio Clipping via Probability-Aware Bounds for LLM Reinforcement Learning](https://arxiv.org/abs/2603.04918)
* **Algorithm:** BandPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** Pending release
* **Local PDF:** [Li et al. - 2026 - BandPO Bridging Trust Regions and Ratio Clipping via Probability-Aware Bounds for LLM Reinforcement.pdf](<papers/Li et al. - 2026 - BandPO Bridging Trust Regions and Ratio Clipping via Probability-Aware Bounds for LLM Reinforcement.pdf>)
* **Rating:** `7.8/10`
> **Description:**
> * **Core Intuition:** Improves PPO-style update control by replacing fixed clipping with probability-aware trust bounds.
> * **Engineering Traits:** Directly relevant to the project's Dimension 2 because it targets update geometry rather than only reward design.
> * **Mathematical Formulation:** BandPO replaces the fixed clip region with adaptive bounds $[\ell_t,u_t]$ depending on action probability and trust-region geometry:
>
> $$L_{BandPO}(\theta)=\hat{\mathbb E}_t[\min(r_t(\theta)\hat A_t,\mathrm{clip}_{[\ell_t,u_t]}(r_t(\theta))\hat A_t)]$$
>
> * **Policy Update:** The policy is updated with larger freedom on informative low-probability actions and tighter constraints on unstable high-ratio updates.

#### 16. [F-GRPO: Don't Let Your Policy Learn the Obvious and Forget the Rare](https://arxiv.org/abs/2602.06717)
* **Algorithm:** F-GRPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** No official code found yet
* **Local PDF:** [Plyusov et al. - 2026 - F-GRPO - Dont Let Your Policy Learn the Obvious and Forget the Rare.pdf](<papers/Plyusov et al. - 2026 - F-GRPO - Dont Let Your Policy Learn the Obvious and Forget the Rare.pdf>)
* **Rating:** `8.2/10`
> **Description:**
> * **Core Intuition:** Applies focal-style reweighting so easy, already-solved prompts do not dominate RLVR updates.
> * **Engineering Traits:** Highly relevant for SLMs because small models are more likely to forget rare-correct modes under group-normalized training.
> * **Mathematical Formulation:** F-GRPO rescales the GRPO advantage by prompt difficulty:
>
> $$\hat A_i^{F-GRPO}=(1-\hat\mu_{pos}(x_i))^\gamma\,\hat A_i^{GRPO}$$
>
> and plugs the reweighted advantage into the usual GRPO clipped objective.
> * **Policy Update:** Easy prompts receive smaller gradients, preserving capacity for rare but useful trajectories.

#### 17. [f-GRPO and Beyond: Divergence-Based Reinforcement Learning Algorithms for General LLM Alignment](https://arxiv.org/abs/2602.05946)
* **Algorithm:** f-GRPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** No official code found yet
* **Local PDF:** [Haldar et al. - 2026 - f-GRPO and Beyond Divergence-Based Reinforcement Learning Algorithms for General LLM Alignment.pdf](<papers/Haldar et al. - 2026 - f-GRPO and Beyond Divergence-Based Reinforcement Learning Algorithms for General LLM Alignment.pdf>)
* **Rating:** `7.7/10`
> **Description:**
> * **Core Intuition:** Generalizes GRPO-style alignment under a broader f-divergence framework and unifies RLVR with preference alignment.
> * **Engineering Traits:** More important theoretically than as a default benchmark, but very useful for the project's interpretive framing.
> * **Mathematical Formulation:** In the variational f-divergence view, one optimizes an objective of the form
>
> $$L_{f-GRPO}(\theta)=\mathbb E[T_\phi(r)]-\mathbb E[f^*(T_\phi(r))]$$
>
> where $T_\phi$ is a variational critic over policy ratios or rewards and $f^*$ is the convex conjugate of the chosen divergence.
> * **Policy Update:** Different $f$ choices yield different gradient weighting schemes, changing how aggressively high-reward trajectories dominate policy updates.

#### 18. [MC-GRPO: Median-Centered Group Relative Policy Optimization for Small-Rollout Reinforcement Learning](https://arxiv.org/abs/2601.22582)
* **Algorithm:** MC-GRPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** [lotusroot-kim/MC-GRPO](https://github.com/lotusroot-kim/MC-GRPO)
* **Local PDF:** [Kim - 2026 - MC-GRPO - Median-Centered Group Relative Policy Optimization for Small-Rollout Reinforcement Learning.pdf](<papers/Kim - 2026 - MC-GRPO - Median-Centered Group Relative Policy Optimization for Small-Rollout Reinforcement Learning.pdf>)
* **Rating:** `8.0/10`
> **Description:**
> * **Core Intuition:** Replaces the mean group baseline with a median-centered one to stabilize low-rollout RL.
> * **Engineering Traits:** Very relevant to realistic SLM settings where rollout budgets are small and baseline noise strongly affects update direction.
> * **Mathematical Formulation:** The robust advantage estimate is
>
> $$\hat A_i=\frac{r_i-\mathrm{median}(r_1,\dots,r_{G+1})}{\mathrm{MAD}(r)+\epsilon}$$
>
> * **Policy Update:** The resulting advantage is then used inside the standard clipped GRPO policy objective, reducing sign flips induced by noisy group means.

#### 19. [AMIR-GRPO: Inducing Implicit Preference Signals into GRPO](https://arxiv.org/abs/2601.03661)
* **Algorithm:** AMIR-GRPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** [AmirHosseinYari2002/AMIR-GRPO](https://github.com/AmirHosseinYari2002/AMIR-GRPO)
* **Local PDF:** [Yari et al. - 2026 - AMIR-GRPO - Inducing Implicit Preference Signals into GRPO.pdf](<papers/Yari et al. - 2026 - AMIR-GRPO - Inducing Implicit Preference Signals into GRPO.pdf>)
* **Rating:** `8.1/10`
> **Description:**
> * **Core Intuition:** Adds an implicit DPO-style contrastive signal on top of GRPO using within-group rankings.
> * **Engineering Traits:** Strong fit for this project because it explicitly connects offline preference signals and online verifiable rewards.
> * **Mathematical Formulation:** The hybrid objective is
>
> $$\mathcal J_{AMIR-GRPO}(\theta)=\mathcal J_{GRPO}(\theta)+\lambda_{reg}\,\mathcal J_{pref}(\theta)$$
>
> where $\mathcal J_{pref}$ is a DPO-like contrastive regularizer built from intra-group reward ordering.
> * **Policy Update:** Policy updates are guided both by scalar group reward advantages and by pairwise ranking relations inside each sampled group.

#### 20. [Rewards as Labels: Revisiting RLVR from a Classification Perspective](https://arxiv.org/abs/2602.05630)
* **Algorithm:** REAL
* **Venue/Year:** ArXiv 2026
* **Reference Code:** No official code found yet
* **Local PDF:** [Zhai et al. - 2026 - Rewards as Labels Revisiting RLVR from a Classification Perspective.pdf](<papers/Zhai et al. - 2026 - Rewards as Labels Revisiting RLVR from a Classification Perspective.pdf>)
* **Rating:** `8.8/10`
> **Description:**
> * **Core Intuition:** Treats verifiable rewards as labels instead of scalar policy weights, reframing RLVR as a classification problem.
> * **Engineering Traits:** Extremely important for the project's Dimension 3 because it directly diagnoses gradient misassignment in GRPO-like methods and reports strong 1.5B results.
> * **Mathematical Formulation:** REAL replaces scalar-weighted policy gradients with a label-based classification loss, e.g.
>
> $$\mathcal L_{REAL}=\mathrm{BCE}(h_\theta(x,y),\,1[r(x,y)>0])$$
>
> optionally with anchor logits for calibration.
> * **Policy Update:** Correct rollouts and incorrect rollouts receive bounded, monotone classification gradients instead of unstable reward-proportional weights.

#### 21. [Length-Unbiased Sequence Policy Optimization: Revealing and Controlling Response Length Variation in RLVR](https://arxiv.org/abs/2602.05261)
* **Algorithm:** LUSPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** No official code found yet
* **Local PDF:** [Liu et al. - 2026 - Length-Unbiased Sequence Policy Optimization Revealing and Controlling Response Length Variation in.pdf](<papers/Liu et al. - 2026 - Length-Unbiased Sequence Policy Optimization Revealing and Controlling Response Length Variation in.pdf>)
* **Rating:** `8.5/10`
> **Description:**
> * **Core Intuition:** Diagnoses and corrects response-length bias in sequence-level RLVR.
> * **Engineering Traits:** Directly aligned with the project's planned pathology analysis around verbosity, reward coupling, and length collapse.
> * **Mathematical Formulation:** LUSPO uses a length-normalized sequence ratio:
>
> $$r_i^{LU}=\exp(\frac{1}{|y_i|}\sum_t \log\frac{\pi_\theta(y_{i,t}\mid x_i,h_{i,t})}{\pi_{old}(y_{i,t}\mid x_i,h_{i,t})})$$
>
> and then applies a clipped sequence-level objective using $r_i^{LU}$.
> * **Policy Update:** This prevents long responses from receiving systematically inflated or deflated updates purely because of token count.

#### 22. [Orchestrating Tokens and Sequences: Dynamic Hybrid Policy Optimization for RLVR](https://arxiv.org/abs/2601.05607)
* **Algorithm:** DHPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** Pending release
* **Local PDF:** [Min et al. - 2026 - Orchestrating Tokens and Sequences Dynamic Hybrid Policy Optimization for RLVR.pdf](<papers/Min et al. - 2026 - Orchestrating Tokens and Sequences Dynamic Hybrid Policy Optimization for RLVR.pdf>)
* **Rating:** `8.3/10`
> **Description:**
> * **Core Intuition:** Mixes token-level and sequence-level importance ratios within one objective.
> * **Engineering Traits:** One of the clearest algorithmic probes for the project's credit-assignment-resolution thesis.
> * **Mathematical Formulation:** A generic DHPO objective has two clipped branches:
>
> $$L_{DHPO}=\alpha\,L_{token}+(1-\alpha)\,L_{seq}$$
>
> where each branch uses its own importance ratio and trust region before the weighted combination.
> * **Policy Update:** The optimizer interpolates between fine-grained token credit and stable sequence-level credit based on a mixing rule or entropy-guided schedule.

#### 23. [SSPO: Subsentence-level Policy Optimization](https://arxiv.org/abs/2511.04256)
* **Algorithm:** SSPO
* **Venue/Year:** ArXiv 2025
* **Reference Code:** No official code found yet
* **Local PDF:** [Yang et al. - 2025 - SSPO Subsentence-level Policy Optimization.pdf](<papers/Yang et al. - 2025 - SSPO Subsentence-level Policy Optimization.pdf>)
* **Rating:** `8.1/10`
> **Description:**
> * **Core Intuition:** Uses sentence-level importance ratios to strike a middle ground between GRPO and GSPO.
> * **Engineering Traits:** Especially relevant when studying whether finer-than-sequence but coarser-than-token credit improves SLM stability.
> * **Mathematical Formulation:** Let $s$ index sentence spans inside a response. SSPO defines the span-level ratio
>
> $$r_{i,s}=\frac{\pi_\theta(y_{i,s}\mid x_i,h_{i,s})}{\pi_{old}(y_{i,s}\mid x_i,h_{i,s})}$$
>
> and applies PPO-style clipping at the sentence level.
> * **Policy Update:** Entire responses are no longer clipped by one scalar, but the algorithm also avoids the highest variance of fully token-level credit assignment.

#### 24. [Set-Level Policy Optimization for Diversity-Preserving LLM Reasoning](https://arxiv.org/abs/2602.01062)
* **Algorithm:** SetPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** No official code found yet
* **Local PDF:** [Li et al. - 2026 - SetPO - Set-Level Policy Optimization for Diversity-Preserving LLM Reasoning.pdf](<papers/Li et al. - 2026 - SetPO - Set-Level Policy Optimization for Diversity-Preserving LLM Reasoning.pdf>)
* **Rating:** `7.7/10`
> **Description:**
> * **Core Intuition:** Adds a set-level diversity reward so RL does not collapse onto a narrow reasoning mode.
> * **Engineering Traits:** Useful for studying diversity retention and exploration, though less central than GSPO/REAL/LUSPO for the main matrix.
> * **Mathematical Formulation:** SetPO augments the base advantage with a leave-one-out diversity contribution:
>
> $$\hat A_i^{SetPO}=\bar A_i+\lambda(D(\Omega)-D(\Omega\setminus\{o_i\}))$$
>
> * **Policy Update:** Trajectories that improve group diversity receive extra positive credit even if their scalar reward is not maximal.

### Multi-reward, Forgetting, Off-Policy RL, and Advanced Extensions

#### 25. [GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization](https://arxiv.org/abs/2601.05242)
* **Algorithm:** GDPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** No official code found yet
* **Local PDF:** [Liu et al. - 2026 - GDPO group reward-decoupled normalization policy optimization for multi-reward RL optimization.pdf](<papers/Liu et al. - 2026 - GDPO group reward-decoupled normalization policy optimization for multi-reward RL optimization.pdf>)
* **Rating:** `8.0/10`
> **Description:**
> * **Core Intuition:** Shows that directly applying GRPO to multiple rewards can collapse heterogeneous reward combinations into indistinguishable normalized advantages. GDPO decouples normalization across reward dimensions so distinct reward signals remain learnable.
> * **Engineering Traits:** Highly relevant to the project's Dimension 3 because it directly addresses feedback-resolution loss under multi-reward supervision, and also relevant to Dimension 2 because multi-objective updates can become unstable on SLMs.
> * **Mathematical Formulation:** Instead of normalizing only a summed reward, GDPO normalizes each reward channel separately and then aggregates them:
>
> $$\hat A_i^{GDPO}=\sum_{m=1}^{M}\lambda_m\frac{r_i^{(m)}-\mu^{(m)}}{\sigma^{(m)}+\epsilon}$$
>
> where $m$ indexes reward channels and $(\mu^{(m)},\sigma^{(m)})$ are group-wise statistics for reward dimension $m$.
> * **Policy Update:** The decoupled advantage is then inserted into a GRPO/PPO-style clipped objective, preserving relative differences among multiple rewards rather than collapsing them into one noisy scalar.

#### 26. [MetaGDPO: Alleviating Catastrophic Forgetting with Metacognitive Knowledge through Group Direct Preference Optimization](https://arxiv.org/abs/2511.12113)
* **Algorithm:** MetaGDPO
* **Venue/Year:** AAAI 2026
* **Reference Code:** No official code found yet
* **Local PDF:** [Zhang et al. - 2025 - MetaGDPO alleviating catastrophic forgetting with metacognitive knowledge through group direct pref.pdf](<papers/Zhang et al. - 2025 - MetaGDPO alleviating catastrophic forgetting with metacognitive knowledge through group direct pref.pdf>)
* **Rating:** `8.4/10`
> **Description:**
> * **Core Intuition:** Couples data curation and a group direct preference objective to reduce catastrophic forgetting in smaller models.
> * **Engineering Traits:** Very relevant to Dimension 1 and one of the few recent papers explicitly centered on sub-8B forgetting behavior.
> * **Mathematical Formulation:** A representative GDPO-style objective compares grouped preferred and non-preferred responses under a reference-regularized margin:
>
> $$\mathcal L_{GDPO}=-\log\sigma(\beta(\Delta_\theta^{group}-\Delta_{ref}^{group}))+\lambda D_{KL}(\pi_\theta\|\pi_{ref})$$
>
> * **Policy Update:** Group-level preference gradients are combined with explicit anchoring to reduce excessive drift and forgetting.

#### 27. [Buffer Matters: Unleashing the Power of Off-Policy Reinforcement Learning in Large Language Model Reasoning](https://arxiv.org/abs/2602.20722)
* **Algorithm:** BAPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** No official code found yet
* **Local PDF:** [Wan et al. - 2026 - Buffer Matters Unleashing the Power of Off-Policy Reinforcement Learning in Large Language Model Re.pdf](<papers/Wan et al. - 2026 - Buffer Matters Unleashing the Power of Off-Policy Reinforcement Learning in Large Language Model Re.pdf>)
* **Rating:** `7.9/10`
> **Description:**
> * **Core Intuition:** Reuses difficult and high-value historical samples instead of discarding them under strict on-policy training.
> * **Engineering Traits:** Important for the project's extension line on off-policy GRPO-style methods and data efficiency under constrained compute.
> * **Mathematical Formulation:** Off-policy trajectories from a replay buffer are reweighted and selected according to policy-improvement criteria, giving a batch objective of the generic form
>
> $$L_{BAPO}(\theta)=\mathbb E_{(x,y)\sim\mathcal B}[w(x,y)\,\hat A(x,y)\,\log\pi_\theta(y\mid x)]$$
>
> * **Policy Update:** The batch sampler adaptively favors historically difficult or high-value samples rather than sampling only fresh on-policy generations.

#### 28. [LLMs Can Learn to Reason Via Off-Policy RL](https://arxiv.org/abs/2602.19362)
* **Algorithm:** OAPL
* **Venue/Year:** ArXiv 2026
* **Reference Code:** No official code found yet
* **Local PDF:** [Ritter et al. - 2026 - LLMs Can Learn to Reason Via Off-Policy RL.pdf](<papers/Ritter et al. - 2026 - LLMs Can Learn to Reason Via Off-Policy RL.pdf>)
* **Rating:** `7.7/10`
> **Description:**
> * **Core Intuition:** Embraces policy lag and explicitly optimizes with off-policy generations rather than trying to hide distribution mismatch.
> * **Engineering Traits:** Strong systems relevance and useful for discussing how real-world RL training is often off-policy by construction.
> * **Mathematical Formulation:** The core update uses a lagged-behavior-policy correction
>
> $$L_{OAPL}(\theta)=\mathbb E_{y\sim\pi_{lag}}[w_\theta(y\mid x)\,\hat A(y,x)],\quad w_\theta(y\mid x)=\frac{\pi_\theta(y\mid x)}{\pi_{lag}(y\mid x)}$$
>
> * **Policy Update:** This makes policy lag an explicit part of the objective rather than a hidden implementation artifact.

#### 29. [Stable Asynchrony: Variance-Controlled Off-Policy RL for LLMs](https://arxiv.org/abs/2602.17616)
* **Algorithm:** VCPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** [mit-han-lab/vcpo](https://github.com/mit-han-lab/vcpo)
* **Local PDF:** [Huang et al. - 2026 - Stable asynchrony variance-controlled off-policy RL for LLMs.pdf](<papers/Huang et al. - 2026 - Stable asynchrony variance-controlled off-policy RL for LLMs.pdf>)
* **Rating:** `7.5/10`
> **Description:**
> * **Core Intuition:** Uses effective sample size and off-policy variance control to stabilize asynchronous RL for LLMs.
> * **Engineering Traits:** More infrastructure-driven than mainline SLM alignment, but highly useful for scaling discussions and asynchronous training caveats.
> * **Mathematical Formulation:** VCPO rescales the policy update with ESS-aware step control:
>
> $$\eta_{eff}=\eta\cdot g(\mathrm{ESS}(w)),\quad \mathrm{ESS}(w)=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}$$
>
> and combines it with a minimum-variance off-policy baseline.
> * **Policy Update:** When stale rollouts cause ESS collapse, the learning rate is automatically damped to avoid unstable updates.

#### 30. [Hindsight-Anchored Policy Optimization: Turning Failure into Feedback in Sparse Reward Settings](https://arxiv.org/abs/2603.11321)
* **Algorithm:** HAPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** No official code found yet
* **Local PDF:** [Wu et al. - 2026 - Hindsight-Anchored Policy Optimization Turning Failure into Feedback in Sparse Reward Settings.pdf](<papers/Wu et al. - 2026 - Hindsight-Anchored Policy Optimization Turning Failure into Feedback in Sparse Reward Settings.pdf>)
* **Rating:** `7.6/10`
> **Description:**
> * **Core Intuition:** Injects hindsight teacher guidance only when the policy fails under sparse reward settings.
> * **Engineering Traits:** Valuable for thinking about sparse-feedback regimes, though it is best treated as an advanced extension rather than a primary benchmark.
> * **Mathematical Formulation:** A generic HAPO-style objective combines RL with hindsight anchoring:
>
> $$\mathcal J_{HAPO}=\mathcal J_{RL}+\lambda_t\,\mathcal J_{anchor}$$
>
> where $\lambda_t$ is adaptively gated and decays as the policy becomes competent.
> * **Policy Update:** Failed trajectories receive temporary teacher-guided correction, but the anchoring term anneals away to recover asymptotically on-policy learning.

#### 31. [Rethinking the Trust Region in LLM Reinforcement Learning](https://arxiv.org/abs/2602.04879) 
* **Algorithm:** DPPO 
* **Venue/Year:** arXiv 2026 
* **Reference Code:** [sail-sg/Stable-RL](https://github.com/sail-sg/Stable-RL) 
* **Local PDF:** [Qi et al. - 2026 - Rethinking the Trust Region in LLM Reinforcement Learning.pdf](<papers/Qi et al. - 2026 - Rethinking the Trust Region in LLM Reinforcement Learning.pdf>) 
* **Rating:** `7.7/10`
* > **Description:**
> * **Core Intuition:** Standard PPO ratio clipping is ill-suited for the expansive vocabularies of LLMs because it over-penalizes low-probability tokens and under-penalizes high-probability ones. DPPO addresses this by substituting the heuristic single-sample ratio clipping with a dynamic mask based on a direct estimation of policy divergence. By utilizing highly efficient Binary and Top-K divergence approximations to minimize memory overhead, it guarantees that updates remain within a theoretically grounded trust region, thereby significantly improving both training stability and efficiency.
> * **Mathematical Formulation:** 
> 
> $$L_{\mu}^{DPPO}(\pi)=\mathbb{E}_{y\sim\mu}[\sum_{t=1}^{|y|} M_t^{DPPO}\cdot \frac{\pi(y_t\mid s_t)}{\mu(y_t\mid s_t)}\cdot \hat A_t]$$

#### 32. [VESPO: Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Training](https://arxiv.org/abs/2602.10693) 
* **Algorithm:** VESPO 
* **Venue/Year:** arXiv 2026 
* **Reference Code:** [FloyedShen/VESPO](https://github.com/FloyedShen/VESPO) 
* **Local PDF:** [Shen et al. - 2026 - VESPO - Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Training.pdf](<papers/Shen et al. - 2026 - VESPO - Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Training.pdf>) 
* **Rating:** `8.1/10`
> Description:
> * Core Intuition: Designed to stabilize off-policy reinforcement learning against severe policy staleness and train-inference mismatches. Rather than relying on heuristic token-level clipping or biased length normalization, it derives a closed-form reshaping kernel from a variational formulation. This kernel operates directly on sequence-level importance weights to smoothly suppress extreme values without memory overhead, enabling highly stable training on small to large models even under fully asynchronous execution and high staleness ratios.
> * Mathematical Formulation:
> 
> $$\nabla\mathcal{J}_{VESPO}=\mathbb{E}_{\tau\sim\mu}[W^{c_1}\exp(c_2(1-W))\cdot A(\tau)\cdot\nabla_\theta\log\pi_\theta(\tau)]$$

#### 33. [Token-Importance Guided Direct Preference Optimization](https://arxiv.org/abs/2505.19653) 
* **Algorithm:** TI-DPO 
* **Venue/Year:** ICLR 2026 
* **Reference Code:** [gracefulning/TIDPO](https://github.com/gracefulning/TIDPO) 
* **Local PDF:** [Yang et al. - 2026 - Token-Importance Guided Direct Preference Optimization.pdf](<papers/Yang et al. - 2026 - Token-Importance Guided Direct Preference Optimization.pdf>) 
* **Rating:** `7.5/10`
> **Description:**
> * **Core Intuition:** Addresses the sequence-level limitations of standard DPO by explicitly modeling token-level importance. It computes a hybrid token weight using gradient attribution and a Gaussian prior to robustly identify critical tokens while mitigating the architectural "Lost-in-the-Middle" bias of LLMs. Furthermore, it integrates a triplet loss to structurally guide the model's intermediate outputs toward preferred responses and away from non-preferred ones, enabling fine-grained semantic alignment at the cost of one additional backward pass per sequence during training.
> * **Mathematical Formulation:**
> 
> $$\mathcal{L}_{TI-DPO} = \mathcal{L}_{DPO-w} + \gamma\mathcal{L}_{triplet}$$


#### 34. [STAPO: Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens](https://arxiv.org/abs/2602.15620)
* **Algorithm:** STAPO 
* **Venue/Year:** arXiv 2026 
* **Reference Code:** N/A
* **Local PDF:** [Liu et al. - 2026 - STAPO - Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens.pdf.pdf](<papers/Liu et al. - 2026 - STAPO - Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens.pdf.pdf>)
* **Rating:** `8.0/10`
> **Description:**
> * **Core Intuition:** RL fine-tuning frequently suffers from instability caused by a tiny fraction (around 0.01%) of "spurious tokens" that contribute little to reasoning correctness but inherit full sequence-level rewards, triggering abnormally amplified gradient updates. STAPO efficiently identifies these tokens (via low probability, low entropy, and positive advantage signals) and masks them to suppress gradient perturbations, proving highly robust and effective for stabilizing RL across various scales, including smaller 1.7B models.
> * **Mathematical Formulation:**
> 
> $$\mathcal J_{STAPO}(\theta)=\mathbb E_{x\sim\mathcal D,\{y_i\}_{i=1}^G\sim\pi_{\theta_{old}}}[\frac{1}{G}\sum_{i=1}^G \frac{1}{\sum_{t=1}^{|y_i|} I_{i,t}^{S2T}} \sum_{t=1}^{|y_i|} I_{i,t}^{S2T}\min(\rho_{i,t}(\theta)\hat A_i,\mathrm{clip}(\rho_{i,t}(\theta),1-\epsilon_{low},1+\epsilon_{high})\hat A_i)]$$

#### 35. [CLIPO: Contrastive Learning in Policy Optimization Generalizes RLVR](https://arxiv.org/abs/2603.10101)
* **Algorithm:** CLIPO 
* **Venue/Year:** arXiv 2026 
* **Reference Code:** [Qwen-Applications/CLIPO](https://github.com/Qwen-Applications/CLIPO) 
* **Local PDF:** [Cui et al. - 2026 - CLIPO - Contrastive Learning in Policy Optimization Generalizes RLVR.pdf](<papers/Cui et al. - 2026 - CLIPO - Contrastive Learning in Policy Optimization Generalizes RLVR.pdf>)
* **Rating:** `8.2/10`
> **Description:**
> * **Core Intuition:** Designed to improve reasoning generalization by addressing the sparse, outcome-only nature of standard RLVR. It introduces a lightweight contrastive head to maximize semantic similarity among successful reasoning trajectories within a rollout group while distancing them from erroneous ones. This acts as a denoising mechanism that distills invariant logical structures into a dense auxiliary reward, effectively mitigating hallucinations and step-level inconsistencies without requiring expensive process-level annotations. It performs exceptionally well on 3B to 8B parameter models.
> * **Mathematical Formulation:**
> 
> $$\max_{\pi_{\theta}} \mathbb{E}[r(x,y)] + \lambda \cdot I(y;\overline{y}|x,\mathcal{P}_{y,\overline{y}})$$

#### 36. [DPO Meets PPO: Reinforced Token Optimization for RLHF](https://arxiv.org/abs/2404.18922)  
* **Algorithm:** RTO (Reinforced Token Optimization) 
* **Venue/Year:** arXiv 2024 
* **Reference Code:** [zkshan2002/RTO](https://github.com/zkshan2002/RTO)  
* **Local PDF:** [DPO Meets PPO - Reinforced Token Optimization for RLHF](<papers/Zhong et al. - 2024 - DPO Meets PPO - Reinforced Token Optimization for RLHF.pdf>)
* **Rating:** 7.8/10
> **Description:**
> * **Core Intuition:** Standard PPO for RLHF struggles with sample inefficiency due to sparse, sentence-level rewards. RTO mitigates this by modeling RLHF as a token-wise Markov Decision Process (MDP). It extracts dense, token-level reward signals from an implicitly learned Direct Preference Optimization (DPO) model and uses them for fine-grained reward shaping. This allows PPO to perform step-by-step optimization, drastically improving sample efficiency while maintaining the robustness of RL training. 
> * **Mathematical Formulation:**
> 
> $$r_{rto}(x,y_{1:h})=\begin{cases}\beta_{1}\log\frac{\pi_{dpo}(y_{h}|x,y_{1:h-1})}{\pi_{ref}(y_{h}|x,y_{1:h-1})}-\beta_{2}\log\frac{\pi(y_{h}|x,y_{1:h-1})}{\pi_{ref}(y_{h}|x,y_{1:h-1})}&h\le H-1\\\beta_{1}\log\frac{\pi_{dpo}(y_{h}|x,y_{1:h-1})}{\pi_{ref}(y_{h}|x,y_{1:h-1})}-\beta_{2}\log\frac{\pi(y_{h}|x,y_{1:h-1})}{\pi_{ref}(y_{h}|x,y_{1:h-1})}+\beta_{3}r_{MLE}(x,y_{1:H})&h=H\end{cases}$$ 

#### 37. [DISPO: Enhancing Training Efficiency and Stability in Reinforcement Learning for Large Language Model Mathematical Reasoning](https://arxiv.org/abs/2602.00983)  
* **Algorithm:** DISPO  
* **Venue/Year:** AISTATS 2026 / arXiv 2026  
* Reference Code: N/A 
* **Local PDF:** [DISPO - Enhancing Training Efficiency and Stability in Reinforcement Learning for Large Language Model Mathematical Reasoning](<papers/Karaman et al. - 2026 - DISPO - Enhancing Training Efficiency and Stability in Reinforcement Learning for Large Language Model Mathematical Reasoning.pdf>) 
* **Rating:** 8.3/10 
> **Description:**
> * **Core Intuition:** Addresses the clear trade-off between stable but slow PPO-style methods and efficient but highly unstable REINFORCE-style methods.  It achieves this by decoupling the clipping of importance sampling (IS) weights into four distinct regimes based on the advantage sign (correct vs. incorrect responses) and IS magnitude.  By tuning these regimes independently, DISPO maintains an optimal balance between exploration and distillation while preventing catastrophic failures like repetitive outputs or vanishing response lengths. 
> * **Mathematical Formulation:**
> 
> $$J_{DISPO}(\theta)=\mathbb E_{(q,a)\sim\mathcal D,\{o_i\}_{i=1}^G\sim\pi_{ref}(\cdot\mid q)}[\frac{1}{\sum_{i=1}^G |o_i|}\sum_{i=1}^G\sum_{t=1}^{|o_i|} \mathrm{sg}(r_{i,t}^d(\theta))\hat A_{i,t}\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})]$$

#### 38. [Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models](https://arxiv.org/abs/2505.23564)
* **Algorithm:** SPO (SPO-chain & SPO-tree)
* **Venue/Year:** NeurIPS 2025 
* **Reference Code:** [AIFrameResearch/SPO](https://github.com/AIFrameResearch/SPO) 
* **Local PDF:** [Segment Policy Optimization - Effective Segment-Level Credit Assignment in RL for Large Language Models](<papers/Guo et al. - 2025 - Segment Policy Optimization - Effective Segment-Level Credit Assignment in RL for Large Language Models.pdf>)
* **Rating:** 8.2/10
> **Description:**
> * **Core Intuition:** Designed to solve the credit assignment problem in LLM reasoning by operating at an intermediate segment granularity. It bridges the gap between inaccurate token-level advantage estimation (which requires training an unstable critic model) and imprecise trajectory-level estimation (like standard GRPO). By partitioning sequences into segments, estimating segment values via Monte Carlo sampling, and applying these advantages exclusively to low-confidence "cutpoint" tokens using a probability mask, it achieves high sample efficiency and highly precise credit assignment on small and medium models without the overhead of a critic.
> * **Mathematical Formulation:**
> 
> $$\mathcal J_{SPO}(\theta)=\mathbb E[\frac{1}{Z}\sum_{k=1}^{K}\sum_{t=t_k}^{t_{k+1}-1}(M_t\cdot \min(r_t(\theta)\hat A_k^{seg},\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_k^{seg})-\beta D_{KL}(\pi_\theta\|\pi_{ref}))]$$

#### 39. [AAPO: Enhancing the Reasoning Capabilities of LLMs with Advantage Momentum](https://arxiv.org/abs/2505.14264) 
* **Algorithm:** AAPO 
* **Venue/Year:** arXiv 2025 
* **Reference Code:** N/A 
* **Local PDF:** [AAPO - Enhancing the Reasoning Capabilities of LLMs with Advantage Momentum](<papers/Xiong et al. - 2025 - AAPO - Enhancing the Reasoning Capabilities of LLMs with Advantage Momentum.pdf>) 
* **Rating:** 8.0/10 
> **Description:**
> * **Core Intuition:** Addresses training inefficiencies—such as zero gradients—in group relative advantage estimation methods (like GRPO) that occur when within-group rewards exhibit low variance (e.g., when all sampled responses are similarly good or bad). It resolves this by introducing an "advantage momentum" term, which compares the rewards of the current policy model's responses against those from a frozen reference model. This effectively provides a reliable optimization signal that prevents the advantage from dropping to zero, ensuring stable and continuous learning even in late-stage RL, proving highly efficient for small language models (like 1.5B, 3B, and 7B architectures).
> * **Mathematical Formulation:**
> 
> $$\hat A_{i,t}^{*}=\frac{r_{\theta_i}-\mathrm{mean}(r_\theta)}{\mathrm{std}(r_\theta)}+\mathrm{clip}(r_{\theta_i}-r_{ref_i},\delta_{low},\delta_{high})$$

#### 40. [SortedRL: Accelerating RL Training for LLMs through Online Length-Aware Scheduling](https://arxiv.org/abs/2603.23414)

  * **Algorithm:** SortedRL
  * **Venue/Year:** arXiv 2026
  * **Reference Code:** N/A
  * **Local PDF:** [Zhang et al. - 2026 - SortedRL - Accelerating RL Training for LLMs through Online Length-Aware Scheduling.pdf](<papers/Zhang et al. - 2026 - SortedRL - Accelerating RL Training for LLMs through Online Length-Aware Scheduling.pdf>)
  * **Rating:** `7.5/10`
> **Description:**
>
>   * **Core Intuition:** SortedRL attacks the systems bottleneck of reasoning RL rather than replacing the base policy objective. It sorts rollout samples online by generated length so short responses can be grouped and consumed first, which lets updates start earlier, shrinks rollout-update bubbles, and creates a near on-policy micro-curriculum. This is a practical fit for small-model policy optimization when wall-clock budget is tight.
>
>   * **Mathematical Formulation:** The policy loss stays in the GRPO/DAPO family, while the data pipeline is changed by a length-aware schedule that sorts responses by output length and forms early update groups from the shortest rollouts first:
>
> $$\sigma=\mathrm{argsort}_i\,|o_i|$$
>
>   * With bounded cache reuse, SortedRL also controls how much stale off-policy data is admitted during asynchronous rollout-update overlap.

#### 41. [Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](https://arxiv.org/abs/2602.23008)

  * **Algorithm:** EMPO2 (Exploratory Memory-Augmented On- and Off-Policy Optimization)
  * **Venue/Year:** ICLR 2026
  * **Reference Code:** [microsoft/agent-lightning](https://github.com/microsoft/agent-lightning/tree/main/contrib/recipes/envs)
  * **Local PDF:** [Liu et al. - 2026 - Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization.pdf](<papers/Liu et al. - 2026 - Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization.pdf>)
  * **Rating:** `6.7/10`
> **Description:**
>
>   * **Core Intuition:** EMPO2 focuses on exploration-heavy agent environments, where pretrained priors are not enough. It adds a self-generated memory buffer of reflective tips and combines on-policy improvement with off-policy consolidation so the agent can benefit from memory during exploration but still remain robust when memory is absent. The match to small-model policy optimization is moderate because it is more agent-centric than pure reasoning RLVR.
>
>   * **Mathematical Formulation:** EMPO2 optimizes a hybrid on/off-policy objective over standard trajectories and memory-augmented replay:
>
> $$\mathcal J_{EMPO2}(\theta)=\mathcal J_{on}(\theta;\tau\sim\pi_{\theta_{old}})+\lambda\mathcal J_{off}(\theta;\tau\sim\mathcal B_{\mathcal M})$$
>
>   * Here $\mathcal{B}_{\mathcal{M}}$ stores memory-augmented experience, and the off-policy branch transfers exploratory discoveries back into the policy.

#### 42. [MHPO: Modulated Hazard-aware Policy Optimization for Stable Reinforcement Learning](https://arxiv.org/abs/2603.16929)

  * **Algorithm:** MHPO
  * **Venue/Year:** arXiv 2026
  * **Reference Code:** N/A
  * **Local PDF:** [Wang et al. - 2026 - MHPO - Modulated Hazard-aware Policy Optimization for Stable Reinforcement Learning.pdf](<papers/Wang et al. - 2026 - MHPO - Modulated Hazard-aware Policy Optimization for Stable Reinforcement Learning.pdf>)
  * **Rating:** `8.0/10`
> **Description:**
>
>   * **Core Intuition:** MHPO is a stability-oriented xPO variant for GRPO-style training. Instead of hard clipping importance ratios, it maps them into a bounded differentiable domain with a Log-Fidelity Modulator and then adds asymmetric hazard penalties to separately control over-expansion and over-contraction. This makes it broadly relevant for small-model policy optimization, where noisy updates can destabilize training quickly.
>
>   * **Mathematical Formulation:** The method replaces raw ratio control with a modulated ratio and hazard-aware regularization:
>
> $$\tilde\rho_{i,t}=\tanh(\alpha\log\rho_{i,t}),\qquad \mathcal L_{MHPO}=-\tilde\rho_{i,t}\hat A_{i,t}+\lambda_+ H_+(\rho_{i,t})+\lambda_- H_-(\rho_{i,t})$$
>
>   * The hazard terms shape a smoother trust-region-like landscape than hard clipping and explicitly suppress extreme policy shifts.

#### 43. [LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](https://arxiv.org/abs/2603.02146)

  * **Algorithm:** LongRLVR
  * **Venue/Year:** ICLR 2026
  * **Reference Code:** [real-absolute-AI/LongRLVR](https://github.com/real-absolute-AI/LongRLVR)
  * **Local PDF:** [Chen et al. - 2026 - LongRLVR - Long-Context Reinforcement Learning Requires Verifiable Context Rewards.pdf](<papers/Chen et al. - 2026 - LongRLVR - Long-Context Reinforcement Learning Requires Verifiable Context Rewards.pdf>)
  * **Rating:** `7.8/10`
> **Description:**
>
>   * **Core Intuition:** LongRLVR shows that answer-only RLVR breaks in long-context settings because the grounding step gets almost no usable gradient. Its fix is to explicitly reward context selection: the policy first identifies support chunks, then receives a dense and verifiable context reward before answer generation. This is especially relevant when small models must reason over retrieved or externally provided long contexts.
>
>   * **Mathematical Formulation:** The paper makes the reward decomposition explicit by augmenting answer reward with a verifiable context reward:
>
> $$r_{total}(y,Z)=r_{ans}(y)+r_{ctx}(Z,G),\qquad \mathcal J(\theta)=\mathbb E_{(X,Q,G)}\,\mathbb E_{(Z,y)\sim\pi_\theta}[r_{ans}(y)+r_{ctx}(Z,G)]$$
>
>   * In practice, $r_{ctx}$ is instantiated with an $F_\beta$-style grounding score over the selected chunk identifiers.

#### 44. [FIPO: Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization](https://arxiv.org/abs/2603.19835)

  * **Algorithm:** FIPO
  * **Venue/Year:** arXiv 2026
  * **Reference Code:** [qwenpilot/FIPO](https://github.com/qwenpilot/FIPO)
  * **Local PDF:** [Ma et al. - 2026 - FIPO - Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization.pdf](<papers/Ma et al. - 2026 - FIPO - Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization.pdf>)
  * **Rating:** `8.7/10`
> **Description:**
>
>   * **Core Intuition:** FIPO targets the coarse credit assignment of GRPO-style RL, where one sequence-level advantage is broadcast across all tokens. It injects discounted future-KL influence into the token update so present tokens are weighted by how much they affect subsequent trajectory behavior. For small-model policy optimization on long reasoning tasks, this is one of the most directly relevant recent ideas.
>
>   * **Mathematical Formulation:** FIPO replaces the standard token advantage with a dense future-aware term that aggregates discounted future-KL influence:
>
> $$\hat A_t^{FIPO}=\hat A_t+\lambda\sum_{k=t+1}^{T}\gamma^{k-t}\,FutureKL_k$$
>
>   * The resulting $\hat A_t^{FIPO}$ is then inserted into a clipped GRPO/DAPO-style surrogate objective together with influence clipping and filtering for stability.

#### 45. [ERPO: Token-Level Entropy-Regulated Policy Optimization for Large Reasoning Models](https://arxiv.org/abs/2603.28204)

  * **Algorithm:** ERPO
  * **Venue/Year:** arXiv 2026
  * **Reference Code:** N/A
  * **Local PDF:** [Yu et al. - 2026 - ERPO - Token-Level Entropy-Regulated Policy Optimization for Large Reasoning Models.pdf](<papers/Yu et al. - 2026 - ERPO - Token-Level Entropy-Regulated Policy Optimization for Large Reasoning Models.pdf>)
  * **Rating:** `8.3/10`
> **Description:**
>
>   * **Core Intuition:** ERPO argues that uniform sequence-level advantages prematurely collapse entropy at critical decision pivots in reasoning traces. It therefore shifts optimization to fine-grained token dynamics by detecting high-entropy decision points, amplifying exploration there, normalizing token signals by progress buckets, and re-synthesizing advantages with outcome anchors. This fine-grained credit assignment is highly relevant to small reasoning models.
>
>   * **Mathematical Formulation:** ERPO defines entropy-aware gating and result-anchored token synthesis to construct its final token-level update signal:
>
> $$W_{i,t}=\sigma(\gamma\,Norm_G(H_{i,t})),\qquad \Psi_{i,t}=W_{i,t}\,\mathrm{sgn}(\hat A_i^{group})\tilde s_{i,t}$$
>
>   * These token-level quantities are then normalized and assembled into the final ERPO advantage used for policy optimization.

#### 46. [HDPO: Hybrid Distillation Policy Optimization via Privileged Self-Distillation](https://arxiv.org/abs/2603.23871)

  * **Algorithm:** HDPO
  * **Venue/Year:** arXiv 2026
  * **Reference Code:** N/A
  * **Local PDF:** [Ding - 2026 - HDPO - Hybrid Distillation Policy Optimization via Privileged Self-Distillation.pdf](<papers/Ding - 2026 - HDPO - Hybrid Distillation Policy Optimization via Privileged Self-Distillation.pdf>)
  * **Rating:** `9.4/10`
> **Description:**
>
>   * **Core Intuition:** HDPO directly targets cliff prompts where ordinary RL has zero gradient because every rollout fails. It uses the same model as a privileged teacher by appending ground-truth information on failed prompts, filters correct privileged rollouts, and distills them back to the student with a mode-covering divergence. Because the empirical study is on Qwen2.5-Math-1.5B-Instruct, the paper is exceptionally well aligned with small-model policy optimization.
>
>   * **Mathematical Formulation:** HDPO augments the base RL objective with privileged self-distillation on cliff prompts:
>
> $$\mathcal J_{HDPO}(\theta)=\mathcal J_{RL}(\theta)+\lambda\,JSD(\pi_\theta^{student}\|\pi_\theta^{teacher,priv})$$
>
>   * The paper further proves that $R=1$ filtered privileged rollouts recover the KL-regularized optimal target distribution in the hard-threshold limit.

#### 47. [Reaching Beyond the Mode: RL for Distributional Reasoning in Language Models](https://arxiv.org/abs/2603.24844)

  * **Algorithm:** Multi-Answer RL
  * **Venue/Year:** arXiv 2026
  * **Reference Code:** [ishapuri/multi_answer_rl](https://github.com/ishapuri/multi_answer_rl)
  * **Local PDF:** [Puri et al. - 2026 - Reaching Beyond the Mode - RL for Distributional Reasoning in Language Models.pdf](<papers/Puri et al. - 2026 - Reaching Beyond the Mode - RL for Distributional Reasoning in Language Models.pdf>)
  * **Rating:** `7.4/10`
> **Description:**
>
>   * **Core Intuition:** This paper argues that standard single-answer RL collapses the model onto one dominant mode even when tasks require multiple plausible answers. Its Multi-Answer RL objective instead trains the model to emit several candidate answers plus confidence estimates in a single forward pass, improving diversity, coverage, and calibration. The match to small-model policy optimization is good when multi-hypothesis reasoning matters, but it is less central for classic single-answer RLVR.
>
>   * **Mathematical Formulation:** The method optimizes a set-valued policy objective over a jointly generated answer set:
>
> $$\mathcal J_{MARL}(\theta)=\mathbb E_{(x,Y)}[r_{set}(Y)],\qquad Y=\{y_1,\dots,y_K\}$$
>
>   * Here $r_{set}$ rewards distributional quality such as coverage, diversity, and calibration rather than only the top-mode answer.

#### 48. [Soft Sequence Policy Optimization](https://arxiv.org/abs/2602.19327)
* **Algorithm:** SoftSPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** N/A
* **Local PDF:** [Glazyrina et al. - 2026 - Soft Sequence Policy Optimization.pdf](<papers/Glazyrina et al. - 2026 - Soft Sequence Policy Optimization.pdf>)
* **Rating:** `8.0/10`
> **Description:**
> * **Core Intuition:** Replaces hard sequence-level ratio clipping with a smoother, soft weighting rule so off-policy or sequence-level RL remains stable without sharp truncation artifacts.
> * **Engineering Traits:** Relevant to SLMs because smooth ratio control can reduce update discontinuities that otherwise trigger instability under low-capacity, low-rollout regimes.
> * **Mathematical Formulation:** A representative soft sequence objective takes the form
>
> $$L_{SoftSPO}(\theta)=\mathbb E_{\tau\sim\mu}[g(W_\theta(\tau))\,A(\tau)]$$
>
> where $W_\theta(\tau)=\pi_\theta(\tau)/\mu(\tau)$ and $g(\cdot)$ is a smooth shaping kernel instead of a hard clip.
> * **Policy Update:** The policy is updated with smoothly down-weighted sequence importance weights, preserving off-policy usability while avoiding abrupt gradient truncation.

#### 49. [Single-stream Policy Optimization](https://arxiv.org/abs/2509.13232)
* **Algorithm:** SSPO-Stream
* **Venue/Year:** ArXiv 2025
* **Reference Code:** N/A
* **Local PDF:** [Xu and Ding - 2025 - Single-stream Policy Optimization.pdf](<papers/Xu and Ding - 2025 - Single-stream Policy Optimization.pdf>)
* **Rating:** `7.6/10`
> **Description:**
> * **Core Intuition:** Simplifies policy optimization into a single streamlined training flow, reducing multi-branch RL system complexity.
> * **Engineering Traits:** More systems-oriented than mechanism-oriented, but still relevant for SLM pipelines where software complexity itself becomes a bottleneck.
> * **Mathematical Formulation:** The paper emphasizes a unified policy update stream rather than a separate critic/reference pipeline, yielding a simplified objective of the generic form
>
> $$L_{SSPO-stream}(\theta)=\mathbb E_{(x,y)}[w_\theta(x,y)\,A(x,y)]$$
>
> * **Policy Update:** A single policy stream handles both rollout credit and update application, reducing synchronization overhead.

#### 50. [Bottom-up Policy Optimization: Your Language Model Policy Secretly Contains Internal Policies](https://arxiv.org/abs/2512.19673)
* **Algorithm:** BuPO
* **Venue/Year:** ArXiv 2025
* **Reference Code:** [Trae1ounG/BuPO](https://github.com/Trae1ounG/BuPO)
* **Local PDF:** [Tan et al. - 2026 - Bottom-up Policy Optimization Your Language Model Policy Secretly Contains Internal Policies.pdf](<papers/Tan et al. - 2026 - Bottom-up Policy Optimization Your Language Model Policy Secretly Contains Internal Policies.pdf>)
* **Rating:** `7.5/10`
> **Description:**
> * **Core Intuition:** Argues that a language model contains multiple latent internal policies and that alignment should be carried out bottom-up rather than as one monolithic global update.
> * **Engineering Traits:** Interesting for the project's knowledge-retention dimension because it suggests a route to more selective updates, though it is less directly standardized than DPO/GRPO-style baselines.
> * **Mathematical Formulation:** The method decomposes a global objective into sub-policy level updates, conceptually of the form
>
> $$\mathcal J_{BuPO}(\theta)=\sum_{m=1}^{M}\lambda_m\,\mathcal J_m(\theta)$$
>
> where each $\mathcal J_m$ corresponds to a latent internal policy or attribute-specific behavior.
> * **Policy Update:** The optimizer encourages targeted sub-policy adaptation instead of uniformly pushing all behaviors with the same global reward.

#### 51. [AT$^2$PO: Agentic Turn-based Policy Optimization via Tree Search](https://arxiv.org/abs/2601.04767)
* **Algorithm:** AT$^2$PO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** [zzfoutofspace/ATPO](https://github.com/zzfoutofspace/ATPO)
* **Local PDF:** [Zong et al. - 2026 - AT$^2$PO Agentic Turn-based Policy Optimization via Tree Search.pdf](<papers/Zong et al. - 2026 - AT$^2$PO Agentic Turn-based Policy Optimization via Tree Search.pdf>)
* **Rating:** `6.8/10`
> **Description:**
> * **Core Intuition:** Extends policy optimization to multi-turn agentic settings by combining turn-level decision making with look-ahead tree search.
> * **Engineering Traits:** Valuable for agentic workflows and tool use, but less central than classic reasoning RLVR for your current SLM project scope.
> * **Mathematical Formulation:** The objective couples policy optimization with turn-level search-derived value estimates:
>
> $$\mathcal J_{AT^2PO}(\theta)=\mathbb E_{\tau}[\sum_{u=1}^{U} \hat A_u^{tree}\,\log\pi_\theta(a_u\mid s_u)]$$
>
> * **Policy Update:** Search-expanded trajectories provide improved turn-level credit, which is then distilled back into the base policy.

### Additional Recent Papers to Watch

#### 52. [$f$-PO: Generalizing Preference Optimization with $f$-divergence Minimization](https://arxiv.org/abs/2410.21662)
* **Algorithm:** f-PO
* **Venue/Year:** ArXiv 2024
* **Reference Code:** N/A
* **Local PDF:** [Han et al. - 2025 - $f$-PO generalizing preference optimization with $f$-divergence minimization.pdf](<papers/Han et al. - 2025 - $f$-PO generalizing preference optimization with $f$-divergence minimization.pdf>)
* **Rating:** `8.1/10`
> **Description:**
> * **Core Intuition:** Unifies many offline preference optimization objectives under a single $f$-divergence perspective, clarifying how DPO-like variants differ mainly in their divergence geometry and induced gradient weighting.
> * **Engineering Traits:** More theoretical than directly benchmark-oriented, but highly useful for your project because it strengthens the bridge from preference feedback form to update geometry, which is exactly what GITA tries to explain.
> * **Mathematical Formulation:** A representative variational form is
>
> $$L_{f-PO}(\theta)=\mathbb E[T_\phi(z_\theta)]-\mathbb E[f^*(T_\phi(z_\theta))]$$
>
> where $z_\theta$ is a preference-induced score and $f^*$ is the convex conjugate of the chosen divergence.
> * **Policy Update:** Different choices of $f$ recover different policy-update behaviors, ranging from conservative anchoring to more aggressive preference sharpening.

#### 53. [Is On-Policy Data always the Best Choice for Direct Preference Optimization-based LM Alignment?](https://arxiv.org/abs/2508.10530)
* **Algorithm:** On-policy DPO Analysis
* **Venue/Year:** ArXiv 2025
* **Reference Code:** N/A
* **Local PDF:** [Sun et al. - 2026 - Is on-policy data always the best choice for direct preference optimization-based LM alignment.pdf](<papers/Sun et al. - 2026 - Is on-policy data always the best choice for direct preference optimization-based LM alignment.pdf>)
* **Rating:** `7.8/10`
> **Description:**
> * **Core Intuition:** Re-examines the common belief that on-policy preference data is always superior for DPO-style alignment, and studies when static data, refreshed data, or on-policy data are actually beneficial.
> * **Engineering Traits:** Very relevant to your project design because it directly informs whether preference data refresh changes update geometry or merely changes effective supervision quality.
> * **Mathematical Formulation:** The paper studies DPO objectives under changing data distributions rather than proposing a wholly new loss, i.e.
>
> $$\mathcal L_{DPO}(\theta;\mathcal D_t)=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D_t}[\log\sigma(\Delta_\theta(x,y_w,y_l))]$$
>
> where the focus is how the choice of $\mathcal D_t$ changes training dynamics.
> * **Policy Update:** The key claim is that the data source itself changes the direction and quality of preference gradients, which matters for small-model stability and retention.

#### 54. [SGPO: Self-Generated Preference Optimization based on Self-Improver](https://arxiv.org/abs/2507.20181)

* **Algorithm:** SGPO
* **Venue/Year:** ArXiv 2025
* **Reference Code:** N/A
* **Local PDF:** [Lee et al. - 2025 - SGPO self-generated preference optimization based on self-improver.pdf](<papers/Lee et al. - 2025 - SGPO self-generated preference optimization based on self-improver.pdf>)
* **Rating:** `7.6/10`
> **Description:**
> * **Core Intuition:** Uses the model's own improving generations to construct preference signals, reducing dependence on externally curated preference datasets.
> * **Engineering Traits:** Interesting for scaling data creation, but less controlled than fixed offline datasets, so it is better treated as a promising extension rather than a primary benchmark for your mechanism study.
> * **Mathematical Formulation:** SGPO still fits the DPO-style pairwise preference pattern, but with self-generated preference pairs:
>
> $$\mathcal L_{SGPO}(\theta)=-\mathbb E_{(x,y^+,y^-)\sim\mathcal D_{self}}[\log\sigma(\Delta_\theta(x,y^+,y^-))]$$
>
> * **Policy Update:** The policy is iteratively improved on preference pairs generated from its own stronger snapshots or self-improving loop.

#### 55. [TAPO: Translation Augmented Policy Optimization for Multilingual Mathematical Reasoning](https://arxiv.org/abs/2603.25419)

* **Algorithm:** TAPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** N/A
* **Local PDF:** [Huang et al. - 2026 - TAPO Translation Augmented Policy Optimization for Multilingual Mathematical Reasoning.pdf](<papers/Huang et al. - 2026 - TAPO Translation Augmented Policy Optimization for Multilingual Mathematical Reasoning.pdf>)
* **Rating:** `7.4/10`
> **Description:**
> * **Core Intuition:** Improves multilingual mathematical reasoning by augmenting policy optimization with translated supervision or translated rollout signals, reducing the language gap in rewardable reasoning trajectories.
> * **Engineering Traits:** More niche than your core English-centric SLM study, but worth tracking because it shows how policy optimization interacts with language transfer and multilingual reward sparsity.
> * **Mathematical Formulation:** A representative TAPO objective can be viewed as a multilingual augmentation of the base RL/RLVR loss:
>
> $$\mathcal J_{TAPO}(\theta)=\mathcal J_{base}(\theta)+\lambda\,\mathcal J_{trans}(\theta)$$
>
> * **Policy Update:** The policy learns from both original and translated reasoning signals, improving reward coverage across languages.

#### 56. [Target Policy Optimization](https://arxiv.org/abs/2604.06159)

* **Algorithm:** TPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** [JeanKaddour/tpo](https://github.com/JeanKaddour/tpo)
* **Local PDF:** [Kaddour - 2026 - Target Policy Optimization.pdf](<papers/Kaddour - 2026 - Target Policy Optimization.pdf>)
* **Rating:** `8.2/10`
> **Description:**
> * **Core Intuition:** Separates the question of which sampled completions should gain probability mass from how the model parameters should move, replacing fragile policy-gradient updates with fitting to an explicitly constructed target distribution.
> * **Engineering Traits:** Very relevant to SLMs because sparse rewards and small rollout groups often make GRPO-style gradients noisy or under-informative; TPO offers a supervised-learning-like update while staying in the online RLVR setting.
> * **Mathematical Formulation:** Given scored completions with old probabilities $p_i^{old}$ and utilities $u_i$, TPO constructs
>
> $$q_i=\frac{p_i^{old}\exp(u_i)}{\sum_j p_j^{old}\exp(u_j)},\qquad \mathcal L_{TPO}(\theta)=-\sum_i q_i\log p_\theta(y_i\mid x)$$
>
> * **Policy Update:** Gradient descent on cross-entropy moves the current policy toward the target distribution $q$ and naturally stops once $p_\theta$ matches the desired redistribution.

#### 57. [Provable and Practical In-Context Policy Optimization for Self-Improvement](https://arxiv.org/abs/2603.01335)

* **Algorithm:** ICPO / ME-ICPO
* **Venue/Year:** ICLR 2026
* **Reference Code:** N/A
* **Local PDF:** [Yu et al. - 2026 - Provable and Practical In-Context Policy Optimization for Self-Improvement.pdf](<papers/Yu et al. - 2026 - Provable and Practical In-Context Policy Optimization for Self-Improvement.pdf>)
* **Rating:** `7.6/10`
> **Description:**
> * **Core Intuition:** Treats multi-round self-reflection at inference time as an in-context policy optimization process, where the model uses its own response and self-assessed reward to refine later responses without parameter updates.
> * **Engineering Traits:** Useful for test-time scaling studies, though it is less directly comparable to train-time RLVR methods because improvement happens through prompting and self-selection rather than optimizer steps.
> * **Mathematical Formulation:** A simplified ICPO loop can be viewed as repeatedly updating the in-context state
>
> $$c_{k+1}=f(c_k,y_k,\hat r_k),\qquad y_{k+1}\sim\pi_\theta(\cdot\mid x,c_{k+1})$$
>
> where ME-ICPO selects low-entropy self-assessed rewards to make the feedback signal more reliable.
> * **Policy Update:** There is no weight update at inference time; the effective policy is improved by conditioning on self-generated reward traces and minimum-entropy response selection.

#### 58. [ESPO: Entropy Importance Sampling Policy Optimization](https://arxiv.org/abs/2512.00499)

* **Algorithm:** ESPO
* **Venue/Year:** ArXiv 2025
* **Reference Code:** [ShopeeLLM/ESPO](https://github.com/ShopeeLLM/ESPO)
* **Local PDF:** [Sheng et al. - 2025 - ESPO - Entropy Importance Sampling Policy Optimization.pdf](<papers/Sheng et al. - 2025 - ESPO - Entropy Importance Sampling Policy Optimization.pdf>)
* **Rating:** `8.3/10`
> **Description:**
> * **Core Intuition:** Groups tokens by predictive entropy so high-uncertainty decision points receive finer optimization while low-entropy tokens avoid unnecessary noisy updates.
> * **Engineering Traits:** Strongly aligned with the project's credit-assignment dimension: it keeps token-level information but uses entropy grouping and adaptive clipping to reduce variance.
> * **Mathematical Formulation:** ESPO can be summarized as applying entropy-conditioned importance sampling and clipping:
>
> $$L_{ESPO}(\theta)=\mathbb E_{t}[\min(w_tA_t,\mathrm{clip}(w_t,1-\epsilon(H_t),1+\epsilon(H_t))A_t)]$$
>
> where $H_t$ controls the token group and the local trust region.
> * **Policy Update:** High-entropy tokens get more carefully weighted updates, improving gradient utilization without letting unstable token-level ratios dominate training.

#### 59. [Trust-Region Adaptive Policy Optimization](https://arxiv.org/abs/2512.17636)

* **Algorithm:** TRAPO
* **Venue/Year:** ArXiv 2025
* **Reference Code:** [Su-my/TRAPO](https://github.com/Su-my/TRAPO)
* **Local PDF:** [Su et al. - 2025 - Trust-Region Adaptive Policy Optimization.pdf](<papers/Su et al. - 2025 - Trust-Region Adaptive Policy Optimization.pdf>)
* **Rating:** `8.0/10`
> **Description:**
> * **Core Intuition:** Interleaves expert-prefix supervision with RL on the model's own completions, addressing the mismatch between rigid SFT imitation and later exploratory RL.
> * **Engineering Traits:** Relevant to SLMs because it tries to preserve useful expert guidance without freezing the model into low-exploration behavior before RL begins.
> * **Mathematical Formulation:** A representative TRAPO objective combines trust-region SFT and RL:
>
> $$\mathcal L_{TRAPO}=\mathcal L_{RL}(y_{model})+\lambda\,\mathcal L_{TrSFT}(y_{expert-prefix})$$
>
> with adaptive prefix selection and trust-region control on the SFT part.
> * **Policy Update:** Each training instance uses SFT where expert prefixes are useful and RL where the model should explore its own completions, reducing the SFT-then-RL inconsistency.

#### 60. [EAPO: Enhancing Policy Optimization with On-Demand Expert Assistance](https://arxiv.org/abs/2509.23730)

* **Algorithm:** EAPO
* **Venue/Year:** ArXiv 2025
* **Reference Code:** N/A
* **Local PDF:** [Song et al. - 2025 - EAPO - Enhancing Policy Optimization with On-Demand Expert Assistance.pdf](<papers/Song et al. - 2025 - EAPO - Enhancing Policy Optimization with On-Demand Expert Assistance.pdf>)
* **Rating:** `7.7/10`
> **Description:**
> * **Core Intuition:** Lets the policy consult external experts during training and learn when assistance is worth using, then evaluates the optimized policy independently without expert calls.
> * **Engineering Traits:** Interesting for low-capacity models because expert assistance can densify exploration, but the extra interaction channel and expert dependency make it more complex than plain GRPO.
> * **Mathematical Formulation:** The training objective can be viewed as optimizing reward under assisted trajectories:
>
> $$\max_\theta\ \mathbb E_{\tau\sim\pi_\theta^{assist}}[R(\tau)-\lambda C_{expert}(\tau)-\beta D_{KL}(\pi_\theta\|\pi_{ref})]$$
>
> where $C_{expert}$ penalizes excessive consultation.
> * **Policy Update:** The model is rewarded for using expert help only when it improves the reasoning trajectory, gradually internalizing expert-guided behavior into the base policy.

#### 61. [Near-Future Policy Optimization](https://arxiv.org/abs/2604.20733)

* **Algorithm:** NPO / AutoNPO
* **Venue/Year:** ArXiv 2026
* **Reference Code:** N/A
* **Local PDF:** [Qin et al. - 2026 - Near-Future Policy Optimization.pdf](<papers/Qin et al. - 2026 - Near-Future Policy Optimization.pdf>)
* **Rating:** `7.5/10`
> **Description:**
> * **Core Intuition:** Uses a model's own near-future checkpoint as the auxiliary trajectory source, aiming for trajectories that are stronger than the current policy but closer than external teachers.
> * **Engineering Traits:** Useful for understanding mixed-policy RLVR, though it requires a training loop that can manage checkpoint selection and intervention timing.
> * **Mathematical Formulation:** NPO frames auxiliary trajectory usefulness through a quality-variance tradeoff:
>
> $$S=\frac{Q}{V},\qquad \mathcal J_{NPO}=\mathcal J_{on-policy}+\alpha\,\mathbb E_{\tau\sim\pi_{future}}[w(\tau)A(\tau)]$$
>
> where the near-future policy is chosen to improve $Q$ without making $V$ too large.
> * **Policy Update:** The current policy learns from its own stronger future snapshot, and AutoNPO triggers the intervention when online signals suggest that the benefit outweighs the variance cost.

#### 62. [GRPO-VPS: Enhancing Group Relative Policy Optimization with Verifiable Process Supervision for Effective Reasoning](https://arxiv.org/abs/2604.20659)

* **Algorithm:** GRPO-VPS
* **Venue/Year:** ICLR 2026
* **Reference Code:** N/A
* **Local PDF:** [Wang et al. - 2026 - GRPO-VPS - Enhancing Group Relative Policy Optimization with Verifiable Process Supervision for Effective Reasoning.pdf](<papers/Wang et al. - 2026 - GRPO-VPS - Enhancing Group Relative Policy Optimization with Verifiable Process Supervision for Effective Reasoning.pdf>)
* **Rating:** `8.1/10`
> **Description:**
> * **Core Intuition:** Adds verifiable process supervision to GRPO by probing how each reasoning segment changes the model's belief in the correct answer.
> * **Engineering Traits:** Very relevant to SLMs because it directly attacks coarse sequence-level credit assignment while avoiding extra learned reward models or expensive Monte Carlo process labels.
> * **Mathematical Formulation:** For reasoning segments $s$, GRPO-VPS estimates progress by the change in correct-answer belief:
>
> $$\Delta_s=\log p_\theta(a^\star\mid x,y_{\le s})-\log p_\theta(a^\star\mid x,y_{<s}),\qquad \hat A_s=\hat A_{seq}+\alpha\Delta_s$$
>
> * **Policy Update:** Segment-level progress refines the usual group-relative advantage, so tokens in helpful reasoning segments receive stronger updates and overthinking segments are suppressed.

---

## 🌟 Useful Resources (Surveys & Codebases)

### 📖 Comprehensive Surveys
* **[A Comprehensive Survey of Direct Preference Optimization: Datasets, Theories, Variants, and Applications](https://arxiv.org/abs/2410.15595)**
  * **Brief**: A must-read survey on DPO-style alignment, its theory, datasets, and rapidly growing family of variants.
* **[A Survey of Reinforcement Learning for Large Reasoning Models](http://arxiv.org/abs/2509.08827)**
  * **Brief**: A strong macro-level survey of the transition from classical RLHF to RLVR and reasoning-centric post-training.
* **[RLHF Algorithms Ranked: An Extensive Evaluation](https://aclanthology.org/2025.emnlp-industry.35/)**
  * **Brief**: A valuable engineering benchmark comparing many XPO algorithms across tasks, rewards, and hyperparameters.

### 🗂️ Datasets for SFT, Preference Learning, and RLVR

* **[Magpie-Qwen2-Pro-200K-English](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2-Pro-200K-English)**
  * **Brief**: A strong candidate for the `general dialogue` portion of your neutral `SFT Base`, especially for instruction-following conversational style.
* **[SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup)**
  * **Brief**: A practical instruction-tuning corpus for the `general instruction / explanation` slice of SFT data.
* **[NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)**
  * **Brief**: A useful source for the `math reasoning` portion of SFT, though it should be filtered carefully so math does not dominate model style.
* **[UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)**
  * **Brief**: One of the most common high-quality preference datasets for offline pairwise or pointwise alignment baselines such as DPO, KTO, ORPO, and SimPO.
* **[UltraFeedback Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)**
  * **Brief**: A convenient preprocessed preference dataset widely used in TRL-style alignment pipelines.
* **[HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)**
  * **Brief**: A classic harmlessness/helpfulness preference dataset; less frontier than UltraFeedback, but still useful as a historical RLHF baseline resource.
* **[PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)**
  * **Brief**: Useful if you want to extend the project toward safety-alignment or study whether safety preference signals behave differently from utility-oriented ones on SLMs.
* **[DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)**
  * **Brief**: A strong candidate for the `online RL / RLVR` stage because it is much better suited for verifiable-reward reasoning than for neutral SFT.
* **[OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)**
  * **Brief**: A useful large-scale reasoning dataset for verifiable math-style RLVR experiments and reward construction.
* **[OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)**
  * **Brief**: A useful reasoning-oriented corpus for studying long-form solutions, chain-of-thought style outputs, and possible verbosity coupling.
* **[GSM8K](https://huggingface.co/datasets/openai/gsm8k)**
  * **Brief**: Primarily an evaluation benchmark in this project, but also important as a decontamination target during SFT and alignment data construction.
* **[HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag)**
  * **Brief**: A key retention benchmark for measuring general knowledge degradation after alignment.
* **[ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc)**
  * **Brief**: Another core retention benchmark for measuring whether target-task gains come with broader reasoning loss.

### 💻 Open-Source Codebases & Frameworks

* **[Hugging Face TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)**
  * **Brief**: The most important unified library for implementing DPO, PPO, KTO, ORPO, and related post-training algorithms.
* **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**
  * **Brief**: A mature fine-tuning framework especially convenient for quick SLM experiments under LoRA and constrained hardware.
* **[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)**
  * **Brief**: Best suited for distributed RLHF and RLVR experiments, especially GRPO, PPO, and REINFORCE++ style pipelines.
* **[VeRL](https://github.com/volcengine/verl)**
  * **Brief**: A rapidly growing post-training framework for RLHF and RLVR, with strong support for scalable rollout-training separation and modern reasoning-model workflows.
* **[vLLM](https://github.com/vllm-project/vllm)**
  * **Brief**: The de facto high-throughput inference engine for LLM evaluation and rollout generation. Particularly useful for your project's unified decoding and batch evaluation pipeline.
* **[DeepSpeed](https://github.com/microsoft/DeepSpeed)**
  * **Brief**: A core systems library for large-scale distributed training, ZeRO memory optimization, and stable full-parameter fine-tuning on limited hardware.
* **[PEFT](https://github.com/huggingface/peft)**
  * **Brief**: Essential for LoRA, QLoRA, and other parameter-efficient fine-tuning methods, directly supporting your `FFT vs LoRA-r16 vs LoRA-r32` comparison axis.
* **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**
  * **Brief**: The standard benchmark harness for automatic evaluation on tasks such as ARC, HellaSwag, GSM8K, and many other academic benchmarks.
* **[Transformers](https://github.com/huggingface/transformers)**
  * **Brief**: The base model and trainer ecosystem underpinning most contemporary alignment stacks, including custom probes and white-box instrumentation.
* **[Unsloth](https://github.com/unslothai/unsloth)**
  * **Brief**: An acceleration-focused toolkit useful for fast iteration on alignment experiments with limited resources.

### 🏗️ Infra, Serving, and Evaluation Stack

* **[Ray](https://github.com/ray-project/ray)**
  * **Brief**: Commonly used for distributed rollout orchestration and actor management in large-scale RLHF / RLVR pipelines.
* **[Accelerate](https://github.com/huggingface/accelerate)**
  * **Brief**: A lightweight distributed training launcher and runtime utility for multi-GPU experiments, rapid prototyping, and reproducible script management.
* **[SGLang](https://github.com/sgl-project/sglang)**
  * **Brief**: A fast serving and programmatic inference framework for structured generation, useful for rollout-heavy reasoning experiments and tool-style prompting.
* **[OpenCompass](https://github.com/open-compass/OpenCompass)**
  * **Brief**: A broad LLM evaluation platform that can complement `lm-eval` when you want larger benchmark coverage or more standardized evaluation workflows.
* **[Lighteval](https://github.com/huggingface/lighteval)**
  * **Brief**: A lightweight evaluation framework from Hugging Face that is convenient for fast benchmark iteration and custom evaluation recipes.
* **[Weights & Biases](https://wandb.ai/site)**
  * **Brief**: Extremely useful for tracking multi-seed experiments, rollout statistics, reward curves, gradient diagnostics, and comparison tables across algorithms.
* **[TensorBoard](https://github.com/tensorflow/tensorboard)**
  * **Brief**: A simple default option for local experiment logging, especially useful if you want lower-overhead tracking than full hosted experiment platforms.
* **[Datasets](https://github.com/huggingface/datasets)**
  * **Brief**: Essential for building reproducible SFT, preference, and RLVR data pipelines with filtering, caching, and manifest-friendly preprocessing.

---

**⏳ To be continued...**
