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
> $$L_{\text{PPO}}(\theta)=\hat{\mathbb{E}}\left[\min\left(r_t(\theta)\hat A_t,\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)\right],\quad r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$$
>
> * **Policy Update:** In RLHF, one typically optimizes
>
> $$\max_\theta \; L_{\text{PPO}}(\theta)-\beta\,\mathbb{E}_{x,y\sim\pi_\theta}\left[D_{\mathrm{KL}}\big(\pi_\theta(\cdot\mid x)\|\pi_{\mathrm{ref}}(\cdot\mid x)\big)\right]$$
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
> $$\mathcal L_{\text{DPO}}=-\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)$$
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
> $$\mathcal L_{\text{KTO}}=-\mathbb E_{(x,y)\sim\mathcal D^+}\left[\log\sigma\big(\beta z_\theta(x,y)-\tau_+\big)\right]-\mathbb E_{(x,y)\sim\mathcal D^-}\left[\log\sigma\big(\tau_- - \beta z_\theta(x,y)\big)\right]$$
>
> where $z_\theta(x,y)=\log \pi_\theta(y\mid x)-\log \pi_{\text{ref}}(y\mid x)$.
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
> $$\mathcal L_{\text{ORPO}}=\mathcal L_{\text{SFT}}(x,y_w)+\lambda\,\mathcal L_{\text{OR}}(x,y_w,y_l)$$
>
> with
>
> $$\mathcal L_{\text{OR}}=-\log\sigma\left(\log\frac{\mathrm{odds}_\theta(y_w\mid x)}{\mathrm{odds}_\theta(y_l\mid x)}\right),\quad \mathrm{odds}_\theta(y\mid x)=\frac{\pi_\theta(y\mid x)}{1-\pi_\theta(y\mid x)}$$
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
> $$\mathcal L_{\text{IPO}}=\left(\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}-\frac{1}{2\beta}\right)^2$$
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
> $$r_{\text{SimPO}}(x,y)=\frac{\beta}{|y|}\log\pi_\theta(y\mid x)$$
>
> and optimize
>
> $$\mathcal L_{\text{SimPO}}=-\log\sigma\left(r_{\text{SimPO}}(x,y_w)-r_{\text{SimPO}}(x,y_l)-\gamma\right)$$
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
> $$\mathcal L_{\text{SLiC-HF}}=-\log\sigma\big(s_\theta(x,y_w)-s_\theta(x,y_l)\big)$$
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
> $$\mathcal L_{\text{TDPO}}=-\sum_t \log\sigma\left(\hat Q_\theta(x,y_w,t)-\hat Q_\theta(x,y_l,t)\right)$$
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
> $$\mathcal L_{\text{PRO}}=-\log\frac{\exp(s_\theta(y^+))}{\sum_{j=1}^K \exp(s_\theta(y_j))}$$
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
> $$\mathcal L_{\text{CPO}}=\mathcal L_{\text{contrastive}}+\lambda\,D_{\mathrm{KL}}\big(\pi_\theta\|\pi_{\text{SFT}}\big)$$
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
> $$\min_\pi\max_\mu\; \mathbb E_{y_1\sim\pi, y_2\sim\mu}\left[P(y_1\succ y_2\mid x)-P(y_2\succ y_1\mid x)\right]$$
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
> $$L_{\text{GRPO}}(\theta)=\hat{\mathbb E}_{i,j,t}\left[\min\left(\rho_{i,j,t}(\theta)\hat A_{i,j},\mathrm{clip}(\rho_{i,j,t}(\theta),1-\epsilon,1+\epsilon)\hat A_{i,j}\right)\right]$$
>
> where $\rho_{i,j,t}(\theta)=\pi_\theta(o_{i,j,t}\mid x_i,o_{i,j,<t})/\pi_{\theta_{\text{old}}}(o_{i,j,t}\mid x_i,o_{i,j,<t})$.
> * **Policy Update:** Update by gradient ascent on $L_{\text{GRPO}}-\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{ref}})$.

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
> $$\nabla J(\theta)=\mathbb E_{x,y\sim\pi_\theta}\left[\left(\frac{r(x,y)-b(x)}{\sigma_R}-\beta D_{\mathrm{KL}}\right)\nabla_\theta\log\pi_\theta(y\mid x)\right]$$
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
> * **Mathematical Formulation:** Define a sequence-level ratio
>
> $$\rho_i^{\text{seq}}(\theta)=\frac{\pi_\theta(y_i\mid x_i)}{\pi_{\theta_{\text{old}}}(y_i\mid x_i)}=\exp\left(\sum_t \log\frac{\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}\mid x_i,y_{i,<t})}\right)$$
>
> and optimize
>
> $$L_{\text{GSPO}}(\theta)=\hat{\mathbb E}_i\left[\min\left(\rho_i^{\text{seq}}\hat A_i,\mathrm{clip}(\rho_i^{\text{seq}},1-\epsilon,1+\epsilon)\hat A_i\right)\right]$$
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
> $$L_{\text{BandPO}}(\theta)=\hat{\mathbb E}_t\left[\min\left(r_t(\theta)\hat A_t,\mathrm{clip}_{[\ell_t,u_t]}(r_t(\theta))\hat A_t\right)\right]$$
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
> $$\hat A_i^{\text{F-GRPO}}=(1-\hat\mu_{\text{pos}}(x_i))^\gamma\,\hat A_i^{\text{GRPO}}$$
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
> $$L_{f\text{-GRPO}}(\theta)=\mathbb E\left[T_\phi(r)\right]-\mathbb E\left[f^*\big(T_\phi(r)\big)\right]$$
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
> $$\mathcal J_{\text{AMIR-GRPO}}(\theta)=\mathcal J_{\text{GRPO}}(\theta)+\lambda_{\text{reg}}\,\mathcal J_{\text{pref}}(\theta)$$
>
> where $\mathcal J_{\text{pref}}$ is a DPO-like contrastive regularizer built from intra-group reward ordering.
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
> $$\mathcal L_{\text{REAL}}=\mathrm{BCE}(h_\theta(x,y),\,\mathbf 1[r(x,y)>0])$$
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
> * **Mathematical Formulation:** LUSPO uses a length-unbiased sequence ratio, conceptually normalizing the log-ratio by sequence length:
>
> $$\rho_i^{\text{LU}}(\theta)=\exp\left(\frac{1}{|y_i|}\sum_t \log\frac{\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}\mid x_i,y_{i,<t})}\right)$$
>
> and then applies a clipped sequence-level objective using $\rho_i^{\text{LU}}$.
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
> $$L_{\text{DHPO}}=\alpha\,L_{\text{token}}+(1-\alpha)\,L_{\text{seq}}$$
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
> * **Mathematical Formulation:** Let $s$ index sentence spans inside a response. SSPO defines
>
> $$\rho_{i,s}(\theta)=\frac{\pi_\theta(y_{i,s}\mid x_i,y_{i,<s})}{\pi_{\theta_{\text{old}}}(y_{i,s}\mid x_i,y_{i,<s})}$$
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
> $$\hat A_i^{\text{SetPO}}=\bar A_i+\lambda\left(D(\Omega)-D(\Omega\setminus\{o_i\})\right)$$
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
> $$\hat A_i^{\text{GDPO}}=\sum_{m=1}^{M}\lambda_m\frac{r_i^{(m)}-\mu^{(m)}}{\sigma^{(m)}+\epsilon}$$
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
> $$\mathcal L_{\text{GDPO}}=-\log\sigma\left(\beta\big(\Delta_\theta^{\text{group}}-\Delta_{\text{ref}}^{\text{group}}\big)\right)+\lambda D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{ref}})$$
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
> $$L_{\text{BAPO}}(\theta)=\mathbb E_{(x,y)\sim\mathcal B}\left[w(x,y)\,\hat A(x,y)\,\log\pi_\theta(y\mid x)\right]$$
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
> $$L_{\text{OAPL}}(\theta)=\mathbb E_{y\sim\pi_{\text{lag}}}\left[w_\theta(y\mid x)\,\hat A(y,x)\right],\quad w_\theta(y\mid x)=\frac{\pi_\theta(y\mid x)}{\pi_{\text{lag}}(y\mid x)}$$
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
> $$\eta_{\text{eff}}=\eta\cdot g(\mathrm{ESS}(w)),\quad \mathrm{ESS}(w)=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}$$
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
> $$\mathcal J_{\text{HAPO}}=\mathcal J_{\text{RL}}+\lambda_t\,\mathcal J_{\text{anchor}}$$
>
> where $\lambda_t$ is adaptively gated and decays as the policy becomes competent.
> * **Policy Update:** Failed trajectories receive temporary teacher-guided correction, but the anchoring term anneals away to recover asymptotically on-policy learning.

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

**⏳ To be continued... (This repository is continuously updated to serve the SLM GITA project.)**
