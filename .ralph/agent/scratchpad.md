# exp_bitnet_serving_path KILLED + reviewed (2026-03-21). Adversarial review PROCEED (kill justified). LoTA-QAF merge erases adapters (0% flips, 116x gap). Runtime LoRA is only viable path. Reviewer notes: K2/K3 trivially satisfied (merged model = base model), float merge fp32 deserves follow-up node.

# Picking next experiment (2026-03-21)
Event: review.killed for exp_bitnet_serving_path → integrated, moving on.
Open bitnet micro nodes: scale_n25 (P2), reasoning_x_domain (P2), clone_compete (P3), basefree_exploration (P3).
None unblock downstream. Tie-break P2: scale_n25 builds on supported N=15, reasoning_x_domain builds on killed task_eval (shaky premise).
**Selected: exp_bitnet_scale_n25** — scaling to N=25 with domains + capabilities.
Will add P5 follow-up node for float merge fp32 serving path (reviewer recommendation, <5 open nodes).
