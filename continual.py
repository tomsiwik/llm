"""Continual learning experiment — 17 configurations × multi-seed.

Configs (a)-(k) systematically ablate architecture, freezing, regularization,
routing topology, and lateral masking. Configs (l)-(q) add cross-disciplinary
novel mechanisms: niche exclusion (ecology), idiotypic regulation (immunology),
Lyapunov criticality (physics), replicator dynamics (evolutionary game theory).
"""
import os, math, random
import mlx.core as mx

from lgme.graph import Node, Graph
from lgme.router import (route_mlp_experts, clone_mlp_expert, init_router_key,
                          kohonen_update, kohonen_update_som, route_with_lateral,
                          cosine_sim, spawn_mlp_expert,
                          niche_overlap, niche_exclusion_penalty, niche_repulsion,
                          route_with_idiotypic, replicator_init, replicator_update,
                          route_replicator, replicator_compute_fitness,
                          init_expert_heads, route_with_expert_heads,
                          build_router_key_matrix, route_dual_process)
from lgme.optimizer import adam_step, evo_adam_step, pareto_adam_step
from lgme.eval import eval_loss
from lgme.data import split_by_initial, split_by_initial_multi
from lgme.ewc import compute_fisher, ewc_penalty
from lgme.si import si_init, si_accumulate, si_consolidate, si_penalty
from lgme.som import som_init, som_grow
from lgme.diagnostics import spin_glass_overlap, approx_lyapunov, lyapunov_regularizer
from lgme.freeze import (init_freeze_map, decay_freeze_map, set_expert_tau,
                         create_offspring, offspring_update_freeze_map)
from lgme.evolution import evolve_experts

# ── Dataset ──
if not os.path.exists('input.txt'):
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(url, 'input.txt')
docs_all = [line.strip() for line in open('input.txt') if line.strip()]

# ── Tokenizer ──
uchars = sorted(set(''.join(docs_all)))
BOS = len(uchars)
vocab_size = len(uchars) + 1

# ── Hyperparams ──
LR, BETA1, BETA2, EPS = 0.01, 0.85, 0.99, 1e-8
NUM_EXPERTS = 4
EVAL_SAMPLE = 200

# ── Scale configs ──
SMALL = dict(n_layer=1, n_embd=16, block_size=16, n_head=4,
             train_steps=500, num_seeds=5, label='small')
LARGE = dict(n_layer=2, n_embd=32, block_size=16, n_head=4,
             train_steps=200, num_seeds=3, label='large')


def make_model(seed, n_layer=1, n_embd=16, block_size=16, n_head=4):
    """Create a fresh model + graph with given seed and scale."""
    rng = random.Random(seed)
    head_dim = n_embd // n_head
    matrix = lambda nout, nin, std=0.08: mx.array(
        [[rng.gauss(0, std) for _ in range(nin)] for _ in range(nout)])
    sd = {
        'wte': matrix(vocab_size, n_embd),
        'wpe': matrix(block_size, n_embd),
        'lm_head': matrix(vocab_size, n_embd),
    }
    nodes = []
    for i in range(n_layer):
        sd[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        sd[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
        nodes.append(Node('attn',
                          [f'layer{i}.attn_wq', f'layer{i}.attn_wk',
                           f'layer{i}.attn_wv', f'layer{i}.attn_wo'],
                          layer_idx=i))
        nodes.append(Node('mlp', [f'layer{i}.mlp_fc1', f'layer{i}.mlp_fc2'],
                          layer_idx=i))
    nodes.append(Node('output', ['lm_head']))
    g = Graph(sd, nodes, n_head=n_head, head_dim=head_dim)
    mx.eval(sd)
    return sd, g


def init_experts(sd, g, n_embd=16, seed=42, expert_heads=False):
    """Clone MLP into NUM_EXPERTS experts, return experts list.

    Args:
        expert_heads: if True, also create per-expert lm_head (output projection)
    """
    clone_rng = random.Random(seed)
    experts = [{
        'id': 0, 'fc1': 'layer0.mlp_fc1', 'fc2': 'layer0.mlp_fc2',
        'router_key': init_router_key(sd, 'layer0.mlp_fc1', n_embd),
        'activation_count': 0, 'som_pos': 0,
    }]
    for eid in range(1, NUM_EXPERTS):
        fc1, fc2 = f'expert{eid}.mlp_fc1', f'expert{eid}.mlp_fc2'
        clone_mlp_expert(sd, 'layer0.mlp_fc1', 'layer0.mlp_fc2', fc1, fc2,
                         rng=clone_rng)
        experts.append({
            'id': eid, 'fc1': fc1, 'fc2': fc2,
            'router_key': init_router_key(sd, fc1, n_embd),
            'activation_count': 0, 'som_pos': eid,
        })
    if expert_heads:
        init_expert_heads(sd, experts, rng=random.Random(seed + 7000))
    mx.eval(sd)
    return experts


def route_fn_simple(x, experts, sd):
    """Simple top-2 routing without logging."""
    return route_mlp_experts(x, experts, sd, top_k=2)[0]


def train(sd, g, docs, steps, adam_m, adam_v, mlp_experts=None,
          frozen_keys=None, ewc_args=None, si_state=None, lambda_si=100.0,
          som_state=None, frozen_router=False, frozen_expert_ids=None,
          use_lateral=False, kohonen_alpha=0.005, train_step_offset=0,
          n_embd=16, block_size=16,
          use_idiotypic=False, use_replicator=False, replicator_state=None,
          use_niche=False, niche_grads_accum=None,
          use_lyapunov=False, lyapunov_mu=0.1,
          use_expert_heads=False,
          use_dual_process=False, p_sequential=0.3, entropy_threshold=2.0,
          guaranteed_expert_ids=None, freeze_map=None,
          evo_training=False, pareto_training=False,
          old_docs=None, evo_n_candidates=5,
          evo_mutation_std=0.02, pareto_slack=0.05):
    """Train for `steps` on `docs`. Returns updated adam buffers."""
    # Selection pressure schedule for replicator dynamics
    if use_replicator and replicator_state is not None:
        beta_init, beta_final = 0.5, 5.0

    # RNG for evolutionary/pareto training (stable across steps)
    _evo_rng = (random.Random(train_step_offset + 7777)
                if (evo_training or pareto_training) else None)

    for step in range(steps):
        global_step = train_step_offset + step
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        routing_log = []

        # Flag for route functions that return logits instead of hidden states
        _route_produces_logits = False

        # Pre-compute router key matrix for vectorized scoring
        _rk_mx = build_router_key_matrix(mlp_experts) if mlp_experts else None

        # Choose route function based on config
        _guaranteed = guaranteed_expert_ids  # capture for closures
        if mlp_experts is not None and use_dual_process:
            # Dual-process: randomly select mode per training step
            # This ensures mx.value_and_grad traces a single deterministic path
            _rng_val = random.random()
            if _rng_val < p_sequential:
                _dp_mode = 'sys2'    # sequential chain (learning mode)
            elif _rng_val < p_sequential + 0.1:
                _dp_mode = 'blend'   # both, blended by confidence
            else:
                _dp_mode = 'sys1'    # parallel fast path (memory mode)
            _dp_uses_heads = use_expert_heads

            def _route_dual(x, experts, sd, _log=routing_log,
                            _frozen=frozen_expert_ids, _mode=_dp_mode,
                            _rk=_rk_mx, _heads=_dp_uses_heads,
                            _thresh=entropy_threshold, _guar=_guaranteed):
                logits, selected = route_dual_process(
                    x, experts, sd, frozen_expert_ids=_frozen,
                    top_k=2, router_keys_mx=_rk, mode=_mode,
                    entropy_threshold=_thresh, use_expert_heads=_heads,
                    guaranteed_ids=_guar)
                _log.append((x.tolist(), [e['id'] for e in selected]))
                return logits
            route_fn = _route_dual
            _route_produces_logits = True
        elif mlp_experts is not None and use_expert_heads:
            def _route_expert_heads(x, experts, sd, _log=routing_log,
                                    _frozen=frozen_expert_ids, _rk=_rk_mx,
                                    _guar=_guaranteed):
                logits, selected = route_with_expert_heads(
                    x, experts, sd, frozen_expert_ids=_frozen, top_k=2,
                    router_keys_mx=_rk, guaranteed_ids=_guar)
                _log.append((x.tolist(), [e['id'] for e in selected]))
                return logits
            route_fn = _route_expert_heads
            _route_produces_logits = True
        elif mlp_experts is not None and use_lateral and som_state is not None:
            def _route_lateral(x, experts, sd, _log=routing_log,
                               _som=som_state, _step=global_step,
                               _frozen=frozen_expert_ids):
                x_out, selected = route_with_lateral(
                    x, experts, sd, _som, _step,
                    frozen_expert_ids=_frozen, top_k=2)
                _log.append((x.tolist(), [e['id'] for e in selected]))
                return x_out
            route_fn = _route_lateral
        elif mlp_experts is not None and use_idiotypic:
            def _route_idiotypic(x, experts, sd, _log=routing_log,
                                 _frozen=frozen_expert_ids):
                x_out, selected = route_with_idiotypic(
                    x, experts, sd, frozen_expert_ids=_frozen,
                    theta_suppress=0.0, top_k=2)
                _log.append((x.tolist(), [e['id'] for e in selected]))
                return x_out
            route_fn = _route_idiotypic
        elif mlp_experts is not None and use_replicator and replicator_state is not None:
            def _route_repl(x, experts, sd, _log=routing_log,
                            _repl=replicator_state, _frozen=frozen_expert_ids):
                x_out, selected = route_replicator(
                    x, experts, sd, _repl,
                    frozen_expert_ids=_frozen, top_k=2)
                _log.append((x.tolist(), [e['id'] for e in selected]))
                return x_out
            route_fn = _route_repl
        elif mlp_experts is not None:
            def _route(x, experts, sd, _log=routing_log, _rk=_rk_mx):
                x_out, selected = route_mlp_experts(x, experts, sd, top_k=2,
                                                     router_keys_mx=_rk)
                _log.append((x.tolist(), [e['id'] for e in selected]))
                return x_out
            route_fn = _route
        else:
            route_fn = None

        # Niche exclusion penalty uses gradients from previous step as estimate
        _niche_grads = niche_grads_accum  # captured for closure

        def loss_fn(sd):
            g.reset_kv()
            losses = []
            for pos_id in range(n):
                token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
                if route_fn is not None:
                    logits = g.forward(token_id, pos_id, sd=sd,
                                       mlp_experts=mlp_experts, route_fn=route_fn,
                                       route_produces_logits=_route_produces_logits)
                else:
                    logits = g.forward(token_id, pos_id, sd=sd)
                probs = mx.softmax(logits)
                losses.append(-mx.log(probs[target_id] + 1e-8))
            loss = mx.mean(mx.stack(losses))

            if ewc_args is not None:
                loss = loss + ewc_penalty(sd, ewc_args['fisher'],
                                          ewc_args['theta_star'], ewc_args['lambda'])
            if si_state is not None:
                loss = loss + si_penalty(sd, si_state, lambda_si)

            # Niche exclusion penalty (uses previous step's gradients)
            if use_niche and mlp_experts is not None and _niche_grads is not None:
                loss = loss + niche_exclusion_penalty(
                    mlp_experts, _niche_grads, sd, tau=0.5, gamma=1.0)

            # Lyapunov criticality regularizer
            if use_lyapunov and n > 0:
                lyap_loss = lyapunov_regularizer(g, sd, tokens[0], 0)
                loss = loss + lyapunov_mu * lyap_loss

            return loss

        loss, grads = mx.value_and_grad(loss_fn)(sd)

        # Update niche gradient accumulator for next step
        if use_niche and mlp_experts is not None and niche_grads_accum is not None:
            for k in grads:
                if k in niche_grads_accum:
                    # Exponential moving average of gradients
                    niche_grads_accum[k] = 0.9 * niche_grads_accum[k] + 0.1 * grads[k]
                else:
                    niche_grads_accum[k] = grads[k]

        # Kohonen update for MoE (gradient-free, uses routing_log side-effect)
        # Skip for replicator routing (uses its own update)
        if (mlp_experts is not None and routing_log and not frozen_router
                and not use_replicator):
            if som_state is not None:
                for x_data, eids in routing_log:
                    for eid in eids:
                        exp_idx = next(
                            (i for i, e in enumerate(mlp_experts) if e['id'] == eid), None)
                        if exp_idx is not None:
                            kohonen_update_som(mlp_experts, exp_idx, x_data,
                                               som_state, kohonen_alpha, global_step)
            else:
                expert_x = {}
                for x_data, eids in routing_log:
                    for eid in eids:
                        expert_x.setdefault(eid, []).append(x_data)
                for exp in mlp_experts:
                    if exp['id'] in expert_x:
                        vecs = expert_x[exp['id']]
                        x_mean = [sum(v[j] for v in vecs) / len(vecs)
                                  for j in range(n_embd)]
                        kohonen_update(exp, x_mean, alpha=kohonen_alpha)

        # Niche repulsion on router keys (gradient-free)
        if use_niche and mlp_experts is not None and not frozen_router:
            niche_repulsion(mlp_experts, tau=0.5, alpha_repel=0.005)

        # Replicator dynamics update (gradient-free)
        if use_replicator and replicator_state is not None and mlp_experts is not None:
            # Schedule selection pressure: low→high
            t_frac = step / max(steps - 1, 1)
            beta_t = beta_init + (beta_final - beta_init) * t_frac
            # Compute fitness from routing log (use expert loss proxy)
            fitness = {}
            for x_data, eids in routing_log:
                x_arr = mx.array(x_data)
                from lgme.model import rmsnorm as _rmsnorm
                x_n = _rmsnorm(x_arr)
                for eid in eids:
                    exp = next((e for e in mlp_experts if e['id'] == eid), None)
                    if exp is not None:
                        h = mx.maximum(sd[exp['fc1']] @ x_n, 0)
                        out = sd[exp['fc2']] @ h
                        # Fitness = negative squared norm (less extreme = better)
                        f = -mx.sum(out * out).item()
                        fitness[eid] = fitness.get(eid, 0) + f
            # Average fitness across routing events
            for eid in fitness:
                count = sum(1 for _, eids in routing_log if eid in eids)
                if count > 0:
                    fitness[eid] /= count
            replicator_update(replicator_state, fitness, beta=beta_t)

        lr_t = LR * (1 - step / steps)

        if (evo_training or pareto_training) and old_docs is not None:
            # Shared: tokenize one old-task doc for evaluation
            _old_doc = old_docs[step % len(old_docs)]
            _old_tokens = [BOS] + [uchars.index(ch) for ch in _old_doc] + [BOS]
            _old_n = min(block_size, len(_old_tokens) - 1)

            def _eval_on_doc(sd_candidate, toks, toks_n):
                """Forward pass loss on a single tokenized doc."""
                g.reset_kv()
                doc_loss = 0.0
                for p in range(toks_n):
                    tid, tgt = toks[p], toks[p + 1]
                    if route_fn is not None:
                        logits = g.forward(tid, p, sd=sd_candidate,
                                           mlp_experts=mlp_experts, route_fn=route_fn,
                                           route_produces_logits=_route_produces_logits)
                    else:
                        logits = g.forward(tid, p, sd=sd_candidate)
                    probs = mx.softmax(logits)
                    doc_loss += (-mx.log(probs[tgt] + 1e-8)).item()
                return doc_loss / max(toks_n, 1)

            if pareto_training:
                # Pareto: separate objectives — maximize learning, constrain forgetting
                def _eval_new(sd_c):
                    return _eval_on_doc(sd_c, tokens, n)

                def _eval_old(sd_c):
                    return _eval_on_doc(sd_c, _old_tokens, _old_n)

                pareto_adam_step(sd, grads, adam_m, adam_v, lr_t, step,
                                eval_new_fn=_eval_new, eval_old_fn=_eval_old,
                                n_candidates=evo_n_candidates,
                                beta1=BETA1, beta2=BETA2, eps=EPS,
                                frozen_keys=frozen_keys,
                                mutation_std=evo_mutation_std,
                                old_loss_slack=pareto_slack,
                                rng=_evo_rng)
            else:
                # EVO: blended combined loss
                def _evo_eval_fn(sd_candidate):
                    new_loss = _eval_on_doc(sd_candidate, tokens, n)
                    old_loss = _eval_on_doc(sd_candidate, _old_tokens, _old_n)
                    return 0.5 * new_loss + 0.5 * old_loss

                evo_adam_step(sd, grads, adam_m, adam_v, lr_t, step,
                              eval_fn=_evo_eval_fn, n_candidates=evo_n_candidates,
                              beta1=BETA1, beta2=BETA2, eps=EPS,
                              frozen_keys=frozen_keys,
                              mutation_std=evo_mutation_std,
                              rng=_evo_rng)
        else:
            adam_step(sd, grads, adam_m, adam_v, lr_t, step, BETA1, BETA2, EPS,
                      frozen_keys=frozen_keys, freeze_map=freeze_map)

        # SI accumulation (after adam/evo_adam step which updates sd)
        if si_state is not None:
            si_accumulate(sd, grads, si_state)

        # Materialize all lazy computations
        eval_targets = [sd, adam_m, adam_v]
        if si_state is not None:
            eval_targets.extend([si_state['running_sum'], si_state['prev_params']])
        mx.eval(*eval_targets)

        if (step + 1) % 100 == 0:
            print(f"  step {step+1}/{steps} loss={loss.item():.4f}", end='\r')
    print()


def evaluate(g, docs, mlp_experts=None, block_size=16,
             route_fn=None, route_produces_logits=False):
    """Eval on a subset of docs.

    Args:
        route_fn: routing function (defaults to cosine top-2 if mlp_experts given)
        route_produces_logits: if True, route_fn returns logits directly
    """
    sample = docs[:EVAL_SAMPLE] if len(docs) > EVAL_SAMPLE else docs
    if route_fn is None and mlp_experts is not None:
        route_fn = lambda x, e, sd: route_fn_simple(x, e, sd)
    return eval_loss(g, sample, uchars, BOS, block_size, mlp_experts, route_fn,
                     route_produces_logits=route_produces_logits)


# ── Config definitions ──
CONFIGS = [
    # Phase A configs
    dict(name='(a) Dense baseline',
         moe=False, freeze=False, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False),
    dict(name='(b) MoE, no freeze',
         moe=True, freeze=False, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False),
    dict(name='(c) MoE + freeze',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False),
    dict(name='(d) MoE + freeze + EWC',
         moe=True, freeze=True, ewc=True, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False),
    dict(name='(e) Dense + freeze-80%',
         moe=False, freeze=False, ewc=False, si=False,
         dense_freeze_pct=0.8, joint=False, frozen_router=False,
         som=False, lateral=False),
    dict(name='(f) Joint A+B',
         moe=False, freeze=False, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=True, frozen_router=False,
         som=False, lateral=False),
    dict(name='(g) MoE + freeze + SI',
         moe=True, freeze=True, ewc=False, si=True,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False),
    # Phase B configs
    dict(name='(h) MoE + freeze + frozenR',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=True,
         som=False, lateral=False),
    dict(name='(i) MoE + SOM + freeze',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=True, lateral=False),
    # Phase C configs
    dict(name='(j) MoE + SOM lateral + freeze',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=True, lateral=True),
    dict(name='(k) MoE + SOM lat + frz + EWC',
         moe=True, freeze=True, ewc=True, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=True, lateral=True),
    # Phase D: Cross-disciplinary novel mechanisms
    dict(name='(l) MoE + frz + niche',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=True, idiotypic=False, replicator=False, lyapunov=False),
    dict(name='(m) MoE + frz + idiotypic',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=True, replicator=False, lyapunov=False),
    dict(name='(n) MoE + frz + Lyapunov',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=False, replicator=False, lyapunov=True),
    dict(name='(o) MoE + frz + replicator',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=False, replicator=True, lyapunov=False),
    dict(name='(p) MoE + frz + niche + repl',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=True, idiotypic=False, replicator=True, lyapunov=False),
    dict(name='(q) MoE + frz + idio + EWC',
         moe=True, freeze=True, ewc=True, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=True, replicator=False, lyapunov=False),
    # Phase E: Expert-specific output heads (addresses lm_head forgetting)
    dict(name='(r) MoE + frz + expert heads',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=False, replicator=False, lyapunov=False,
         expert_heads=True),
    dict(name='(s) MoE + frz + exp heads + EWC',
         moe=True, freeze=True, ewc=True, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=False, replicator=False, lyapunov=False,
         expert_heads=True),
    dict(name='(t) exp heads + spawn',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=False, replicator=False, lyapunov=False,
         expert_heads=True, spawn_new=True),
    dict(name='(u) exp heads + spawn + EWC',
         moe=True, freeze=True, ewc=True, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=False, replicator=False, lyapunov=False,
         expert_heads=True, spawn_new=True),
    # Phase F: Dual-process expert composition (System 1/System 2)
    dict(name='(v) dual-process',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=False, replicator=False, lyapunov=False,
         dual_process=True, expert_heads=False),
    dict(name='(w) dual + exp heads',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=False, replicator=False, lyapunov=False,
         dual_process=True, expert_heads=True),
    dict(name='(x) dual + exp heads + EWC',
         moe=True, freeze=True, ewc=True, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=False, replicator=False, lyapunov=False,
         dual_process=True, expert_heads=True),
    # Phase G: Spawn + guaranteed routing (fresh capacity with routing guarantee)
    dict(name='(y) spawn + guaranteed',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=False, replicator=False, lyapunov=False,
         expert_heads=True, spawn_new=True, guaranteed_routing=True),
    dict(name='(z) spawn+guar+EWC',
         moe=True, freeze=True, ewc=True, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=False, replicator=False, lyapunov=False,
         expert_heads=True, spawn_new=True, guaranteed_routing=True),
    dict(name='(aa) spawn+guar+dual',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=False, replicator=False, lyapunov=False,
         expert_heads=True, spawn_new=True, guaranteed_routing=True,
         dual_process=True),
    # Phase H: Selective freeze — only freeze top-N most active experts
    dict(name='(ab) sel-frz-2 + exp heads',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=False, replicator=False, lyapunov=False,
         expert_heads=True, selective_freeze=2),
    dict(name='(ac) sel-frz-2 + heads + EWC',
         moe=True, freeze=True, ewc=True, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=False, replicator=False, lyapunov=False,
         expert_heads=True, selective_freeze=2),
    dict(name='(ad) spwn+guar+sel-frz-2',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         niche=False, idiotypic=False, replicator=False, lyapunov=False,
         expert_heads=True, spawn_new=True, guaranteed_routing=True,
         selective_freeze=2),
    # Phase AFMEI: Adaptive Freeze Maps
    dict(name='(ae) AFMEI d=0.3',
         moe=True, freeze=False, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True, afmei=True, afmei_decay=0.3),
    dict(name='(af) AFMEI d=0.5',
         moe=True, freeze=False, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True, afmei=True, afmei_decay=0.5),
    dict(name='(ag) AFMEI d=0.7',
         moe=True, freeze=False, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True, afmei=True, afmei_decay=0.7),
    dict(name='(ah) AFMEI d=0.9',
         moe=True, freeze=False, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True, afmei=True, afmei_decay=0.9),
    dict(name='(ai) AFMEI d=0.7+EWC',
         moe=True, freeze=False, ewc=True, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True, afmei=True, afmei_decay=0.7),
    dict(name='(aj) AFMEI d=0.5+spawn',
         moe=True, freeze=False, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True, afmei=True, afmei_decay=0.5,
         spawn_new=True),
    # Phase AFMEI+Offspring: freeze map guided by offspring evaluation
    dict(name='(ak) AFMEI+child d=0.3',
         moe=True, freeze=False, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True, afmei=True, afmei_decay=0.3,
         afmei_offspring=True, afmei_offspring_steps=50),
    dict(name='(al) AFMEI+child d=0.5',
         moe=True, freeze=False, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True, afmei=True, afmei_decay=0.5,
         afmei_offspring=True, afmei_offspring_steps=50),
    dict(name='(am) AFMEI+child d=0.7',
         moe=True, freeze=False, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True, afmei=True, afmei_decay=0.7,
         afmei_offspring=True, afmei_offspring_steps=50),
    dict(name='(an) AFMEI+child+EWC d=0.5',
         moe=True, freeze=False, ewc=True, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True, afmei=True, afmei_decay=0.5,
         afmei_offspring=True, afmei_offspring_steps=50),
    # Phase EVO: DEAP-powered evolutionary expert optimization
    dict(name='(ao) EVO pop40 gen20',
         moe=True, freeze=False, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True, evo=True,
         evo_pop=40, evo_gen=20, evo_eval_sample=30),
    dict(name='(ap) EVO pop60 gen30',
         moe=True, freeze=False, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True, evo=True,
         evo_pop=60, evo_gen=30, evo_eval_sample=30),
    dict(name='(aq) EVO+freeze',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True, evo=True,
         evo_pop=40, evo_gen=20, evo_eval_sample=30),
    dict(name='(ar) EVO+EWC',
         moe=True, freeze=False, ewc=True, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True, evo=True,
         evo_pop=40, evo_gen=20, evo_eval_sample=30),
    # Phase EVO-train: evolutionary selection INSIDE the gradient training loop
    dict(name='(as) EVO-train c=5',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True,
         evo_training=True, evo_n_candidates=5, evo_mutation_std=0.02),
    dict(name='(at) EVO-train c=3',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True,
         evo_training=True, evo_n_candidates=3, evo_mutation_std=0.01),
    dict(name='(au) EVO-train+EWC',
         moe=True, freeze=True, ewc=True, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True,
         evo_training=True, evo_n_candidates=5, evo_mutation_std=0.02),
    dict(name='(av) EVO-train nofreeze',
         moe=True, freeze=False, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True,
         evo_training=True, evo_n_candidates=5, evo_mutation_std=0.02),
    # Refined EVO-train: conservative variants
    dict(name='(aw) EVO-train c=2',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True,
         evo_training=True, evo_n_candidates=2, evo_mutation_std=0.0),
    dict(name='(ax) EVO-train c=3 m=0.005',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True,
         evo_training=True, evo_n_candidates=3, evo_mutation_std=0.005),
    # Phase PARETO: Pareto selection — maximize learning, constrain forgetting
    dict(name='(ay) Pareto s=0.05',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True,
         pareto_training=True, evo_n_candidates=4, evo_mutation_std=0.01,
         pareto_slack=0.05),
    dict(name='(az) Pareto s=0.10',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True,
         pareto_training=True, evo_n_candidates=4, evo_mutation_std=0.01,
         pareto_slack=0.10),
    dict(name='(ba) Pareto s=0.02',
         moe=True, freeze=True, ewc=False, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True,
         pareto_training=True, evo_n_candidates=4, evo_mutation_std=0.01,
         pareto_slack=0.02),
    dict(name='(bb) Pareto+EWC s=0.05',
         moe=True, freeze=True, ewc=True, si=False,
         dense_freeze_pct=0.0, joint=False, frozen_router=False,
         som=False, lateral=False,
         expert_heads=True,
         pareto_training=True, evo_n_candidates=4, evo_mutation_std=0.01,
         pareto_slack=0.05),
]


def run_two_task(cfg, seed, scale):
    """Run a single config+seed on the 2-task (A/B) protocol.

    Returns dict with L_A_before, L_A_after, L_B_final, BWT, FWT,
    plus diagnostic metrics (spin_glass_q, lyapunov_max) when applicable.
    """
    n_layer = scale['n_layer']
    n_embd = scale['n_embd']
    block_size = scale['block_size']
    n_head = scale['n_head']
    train_steps = scale['train_steps']

    # Extract new mechanism flags (default False for old configs)
    use_niche = cfg.get('niche', False)
    use_idiotypic = cfg.get('idiotypic', False)
    use_replicator = cfg.get('replicator', False)
    use_lyapunov = cfg.get('lyapunov', False)
    use_dual_process = cfg.get('dual_process', False)

    docs = list(docs_all)
    random.Random(seed).shuffle(docs)
    set_a, set_b = split_by_initial(docs, boundary='n')

    sd, g = make_model(seed, n_layer, n_embd, block_size, n_head)

    use_expert_heads = cfg.get('expert_heads', False)

    # Joint training: combine A+B, train once, measure
    if cfg['joint']:
        combined = set_a + set_b
        random.Random(seed).shuffle(combined)
        adam_m = {k: mx.zeros_like(sd[k]) for k in sd}
        adam_v = {k: mx.zeros_like(sd[k]) for k in sd}
        train(sd, g, combined, train_steps * 2, adam_m, adam_v,
              n_embd=n_embd, block_size=block_size)
        L_A = evaluate(g, set_a, block_size=block_size)
        L_B = evaluate(g, set_b, block_size=block_size)
        return {
            'L_A_before': L_A, 'L_A_after': L_A, 'L_B_final': L_B,
            'BWT': 0.0, 'FWT': L_B,
        }

    experts = (init_experts(sd, g, n_embd, seed, expert_heads=use_expert_heads)
               if cfg['moe'] else None)
    adam_m = {k: mx.zeros_like(sd[k]) for k in sd}
    adam_v = {k: mx.zeros_like(sd[k]) for k in sd}

    # SOM state
    som_state = None
    if cfg['som'] and experts:
        som_state = som_init(len(experts), total_steps=train_steps * 2)

    # SI state
    si_state_obj = None
    if cfg['si']:
        attn_keys = g.get_attn_keys()
        si_state_obj = si_init(sd, attn_keys)

    # Replicator state
    replicator_state = None
    if use_replicator and experts:
        replicator_state = replicator_init(experts)

    # Niche gradient accumulator
    niche_grads_accum = None
    if use_niche and experts:
        niche_grads_accum = {}

    # Build eval route_fn
    eval_route_fn = None
    eval_route_produces_logits = False
    if use_dual_process and experts is not None:
        # Eval always uses blend mode for dual-process
        def _eval_route_dp(x, e, sd):
            logits, _ = route_dual_process(
                x, e, sd, top_k=2, mode='blend',
                entropy_threshold=2.0, use_expert_heads=use_expert_heads)
            return logits
        eval_route_fn = _eval_route_dp
        eval_route_produces_logits = True
    elif use_expert_heads and experts is not None:
        def _eval_route_heads(x, e, sd):
            logits, _ = route_with_expert_heads(x, e, sd, top_k=2)
            return logits
        eval_route_fn = _eval_route_heads
        eval_route_produces_logits = True

    # ── Phase 1: Train on set A ──
    train(sd, g, set_a, train_steps, adam_m, adam_v, mlp_experts=experts,
          si_state=si_state_obj, som_state=som_state,
          use_idiotypic=use_idiotypic, use_replicator=use_replicator,
          replicator_state=replicator_state,
          use_niche=use_niche, niche_grads_accum=niche_grads_accum,
          use_lyapunov=use_lyapunov,
          use_expert_heads=use_expert_heads,
          use_dual_process=use_dual_process,
          n_embd=n_embd, block_size=block_size)
    L_A_before = evaluate(g, set_a, experts, block_size=block_size,
                          route_fn=eval_route_fn,
                          route_produces_logits=eval_route_produces_logits)
    L_B_init = evaluate(g, set_b, experts, block_size=block_size,
                        route_fn=eval_route_fn,
                        route_produces_logits=eval_route_produces_logits)

    # Diagnostic: snapshot weights after task A for spin glass overlap
    sd_after_A = {k: mx.array(sd[k]) for k in sd}

    # Diagnostic: measure Lyapunov exponent after task A
    lyap_after_A = None
    if experts is not None:
        doc0 = set_a[0] if set_a else 'a'
        tok = [BOS] + [uchars.index(ch) for ch in doc0] + [BOS]
        lyap_after_A = approx_lyapunov(
            g, sd, tok[:3], list(range(min(3, len(tok)))),
            seed=seed)

    # Consolidate SI at task boundary
    if si_state_obj is not None:
        si_consolidate(sd, si_state_obj)

    # ── Prepare phase 2 ──
    frozen = None
    ewc_args = None
    frozen_expert_ids = set()

    # Freeze expert params (including expert-specific lm_heads if used)
    selective_n = cfg.get('selective_freeze', None)
    if cfg['freeze'] and experts is not None:
        frozen = set()
        if selective_n is not None:
            # Only freeze the top-N most activated experts
            sorted_exps = sorted(experts,
                                 key=lambda e: e.get('activation_count', 0),
                                 reverse=True)
            to_freeze = sorted_exps[:selective_n]
        else:
            to_freeze = experts
        for exp in to_freeze:
            frozen |= g.get_expert_keys(exp['id'])
            if use_expert_heads and 'lm_head' in exp:
                frozen.add(exp['lm_head'])
            frozen_expert_ids.add(exp['id'])

    # Dense freeze: randomly freeze a percentage of all keys
    if cfg['dense_freeze_pct'] > 0:
        rng_freeze = random.Random(seed + 1000)
        all_keys = list(sd.keys())
        n_freeze = int(len(all_keys) * cfg['dense_freeze_pct'])
        frozen = set(rng_freeze.sample(all_keys, n_freeze))

    # Frozen router: freeze router keys after phase A
    frozen_router = cfg['frozen_router']

    # EWC
    if cfg['ewc']:
        attn_keys = g.get_attn_keys()
        fisher = compute_fisher(g, set_a[:50], uchars, BOS,
                                block_size, attn_keys)
        theta_star = {k: sd[k] for k in attn_keys}
        ewc_args = {'fisher': fisher, 'theta_star': theta_star, 'lambda': 100.0}

    # Evolutionary expert optimization at task boundary
    if cfg.get('evo', False) and experts is not None and len(experts) >= 2:
        evo_result = evolve_experts(
            sd, experts, g, old_docs=set_a, new_docs=set_b,
            uchars=uchars, BOS=BOS, block_size=block_size,
            pop_size=cfg.get('evo_pop', 40),
            n_gen=cfg.get('evo_gen', 20),
            eval_sample=cfg.get('evo_eval_sample', 30),
            use_expert_heads=use_expert_heads,
            seed=seed + 6000)
        if evo_result is not None:
            # Apply evolved weights to the target expert
            for sd_key, new_val in evo_result['weights'].items():
                sd[sd_key] = new_val
                # Reset adam buffers for evolved params (new starting point)
                adam_m[sd_key] = mx.zeros_like(new_val)
                adam_v[sd_key] = mx.zeros_like(new_val)
            mx.eval(sd)

    # AFMEI: continuous freeze maps (replace binary freeze)
    freeze_map = None
    if cfg.get('afmei', False) and experts is not None:
        freeze_map = init_freeze_map(sd)
        # Decay at task boundary: consolidate knowledge from Phase A
        decay_rate = cfg.get('afmei_decay', 0.7)
        # Exclude shared attention params from decay (they're not per-expert)
        attn_keys = g.get_attn_keys()
        decay_freeze_map(freeze_map, decay_rate=decay_rate,
                         exclude_keys=attn_keys)

        # AFMEI offspring: create child, train briefly, update parent freeze maps
        if cfg.get('afmei_offspring', False) and len(experts) >= 2:
            # Pick two parents: top-2 most activated experts
            sorted_exps = sorted(experts,
                                 key=lambda e: e.get('activation_count', 0),
                                 reverse=True)
            parent_a, parent_b = sorted_exps[0], sorted_exps[1]

            # Create offspring
            child_id = max(e['id'] for e in experts) + 100  # high id to avoid collisions
            child = create_offspring(sd, parent_a, parent_b, child_id, n_embd,
                                     strategy='blend', alpha=0.5,
                                     rng=random.Random(seed + 5000))
            # Create lm_head for child if parents have them
            if use_expert_heads and 'lm_head' not in child:
                child_head = f'expert{child_id}.lm_head'
                sd[child_head] = 0.5 * (sd[parent_a['lm_head']] + sd[parent_b['lm_head']])
                child['lm_head'] = child_head

            # Extend adam buffers for child
            child_keys = [child['fc1'], child['fc2']]
            if 'lm_head' in child:
                child_keys.append(child['lm_head'])
            for ck in child_keys:
                adam_m[ck] = mx.zeros_like(sd[ck])
                adam_v[ck] = mx.zeros_like(sd[ck])
            mx.eval(sd)

            # Add child to expert list temporarily
            child_experts = list(experts) + [child]

            # Child freeze map: fully trainable on child, frozen on parents
            child_freeze_map = dict(freeze_map)
            for ck in child_keys:
                child_freeze_map[ck] = 1.0
            # Freeze parent params during child training (child trains alone)
            for exp in experts:
                set_expert_tau(child_freeze_map, exp, 0.0)

            # Train child briefly on new task (preview)
            offspring_steps = cfg.get('afmei_offspring_steps', 50)
            train(sd, g, set_b[:100], offspring_steps, adam_m, adam_v,
                  mlp_experts=child_experts,
                  freeze_map=child_freeze_map,
                  use_expert_heads=use_expert_heads,
                  n_embd=n_embd, block_size=block_size)

            # Evaluate child vs each parent on OLD task data
            def _eval_route_child(x, e, sd):
                logits, _ = route_with_expert_heads(x, e, sd, top_k=2)
                return logits
            child_eval_route = _eval_route_child if use_expert_heads else None
            child_eval_logits = use_expert_heads

            child_loss_old = evaluate(g, set_a[:50], child_experts,
                                      block_size=block_size,
                                      route_fn=child_eval_route,
                                      route_produces_logits=child_eval_logits)

            for parent in [parent_a, parent_b]:
                parent_loss_old = evaluate(g, set_a[:50], [parent],
                                           block_size=block_size,
                                           route_fn=child_eval_route,
                                           route_produces_logits=child_eval_logits)
                offspring_update_freeze_map(
                    freeze_map, parent,
                    child_loss=child_loss_old,
                    parent_loss=parent_loss_old,
                    thaw_amount=cfg.get('afmei_thaw', 0.3),
                    freeze_amount=cfg.get('afmei_freeze_amt', 0.2))

            # Remove child from sd to avoid polluting the main training
            # (child was temporary — just used to probe which params to protect)
            for ck in child_keys:
                del sd[ck]
                del adam_m[ck]
                del adam_v[ck]

    # Spawn new expert with expert-specific head for the new task
    guaranteed_expert_ids = None
    if use_expert_heads and cfg.get('spawn_new', False) and experts is not None:
        new_id = max(e['id'] for e in experts) + 1
        # Clone from the expert with highest activation count (most used)
        best_exp = max(experts, key=lambda e: e.get('activation_count', 0))
        new_exp = spawn_mlp_expert(sd, best_exp, new_id, n_embd,
                                   rng=random.Random(seed + 3000))
        # Create expert-specific lm_head for the new expert
        head_key = f'expert{new_id}.lm_head'
        shared_lm = sd['lm_head']
        rng_head = random.Random(seed + 4000)
        rows, cols = shared_lm.shape
        noise = mx.array([[rng_head.gauss(0, 0.02) for _ in range(cols)]
                          for _ in range(rows)])
        sd[head_key] = shared_lm + noise
        new_exp['lm_head'] = head_key
        experts.append(new_exp)
        # Extend adam buffers
        for new_key in [new_exp['fc1'], new_exp['fc2'], head_key]:
            adam_m[new_key] = mx.zeros_like(sd[new_key])
            adam_v[new_key] = mx.zeros_like(sd[new_key])
        mx.eval(sd)
        # AFMEI: set new expert params fully trainable in freeze map
        if freeze_map is not None:
            set_expert_tau(freeze_map, new_exp, 1.0)
            # Also add the new keys to freeze map (they weren't in sd when map was created)
            for new_key in [new_exp['fc1'], new_exp['fc2'], head_key]:
                freeze_map[new_key] = 1.0
        # Guaranteed routing: always include new expert in top-K
        if cfg.get('guaranteed_routing', False):
            guaranteed_expert_ids = {new_id}
        # Update eval route fn to include new expert
        _new_guar = guaranteed_expert_ids
        def _eval_route_heads_updated(x, e, sd, _guar=_new_guar):
            logits, _ = route_with_expert_heads(x, e, sd, top_k=2,
                                                 guaranteed_ids=_guar)
            return logits
        eval_route_fn = _eval_route_heads_updated

    # Cascade: spawn new expert from nearest frozen at highest-error SOM position
    if cfg['lateral'] and som_state is not None and experts is not None:
        new_pos = som_grow(som_state, experts)
        nearest_idx = min(range(len(experts)),
                          key=lambda i: abs(experts[i]['som_pos'] - new_pos))
        new_id = max(e['id'] for e in experts) + 1
        new_exp = spawn_mlp_expert(sd, experts[nearest_idx], new_id, n_embd,
                                   rng=random.Random(seed + 2000))
        new_exp['som_pos'] = new_pos
        experts.append(new_exp)
        # Extend adam buffers for new params
        for new_key in [new_exp['fc1'], new_exp['fc2']]:
            adam_m[new_key] = mx.zeros_like(sd[new_key])
            adam_v[new_key] = mx.zeros_like(sd[new_key])

    # Add new expert to replicator allocation if it was spawned
    if replicator_state is not None and experts:
        for exp in experts:
            if exp['id'] not in replicator_state['allocation']:
                replicator_state['allocation'][exp['id']] = 1.0 / len(experts)
        # Re-normalize
        total_alloc = sum(replicator_state['allocation'].values())
        for eid in replicator_state['allocation']:
            replicator_state['allocation'][eid] /= total_alloc

    # ── Phase 2: Train on set B ──
    _evo_training = cfg.get('evo_training', False)
    _pareto_training = cfg.get('pareto_training', False)
    _needs_old_docs = _evo_training or _pareto_training
    train(sd, g, set_b, train_steps, adam_m, adam_v, mlp_experts=experts,
          frozen_keys=frozen, ewc_args=ewc_args,
          si_state=si_state_obj, som_state=som_state,
          frozen_router=frozen_router, frozen_expert_ids=frozen_expert_ids,
          use_lateral=cfg['lateral'], train_step_offset=train_steps,
          use_idiotypic=use_idiotypic, use_replicator=use_replicator,
          replicator_state=replicator_state,
          use_niche=use_niche, niche_grads_accum=niche_grads_accum,
          use_lyapunov=use_lyapunov,
          use_expert_heads=use_expert_heads,
          use_dual_process=use_dual_process,
          guaranteed_expert_ids=guaranteed_expert_ids,
          freeze_map=freeze_map,
          evo_training=_evo_training,
          pareto_training=_pareto_training,
          old_docs=set_a if _needs_old_docs else None,
          evo_n_candidates=cfg.get('evo_n_candidates', 4),
          evo_mutation_std=cfg.get('evo_mutation_std', 0.01),
          pareto_slack=cfg.get('pareto_slack', 0.05),
          n_embd=n_embd, block_size=block_size)
    L_A_after = evaluate(g, set_a, experts, block_size=block_size,
                         route_fn=eval_route_fn,
                         route_produces_logits=eval_route_produces_logits)
    L_B_final = evaluate(g, set_b, experts, block_size=block_size,
                         route_fn=eval_route_fn,
                         route_produces_logits=eval_route_produces_logits)

    # BWT: negative = forgetting (standard CL convention)
    BWT = L_A_before - L_A_after
    FWT = L_B_init

    # Diagnostic: spin glass overlap between task A and task B weights
    q_overlap = spin_glass_overlap(sd_after_A, sd)

    # Diagnostic: Lyapunov exponent after task B
    lyap_after_B = None
    if experts is not None:
        doc0 = set_b[0] if set_b else 'a'
        tok = [BOS] + [uchars.index(ch) for ch in doc0] + [BOS]
        lyap_after_B = approx_lyapunov(
            g, sd, tok[:3], list(range(min(3, len(tok)))),
            seed=seed)

    result = {
        'L_A_before': L_A_before, 'L_A_after': L_A_after,
        'L_B_final': L_B_final, 'BWT': BWT, 'FWT': FWT,
        'spin_glass_q': q_overlap,
    }
    if lyap_after_A is not None:
        result['lyap_after_A'] = lyap_after_A
    if lyap_after_B is not None:
        result['lyap_after_B'] = lyap_after_B

    return result


def run_five_task(cfg, seed, scale):
    """Run a single config+seed on the 5-task sequential protocol.

    Returns forgetting matrix: losses[i][j] = loss on task i after training on task j.
    """
    n_layer = scale['n_layer']
    n_embd = scale['n_embd']
    block_size = scale['block_size']
    n_head = scale['n_head']
    train_steps = scale['train_steps']

    use_niche = cfg.get('niche', False)
    use_idiotypic = cfg.get('idiotypic', False)
    use_replicator = cfg.get('replicator', False)
    use_lyapunov = cfg.get('lyapunov', False)
    use_expert_heads = cfg.get('expert_heads', False)
    use_dual_process = cfg.get('dual_process', False)

    docs = list(docs_all)
    random.Random(seed).shuffle(docs)
    task_sets = split_by_initial_multi(docs, ['f', 'k', 'p', 'u'])
    n_tasks = len(task_sets)
    task_labels = ['a-e', 'f-j', 'k-o', 'p-t', 'u-z']

    sd, g = make_model(seed, n_layer, n_embd, block_size, n_head)

    # Joint: train on everything
    if cfg['joint']:
        combined = [d for ts in task_sets for d in ts]
        random.Random(seed).shuffle(combined)
        adam_m = {k: mx.zeros_like(sd[k]) for k in sd}
        adam_v = {k: mx.zeros_like(sd[k]) for k in sd}
        train(sd, g, combined, train_steps * n_tasks, adam_m, adam_v,
              n_embd=n_embd, block_size=block_size)
        losses = {}
        for i in range(n_tasks):
            for j in range(n_tasks):
                losses[(i, j)] = evaluate(g, task_sets[i], block_size=block_size)
        return losses

    experts = (init_experts(sd, g, n_embd, seed, expert_heads=use_expert_heads)
               if cfg['moe'] else None)
    adam_m = {k: mx.zeros_like(sd[k]) for k in sd}
    adam_v = {k: mx.zeros_like(sd[k]) for k in sd}

    som_state = None
    if cfg['som'] and experts:
        som_state = som_init(len(experts), total_steps=train_steps * n_tasks)

    si_state_obj = None
    if cfg['si']:
        attn_keys = g.get_attn_keys()
        si_state_obj = si_init(sd, attn_keys)

    replicator_state = None
    if use_replicator and experts:
        replicator_state = replicator_init(experts)

    niche_grads_accum = None
    if use_niche and experts:
        niche_grads_accum = {}

    # Build eval route_fn
    eval_route_fn = None
    eval_route_produces_logits = False
    if use_dual_process and experts is not None:
        def _eval_route_dp_5t(x, e, sd):
            logits, _ = route_dual_process(
                x, e, sd, top_k=2, mode='blend',
                entropy_threshold=2.0, use_expert_heads=use_expert_heads)
            return logits
        eval_route_fn = _eval_route_dp_5t
        eval_route_produces_logits = True
    elif use_expert_heads and experts is not None:
        def _eval_route_heads_5t(x, e, sd):
            logits, _ = route_with_expert_heads(x, e, sd, top_k=2)
            return logits
        eval_route_fn = _eval_route_heads_5t
        eval_route_produces_logits = True

    losses = {}  # losses[(i, j)] = loss on task i after training through task j
    frozen_expert_ids = set()
    frozen = None
    ewc_args = None
    freeze_map_5t = None  # AFMEI freeze map (initialized at first task boundary)

    for task_idx in range(n_tasks):
        task_docs = task_sets[task_idx]
        if not task_docs:
            continue

        # For tasks after the first, apply freezing / regularization
        if task_idx > 0:
            selective_n_5t = cfg.get('selective_freeze', None)
            if cfg['freeze'] and experts is not None:
                frozen = set()
                if selective_n_5t is not None:
                    sorted_exps = sorted(
                        experts,
                        key=lambda e: e.get('activation_count', 0),
                        reverse=True)
                    to_freeze = sorted_exps[:selective_n_5t]
                else:
                    to_freeze = experts
                for exp in to_freeze:
                    if exp['id'] not in frozen_expert_ids:
                        frozen_expert_ids.add(exp['id'])
                    frozen |= g.get_expert_keys(exp['id'])
                    if use_expert_heads and 'lm_head' in exp:
                        frozen.add(exp['lm_head'])

            if cfg['dense_freeze_pct'] > 0:
                rng_freeze = random.Random(seed + 1000 + task_idx)
                all_keys = list(sd.keys())
                n_freeze = int(len(all_keys) * cfg['dense_freeze_pct'])
                frozen = set(rng_freeze.sample(all_keys, n_freeze))

            if cfg['ewc']:
                attn_keys = g.get_attn_keys()
                prev_docs = task_sets[task_idx - 1][:50]
                fisher = compute_fisher(g, prev_docs, uchars, BOS,
                                        block_size, attn_keys)
                theta_star = {k: sd[k] for k in attn_keys}
                ewc_args = {'fisher': fisher, 'theta_star': theta_star,
                            'lambda': 100.0}

            if si_state_obj is not None:
                si_consolidate(sd, si_state_obj)

            # AFMEI: continuous freeze maps
            if cfg.get('afmei', False) and experts is not None:
                if freeze_map_5t is None:
                    freeze_map_5t = init_freeze_map(sd)
                decay_rate = cfg.get('afmei_decay', 0.7)
                attn_keys_5t = g.get_attn_keys()
                decay_freeze_map(freeze_map_5t, decay_rate=decay_rate,
                                 exclude_keys=attn_keys_5t)

            # Evolutionary expert optimization
            if cfg.get('evo', False) and experts is not None and len(experts) >= 2:
                old_docs_5t = []
                for prev_idx in range(task_idx):
                    old_docs_5t.extend(task_sets[prev_idx][:20])
                evo_result = evolve_experts(
                    sd, experts, g,
                    old_docs=old_docs_5t, new_docs=task_docs,
                    uchars=uchars, BOS=BOS, block_size=block_size,
                    pop_size=cfg.get('evo_pop', 40),
                    n_gen=cfg.get('evo_gen', 20),
                    eval_sample=cfg.get('evo_eval_sample', 15),
                    use_expert_heads=use_expert_heads,
                    seed=seed + 6000 + task_idx)
                if evo_result is not None:
                    for sd_key, new_val in evo_result['weights'].items():
                        sd[sd_key] = new_val
                        adam_m[sd_key] = mx.zeros_like(new_val)
                        adam_v[sd_key] = mx.zeros_like(new_val)
                    mx.eval(sd)

            # Cascade: spawn new expert
            if cfg['lateral'] and som_state is not None and experts is not None:
                new_pos = som_grow(som_state, experts)
                nearest_idx = min(range(len(experts)),
                                  key=lambda i: abs(experts[i]['som_pos'] - new_pos))
                new_id = max(e['id'] for e in experts) + 1
                new_exp = spawn_mlp_expert(sd, experts[nearest_idx], new_id,
                                           n_embd,
                                           rng=random.Random(seed + 2000 + task_idx))
                new_exp['som_pos'] = new_pos
                experts.append(new_exp)
                for new_key in [new_exp['fc1'], new_exp['fc2']]:
                    adam_m[new_key] = mx.zeros_like(sd[new_key])
                    adam_v[new_key] = mx.zeros_like(sd[new_key])
                if frozen is not None:
                    frozen = set()
                    for exp in experts:
                        if exp['id'] in frozen_expert_ids:
                            frozen |= g.get_expert_keys(exp['id'])

            # Spawn new expert with guaranteed routing (expert heads configs)
            guaranteed_expert_ids_5t = None
            if (use_expert_heads and cfg.get('spawn_new', False)
                    and experts is not None):
                new_id = max(e['id'] for e in experts) + 1
                best_exp = max(experts, key=lambda e: e.get('activation_count', 0))
                new_exp = spawn_mlp_expert(sd, best_exp, new_id, n_embd,
                                           rng=random.Random(seed + 3000 + task_idx))
                head_key = f'expert{new_id}.lm_head'
                shared_lm = sd['lm_head']
                rng_head = random.Random(seed + 4000 + task_idx)
                rows, cols = shared_lm.shape
                noise = mx.array([[rng_head.gauss(0, 0.02) for _ in range(cols)]
                                  for _ in range(rows)])
                sd[head_key] = shared_lm + noise
                new_exp['lm_head'] = head_key
                experts.append(new_exp)
                for new_key in [new_exp['fc1'], new_exp['fc2'], head_key]:
                    adam_m[new_key] = mx.zeros_like(sd[new_key])
                    adam_v[new_key] = mx.zeros_like(sd[new_key])
                mx.eval(sd)
                # AFMEI: set new expert params fully trainable
                if freeze_map_5t is not None:
                    set_expert_tau(freeze_map_5t, new_exp, 1.0)
                    for new_key in [new_exp['fc1'], new_exp['fc2'], head_key]:
                        freeze_map_5t[new_key] = 1.0
                if cfg.get('guaranteed_routing', False):
                    guaranteed_expert_ids_5t = {new_id}
                # Update eval route fn
                _guar_5t = guaranteed_expert_ids_5t
                if use_dual_process:
                    def _eval_route_dp_5t_upd(x, e, sd, _g=_guar_5t):
                        logits, _ = route_dual_process(
                            x, e, sd, top_k=2, mode='blend',
                            entropy_threshold=2.0,
                            use_expert_heads=True, guaranteed_ids=_g)
                        return logits
                    eval_route_fn = _eval_route_dp_5t_upd
                else:
                    def _eval_route_heads_5t_upd(x, e, sd, _g=_guar_5t):
                        logits, _ = route_with_expert_heads(
                            x, e, sd, top_k=2, guaranteed_ids=_g)
                        return logits
                    eval_route_fn = _eval_route_heads_5t_upd
                eval_route_produces_logits = True

            # Update replicator allocation for new experts
            if replicator_state is not None and experts:
                for exp in experts:
                    if exp['id'] not in replicator_state['allocation']:
                        replicator_state['allocation'][exp['id']] = 1.0 / len(experts)
                total_alloc = sum(replicator_state['allocation'].values())
                for eid in replicator_state['allocation']:
                    replicator_state['allocation'][eid] /= total_alloc
        else:
            guaranteed_expert_ids_5t = None

        # Collect old docs for evo/pareto training (all previous tasks)
        _evo_training_5t = cfg.get('evo_training', False)
        _pareto_training_5t = cfg.get('pareto_training', False)
        _needs_old_5t = (_evo_training_5t or _pareto_training_5t) and task_idx > 0
        _old_docs_5t = None
        if _needs_old_5t:
            _old_docs_5t = []
            for prev_idx in range(task_idx):
                _old_docs_5t.extend(task_sets[prev_idx])

        train(sd, g, task_docs, train_steps, adam_m, adam_v,
              mlp_experts=experts, frozen_keys=frozen, ewc_args=ewc_args,
              si_state=si_state_obj, som_state=som_state,
              frozen_router=cfg['frozen_router'],
              frozen_expert_ids=frozen_expert_ids,
              use_lateral=cfg['lateral'],
              train_step_offset=task_idx * train_steps,
              use_idiotypic=use_idiotypic, use_replicator=use_replicator,
              replicator_state=replicator_state,
              use_niche=use_niche, niche_grads_accum=niche_grads_accum,
              use_lyapunov=use_lyapunov,
              use_expert_heads=use_expert_heads,
              use_dual_process=use_dual_process,
              guaranteed_expert_ids=guaranteed_expert_ids_5t,
              freeze_map=freeze_map_5t,
              evo_training=_evo_training_5t and task_idx > 0,
              pareto_training=_pareto_training_5t and task_idx > 0,
              old_docs=_old_docs_5t,
              evo_n_candidates=cfg.get('evo_n_candidates', 4),
              evo_mutation_std=cfg.get('evo_mutation_std', 0.01),
              pareto_slack=cfg.get('pareto_slack', 0.05),
              n_embd=n_embd, block_size=block_size)

        # Evaluate on all tasks seen so far
        for eval_idx in range(task_idx + 1):
            losses[(eval_idx, task_idx)] = evaluate(
                g, task_sets[eval_idx], experts, block_size=block_size,
                route_fn=eval_route_fn,
                route_produces_logits=eval_route_produces_logits)

    return losses


def compute_cl_metrics(losses, n_tasks):
    """Compute standard CL metrics from forgetting matrix.

    Args:
        losses: dict mapping (task_i, after_task_j) → loss
        n_tasks: number of tasks

    Returns:
        dict with ACC (mean final accuracy), BWT (backward transfer),
        FWT (forward transfer)
    """
    # ACC: mean loss on all tasks after training on all
    final_losses = []
    for i in range(n_tasks):
        key = (i, n_tasks - 1)
        if key in losses:
            final_losses.append(losses[key])
    acc = sum(final_losses) / len(final_losses) if final_losses else float('inf')

    # BWT: mean(L_i_after_i - L_i_after_last) for i < n_tasks-1
    # Negative = forgetting
    bwt_vals = []
    for i in range(n_tasks - 1):
        key_after_own = (i, i)
        key_after_last = (i, n_tasks - 1)
        if key_after_own in losses and key_after_last in losses:
            bwt_vals.append(losses[key_after_own] - losses[key_after_last])
    bwt = sum(bwt_vals) / len(bwt_vals) if bwt_vals else 0.0

    return {'ACC': acc, 'BWT': bwt}


def run_experiment(configs, scale, protocol='two_task'):
    """Run all configs × seeds and aggregate results."""
    num_seeds = scale['num_seeds']
    seeds = list(range(42, 42 + num_seeds))
    all_results = {}

    for cfg in configs:
        name = cfg['name']
        seed_results = []
        print(f"\n{'='*60}")
        print(f"Config: {name} ({scale['label']}, {protocol})")
        print(f"{'='*60}")

        for si, seed in enumerate(seeds):
            print(f"  Seed {si+1}/{num_seeds} (seed={seed})")

            if protocol == 'two_task':
                r = run_two_task(cfg, seed, scale)
                seed_results.append(r)
                print(f"    BWT={r['BWT']:+.4f}  L_B={r['L_B_final']:.4f}")
            else:
                losses = run_five_task(cfg, seed, scale)
                n_tasks = 5
                metrics = compute_cl_metrics(losses, n_tasks)
                seed_results.append(metrics)
                print(f"    ACC={metrics['ACC']:.4f}  BWT={metrics['BWT']:+.4f}")

        all_results[name] = seed_results

    return all_results


def summarize_results(all_results, protocol='two_task'):
    """Print mean ± std summary table."""
    print(f"\n{'='*110}")
    if protocol == 'two_task':
        print(f"{'Config':<32} {'L_A_before':>11} {'L_A_after':>11} "
              f"{'L_B_final':>11} {'BWT':>14} {'q_overlap':>11} {'λ_max_B':>11}")
        print(f"{'-'*110}")
        for name, runs in all_results.items():
            def stat(key):
                vals = [r[key] for r in runs if key in r]
                if not vals:
                    return None, None
                mu = sum(vals) / len(vals)
                if len(vals) > 1:
                    std = (sum((v - mu)**2 for v in vals) / (len(vals) - 1)) ** 0.5
                else:
                    std = 0.0
                return mu, std
            la_b = stat('L_A_before')
            la_a = stat('L_A_after')
            lb = stat('L_B_final')
            bwt = stat('BWT')
            q = stat('spin_glass_q')
            lyap = stat('lyap_after_B')
            line = (f"{name:<32} {la_b[0]:>5.3f}±{la_b[1]:.3f} "
                    f"{la_a[0]:>5.3f}±{la_a[1]:.3f} "
                    f"{lb[0]:>5.3f}±{lb[1]:.3f} "
                    f"{bwt[0]:>+6.3f}±{bwt[1]:.3f}")
            if q[0] is not None:
                line += f" {q[0]:>5.3f}±{q[1]:.3f}"
            else:
                line += f" {'n/a':>11}"
            if lyap[0] is not None:
                line += f" {lyap[0]:>+5.3f}±{lyap[1]:.3f}"
            else:
                line += f" {'n/a':>11}"
            print(line)
    else:
        print(f"{'Config':<32} {'ACC':>14} {'BWT':>14}")
        print(f"{'-'*90}")
        for name, runs in all_results.items():
            def stat(key):
                vals = [r[key] for r in runs]
                mu = sum(vals) / len(vals)
                if len(vals) > 1:
                    std = (sum((v - mu)**2 for v in vals) / (len(vals) - 1)) ** 0.5
                else:
                    std = 0.0
                return mu, std
            acc = stat('ACC')
            bwt = stat('BWT')
            print(f"{name:<32} {acc[0]:>6.3f}±{acc[1]:.3f} "
                  f"{bwt[0]:>+6.3f}±{bwt[1]:.3f}")
    print(f"{'='*110}")


def plot_results(all_results, filename='continual_results.png'):
    """Generate bar charts for BWT and L_B_final with error bars."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping chart")
        return

    names = list(all_results.keys())
    short_names = [n.split(')')[0] + ')' for n in names]

    def get_stats(key):
        means, stds = [], []
        for name in names:
            vals = [r[key] for r in all_results[name] if key in r]
            mu = sum(vals) / len(vals) if vals else 0
            std = ((sum((v - mu)**2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
                   if len(vals) > 1 else 0)
            means.append(mu)
            stds.append(std)
        return means, stds

    # Check if this is two_task data
    first_run = list(all_results.values())[0][0]
    is_two_task = 'BWT' in first_run and 'L_B_final' in first_run

    if is_two_task:
        fig, axes = plt.subplots(1, 2, figsize=(max(12, len(names) * 1.2), 5))
        colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6',
                  '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4', '#8bc34a',
                  '#ff5722', '#607d8b', '#795548', '#cddc39', '#ff9800', '#9c27b0',
                  '#4caf50', '#2196f3', '#673ab7', '#009688',
                  '#ff6f00', '#304ffe', '#00c853']

        bwt_m, bwt_s = get_stats('BWT')
        ax = axes[0]
        bars = ax.bar(range(len(names)), bwt_m, yerr=bwt_s,
                      color=colors[:len(names)], capsize=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('BWT (neg = forgetting)')
        ax.set_title('Backward Transfer')
        ax.axhline(y=0, color='black', linewidth=0.5)

        lb_m, lb_s = get_stats('L_B_final')
        ax = axes[1]
        ax.bar(range(len(names)), lb_m, yerr=lb_s,
               color=colors[:len(names)], capsize=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('L_B_final')
        ax.set_title('New Task Learning (lower = better)')

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"\nChart saved to {filename}")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(max(12, len(names) * 1.2), 5))
        colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6',
                  '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4', '#8bc34a',
                  '#ff5722', '#607d8b', '#795548', '#cddc39', '#ff9800', '#9c27b0',
                  '#4caf50', '#2196f3', '#673ab7', '#009688',
                  '#ff6f00', '#304ffe', '#00c853']

        acc_m, acc_s = get_stats('ACC')
        ax = axes[0]
        ax.bar(range(len(names)), acc_m, yerr=acc_s,
               color=colors[:len(names)], capsize=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('ACC (loss, lower = better)')
        ax.set_title('Final Accuracy')

        bwt_m, bwt_s = get_stats('BWT')
        ax = axes[1]
        ax.bar(range(len(names)), bwt_m, yerr=bwt_s,
               color=colors[:len(names)], capsize=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('BWT (neg = forgetting)')
        ax.set_title('Backward Transfer')
        ax.axhline(y=0, color='black', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"\nChart saved to {filename}")


# ── Main ──
if __name__ == '__main__':
    # Round 1: 2-task protocol at small scale (all 11 configs × 5 seeds)
    print("\n" + "="*80)
    print("ROUND 1: 2-task protocol, small scale (1 layer, 16-dim)")
    print("="*80)
    results_small = run_experiment(CONFIGS, SMALL, protocol='two_task')
    summarize_results(results_small, protocol='two_task')
    plot_results(results_small, filename='continual_2task_small.png')

    # Round 2: 5-task protocol at small scale
    print("\n" + "="*80)
    print("ROUND 2: 5-task protocol, small scale (1 layer, 16-dim)")
    print("="*80)
    results_5task = run_experiment(CONFIGS, SMALL, protocol='five_task')
    summarize_results(results_5task, protocol='five_task')
    plot_results(results_5task, filename='continual_5task_small.png')

    # Round 3: 2-task protocol at large scale (2 layers, 32-dim)
    print("\n" + "="*80)
    print("ROUND 3: 2-task protocol, large scale (2 layers, 32-dim)")
    print("="*80)
    results_large = run_experiment(CONFIGS, LARGE, protocol='two_task')
    summarize_results(results_large, protocol='two_task')
    plot_results(results_large, filename='continual_2task_large.png')
