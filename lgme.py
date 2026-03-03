"""LGME — Living Graph of Micro-Experts"""
import os, math, random
random.seed(42)

# ── Building blocks ──
from lgme.autograd import Value
from lgme.model import softmax, embed, attn_forward, mlp_forward, output_forward
from lgme.graph import Node, Graph
from lgme.art import ART
from lgme.router import (route_mlp_experts, clone_mlp_expert, init_router_key,
                          kohonen_update, spawn_mlp_expert, consolidate_experts,
                          cosine_sim)
from lgme.optimizer import adam_step

# ── Config flags ──
EXPERT_ROUTING = True
EXPERT_SPAWN   = True

# ── Dataset ──
if not os.path.exists('input.txt'):
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ── Tokenizer ──
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ── Hyperparams & state dict ──
n_layer, n_embd, block_size, n_head = 1, 16, 16, 4
head_dim = n_embd // n_head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

# ── Graph ──
graph = Graph(state_dict, [
    Node('attn',   ['layer0.attn_wq','layer0.attn_wk','layer0.attn_wv','layer0.attn_wo']),
    Node('mlp',    ['layer0.mlp_fc1','layer0.mlp_fc2']),
    Node('output', ['lm_head']),
], n_head=n_head, head_dim=head_dim)

# ── MoE init: clone MLP into 4 experts ──
NUM_INITIAL_EXPERTS = 4
mlp_experts = []
clone_rng = random.Random(42)
if EXPERT_ROUTING:
    mlp_experts.append({
        'id': 0, 'fc1': 'layer0.mlp_fc1', 'fc2': 'layer0.mlp_fc2',
        'router_key': init_router_key(state_dict, 'layer0.mlp_fc1', n_embd),
        'activation_count': 0,
    })
    for eid in range(1, NUM_INITIAL_EXPERTS):
        fc1 = f'expert{eid}.mlp_fc1'
        fc2 = f'expert{eid}.mlp_fc2'
        clone_mlp_expert(state_dict, 'layer0.mlp_fc1', 'layer0.mlp_fc2', fc1, fc2, rng=clone_rng)
        mlp_experts.append({
            'id': eid, 'fc1': fc1, 'fc2': fc2,
            'router_key': init_router_key(state_dict, fc1, n_embd),
            'activation_count': 0,
        })
    graph.rebuild_params()

next_expert_id = NUM_INITIAL_EXPERTS if EXPERT_ROUTING else 1
print(f"num params: {len(graph.params)}")

# ── ART (novelty detection only — no LR modulation) ──
art = ART(initial_mu=math.log(vocab_size))

# ── Training ──
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
adam_m = [0.0] * len(graph.params)
adam_v = [0.0] * len(graph.params)
num_steps = 1000
warmup_steps = 200
spawn_count = 0
merge_count = 0
spawn_rng = random.Random(123)

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # ── Forward pass ──
    graph.reset_kv()
    losses = []
    routing_log = []

    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        if EXPERT_ROUTING:
            def _route(x, experts, sd, _log=routing_log):
                x_out, selected = route_mlp_experts(x, experts, sd, top_k=2)
                _log.append(([xi.data for xi in x], [e['id'] for e in selected]))
                return x_out
            logits = graph.forward(token_id, pos_id, mlp_experts, _route)
        else:
            logits = graph.forward(token_id, pos_id)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    # ── ART classify (novelty detection, no LR scaling) ──
    art_tag = art.classify(loss.data)

    # ── Backward ──
    loss.backward()

    # ── Kohonen update ──
    if EXPERT_ROUTING and routing_log:
        expert_x = {}
        for x_data, eids in routing_log:
            for eid in eids:
                expert_x.setdefault(eid, []).append(x_data)
        for exp in mlp_experts:
            if exp['id'] in expert_x:
                vecs = expert_x[exp['id']]
                x_mean = [sum(v[j] for v in vecs) / len(vecs) for j in range(n_embd)]
                kohonen_update(exp, x_mean, alpha=0.005)

    # ── Adam optimizer (no ART-LR modulation) ──
    lr_t = learning_rate * (1 - step / num_steps)
    adam_step(graph.params, adam_m, adam_v, lr_t, step, beta1, beta2, eps_adam)

    # ── Spawn + consolidation ──
    if EXPERT_SPAWN and EXPERT_ROUTING and step > warmup_steps:
        if art_tag == 'NOVEL' and len(mlp_experts) < 12:
            if routing_log:
                all_x = [xd for xd, _ in routing_log]
                input_mean = [sum(xd[j] for xd in all_x) / len(all_x) for j in range(n_embd)]
                src = max(mlp_experts, key=lambda e: cosine_sim(e['router_key'], input_mean))
            else:
                src = mlp_experts[0]
            new_exp = spawn_mlp_expert(state_dict, src, next_expert_id, n_embd, rng=spawn_rng)
            mlp_experts.append(new_exp)
            next_expert_id += 1
            spawn_count += 1
            graph.rebuild_params()
            adam_m = adam_m + [0.0] * (len(graph.params) - len(adam_m))
            adam_v = adam_v + [0.0] * (len(graph.params) - len(adam_v))

        if step % 200 == 0 and step > 0:
            mlp_experts, removed = consolidate_experts(mlp_experts, state_dict, threshold=0.9)
            if removed:
                merge_count += len(removed)
                graph.rebuild_params()
                new_len = len(graph.params)
                adam_m = (adam_m[:new_len] + [0.0] * max(0, new_len - len(adam_m)))[:new_len]
                adam_v = (adam_v[:new_len] + [0.0] * max(0, new_len - len(adam_v)))[:new_len]

    print(f"step {step+1:4d}/{num_steps} | loss {loss.data:.4f} | {art_tag:8s} | "
          f"experts:{len(mlp_experts)}", end='\r')

# ── Inference ──
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    graph.reset_kv()
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        if EXPERT_ROUTING:
            logits = graph.forward(token_id, pos_id, mlp_experts,
                                   lambda x, exp, sd: route_mlp_experts(x, exp, sd, top_k=2)[0])
        else:
            logits = graph.forward(token_id, pos_id)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS: break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")

# ── Diagnostics ──
print("\n--- diagnostics ---")
print(f"ART vigilance:  {art.counts}")
print(f"  → {art.counts['KNOWN']/num_steps*100:.0f}% known, "
      f"{art.counts['ADJACENT']/num_steps*100:.0f}% adjacent, "
      f"{art.counts['NOVEL']/num_steps*100:.0f}% novel")
if EXPERT_ROUTING:
    print(f"MLP experts:    {len(mlp_experts)} active")
    for exp in mlp_experts:
        print(f"  expert {exp['id']:2d}: {exp.get('activation_count',0):5d} activations")
if EXPERT_SPAWN:
    print(f"Spawned: {spawn_count} | Merged: {merge_count}")
