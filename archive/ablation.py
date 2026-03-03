"""LGME Ablation Study — all 32 phase combinations.

Runs training with every combination of the 5 phase flags and records loss curves.
Produces: ablation_results.csv, ablation_chart.png, and summary statistics.
"""
import os, math, random, time, csv, sys
from itertools import product
from collections import Counter

# ── Dataset (loaded once, copied per trial) ──
if not os.path.exists('input.txt'):
    import urllib.request
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt', 'input.txt')
RAW_DOCS = [line.strip() for line in open('input.txt') if line.strip()]
UCHARS = sorted(set(''.join(RAW_DOCS)))
BOS = len(UCHARS)
VOCAB_SIZE = len(UCHARS) + 1

from lgme.autograd import Value
from lgme.model import softmax
from lgme.graph import Node, Graph
from lgme.art import ART
from lgme.bloom import BloomFilter
from lgme.splay import SplayTree, stable_hash
from lgme.router import (route_mlp_experts, clone_mlp_expert, init_router_key,
                          kohonen_update, spawn_mlp_expert, consolidate_experts, cosine_sim)
from lgme.optimizer import adam_step

# ── Config ──
N_EMBD, BLOCK_SIZE, N_HEAD = 16, 16, 4
HEAD_DIM = N_EMBD // N_HEAD
NUM_STEPS = 200
NUM_INITIAL_EXPERTS = 4
WARMUP = 50
CONSOL_EVERY = 75
MAX_EXPERTS = 10
LR, BETA1, BETA2, EPS = 0.01, 0.85, 0.99, 1e-8
PHASE_NAMES = ['P1:ART-LR', 'P2:Bloom', 'P3:Splay', 'P4:MoE', 'P5:Spawn']


def run_trial(p1, p2, p3, p4, p5):
    """Run one training trial. Returns list of per-step loss values."""
    random.seed(42)
    docs = list(RAW_DOCS)
    random.shuffle(docs)

    # Init state dict (deterministic from seed)
    matrix = lambda nout, nin, std=0.08: [
        [Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
    sd = {
        'wte': matrix(VOCAB_SIZE, N_EMBD),
        'wpe': matrix(BLOCK_SIZE, N_EMBD),
        'lm_head': matrix(VOCAB_SIZE, N_EMBD),
    }
    sd['layer0.attn_wq'] = matrix(N_EMBD, N_EMBD)
    sd['layer0.attn_wk'] = matrix(N_EMBD, N_EMBD)
    sd['layer0.attn_wv'] = matrix(N_EMBD, N_EMBD)
    sd['layer0.attn_wo'] = matrix(N_EMBD, N_EMBD)
    sd['layer0.mlp_fc1'] = matrix(4 * N_EMBD, N_EMBD)
    sd['layer0.mlp_fc2'] = matrix(N_EMBD, 4 * N_EMBD)

    graph = Graph(sd, [
        Node('attn', ['layer0.attn_wq', 'layer0.attn_wk', 'layer0.attn_wv', 'layer0.attn_wo']),
        Node('mlp', ['layer0.mlp_fc1', 'layer0.mlp_fc2']),
        Node('output', ['lm_head']),
    ], n_head=N_HEAD, head_dim=HEAD_DIM)

    # Phase 4: clone MLP into experts
    mlp_experts = []
    clone_rng = random.Random(42)
    spawn_rng = random.Random(123)
    if p4:
        mlp_experts.append({
            'id': 0, 'fc1': 'layer0.mlp_fc1', 'fc2': 'layer0.mlp_fc2',
            'router_key': init_router_key(sd, 'layer0.mlp_fc1', N_EMBD),
            'activation_count': 0,
        })
        for eid in range(1, NUM_INITIAL_EXPERTS):
            fc1 = f'expert{eid}.mlp_fc1'
            fc2 = f'expert{eid}.mlp_fc2'
            clone_mlp_expert(sd, 'layer0.mlp_fc1', 'layer0.mlp_fc2', fc1, fc2, rng=clone_rng)
            mlp_experts.append({
                'id': eid, 'fc1': fc1, 'fc2': fc2,
                'router_key': init_router_key(sd, fc1, N_EMBD),
                'activation_count': 0,
            })
        graph.rebuild_params()
    next_expert_id = NUM_INITIAL_EXPERTS if p4 else 1

    # Cognitive stack
    art = ART(initial_mu=math.log(VOCAB_SIZE))
    bf = BloomFilter()
    splay = SplayTree()

    # Optimizer
    adam_m = [0.0] * len(graph.params)
    adam_v = [0.0] * len(graph.params)

    loss_history = []

    for step in range(NUM_STEPS):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [UCHARS.index(ch) for ch in doc] + [BOS]
        n = min(BLOCK_SIZE, len(tokens) - 1)

        # Phase 2: Bloom
        novel_bgs = bf.check_name(doc) if p2 else []
        bloom_path = 'novel' if (p2 and novel_bgs) else 'familiar'

        # Phase 3: Splay cache (only with MoE)
        cached_routing = None
        if p3 and p4 and bloom_path == 'familiar':
            cached_routing = splay.lookup(stable_hash(tokens))

        # Forward
        graph.reset_kv()
        losses = []
        routing_log = []

        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            if p4:
                def _route(x, experts, sd_arg, _forced=cached_routing, _log=routing_log):
                    x_out, selected = route_mlp_experts(
                        x, experts, sd_arg, top_k=2, forced_ids=_forced)
                    _log.append(([xi.data for xi in x], [e['id'] for e in selected]))
                    return x_out
                logits = graph.forward(token_id, pos_id, mlp_experts, _route)
            else:
                logits = graph.forward(token_id, pos_id)
            probs = softmax(logits)
            losses.append(-probs[target_id].log())
        loss = (1 / n) * sum(losses)

        # Phase 1: ART LR
        art_tag = art.classify(loss.data)
        lr_scale = {'KNOWN': 0.1, 'ADJACENT': 1.0, 'NOVEL': 1.5}[art_tag] if p1 else 1.0

        # Phase 3: cache actual routing
        if p3 and p4 and routing_log:
            all_sel = [eid for _, eids in routing_log for eid in eids]
            most_common = [eid for eid, _ in Counter(all_sel).most_common(2)]
            splay.insert(stable_hash(tokens), most_common)

        # Backward
        loss.backward()

        # Phase 4: Kohonen with actual data
        if p4 and routing_log:
            expert_x = {}
            for x_data, eids in routing_log:
                for eid in eids:
                    expert_x.setdefault(eid, []).append(x_data)
            for exp in mlp_experts:
                if exp['id'] in expert_x:
                    vecs = expert_x[exp['id']]
                    x_mean = [sum(v[j] for v in vecs) / len(vecs) for j in range(N_EMBD)]
                    kohonen_update(exp, x_mean, alpha=0.005)

        # Adam
        lr_t = LR * lr_scale * (1 - step / NUM_STEPS)
        adam_step(graph.params, adam_m, adam_v, lr_t, step, BETA1, BETA2, EPS)

        # Phase 5: Spawn + consolidate
        if p5 and p4 and step > WARMUP:
            if art_tag == 'NOVEL' and len(mlp_experts) < MAX_EXPERTS:
                if routing_log:
                    all_x = [xd for xd, _ in routing_log]
                    input_mean = [sum(xd[j] for xd in all_x) / len(all_x) for j in range(N_EMBD)]
                    src = max(mlp_experts, key=lambda e: cosine_sim(e['router_key'], input_mean))
                else:
                    src = mlp_experts[0]
                new_exp = spawn_mlp_expert(sd, src, next_expert_id, N_EMBD, rng=spawn_rng)
                mlp_experts.append(new_exp)
                next_expert_id += 1
                graph.rebuild_params()
                adam_m = adam_m + [0.0] * (len(graph.params) - len(adam_m))
                adam_v = adam_v + [0.0] * (len(graph.params) - len(adam_v))

            if step % CONSOL_EVERY == 0 and step > 0:
                mlp_experts, removed = consolidate_experts(mlp_experts, sd, threshold=0.9)
                if removed:
                    graph.rebuild_params()
                    new_len = len(graph.params)
                    adam_m = [0.0] * new_len
                    adam_v = [0.0] * new_len

        loss_history.append(loss.data)

    return loss_history


# ── Run all 32 combinations ──
print(f"LGME Ablation: {NUM_STEPS} steps x 32 combinations")
print(f"Phase flags: {' | '.join(PHASE_NAMES)}")
print("-" * 70)

all_flags = list(product([False, True], repeat=5))
results = {}
total_t0 = time.time()

for i, flags in enumerate(all_flags):
    label = ''.join(str(int(f)) for f in flags)
    t0 = time.time()
    losses = run_trial(*flags)
    elapsed = time.time() - t0
    results[label] = losses
    final_50 = sum(losses[-50:]) / 50  # smoothed final loss
    phases_on = [PHASE_NAMES[j] for j in range(5) if flags[j]]
    desc = '+'.join(phases_on) if phases_on else 'BASELINE'
    print(f"[{i+1:2d}/32] {label} | final={final_50:.4f} | {elapsed:5.1f}s | {desc}")

total_elapsed = time.time() - total_t0
print(f"\nTotal: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

# ── Save CSV ──
csv_path = 'ablation_results.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['flags', 'P1', 'P2', 'P3', 'P4', 'P5', 'step', 'loss'])
    for label, losses in results.items():
        for step, loss_val in enumerate(losses):
            w.writerow([label, *[int(c) for c in label], step, f"{loss_val:.6f}"])
print(f"CSV saved: {csv_path}")

# ── Marginal effects ──
print("\n" + "=" * 70)
print("MARGINAL EFFECTS (negative = helps)")
print("=" * 70)
for pi in range(5):
    on_finals = [sum(results[l][-50:]) / 50 for l in results if l[pi] == '1']
    off_finals = [sum(results[l][-50:]) / 50 for l in results if l[pi] == '0']
    delta = sum(on_finals) / len(on_finals) - sum(off_finals) / len(off_finals)
    bar = "+" * int(abs(delta) * 50) if delta > 0 else "-" * int(abs(delta) * 50)
    direction = "HURTS" if delta > 0 else "HELPS"
    print(f"  {PHASE_NAMES[pi]:12s}: {delta:+.4f}  {direction:5s}  {bar}")

# ── Summary table ──
print("\n" + "=" * 70)
print("TOP 5 BEST / WORST CONFIGURATIONS")
print("=" * 70)
ranked = sorted(results.items(), key=lambda t: sum(t[1][-50:]) / 50)
print("BEST:")
for label, losses in ranked[:5]:
    final = sum(losses[-50:]) / 50
    phases = [PHASE_NAMES[j] for j in range(5) if label[j] == '1']
    print(f"  {label} = {final:.4f}  ({'+'.join(phases) if phases else 'BASELINE'})")
print("WORST:")
for label, losses in ranked[-5:]:
    final = sum(losses[-50:]) / 50
    phases = [PHASE_NAMES[j] for j in range(5) if label[j] == '1']
    print(f"  {label} = {final:.4f}  ({'+'.join(phases) if phases else 'BASELINE'})")

# ── Chart ──
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def smooth(vals, w=15):
        out = []
        for i in range(len(vals)):
            start = max(0, i - w + 1)
            out.append(sum(vals[start:i+1]) / (i - start + 1))
        return out

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # ── Left: MoE OFF ──
    ax = axes[0]
    ax.set_title('Phase 4 (MoE) OFF', fontsize=13, fontweight='bold')
    for label, losses in sorted(results.items()):
        if label[3] == '0':
            ls = smooth(losses)
            if label == '00000':
                ax.plot(ls, 'k-', lw=2.5, label='BASELINE', zorder=10)
            elif sum(int(c) for c in label) == 1:
                idx = label.index('1')
                ax.plot(ls, lw=1.8, label=PHASE_NAMES[idx], zorder=5)
            else:
                n_on = sum(int(c) for c in label)
                ax.plot(ls, color='gray', alpha=0.25 + 0.1 * n_on, lw=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(bottom=1.5)
    ax.grid(True, alpha=0.3)

    # ── Middle: MoE ON ──
    ax = axes[1]
    ax.set_title('Phase 4 (MoE) ON', fontsize=13, fontweight='bold')
    for label, losses in sorted(results.items()):
        if label[3] == '1':
            ls = smooth(losses)
            if label == '00010':
                ax.plot(ls, 'k-', lw=2.5, label='MoE only', zorder=10)
            elif label == '11111':
                ax.plot(ls, 'r-', lw=2.5, label='ALL ON', zorder=10)
            elif sum(int(c) for c in label) == 2:  # MoE + one other
                other_idx = [j for j in range(5) if j != 3 and label[j] == '1'][0]
                ax.plot(ls, lw=1.8, label=f'MoE+{PHASE_NAMES[other_idx]}', zorder=5)
            else:
                n_on = sum(int(c) for c in label)
                ax.plot(ls, color='gray', alpha=0.25 + 0.08 * n_on, lw=0.7)
    ax.set_xlabel('Step')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(bottom=1.5)
    ax.grid(True, alpha=0.3)

    # ── Right: Marginal effects ──
    ax = axes[2]
    ax.set_title('Marginal Effect per Phase', fontsize=13, fontweight='bold')
    deltas = []
    for pi in range(5):
        on_f = [sum(results[l][-50:]) / 50 for l in results if l[pi] == '1']
        off_f = [sum(results[l][-50:]) / 50 for l in results if l[pi] == '0']
        deltas.append(sum(on_f) / len(on_f) - sum(off_f) / len(off_f))
    colors = ['#2ecc71' if d < 0 else '#e74c3c' for d in deltas]
    bars = ax.barh(PHASE_NAMES, deltas, color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0, color='black', lw=1)
    ax.set_xlabel('Δ Loss (negative = helps)')
    for bar, delta in zip(bars, deltas):
        ax.text(bar.get_width() + 0.002 * (-1 if delta < 0 else 1), bar.get_y() + bar.get_height()/2,
                f'{delta:+.4f}', va='center', ha='left' if delta >= 0 else 'right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    chart_path = 'ablation_chart.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\nChart saved: {chart_path}")

except ImportError:
    print("\nmatplotlib not available — results in CSV only")
