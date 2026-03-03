"""
micromoe.py — Mixture of Experts GPT, from scratch, in pure Python.

Same as microgpt.py (@karpathy) but the MLP is replaced with a Mixture
of Experts: N experts exist, a learned router picks top-k per token,
only those k run, outputs are weighted-summed. This is how Mixtral works.
"""

import os, math, random
random.seed(42)

# --- Dataset + Tokenizer (from microgpt.py) -----------------------------------

if not os.path.exists('input.txt'):
    import urllib.request
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt', 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
uchars = sorted(set(''.join(docs)))
BOS, vocab_size = len(uchars), len(uchars) + 1

# --- Scalar Autograd (from microgpt.py) ---------------------------------------

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data, self.grad = data, 0
        self._children, self._local_grads = children, local_grads
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return Value(self.data + o.data, (self, o), (1, 1))
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return Value(self.data * o.data, (self, o), (o.data, self.data))
    def __pow__(self, n): return Value(self.data**n, (self,), (n * self.data**(n-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, o): return self + o
    def __sub__(self, o): return self + (-o)
    def __rsub__(self, o): return o + (-self)
    def __rmul__(self, o): return self * o
    def __truediv__(self, o): return self * o**-1
    def __rtruediv__(self, o): return o * self**-1
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children: build(c)
                topo.append(v)
        build(self); self.grad = 1
        for v in reversed(topo):
            for c, g in zip(v._children, v._local_grads): c.grad += g * v.grad

def linear(x, w): return [sum(wi*xi for wi, xi in zip(r, x)) for r in w]

def softmax(logits):
    mx = max(v.data for v in logits)
    exps = [(v - mx).exp() for v in logits]; s = sum(exps)
    return [e / s for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    return [xi * (ms + 1e-5)**-0.5 for xi in x]

# --- Mixture of Experts (the only addition to a standard GPT) -----------------

def expert_forward(x, w1, w2):
    """One expert = one 2-layer MLP with ReLU."""
    return linear([h.relu() for h in linear(x, w1)], w2)

def top_k_gate(scores, k):
    """Select top-k experts by score, renormalize their softmax weights.
    Selection is non-differentiable; gradients flow through the gate weights."""
    probs = softmax(scores)
    top = sorted(range(len(scores)), key=lambda i: scores[i].data)[-k:]
    total = sum(probs[i] for i in top)
    return [(i, probs[i] / total) for i in top], probs

def moe(x, li, load):
    """MoE layer: route → gate → run top-k experts → weighted sum."""
    scores = linear(x, W[f'{li}.router'])              # learned router
    selected, probs = top_k_gate(scores, TOP_K)
    out = [Value(0.0)] * D
    fired = []
    for i, w in selected:
        load[i] += 1; fired.append(i)
        e = expert_forward(x, W[f'{li}.e{i}.w1'], W[f'{li}.e{i}.w2'])
        out = [o + w * ei for o, ei in zip(out, e)]
    return out, probs, fired

def balance_loss(gate_log, n_exp):
    """L = N * sum(mean_prob_i^2). Minimized at uniform 1/N. Prevents collapse."""
    loss = Value(0.0)
    for lp in gate_log:
        if not lp: continue
        for e in range(n_exp):
            avg = sum(p[e] for p in lp) / len(lp)
            loss = loss + avg * avg
    return loss * n_exp

# --- Model: GPT with MoE feedforward ------------------------------------------

n_layer, D, B, n_head = 1, 16, 16, 4   # layers, embed dim, block size, heads
HD = D // n_head
N_EXP, TOP_K = 4, 2                     # 4 experts, 2 active per token

mat = lambda r, c: [[Value(random.gauss(0, .08)) for _ in range(c)] for _ in range(r)]
W = {'wte': mat(vocab_size, D), 'wpe': mat(B, D), 'lm_head': mat(vocab_size, D)}
for li in range(n_layer):
    for k in ('wq','wk','wv','wo'): W[f'{li}.{k}'] = mat(D, D)
    for e in range(N_EXP):                               # N experts replace 1 MLP
        W[f'{li}.e{e}.w1'] = mat(4*D, D)
        W[f'{li}.e{e}.w2'] = mat(D, 4*D)
    W[f'{li}.router'] = mat(N_EXP, D)                    # learned routing

params = [p for m in W.values() for r in m for p in r]
print(f"docs:{len(docs)} vocab:{vocab_size} params:{len(params)} experts:{N_EXP}x top-{TOP_K}")

def forward(tok, pos, kk, kv, glog, load):
    x = [t + p for t, p in zip(W['wte'][tok], W['wpe'][pos])]
    x = rmsnorm(x)
    trace = []
    for li in range(n_layer):
        # attention (identical to GPT)
        xr = x; x = rmsnorm(x)
        q, k, v = linear(x, W[f'{li}.wq']), linear(x, W[f'{li}.wk']), linear(x, W[f'{li}.wv'])
        kk[li].append(k); kv[li].append(v)
        xa = []
        for h in range(n_head):
            s = h * HD
            qh, kh = q[s:s+HD], [ki[s:s+HD] for ki in kk[li]]
            vh = [vi[s:s+HD] for vi in kv[li]]
            a = softmax([sum(qh[j]*kh[t][j] for j in range(HD)) / HD**.5
                         for t in range(len(kh))])
            xa.extend([sum(a[t]*vh[t][j] for t in range(len(vh))) for j in range(HD)])
        x = [a + b for a, b in zip(linear(xa, W[f'{li}.wo']), xr)]
        # MoE feedforward (replaces the MLP)
        xr = x; x = rmsnorm(x)
        x, gp, fired = moe(x, li, load)
        glog[li].append(gp); trace.append(fired)
        x = [a + b for a, b in zip(x, xr)]
    return linear(x, W['lm_head']), trace

# --- Training (same Adam as microgpt + balance loss) --------------------------

lr, b1, b2, eps = 0.01, 0.85, 0.99, 1e-8
mo, ve = [0.0]*len(params), [0.0]*len(params)

for step in range(1000):
    doc = docs[step % len(docs)]
    toks = [BOS] + [uchars.index(c) for c in doc] + [BOS]
    n = min(B, len(toks) - 1)
    kk, kv, glog = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    load, losses = [0]*N_EXP, []
    for pos in range(n):
        logits, _ = forward(toks[pos], pos, kk, kv, glog, load)
        p = softmax(logits)
        losses.append(-p[toks[pos+1]].log())
    tloss = sum(losses) / n
    bloss = balance_loss(glog, N_EXP)
    loss = tloss + 0.01 * bloss
    loss.backward()
    lr_t = lr * (1 - step / 1000)
    for i, p in enumerate(params):
        mo[i] = b1*mo[i] + (1-b1)*p.grad
        ve[i] = b2*ve[i] + (1-b2)*p.grad**2
        p.data -= lr_t * (mo[i]/(1-b1**(step+1))) / ((ve[i]/(1-b2**(step+1)))**.5 + eps)
        p.grad = 0
    if (step+1) % 100 == 0:
        t = max(sum(load), 1)
        ld = ' '.join(f'e{i}:{load[i]*100//t}%' for i in range(N_EXP))
        print(f"step {step+1:4d} | loss {tloss.data:.4f} | bal {bloss.data:.4f} | {ld}")

# --- Inference: generate names + show which experts fire per character ---------

print("\n--- generated names (expert routing per character) ---")
for si in range(20):
    kk, kv, glog = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    load = [0]*N_EXP; tid = BOS; chars, route = [], []
    for pos in range(B):
        logits, fired = forward(tid, pos, kk, kv, glog, load)
        p = softmax([l / 0.5 for l in logits])
        tid = random.choices(range(vocab_size), weights=[pi.data for pi in p])[0]
        if tid == BOS: break
        chars.append(uchars[tid])
        route.append(','.join(str(e) for e in fired[0]))
    print(f"  {''.join(chars):12s}  {' '.join(route)}")
