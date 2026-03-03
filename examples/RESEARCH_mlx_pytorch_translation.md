# PyTorch → MLX Translation Guide for MoE

Quick reference for porting the 12 PyTorch example extracts to MLX.

## 1. Tensor Basics

```python
# PyTorch                          # MLX
torch.tensor([1, 2, 3])           mx.array([1, 2, 3])
torch.zeros(3, 4)                 mx.zeros((3, 4))          # tuple!
torch.randn(3, 4)                 mx.random.normal((3, 4))
torch.arange(10)                  mx.arange(10)
```

**No device management** — MLX uses unified memory. No `.to(device)`, `.cuda()`.

**Lazy evaluation** — nothing computes until `mx.eval()`. Must call at iteration boundaries.

## 2. Indexing & Gathering

```python
# PyTorch                                    # MLX
torch.gather(x, dim=1, index=idx)           mx.take_along_axis(x, idx, axis=1)
torch.index_select(x, 0, idx)               mx.take(x, idx, axis=0)  # or x[idx]
out.scatter_(1, idx, src)                    mx.put_along_axis(out, idx, src, axis=1)
out.scatter_add_(0, idx, src)               out.at[idx].add(src)  # nondeterministic!
```

**No boolean mask selection** — `x[bool_mask]` for variable-length output not supported. Use `mx.where()`.

**No `in` operator** — use `mx.any(arr == val).item()`.

## 3. Linear Layers — IDENTICAL

Both store weight as `(output, input)`, compute `y = x @ W.T + b`.

```python
# PyTorch: nn.Linear(256, 128)  → weight.shape = (128, 256)
# MLX:     nn.Linear(256, 128)  → weight.shape = (128, 256)  ✓
```

**Conv2d differs**: MLX expects NHWC, PyTorch expects NCHW.

## 4. Softmax / Gating

```python
# PyTorch                                    # MLX
F.softmax(logits, dim=-1)                   mx.softmax(logits, axis=-1)
values, indices = torch.topk(x, k=2)       # NO direct equivalent!
torch.multinomial(probs, 1)                 mx.random.categorical(logits)  # takes logits!
```

**Top-k workaround** (the Mixtral pattern):
```python
indices = mx.argpartition(-scores, kth=k-1, axis=-1)[:, :k]
values = mx.take_along_axis(scores, indices, axis=-1)
```

## 5. Autograd — Fundamentally Different

PyTorch: implicit tape, `.backward()`, `.grad`
MLX: explicit function transforms (JAX-style)

```python
# PyTorch                              # MLX
loss = model(x)                        def loss_fn(model, X, y):
loss.backward()                            return nn.losses.cross_entropy(model(X), y).mean()
optimizer.step()                       loss_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer.zero_grad()                  loss, grads = loss_grad_fn(model, X, y)
                                       optimizer.update(model, grads)
                                       mx.eval(model.parameters(), optimizer.state)
```

- `mx.stop_gradient(x)` = `x.detach()`
- No `torch.no_grad()` needed — no implicit tape
- **Model MUST be first arg** to loss_fn for `nn.value_and_grad`

## 6. In-Place Operations — None in MLX

MLX is purely functional. Every operation returns a new array.

```python
# PyTorch              # MLX
x.add_(1.0)           x = x + 1.0
x.zero_()             x = mx.zeros_like(x)
x.scatter_add_(...)   x = x.at[idx].add(val)
```

## 7. Einsum / Matmul — Mostly Same

```python
# PyTorch                              # MLX
torch.einsum('bse,bse->bs', x, y)    mx.einsum('bse,bse->bs', x, y)
torch.bmm(a, b)                       a @ b  # or mx.matmul(a, b)
```

**MoE-specific: `mx.gather_mm`** — fused gather + matmul, no PyTorch equivalent:
```python
out = mx.gather_mm(tokens, expert_weights.swapaxes(-1, -2),
                   rhs_indices=expert_indices)
```

## 8. Parameter Management

```python
# PyTorch                              # MLX
nn.Parameter(tensor)                   self.w = mx.array(...)  # auto-detected
nn.ParameterList([...])               [nn.Linear(...), ...]    # plain list!
model.state_dict()                     model.parameters()       # nested dict
model.load_state_dict(sd)             model.update(params)
p.requires_grad = False               model.freeze() / model.unfreeze()
```

## 9. Custom Modules

```python
# PyTorch                              # MLX
class M(nn.Module):                    class M(nn.Module):
    def forward(self, x):                 def __call__(self, x):  # NOT forward
        ...                                   ...
```

No `register_buffer` — use private `self._buf` or `freeze(keys=[...])`.

## 10. MoE Dispatch Patterns in MLX

### Pattern A: Loop (simple, from mixtral example)
```python
for xt, st, it in zip(x, scores, inds.tolist()):
    yt = mx.concatenate([self.experts[e](xt) for e in it], axis=-1)
    y.append((yt * st).sum(axis=-1))
```

### Pattern B: `SwitchLinear` with `gather_mm` (production, from mlx-lm)
```python
# Stack all expert weights into (num_experts, out, in)
# Use gather_mm for fused dispatch — no loop
x = mx.gather_mm(x, self.weight.swapaxes(-1, -2),
                 rhs_indices=indices, sorted_indices=True)
```

### Pattern C: Sort tokens by expert for cache locality
```python
order = mx.argsort(indices.flatten())
sorted_x = x.flatten(0, -3)[order // M]
# ... dispatch with sorted_indices=True ...
result = result[mx.argsort(order)]  # unsort
```

## 11. Gradient Manipulation

```python
# Clip grads
grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)

# Total grad norm
leaves = mlx.utils.tree_flatten(grads)
total_norm = mx.sqrt(sum(mx.sum(g * g) for _, g in leaves))

# Freeze/unfreeze (replaces requires_grad)
model.freeze()
model.experts[3].unfreeze()
```

## 12. Gotcha Summary

| # | Gotcha | Fix |
|---|--------|-----|
| 1 | `mx.topk` returns values only | Use `mx.argpartition` for indices |
| 2 | No `x[bool_mask]` selection | Use `mx.where()` |
| 3 | `in` operator fails on mx.array | Use `mx.any(arr == val)` |
| 4 | Lazy eval — control flow forces eval | Batch `mx.eval()` calls |
| 5 | Conv2d is NHWC not NCHW | Transpose inputs |
| 6 | Must call `mx.eval()` each step | Or graph grows unbounded → OOM |
| 7 | Scatter updates nondeterministic | Sort indices first if needed |
| 8 | Model must be first arg to loss_fn | For `nn.value_and_grad` |
| 9 | No DataLoader | Write your own batching |
| 10 | `.npz`/`.safetensors` not `.pt` | Convert via numpy |
| 11 | No `register_buffer` | Private attrs or `freeze(keys=)` |
| 12 | Override `__call__` not `forward` | MLX has no hook system |

## Key MLX MoE Implementations to Study

1. **`mlx-examples/llms/mixtral/mixtral.py`** — loop-based, reference pattern
2. **`mlx-lm/mlx_lm/models/switch_layers.py`** — production `gather_mm` dispatch (SwitchLinear, SwitchGLU, SwitchMLP)
3. **Multi-node expert parallelism** — [arXiv:2506.23635](https://arxiv.org/html/2506.23635v1)
