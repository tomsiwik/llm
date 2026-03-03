"""Knowledge distillation between experts.

Transfer knowledge from teacher to student by training the student
to match the teacher's outputs on a set of probe inputs.
"""

import mlx.core as mx
from tribe.expert import forward_batch


def distill(teacher, student, probe_inputs, steps=100, lr=0.015, fwd=None):
    """Transfer knowledge from teacher to student on probe inputs.

    Student learns to match teacher's outputs (not ground truth).
    This is how neighbors absorb a dying member's knowledge.

    Cost: `steps` gradient updates on `len(probe_inputs)` examples.
    At scale, this replaces full retraining with targeted transfer.
    """
    if fwd is None:
        fwd = forward_batch
    X = mx.stack(probe_inputs)
    T = fwd(teacher, X)
    mx.eval(T)

    for _ in range(steps):
        def loss_fn(w):
            preds = fwd(w, X)
            return mx.mean((preds - T) ** 2)
        _, grads = mx.value_and_grad(loss_fn)(student)
        for k in student:
            student[k] = student[k] - lr * grads[k]
        mx.eval(*[student[k] for k in student])
