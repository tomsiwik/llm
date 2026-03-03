"""ART (Adaptive Resonance Theory) — novelty classifier."""

import math


class ART:
    """Tracks running surprise statistics and classifies each input as KNOWN/ADJACENT/NOVEL."""
    def __init__(self, rho=0.7, beta=0.05, initial_mu=None):
        self.rho = rho
        self.mu = initial_mu if initial_mu is not None else math.log(27)  # default: ln(vocab_size) ≈ 3.30
        self.sigma = 0.5
        self.beta = beta
        self.counts = {'KNOWN': 0, 'ADJACENT': 0, 'NOVEL': 0}

    def classify(self, loss_val):
        s = loss_val
        self.mu = (1 - self.beta) * self.mu + self.beta * s
        self.sigma = (1 - self.beta) * self.sigma + self.beta * abs(s - self.mu)
        if s < self.mu - self.sigma:
            tag = 'KNOWN'
        elif s > self.mu + 2 * self.sigma:
            tag = 'NOVEL'
        else:
            tag = 'ADJACENT'
        self.counts[tag] += 1
        return tag
