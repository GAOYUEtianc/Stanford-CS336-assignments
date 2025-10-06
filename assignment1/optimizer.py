import torch
import math

class SimpleSGD:
    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr = lr
        # iteration counter
        self.t = 0

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            # in-place update with decayed learning rate
            p.data -= self.lr / math.sqrt(self.t) * p.grad.data

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


def run_experiment(lr, steps=10):
    print(f"\n=== Running with learning rate = {lr} ===")
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SimpleSGD([weights], lr=lr)

    for t in range(steps):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(f"Step {t:02d}: loss = {loss.item():.6f}")
        loss.backward()
        opt.step()


if __name__ == "__main__":
    for lr in [1, 1e1, 1e2, 1e3]:
        run_experiment(lr, steps=10)
