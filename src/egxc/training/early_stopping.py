import jax.numpy as jnp

class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def stop(self, loss: float) -> bool:
        if jnp.isnan(loss):
            print("Loss is NaN, stopping early")
            return True
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
