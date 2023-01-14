# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_actor.ipynb.

# %% auto 0
__all__ = ['Actor']

# %% ../nbs/02_actor.ipynb 3
import torch
from torch import nn

# %% ../nbs/02_actor.ipynb 4
class Actor:
    def __init__(self, env, model, tokenizer):
        self.agent = None
        # TODO: why use max?
        self.n_actions = max(model.config.vocab_size, tokenizer.vocab_size)
        self.env = env
        self.model = model
        self.obs_size = model.config.hidden_size
        self.converter = nn.Linear(self.obs_size, self.n_actions)
        # self.converter.weight = cop