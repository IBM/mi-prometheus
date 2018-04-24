"""NTM Memory"""
import torch
from models.ntm.tensor_utils import sim, outer_prod


class Memory:
    def __init__(self, mem_t):
        self._memory = mem_t

    def attention_read(self, wt):
        return sim(wt, self._memory)

    def add_weighted(self, add, wt):
        # memory = memory + sum_{head h} weighted add(h)
        self._memory = self._memory + torch.sum(outer_prod(add, wt), dim=-3)

    def erase_weighted(self, erase, wt):
        # memory = memory * product_{head h} (1 - weighted erase(h))
        self._memory = self._memory * torch.prod(1 - outer_prod(erase, wt), dim=-3)

    def content_similarity(self, k):
        return sim(k, self._memory, l2_normalize=True, aligned=False)

    @property
    def size(self):
        return self._memory.size()

    @property
    def content(self):
        return self._memory
