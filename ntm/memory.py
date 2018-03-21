"""NTM Memory"""
import torch
from ntm.tensor_utils import sim, outer_prod


class Memory:
    def __init__(self, mem_t):
        self._memory = mem_t

    def attention_read(self, wt):
        return sim(wt, self._memory, transpose=True)

    def attention_add(self, wt, add):
        # memory = memory + sum_{head h} weighted add(h)
        self._memory = self._memory + torch.sum(outer_prod(wt, add), dim=-3)

    def attention_erase(self, wt, erase):
        # memory = memory * product_{head h} (1 - weighted erase(h))
        self._memory = self._memory * torch.prod(1 - outer_prod(wt, erase), dim=-3)

    def content_similarity(self, k):
        return sim(k, self._memory, l2_normalize=True)

    @property
    def size(self):
        return self._memory.size()

    @property
    def content(self):
        return self._memory
