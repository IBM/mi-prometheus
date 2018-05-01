"""NTM Memory"""
import torch
from models.ntm.tensor_utils import sim, outer_prod


class Memory:
    def __init__(self, mem_t):
        """Initializes the memory

        :param mem_t: the memory [BATCH_SIZE x CONTENT_SIZE X MEMORY_SIZE]
        """

        self._memory = mem_t

    def attention_read(self, wt):
        """Returns the data read from memory       
 
        :param wt: the read weight [BATCH_SIZE x MEMORY_SIZE ] 
        :return: the read data [BATCH_SIZE X CONTENT_SIZE]
        """

        return sim(wt, self._memory)

    def add_weighted(self, add, wt):
        """Writes data to memory

        :param wt: the write weight [ BATCH_SIZE x MEMORY_SIZE]
        :param add: the data to be added to memory [BATCH_SIZE x CONTENT_SIZE] 
        """

        # memory = memory + sum_{head h} weighted add(h)
        self._memory = self._memory + torch.sum(outer_prod(add, wt), dim=-3)

    def erase_weighted(self, erase, wt):
        """Erases elements from memory

        :param wt: the write weight [ BATCH_SIZE x MEMORY_SIZE]
        :param wt: data to be erased from memory [ BATCH_SIZE x CONTENT_SIZE]
        """

        # memory = memory * product_{head h} (1 - weighted erase(h))
        self._memory = self._memory * torch.prod(1 - outer_prod(erase, wt), dim=-3)

    def content_similarity(self, k):
        """Calculates the dot product for Content aware addressing

        :param k: the keys emitted by the controller [BATCH_SIZE x CONTENT_SIZE] 
        :return: the dot product between the keys and query [BATCH_SIZE x MEMORY_SIZE]
        """

        return sim(k, self._memory, l2_normalize=True, aligned=False)

    @property
    def size(self):
        """Returns the size of the memory

        :return: Int size of the memory
        """

        return self._memory.size()

    @property
    def content(self):
        """Returns the entire memory

        :return: the memory []
        """

        return self._memory
