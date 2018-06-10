"""DWM Memory"""
import torch
from models.dwm.tensor_utils import sim, outer_prod


class Memory:
    def __init__(self, mem_t):
        """Initializes the memory

        :param mem_t of shape (batch_size, memory_addresses_size, memory_content_size): the memory at time t
        """

        self._memory = mem_t

    def attention_read(self, wt):
        """Returns the data read from memory       
 
        :param wt of shape (batch_size, num_heads, memory_addresses_size)  : head's weights
        :return: the read data of shape (batch_size, num_heads, memory_content_size)
        """

        return sim(wt, self._memory)

    def add_weighted(self, add, wt):
        """Writes data to memory

        :param wt of shape (batch_size, num_heads, memory_addresses_size)  : head's weights
        :param add of shape (batch_size, num_heads, memory_content_size) : the data to be added to memory

        :return the updated memory of shape (batch_size, memory_addresses_size, memory_content_size)
        """

        # memory = memory + sum_{head h} weighted add(h)
        self._memory = self._memory + torch.sum(outer_prod(add, wt), dim=-3)

    def erase_weighted(self, erase, wt):
        """Erases elements from memory

        :param wt of shape (batch_size, num_heads, memory_addresses_size)  : head's weights
        :param erase of shape (batch_size, num_heads, memory_content_size) : data to be erased from memory

        :return the updated memory of shape (batch_size, memory_addresses_size, memory_content_size)
        """

        # memory = memory * product_{head h} (1 - weighted erase(h))
        self._memory = self._memory * torch.prod(1 - outer_prod(erase, wt), dim=-3)

    def content_similarity(self, k):
        """Calculates the dot product for Content aware addressing

        :param k of shape (batch_size, num_heads, memory_content_size): the keys emitted by the controller
        :return: the dot product between the keys and query of shape (batch_size, num_heads, memory_addresses_size)
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
