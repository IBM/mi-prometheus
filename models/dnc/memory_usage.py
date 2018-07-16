import torch
from misc.app_state import AppState

# Ensure values are greater than epsilon to avoid numerical instability.
_EPSILON = 1e-6

class MemoryUsage(object):
    """Memory usage that is increased by writing and decreased by reading.
    This module has a state is a tensor with values in
    the range [0, 1] indicating the usage of each of `memory_size` memory slots.
    The usage is:
    *   Increased by writing, where usage is increased towards 1 at the write
        addresses.
    *   Decreased by reading, where usage is decreased after reading from a
        location when free_gate is close to 1.
    The function `write_allocation_weights` can be invoked to get free locations
    to write to for a number of write heads.
    """
  
    def __init__(self, name='MemoryUsage'):
        """Creates a MemoryUsages module.
          :param name: Name of the module.
        """
        super(MemoryUsage, self).__init__()
  
    def init_state(self, memory_address_size, batch_size):
        """
        Returns 'zero' (initial) state tuple.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of InterfaceStateTuple class.
        """
        dtype = AppState().dtype
        self._memory_size=memory_address_size

        usage = torch.zeros((batch_size,  memory_address_size)).type(dtype)
  
        return usage
  
  
    def calculate_usage(self, write_weights, free_gate, read_weights, prev_usage):
        """Calculates the new memory usage u_t.
        Memory that was written to in the previous time step will have its usage
        increased; memory that was read from and the controller says can be "freed"
        will have its usage decreased.
        Args:
          :param write_weights: tensor of shape `[batch_size, num_writes,
              memory_size]` giving write weights at previous time step.
          :param free_gate: tensor of shape `[batch_size, num_reads]` which indicates
              which read heads read memory that can now be freed.
          :param read_weights: tensor of shape `[batch_size, num_reads,
              memory_size]` giving read weights at previous time step.
          :param prev_usage: tensor of shape `[batch_size, memory_size]` giving
              usage u_{t - 1} at the previous time step, with entries in range
              [0, 1].
        Returns:
          :returns: tensor of shape `[batch_size, memory_size]` representing updated memory
          usage.
        """ 
       # Calculation of usage is not differentiable with respect to write weights.
        with torch.no_grad():
            usage = self._usage_after_write(prev_usage, write_weights)
            usage = self._usage_after_read(usage, free_gate, read_weights)
        return usage
  
    def write_allocation_weights(self, usage, write_gates, num_writes):
        """Calculates freeness-based locations for writing to.
        This finds unused memory by ranking the memory locations by usage, for each
        write head. (For more than one write head, we use a "simulated new usage"
        which takes into account the fact that the previous write head will increase
        the usage in that area of the memory.)
        Args:
          :param usage: A tensor of shape `[batch_size, memory_size]` representing
              current memory usage.
          :param write_gates: A tensor of shape `[batch_size, num_writes]` with values in
              the range [0, 1] indicating how much each write head does writing
              based on the address returned here (and hence how much usage
              increases).
          :param num_writes: The number of write heads to calculate write weights for.
        Returns:
          :returns: tensor of shape `[batch_size, num_writes, memory_size]` containing the
              freeness-based write locations. Note that this isn't scaled by
              `write_gate`; this scaling must be applied externally.
        """
  
        allocation_weights = []
        for i in range(num_writes):
            allocation_weights.append(self._allocation(usage))
            # update usage to take into account writing to this new allocation
            usage = usage + ((1 - usage) * write_gates[:, i, :] * allocation_weights[i])
 
        # Pack the allocation weights for the write heads into one tensor.
        full_weights=torch.stack(allocation_weights, dim=1)

        return full_weights
  
    def _usage_after_write(self, prev_usage, write_weights):
        """Calculates the new usage after writing to memory.
        Args:
          :param prev_usage: tensor of shape `[batch_size, memory_size]`.
          :param write_weights: tensor of shape `[batch_size, num_writes, memory_size]`.
        Returns:
          :returns: New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        # Calculate the aggregated effect of all write heads
        write_weights2 = 1 - torch.prod(1 - write_weights, 1)
        return prev_usage + (1 - prev_usage) * write_weights2
  
    def _usage_after_read(self, prev_usage, free_gate, read_weights):
        """Calculates the new usage after reading and freeing from memory.
        Args:
          :param prev_usage: tensor of shape `[batch_size, memory_size]`.
          :param free_gate: tensor of shape `[batch_size, num_reads]` with entries in the
              range [0, 1] indicating the amount that locations read from can be
              freed.
          :param read_weights: tensor of shape `[batch_size, num_reads, memory_size]`.
        Returns:
          :returns: New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        free_read_weights = free_gate * read_weights
        phi = torch.prod(1 - free_read_weights, 1)
        
        return prev_usage * phi
  
    def _allocation(self, usage):
        r"""Computes allocation by sorting `usage`.
        This corresponds to the value a = a_t[\phi_t[j]] in the paper.
        Args:
          :param usage: tensor of shape `[batch_size, memory_size]` indicating current
              memory usage. This is equal to u_t in the paper when we only have one
              write head, but for multiple write heads, one should update the usage
              while iterating through the write heads to take into account the
              allocation returned by this function.
        Returns:
          :returns: Tensor of shape `[batch_size, memory_size]` corresponding to allocation.
        """
        # Ensure values are not too small prior to cumprod.
        usage = _EPSILON + (1 - _EPSILON) * usage
  
        #sorts usage along the last index
        sorted_usage, indices = torch.sort(usage, descending=False)
        sorted_nonusage=1-sorted_usage
  
        #this computes the exclusive cumulative product
        prod_sorted_usage = self.exclusive_cumprod_temp(sorted_usage)
        
        # Weights 1-usage by the exclusive product of the sorted usage
        # This is a trick to choose the first point
        # as anything other than the first element of the exclusive product should be negligibly small
        # and this operation isn't differentiable anyways due to the sorting above
        sorted_allocation = sorted_nonusage * prod_sorted_usage
  
        # This final line "unsorts" sorted_allocation, so that the indexing
        # corresponds to the original indexing of `usage`. 
        unsorted_all = sorted_allocation.new(*sorted_allocation.size())
        unsorted_all.scatter_(1, indices, sorted_allocation)
         
        return unsorted_all

    def exclusive_cumprod_temp(self, sorted_usage, dim=1):
        """Applies the exclusive cumultative product (at the moment it assumes the shape of the input)
        Args:
            :param sorted_usage: tensor of shape `[batch_size, memory_size]` indicating current
              memory usage sorted in ascending order.
        Returns:
            :returns: Tensor of shape `[batch_size, memory_size]` that is exclusive pruduct of the sorted usage
                     i.e. = [1, u1, u1*u2, u1*u2*u3, ....]

        """
        #TODO: expand this so it works for any dim
        dtype = AppState().dtype
        a=torch.ones((sorted_usage.shape[0],1)).type(dtype)
        b=torch.cat((a,sorted_usage),dim=dim).type(dtype)
        prod_sorted_usage = torch.cumprod(b, dim=dim)[:,:-1]
        return prod_sorted_usage

    @property
    def state_size(self):
        """Returns the shape of the state tensor."""
        return (self._memory_size)
