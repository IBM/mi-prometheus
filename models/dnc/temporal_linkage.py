class TemporalLinkage():
    """Keeps track of write order for forward and backward addressing.
    This is a pseudo-RNNCore module, whose state is a pair `(link,
    precedence_weights)`, where `link` is a (collection of) graphs for (possibly
    multiple) write heads (represented by a tensor with values in the range
    [0, 1]), and `precedence_weights` records the "previous write locations" used
    to build the link graphs.
    The function `directional_read_weights` computes addresses following the
    forward and backward directions in the link graphs.
    """
  
    def __init__(self, memory_size, num_writes, name='temporal_linkage'):
        """Construct a TemporalLinkage module.
        Args:
          memory_size: The number of memory slots.
          num_writes: The number of write heads.
          name: Name of the module.
        """
        super(TemporalLinkage, self).__init__(name=name)
        self._memory_size = memory_size
        self._num_writes = num_writes
        
     
  
    def _build(self, write_weights, prev_state):
        """Calculate the updated linkage state given the write weights.
        Args:
          write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
              containing the memory addresses of the different write heads.
          prev_state: `TemporalLinkageState` tuple containg a tensor `link` of
              shape `[batch_size, num_writes, memory_size, memory_size]`, and a
              tensor `precedence_weights` of shape `[batch_size, num_writes,
              memory_size]` containing the aggregated history of recent writes.
        Returns:
          A `TemporalLinkageState` tuple `next_state`, which contains the updated
          link and precedence weights.
        """
        link = self._link(prev_state.link, prev_state.precedence_weights,
                          write_weights)
        precedence_weights = self._precedence_weights(prev_state.precedence_weights,
                                                      write_weights)
        return TemporalLinkageState(
            link=link, precedence_weights=precedence_weights)
    
    def directional_read_weights(self, link, prev_read_weights, forward):
        """Calculates the forward or the backward read weights.
        For each read head (at a given address), there are `num_writes` link graphs
        to follow. Thus this function computes a read address for each of the
        `num_reads * num_writes` pairs of read and write heads.
        Args:
          link: tensor of shape `[batch_size, num_writes, memory_size,
              memory_size]` representing the link graphs L_t.
          prev_read_weights: tensor of shape `[batch_size, num_reads,
              memory_size]` containing the previous read weights w_{t-1}^r.
          forward: Boolean indicating whether to follow the "future" direction in
              the link graph (True) or the "past" direction (False).
        Returns:
          tensor of shape `[batch_size, num_reads, num_writes, memory_size]`
        """
        with tf.name_scope('directional_read_weights'):
          # We calculate the forward and backward directions for each pair of
          # read and write heads; hence we need to tile the read weights and do a
          # sort of "outer product" to get this.
          expanded_read_weights = tf.stack([prev_read_weights] * self._num_writes,
                                           1)
          result = tf.matmul(expanded_read_weights, link, adjoint_b=forward)
          # Swap dimensions 1, 2 so order is [batch, reads, writes, memory]:
          return tf.transpose(result, perm=[0, 2, 1, 3])
    
    def _link(self, prev_link, prev_precedence_weights, write_weights):
        """Calculates the new link graphs.
        For each write head, the link is a directed graph (represented by a matrix
        with entries in range [0, 1]) whose vertices are the memory locations, and
        an edge indicates temporal ordering of writes.
        Args:
          prev_link: A tensor of shape `[batch_size, num_writes, memory_size,
              memory_size]` representing the previous link graphs for each write
              head.
          prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
              memory_size]` which is the previous "aggregated" write weights for
              each write head.
          write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
              containing the new locations in memory written to.
        Returns:
          A tensor of shape `[batch_size, num_writes, memory_size, memory_size]`
          containing the new link graphs for each write head.
        """
        with tf.name_scope('link'):
          batch_size = prev_link.get_shape()[0].value
          write_weights_i = tf.expand_dims(write_weights, 3)
          write_weights_j = tf.expand_dims(write_weights, 2)
          prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights, 2)
          prev_link_scale = 1 - write_weights_i - write_weights_j
          new_link = write_weights_i * prev_precedence_weights_j
          link = prev_link_scale * prev_link + new_link
          # Return the link with the diagonal set to zero, to remove self-looping
          # edges.
          return tf.matrix_set_diag(
              link,
              tf.zeros(
                  [batch_size, self._num_writes, self._memory_size],
                  dtype=link.dtype))
    
    def _precedence_weights(self, prev_precedence_weights, write_weights):
        """Calculates the new precedence weights given the current write weights.
        The precedence weights are the "aggregated write weights" for each write
        head, where write weights with sum close to zero will leave the precedence
        weights unchanged, but with sum close to one will replace the precedence
        weights.
        Args:
          prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
              memory_size]` containing the previous precedence weights.
          write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
              containing the new write weights.
        Returns:
          A tensor of shape `[batch_size, num_writes, memory_size]` containing the
          new precedence weights.
        """
        with tf.name_scope('precedence_weights'):
          write_sum = tf.reduce_sum(write_weights, 2, keep_dims=True)
          return (1 - write_sum) * prev_precedence_weights + write_weights
    
    @property
    def state_size(self):
        """Returns a `TemporalLinkageState` tuple of the state tensors' shapes."""
        return TemporalLinkageState(
            link=tf.TensorShape(
                [self._num_writes, self._memory_size, self._memory_size]),
            precedence_weights=tf.TensorShape([self._num_writes,
                                               self._memory_size]),)
