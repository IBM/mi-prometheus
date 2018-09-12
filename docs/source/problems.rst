.. role:: hidden
    :class: hidden-section

Problems
============

.. automodule:: problems
.. currentmodule:: problems

Base Problem
--------------

.. automodule:: problems.problem
    :members:
    :special-members:
    :exclude-members: __dict__,__weakref__

ImageTextToClass Problems
----------------------------

.. automodule:: problems.image_text_to_class.image_text_to_class_problem
    :members:
    :special-members:
    :exclude-members: __dict__,__weakref__

:hidden:`CLEVR`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: problems.image_text_to_class.clevr
    :members:

.. automodule:: problems.image_text_to_class.clevr_dataset
    :members:

.. automodule:: problems.image_text_to_class.generate_feature_maps
    :members:

:hidden:`ShapeColorQuery`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: problems.image_text_to_class.shape_color_query
    :members:

:hidden:`Sort-Of-CLEVR`
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: problems.image_text_to_class.sort_of_clevr
    :members:

ImageToClass Problems
----------------------------

.. automodule:: problems.image_to_class.image_to_class_problem
    :members:

:hidden:`CIFAR-10`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: problems.image_to_class.cifar10
    :members:

:hidden:`MNIST`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: problems.image_to_class.mnist
    :members:

SequenceToSequence Problems
----------------------------

.. automodule:: problems.seq_to_seq.seq_to_seq_problem
    :members:

Algorithmic SequenceToSequence Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: problems.seq_to_seq.algorithmic.algorithmic_seq_to_seq_problem
    :members:

:hidden:`MAES baselines`
`````````````````````````````
.. automodule:: problems.seq_to_seq.algorithmic.maes_baselines.dual_serial_reverse_recall_cl
    :members:

.. automodule:: problems.seq_to_seq.algorithmic.maes_baselines.repeat_reverse_recall_cl
    :members:

.. automodule:: problems.seq_to_seq.algorithmic.maes_baselines.repeat_serial_recall_cl
    :members:

.. automodule:: problems.seq_to_seq.algorithmic.maes_baselines.reverse_recall_cl
    :members:

.. automodule:: problems.seq_to_seq.algorithmic.maes_baselines.sequence_comparison_cl
    :members:

.. automodule:: problems.seq_to_seq.algorithmic.maes_baselines.sequence_equality_cl
    :members:

.. automodule:: problems.seq_to_seq.algorithmic.maes_baselines.sequence_symmetry_cl
    :members:

.. automodule:: problems.seq_to_seq.algorithmic.maes_baselines.serial_recall_cl
    :members:

.. automodule:: problems.seq_to_seq.algorithmic.maes_baselines.skip_recall_cl
    :members:

.. automodule:: problems.seq_to_seq.algorithmic.maes_baselines.sequence_symmetry_cl
    :members:

:hidden:`Distraction Carry`
`````````````````````````````
.. automodule:: problems.seq_to_seq.algorithmic.distraction_carry
    :members:

:hidden:`Distraction Forget`
`````````````````````````````
.. automodule:: problems.seq_to_seq.algorithmic.distraction_forget
    :members:

:hidden:`Distraction Ignore`
`````````````````````````````
.. automodule:: problems.seq_to_seq.algorithmic.distraction_ignore
    :members:

:hidden:`Interruption Not`
`````````````````````````````
.. automodule:: problems.seq_to_seq.algorithmic.interruption_not
    :members:

:hidden:`Interruption Reverse Recall`
```````````````````````````````````````
.. automodule:: problems.seq_to_seq.algorithmic.interruption_reverse_recall
    :members:

:hidden:`Interruption Swap Recall`
```````````````````````````````````````
.. automodule:: problems.seq_to_seq.algorithmic.interruption_swap_recall
    :members:

:hidden:`Manipulation Spatial Not`
```````````````````````````````````````
.. automodule:: problems.seq_to_seq.algorithmic.manipulation_spatial_not
    :members:

:hidden:`Manipulation Spatial Rotation`
```````````````````````````````````````````
.. automodule:: problems.seq_to_seq.algorithmic.manipulation_spatial_rotation
    :members:

:hidden:`Manipulation Temporal Swap`
```````````````````````````````````````
.. automodule:: problems.seq_to_seq.algorithmic.manipulation_temporal_swap
    :members:

:hidden:`Operation Span`
```````````````````````````````````````
.. automodule:: problems.seq_to_seq.algorithmic.operation_span
    :members:

:hidden:`Reading Span`
```````````````````````````````````````
.. automodule:: problems.seq_to_seq.algorithmic.reading_span
    :members:

:hidden:`Reverse Recall`
```````````````````````````````````````
.. automodule:: problems.seq_to_seq.algorithmic.reverse_recall
    :members:

:hidden:`Scrath Pad`
```````````````````````````````````````
.. automodule:: problems.seq_to_seq.algorithmic.scratch_pad
    :members:

:hidden:`Serial Recall`
```````````````````````````````````````
.. automodule:: problems.seq_to_seq.algorithmic.serial_recall
    :members:

:hidden:`Serial Recall Simplified`
```````````````````````````````````````
.. automodule:: problems.seq_to_seq.algorithmic.serial_recall_simplified
    :members:

TextToText Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: problems.seq_to_seq.text2text.text_to_text_problem
    :members:

:hidden:`Translation`
`````````````````````````````
.. automodule:: problems.seq_to_seq.text2text.translation
    :members:

Utils
----------------------------

.. automodule:: problems.utils.language
    :members:

VideoToClass Problems
----------------------------

.. automodule:: problems.video_to_class.video_to_class_problem
    :members:

:hidden:`Sequential MNIST`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: problems.video_to_class.seq_mnist_to_class.permuted_sequential_row_mnist
    :members:

.. automodule:: problems.video_to_class.seq_mnist_to_class.sequential_pixel_mnist
    :members:

.. automodule:: problems.video_to_class.seq_mnist_to_class.sequential_row_mnist
    :members:


ProblemFactory
-----------------
.. currentmodule:: problems
.. autoclass:: ProblemFactory
    :members:
