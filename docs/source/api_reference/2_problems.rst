.. role:: hidden
    :class: hidden-section

miprometheus.problems
********************************************

.. automodule:: miprometheus.problems
.. currentmodule:: miprometheus.problems


Problem
============================================
.. autoclass:: miprometheus.problems.Problem

ProblemFactory
============================================
.. autoclass:: ProblemFactory

ImageTextToClass Problems
============================================
.. autoclass:: miprometheus.problems.ImageTextToClassProblem

CLEVR
--------------------------------------------
.. autoclass:: CLEVR

Sort-Of-CLEVR
--------------------------------------------
.. autoclass:: SortOfCLEVR

ShapeColorQuery
--------------------------------------------
.. autoclass:: ShapeColorQuery



ImageToClass Problems
============================================
.. autoclass:: ImageToClassProblem

CIFAR-10
--------------------------------------------
.. autoclass:: CIFAR10

MNIST
--------------------------------------------
.. autoclass:: MNIST


SequenceToSequence Problems
============================================
.. autoclass:: SeqToSeqProblem


VideoTextToClass Problems
--------------------------------------------
.. autoclass:: VQAProblem

COG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: COG


Algorithmic SequenceToSequence Problems
============================================
.. autoclass:: AlgorithmicSeqToSeqProblem

Dual Comparison
--------------------------------------------

SequenceComparisonCommandLines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SequenceComparisonCommandLines

SequenceEqualityCommandLines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SequenceEqualityCommandLines

SequenceSymmetryCommandLines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SequenceSymmetryCommandLines


Dual Distraction
--------------------------------------------

DistractionCarry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: DistractionCarry

DistractionForget
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: DistractionForget

DistractionIgnore
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: DistractionIgnore


Dual Ignore
--------------------------------------------

InterruptionNot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: InterruptionNot

InterruptionReverseRecall
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: InterruptionReverseRecall

InterruptionSwapRecall
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: InterruptionSwapRecall


Manipulation Spatial
--------------------------------------------

ManipulationSpatialNot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ManipulationSpatialNot

ManipulationSpatialRotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ManipulationSpatialRotation

Manipulation Temporal
--------------------------------------------

ManipulationTemporalSwap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ManipulationTemporalSwap

SkipRecallCommandLines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SkipRecallCommandLines

Recall
--------------------------------------------

OperationSpan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: OperationSpan

ReadingSpan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ReadingSpan

RepeatReverseRecallCommandLines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RepeatReverseRecallCommandLines

RepeatSerialRecallCommandLines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RepeatSerialRecallCommandLines

ReverseRecallCommandLines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ReverseRecallCommandLines

ScratchPadCommandLines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ScratchPadCommandLines

SerialRecallCommandLines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SerialRecallCommandLines


TextToText Problems
============================================
.. automodule:: miprometheus.problems.seq_to_seq.text2text.text_to_text_problem
    :members:
    :special-members:
    :exclude-members: __dict__,__weakref__

TranslationAnki
--------------------------------------------
.. autoclass:: miprometheus.problems.seq_to_seq.TranslationAnki


VideoToClass Problems
============================================
.. autoclass:: miprometheus.problems.VideoToClassProblem

Permuted Sequential Row Mnist
--------------------------------------------
.. autoclass:: miprometheus.problems.video_to_class.PermutedSequentialRowMnist

Sequential Pixel MNIST
--------------------------------------------
.. autoclass:: miprometheus.problems.video_to_class.SequentialPixelMNIST
