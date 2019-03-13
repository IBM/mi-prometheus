# Main imports.
from .problem import Problem
from .problem_factory import ProblemFactory

# Imports from the different domains.
# image_text_to_class
from .image_text_to_class.clevr import CLEVR
from .image_text_to_class.image_text_to_class_problem import ObjectRepresentation, ImageTextToClassProblem
from .image_text_to_class.sort_of_clevr import SortOfCLEVR
from .image_text_to_class.shape_color_query import ShapeColorQuery
from .image_text_to_class.vqa_med_2019 import VQAMED

# image_to_class
from .image_to_class.cifar10 import CIFAR10
from .image_to_class.image_to_class_problem import ImageToClassProblem
from .image_to_class.mnist import MNIST

# seq_to_seq
from .seq_to_seq.seq_to_seq_problem import SeqToSeqProblem

# seq_to_seq.algorithmic
from .seq_to_seq.algorithmic.algorithmic_seq_to_seq_problem import AlgorithmicSeqToSeqProblem

# .seq_to_seq.algorithmic.dual_comparison
from .seq_to_seq.algorithmic.dual_comparison.sequence_comparison_cl import SequenceComparisonCommandLines
from .seq_to_seq.algorithmic.dual_comparison.sequence_equality_cl import SequenceEqualityCommandLines
from .seq_to_seq.algorithmic.dual_comparison.sequence_symmetry_cl import SequenceSymmetryCommandLines

# .seq_to_seq.algorithmic.dual_distraction
from .seq_to_seq.algorithmic.dual_distraction.distraction_carry import DistractionCarry
from .seq_to_seq.algorithmic.dual_distraction.distraction_forget import DistractionForget
from .seq_to_seq.algorithmic.dual_distraction.distraction_ignore import DistractionIgnore
from .seq_to_seq.algorithmic.dual_distraction.reading_span import ReadingSpan

# .seq_to_seq.algorithmic.dual_interruption
from .seq_to_seq.algorithmic.dual_interruption.interruption_not import InterruptionNot
from .seq_to_seq.algorithmic.dual_interruption.interruption_reverse_recall import InterruptionReverseRecall
from .seq_to_seq.algorithmic.dual_interruption.interruption_swap_recall import InterruptionSwapRecall
from .seq_to_seq.algorithmic.dual_interruption.operation_span import OperationSpan

# .seq_to_seq.algorithmic.manipulation_spatial
from .seq_to_seq.algorithmic.manipulation_spatial.manipulation_spatial_not import ManipulationSpatialNot
from .seq_to_seq.algorithmic.manipulation_spatial.manipulation_spatial_rotation import ManipulationSpatialRotation

# .seq_to_seq.algorithmic.manipulation_temporal
from .seq_to_seq.algorithmic.manipulation_temporal.manipulation_temporal_rotation import ManipulationTemporalRotation
from .seq_to_seq.algorithmic.manipulation_temporal.repeat_reverse_recall_cl import RepeatReverseRecallCommandLines
from .seq_to_seq.algorithmic.manipulation_temporal.reverse_recall_cl import ReverseRecallCommandLines
from .seq_to_seq.algorithmic.manipulation_temporal.skip_recall_cl import SkipRecallCommandLines

# .seq_to_seq.algorithmic.recall
from .seq_to_seq.algorithmic.recall.repeat_serial_recall_cl import RepeatSerialRecallCommandLines
from .seq_to_seq.algorithmic.recall.scratch_pad_cl import ScratchPadCommandLines
from .seq_to_seq.algorithmic.recall.serial_recall_cl import SerialRecallCommandLines

# .seq_to_seq.text_to_text
from .seq_to_seq.text_to_text.text_to_text_problem import TextToTextProblem
from .seq_to_seq.text_to_text.translation_anki import TranslationAnki

# .seq_to_seq.video_text_to_class
from .seq_to_seq.video_text_to_class.video_text_to_class_problem import VideoTextToClassProblem
from .seq_to_seq.video_text_to_class.cog.cog import COG

# .video_to_class
from .video_to_class.video_to_class_problem import VideoToClassProblem
# .video_to_class.seq_mnist_to_class
from .video_to_class.seq_mnist_to_class.permuted_sequential_row_mnist import PermutedSequentialRowMnist
from .video_to_class.seq_mnist_to_class.sequential_pixel_mnist import SequentialPixelMNIST


__all__ = [
    'Problem',
    'ProblemFactory',
    # image_text_to_class
    'CLEVR', 'ObjectRepresentation', 'ImageTextToClassProblem', 'SortOfCLEVR', 'ShapeColorQuery', 'VQAMED',
    # image_to_class
    'CIFAR10', 'ImageToClassProblem', 'MNIST',
    # seq_to_seq
    'SeqToSeqProblem',
    # seq_to_seq.algorithmic
    'AlgorithmicSeqToSeqProblem',
    # .seq_to_seq.algorithmic.dual_comparison
    'SequenceComparisonCommandLines', 'SequenceEqualityCommandLines', 'SequenceSymmetryCommandLines',
    # .seq_to_seq.algorithmic.dual_distraction
    'DistractionCarry', 'DistractionForget', 'DistractionIgnore', 'ReadingSpan',
    # .seq_to_seq.algorithmic.dual_interruption
    'InterruptionNot', 'InterruptionReverseRecall', 'InterruptionSwapRecall', 'OperationSpan',
    # .seq_to_seq.algorithmic.manipulation_spatial
    'ManipulationSpatialNot', 'ManipulationSpatialRotation',
    # .seq_to_seq.algorithmic.manipulation_temporal
    'ManipulationTemporalRotation', 'RepeatReverseRecallCommandLines', 'ReverseRecallCommandLines', 'SkipRecallCommandLines',
    # .seq_to_seq.algorithmic.recall
    'RepeatSerialRecallCommandLines', 'ScratchPadCommandLines', 'SerialRecallCommandLines',
    # .seq_to_seq.text2text
    'TextToTextProblem', 'TranslationAnki',
    # .seq_to_seq.video_text_to_class
    'VideoTextToClassProblem',
    'COG',
    # .video_to_class
    'VideoToClassProblem',
    # .video_to_class.seq_mnist_to_class
    'PermutedSequentialRowMnist', 'SequentialPixelMNIST',
    ]
