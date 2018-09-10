from .clevr import CLEVR
from .clevr_dataset import CLEVRDataset
from .generate_feature_maps import GenerateFeatureMaps
from .image_text_to_class_problem import ImageTextTuple, SceneDescriptionTuple, ObjectRepresentation, \
    ImageTextToClassProblem
from .sort_of_clevr import SortOfCLEVR
from .shape_color_query import ShapeColorQuery

__all__ = [
    'CLEVR',
    'CLEVRDataset',
    'GenerateFeatureMaps',
    'ImageTextTuple',
    'SceneDescriptionTuple',
    'ObjectRepresentation',
    'ImageTextToClassProblem',
    'SortOfCLEVR',
    'ShapeColorQuery']
