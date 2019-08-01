"""
This module provides data loaders and transformers for popular vision datasets.
"""
from . import transforms
from . import batchify

from .imagenet.classification import ImageNet, ImageNet1kAttr
from .dataloader import DetectionDataLoader, RandomTransformDataLoader
from .pascal_voc.detection import VOCDetection
from .mscoco.detection import COCODetection
from .segbase import ms_batchify_fn
from .mixup.detection import MixupDetection
