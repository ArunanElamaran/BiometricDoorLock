"""
Facial_Rec_Development package.

Provides:
- ImagePreprocessor: image loading, alignment and preprocessing utilities.
- LightweightFaceNet: lightweight VGG-style CNN backbone.
- FaceRecognitionSystem: high-level training/inference interface.
"""

from .ImageProcessor import ImagePreprocessor
from .model import LightweightFaceNet, FaceRecognitionSystem, print_model_info

__all__ = [
    "ImagePreprocessor",
    "LightweightFaceNet",
    "FaceRecognitionSystem",
    "print_model_info",
]

