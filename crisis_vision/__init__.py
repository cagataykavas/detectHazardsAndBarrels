"""Explainable hazmat placard and colored-barrel tracking."""

from crisis_vision.config import CrisisConfig
from crisis_vision.pipeline import FrameProcessor, analyze_frames, analyze_video

__all__ = ["CrisisConfig", "FrameProcessor", "analyze_frames", "analyze_video"]
__version__ = "1.0.0"
