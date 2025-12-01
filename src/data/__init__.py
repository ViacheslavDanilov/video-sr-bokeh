"""ICP Prediction Application

A streamlined application for predicting Intracranial Pressure (ICP)
from Blood Flow Index (BFI) time series data using machine learning models.
"""  # noqa: D415

__version__ = "0.1.0"
__author__ = "Safe-ICP Team"

from .predictor import ICPPredictor

__all__ = ["ICPPredictor"]
