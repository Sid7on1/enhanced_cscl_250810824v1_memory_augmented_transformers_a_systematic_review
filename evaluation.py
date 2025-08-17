import logging
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple

# Define constants
VELOCITY_THRESHOLD = 0.5
FLOW_THEORY_THRESHOLD = 0.8

# Define exception classes
class EvaluationException(Exception):
    """Base class for evaluation exceptions"""
    pass

class InvalidMetricException(EvaluationException):
    """Raised when an invalid metric is specified"""
    pass

class InvalidDataException(EvaluationException):
    """Raised when invalid data is provided"""
    pass

# Define data structures/models
class EvaluationResult:
    """Represents the result of an evaluation"""
    def __init__(self, metric: str, value: float):
        self.metric = metric
        self.value = value

# Define validation functions
def validate_metric(metric: str) -> bool:
    """Validates a metric"""
    valid_metrics = ["accuracy", "precision", "recall", "f1_score"]
    return metric in valid_metrics

def validate_data(data: List[float]) -> bool:
    """Validates data"""
    return all(isinstance(x, (int, float)) for x in data)

# Define utility methods
def calculate_velocity(data: List[float]) -> float:
    """Calculates the velocity of a sequence of data points"""
    if len(data) < 2:
        raise InvalidDataException("Insufficient data to calculate velocity")
    return (data[-1] - data[0]) / (len(data) - 1)

def calculate_flow_theory(data: List[float]) -> float:
    """Calculates the flow theory of a sequence of data points"""
    if len(data) < 2:
        raise InvalidDataException("Insufficient data to calculate flow theory")
    return sum((data[i] - data[i-1]) ** 2 for i in range(1, len(data))) / (len(data) - 1)

# Define main class
class Evaluator:
    """Evaluates agent performance using various metrics"""
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def evaluate(self, data: List[float], metric: str) -> EvaluationResult:
        """Evaluates the agent performance using the specified metric"""
        if not validate_metric(metric):
            raise InvalidMetricException("Invalid metric specified")
        if not validate_data(data):
            raise InvalidDataException("Invalid data provided")

        if metric == "accuracy":
            return self._evaluate_accuracy(data)
        elif metric == "precision":
            return self._evaluate_precision(data)
        elif metric == "recall":
            return self._evaluate_recall(data)
        elif metric == "f1_score":
            return self._evaluate_f1_score(data)

    def _evaluate_accuracy(self, data: List[float]) -> EvaluationResult:
        """Evaluates the accuracy of the agent"""
        velocity = calculate_velocity(data)
        if velocity > VELOCITY_THRESHOLD:
            return EvaluationResult("accuracy", 1.0)
        else:
            return EvaluationResult("accuracy", 0.0)

    def _evaluate_precision(self, data: List[float]) -> EvaluationResult:
        """Evaluates the precision of the agent"""
        flow_theory = calculate_flow_theory(data)
        if flow_theory > FLOW_THEORY_THRESHOLD:
            return EvaluationResult("precision", 1.0)
        else:
            return EvaluationResult("precision", 0.0)

    def _evaluate_recall(self, data: List[float]) -> EvaluationResult:
        """Evaluates the recall of the agent"""
        velocity = calculate_velocity(data)
        if velocity > VELOCITY_THRESHOLD:
            return EvaluationResult("recall", 1.0)
        else:
            return EvaluationResult("recall", 0.0)

    def _evaluate_f1_score(self, data: List[float]) -> EvaluationResult:
        """Evaluates the F1 score of the agent"""
        precision = self._evaluate_precision(data).value
        recall = self._evaluate_recall(data).value
        if precision + recall == 0:
            return EvaluationResult("f1_score", 0.0)
        else:
            return EvaluationResult("f1_score", 2 * (precision * recall) / (precision + recall))

# Define configuration support
class Configuration:
    """Represents the configuration for the evaluator"""
    def __init__(self, config: Dict[str, str]):
        self.config = config

    def get_config(self) -> Dict[str, str]:
        """Returns the configuration"""
        return self.config

# Define unit test compatibility
import unittest

class TestEvaluator(unittest.TestCase):
    def test_evaluate(self):
        evaluator = Evaluator({"metric": "accuracy"})
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = evaluator.evaluate(data, "accuracy")
        self.assertEqual(result.metric, "accuracy")
        self.assertAlmostEqual(result.value, 1.0)

    def test_validate_metric(self):
        self.assertTrue(validate_metric("accuracy"))
        self.assertFalse(validate_metric("invalid_metric"))

    def test_validate_data(self):
        self.assertTrue(validate_data([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.assertFalse(validate_data([1.0, "2.0", 3.0, 4.0, 5.0]))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {"metric": "accuracy"}
    evaluator = Evaluator(config)
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = evaluator.evaluate(data, "accuracy")
    print(f"Metric: {result.metric}, Value: {result.value}")
    unittest.main(argv=[''], verbosity=2, exit=False)