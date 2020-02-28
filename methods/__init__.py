import random
from typing import Dict, Type

from automl.components.base import EstimatorComponent
from automl.components.classification import ClassifierChoice
from automl.components.data_preprocessing import DataPreprocessorChoice
from automl.components.feature_preprocessing import FeaturePreprocessorChoice

ALGORITHMS: Dict[str, Type[EstimatorComponent]] = {}
ALGORITHMS.update(ClassifierChoice().get_components())
ALGORITHMS.update(DataPreprocessorChoice().get_components())
ALGORITHMS.update(FeaturePreprocessorChoice().get_components())
