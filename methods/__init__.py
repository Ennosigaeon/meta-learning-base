from typing import Dict, Type

from dswizard.components.base import EstimatorComponent
from dswizard.components.classification import ClassifierChoice
from dswizard.components.data_preprocessing import DataPreprocessorChoice
from dswizard.components.feature_preprocessing import FeaturePreprocessorChoice

"""Generates dict with all available Algorithms from sklearn-components"""
ALGORITHMS: Dict[str, Type[EstimatorComponent]] = {}
ALGORITHMS.update(ClassifierChoice().get_components())
ALGORITHMS.update(DataPreprocessorChoice().get_components())
ALGORITHMS.update(FeaturePreprocessorChoice().get_components())

CLASSIFIERS: Dict[str, Type[EstimatorComponent]] = {}
CLASSIFIERS.update(ClassifierChoice().get_components())
