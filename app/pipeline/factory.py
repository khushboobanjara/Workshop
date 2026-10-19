from pathlib import Path

from app.extraction.learned import DEFAULT_MODEL_PATH, LearnedExtractor
from app.pipeline.anomaly import AnomalyStage
from app.pipeline.base import Pipeline
from app.pipeline.classify import ClassifyStage
from app.pipeline.decision import DecisionStage
from app.pipeline.duplicates import DuplicateStage
from app.pipeline.extract import ExtractStage, regex_extractor
from app.pipeline.risk import RiskStage
from app.pipeline.validate import ValidateStage


def build_extractor():
    """Learned extractor if its model has been trained (python -m app.ml.train_extractor), else the regex."""
    if Path(DEFAULT_MODEL_PATH).exists():
        return LearnedExtractor.load(DEFAULT_MODEL_PATH)
    return regex_extractor


def build_pipeline() -> Pipeline:
    """The production pipeline, in architecture order."""
    return Pipeline([ClassifyStage(), ExtractStage(build_extractor()), ValidateStage(), DuplicateStage(),
                     AnomalyStage(), RiskStage(), DecisionStage()])
