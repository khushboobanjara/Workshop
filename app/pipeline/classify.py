import joblib

from app.ml.versioning import file_version

from app.config import settings
from app.ml.train_classifier import email_text
from app.pipeline.base import PipelineContext, Stage


class ClassifyStage(Stage):
    name = "classify"

    def __init__(self, model_path: str | None = None, min_confidence: float | None = None):
        self.model = joblib.load(model_path or settings.classifier_model_path)
        self.model_version = file_version(model_path or settings.classifier_model_path)
        self.min_conf = min_confidence if min_confidence is not None else settings.classifier_min_confidence

    def run(self, ctx: PipelineContext) -> PipelineContext:
        proba = self.model.predict_proba([email_text(ctx.email)])[0]
        i = int(proba.argmax())
        category, conf = str(self.model.classes_[i]), float(proba[i])
        ctx.email["category"], ctx.email["category_confidence"] = category, round(conf, 4)

        if conf < self.min_conf:
            ctx.flags.append(f"Low classification confidence ({conf:.2f}) - needs human check")
            ctx.decision = "human_review"
            ctx.stop = True
        elif category != "invoice":
            ctx.flags.append(f"Not an invoice (classified as {category})")
            ctx.decision = "not_invoice"
            ctx.stop = True
        return ctx
