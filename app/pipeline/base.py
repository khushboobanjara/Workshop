"""Pipeline skeleton: every stage has the same interface, and the runner
handles timing and audit logging so stages cannot forget to log."""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.db import AuditLog, SessionLocal


@dataclass
class PipelineContext:
    email: dict[str, Any]
    email_id: int | None = None
    invoice: dict[str, Any] = field(default_factory=dict)
    risk_signals: dict[str, float] = field(default_factory=dict)  # name -> 0..1
    flags: list[str] = field(default_factory=list)                # human-readable issues
    decision: str | None = None
    stop: bool = False                                            # e.g. not an invoice


class Stage(ABC):
    name: str = "stage"

    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext: ...


class Pipeline:
    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self, ctx: PipelineContext) -> PipelineContext:
        for stage in self.stages:
            if ctx.stop:
                break
            start = time.perf_counter()
            ctx = stage.run(ctx)
            ms = (time.perf_counter() - start) * 1000
            self._audit(ctx, stage.name, ms, getattr(stage, "model_version", None))
        return ctx

    @staticmethod
    def _audit(ctx: PipelineContext, stage: str, ms: float, model_version: str | None = None) -> None:
        with SessionLocal() as s:
            s.add(AuditLog(
                entity_type="email", entity_id=ctx.email_id, stage=stage,
                action="completed", duration_ms=round(ms, 2), model_version=model_version,
                details={"flags": list(ctx.flags), "signals": dict(ctx.risk_signals)},
            ))
            s.commit()
