"""Promote a candidate model to production, after a person has looked at the numbers.

    python -m scripts.promote_model --name classifier --candidate models/candidates/classifier-v2.joblib        # dry run
    python -m scripts.promote_model --name classifier --candidate models/candidates/classifier-v2.joblib --yes  # do it

Nothing trains or deploys automatically. This copies the candidate over the serving file (keeping the old one in
models/archive/) and tells you to restart the service. Train a candidate with, for example:
    python -m app.ml.train_classifier --model-out models/candidates/classifier-v2.joblib"""
import argparse
import shutil
import sys
from pathlib import Path

from app.ml import registry
from app.ml.versioning import file_version


def promote(name: str, candidate: str, yes: bool = False, archive_dir: str = "models/archive") -> int:
    serving = registry.serving_paths().get(name)
    if serving is None:
        print(f"Unknown model '{name}'. Choose from: {', '.join(registry.serving_paths())}")
        return 2
    cand_meta = registry.read_meta(candidate)
    if cand_meta is None:
        print("Refusing: the candidate has no valid metadata file (or it does not match the model file).")
        print("A model that was not evaluated and recorded by its training run cannot be promoted.")
        return 1
    if cand_meta["model_name"] != name:
        print(f"Refusing: the candidate is a '{cand_meta['model_name']}' model, not '{name}'.")
        return 1
    current = registry.read_meta(serving)
    print(f"Current  {name}: {file_version(serving)}  {registry.headline(current) or 'no recorded metrics'}")
    print(f"Candidate {name}: {cand_meta['version']}  {registry.headline(cand_meta) or 'no recorded metrics'}")
    print(f"Both measured on: {registry.DATA_NOTICE}")
    if not yes:
        print("\nDry run only. Re-run with --yes to promote.")
        return 0
    if file_version(serving):
        Path(archive_dir).mkdir(parents=True, exist_ok=True)
        old = Path(archive_dir) / f"{name}-{file_version(serving)}{Path(serving).suffix}"
        shutil.copy2(serving, old)
        if registry.meta_path(serving).exists():
            shutil.copy2(registry.meta_path(serving), registry.meta_path(old))
        print(f"Archived the current model to {old}")
    shutil.copy2(candidate, serving)
    shutil.copy2(registry.meta_path(candidate), registry.meta_path(serving))
    print(f"Promoted {cand_meta['version']} to {serving}. Restart the service to load it.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    return promote(a.name, a.candidate, a.yes)


if __name__ == "__main__":
    sys.exit(main())
