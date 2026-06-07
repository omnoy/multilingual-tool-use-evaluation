"""
Evaluation runner for multilingual benchmarks.

This module is a thin adapter over BFCL's eval_runner. It:
  1. Resolves multilingual dataset paths (data/benchmarks/<category>/<locale>.json)
     instead of the default BFCL data directory.
  2. Writes results under results/<model>/<locale>/<category>/ so locale-specific
     scores are stored separately and can be compared across languages.
  3. Delegates all actual scoring logic to BFCL's existing checkers unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from bfcl_eval.eval_checker import eval_runner as bfcl_runner
from bfcl_eval.constants.eval_config import RESULT_PATH, SCORE_PATH

from multilingual_bfcl.benchmark_builder import output_path as benchmark_output_path
from multilingual_bfcl.localization.locale_config import get_locale

_PACKAGE_ROOT = Path(__file__).parent.parent.parent
_RESULT_ROOT = _PACKAGE_ROOT / "results"


def multilingual_result_path(model: str, locale_code: str, category: str) -> Path:
    return _RESULT_ROOT / model / locale_code / category


def run_evaluation(
    model: str,
    category: str,
    locale_code: str,
    result_file: Path,
) -> dict[str, Any]:
    """
    Evaluate model outputs stored in result_file against BFCL ground truth.

    Args:
        model:        Model registry name (e.g. "claude-opus-4-8-FC").
        category:     BFCL category name (e.g. "simple_python").
        locale_code:  BCP-47 locale of the dataset (e.g. "he").
        result_file:  Path to the JSON file with model responses
                      (same format as BFCL result files).

    Returns:
        Score dict as returned by BFCL's eval_runner.
    """
    locale = get_locale(locale_code)  # validates locale early

    # BFCL's eval_runner locates the dataset by model+category using its own
    # path conventions. We patch RESULT_PATH temporarily so BFCL writes scores
    # to our locale-aware results directory rather than its own.
    out_dir = multilingual_result_path(model, locale_code, category)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy result file into a temp directory that mirrors BFCL's expected layout
    # so we can call bfcl_runner without modifying its internals.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_model_dir = Path(tmp) / model
        tmp_model_dir.mkdir()
        shutil.copy(result_file, tmp_model_dir / result_file.name)

        original_result_path = os.environ.get("BFCL_RESULT_PATH", "")
        original_score_path = os.environ.get("BFCL_SCORE_PATH", "")
        try:
            os.environ["BFCL_RESULT_PATH"] = tmp
            os.environ["BFCL_SCORE_PATH"] = str(out_dir)
            scores = bfcl_runner.runner(
                model=model,
                test_category=category,
                api_sanity_check=False,
            )
        finally:
            os.environ["BFCL_RESULT_PATH"] = original_result_path
            os.environ["BFCL_SCORE_PATH"] = original_score_path

    return scores
