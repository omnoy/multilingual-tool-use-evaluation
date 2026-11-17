"""Experiment (benchmark file) naming: localization level <-> file stem.

A translated benchmark file is named ``<locale>_<experiment>.jsonl``, where the experiment
is derived from the localization level stored in each entry's ``localization_level`` field:

    level             experiment                      e.g.
    ---------------   -----------------------------   -------------------------------
    (English source)  en_baseline                     en/en_baseline.jsonl
    query             native_query_eng_target         he/he_native_query_eng_target.jsonl
    full              native_query_native_target      he/he_native_query_native_target.jsonl
    full_func_desc    native_qt_func_descs            he/he_native_qt_func_descs.jsonl
    full_lang_desc    native_qt_lang_hints            he/he_native_qt_lang_hints.jsonl
    full_full_full    native_qt_max_context           he/he_native_qt_max_context.jsonl
    func_desc         native_query_func_descs         he/he_native_query_func_descs.jsonl

Levels without a mapping fall back to ``<locale>_<level>``.
"""
from __future__ import annotations

BASELINE = "en_baseline"

LEVEL_TO_EXPERIMENT: dict[str, str] = {
    "query": "native_query_eng_target",
    "full": "native_query_native_target",
    "full_func_desc": "native_qt_func_descs",
    "full_lang_desc": "native_qt_lang_hints",
    "full_full_full": "native_qt_max_context",
    "func_desc": "native_query_func_descs",
}
EXPERIMENT_TO_LEVEL: dict[str, str] = {v: k for k, v in LEVEL_TO_EXPERIMENT.items()}

# Display order of experiments (without the locale prefix) in summaries.
EXPERIMENT_ORDER: list[str] = [
    BASELINE,
    "native_query_eng_target",
    "native_query_native_target",
    "native_qt_func_descs",
    "native_qt_lang_hints",
    "native_qt_max_context",
    "native_query_func_descs",
]


def experiment_stem(locale: str, level: str) -> str:
    """('he', 'full') -> 'he_native_query_native_target'."""
    return f"{locale}_{LEVEL_TO_EXPERIMENT.get(level, level)}"


def split_stem(stem: str, locale: str) -> str:
    """'he_native_qt_func_descs' -> 'native_qt_func_descs' (strip the locale prefix).

    English baseline keeps its full name ('en_baseline'); stems without the prefix are returned as is.
    """
    if stem == BASELINE:
        return stem
    prefix = f"{locale}_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def level_of_stem(stem: str, locale: str) -> str | None:
    """'he_native_query_native_target' -> 'full'; None if the stem isn't a known experiment."""
    return EXPERIMENT_TO_LEVEL.get(split_stem(stem, locale))


def derived_stem(stem: str, locale: str, variant: str) -> str:
    """Name of a variant built on top of an experiment file.

    ('he_native_query_native_target', 'he', 'lang_desc') -> 'he_native_qt_lang_hints'
    (level 'full' + '_lang_desc' = 'full_lang_desc'). Falls back to '<stem>_<variant>'
    when the combination isn't a known level.
    """
    level = level_of_stem(stem, locale)
    if level is not None and f"{level}_{variant}" in LEVEL_TO_EXPERIMENT:
        return experiment_stem(locale, f"{level}_{variant}")
    return f"{stem}_{variant}"
