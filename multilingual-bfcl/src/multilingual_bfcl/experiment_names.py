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
    full_full_full    native_qt_all_cues              he/he_native_qt_all_cues.jsonl
    func_desc         native_query_func_descs         he/he_native_query_func_descs.jsonl

Levels without a mapping fall back to ``<locale>_<level>``.

The levels above are the internal vocabulary used by scripts and the CLI (--level). The
``localization_level`` field stored in each entry holds the *experiment* name
(e.g. "native_qt_all_cues"); older files stored legacy values instead, see LEGACY_LEVEL_VALUES.
"""
from __future__ import annotations

BASELINE = "en_baseline"

LEVEL_TO_EXPERIMENT: dict[str, str] = {
    "query": "native_query_eng_target",
    "full": "native_query_native_target",
    "full_func_desc": "native_qt_func_descs",
    "full_lang_desc": "native_qt_lang_hints",
    "full_full_full": "native_qt_all_cues",
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
    "native_qt_all_cues",
    "native_query_func_descs",
]

# Values older files stored in the `localization_level` field -> experiment name.
# ("full" was also written into *_native_qt_lang_hints files, so prefer the file name when known.)
LEGACY_LEVEL_VALUES: dict[str, str] = {
    **LEVEL_TO_EXPERIMENT,
    "full_and_function_descriptions": "native_qt_func_descs",
    "function_descriptions": "native_query_func_descs",
    # Old experiment name, before the rename. Split so rename_translation_levels.py doesn't rewrite it.
    "native_qt_max" "_context": "native_qt_all_cues",
}

# Experiments whose ground-truth values are translated (query + target in the native language).
TRANSLATED_TARGET_EXPERIMENTS: frozenset[str] = frozenset({
    "native_query_native_target",
    "native_qt_func_descs",
    "native_qt_lang_hints",
    "native_qt_all_cues",
})


def experiment_of_level(level: str) -> str:
    """'full' -> 'native_query_native_target'; the value to store in `localization_level`."""
    return LEVEL_TO_EXPERIMENT.get(level, level)


def normalize_level_value(value: str) -> str:
    """Map a stored `localization_level` (new or legacy, optionally '_shuffled') to an experiment name."""
    shuffled = value.endswith("_shuffled")
    base = value[: -len("_shuffled")] if shuffled else value
    return LEGACY_LEVEL_VALUES.get(base, base) + ("_shuffled" if shuffled else "")


def has_translated_target(value: str) -> bool:
    """True if entries with this `localization_level` (new or legacy) have translated ground truth."""
    base = normalize_level_value(value)
    if base.endswith("_shuffled"):
        base = base[: -len("_shuffled")]
    return base in TRANSLATED_TARGET_EXPERIMENTS


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
