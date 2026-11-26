"""Detect whether a rejected function call used the wrong *language* for its free-text
parameters. Used by run_robustness_eval.py --error-lvl lang, so the "invalid language"
error is only sent when it is actually true.

A parameter is checked only if it is a free-text string: `type: string` (or an array of
strings) in the function schema, with no `enum`. Enum-constrained parameters are never
checked, since their values are fixed tokens rather than natural language.

For each checked parameter whose model value is not already an accepted ground-truth
value, two checks run in order:

1) Counterpart ground truth. Every entry exists in an English-target and a
   native-target variant (counterpart_experiment). If the model's value equals an
   accepted answer of the SAME entry's counterpart variant, the model gave the right
   answer in the wrong language -> mismatch. No language detection is needed.

2) lingua, only when (1) did not match. The benchmark's target language (English for
native_query_eng_target / native_query_func_descs, the locale's language for the
translated-target experiments, see experiment_names.has_translated_target) is the
prior. We detect with lingua
(restricted to English + the locale's language) the language of the accepted values
and of the model's value. It is a language mismatch only when
  * the accepted values are detected in the target language (so this parameter's
    answer really is in that language, not an untranslated name), and
  * the model's value is confidently detected in a language none of the accepted
    values are in.
Anything uncertain (short Latin tokens or codes, proper nouns spelled the same in both
languages, ground truth accepting both languages) counts as no mismatch, so the hint
errs on the side of not being sent.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from bfcl_eval.eval_checker.ast_eval.ast_checker import standardize_string

from multilingual_bfcl.experiment_names import has_translated_target

# Below this many letters a Latin-script value is a code / abbreviation ("CA", "GOOG",
# "A"), which lingua cannot place reliably, so it is never judged. Non-Latin scripts
# (Hebrew, CJK) identify the language from a single character, so they are exempt.
MIN_LATIN_LETTERS = 3
# lingua's minimum relative distance: with two candidate languages, a detection closer
# than this is returned as None (unsure) instead of a coin-flip guess.
MIN_RELATIVE_DISTANCE = 0.3


@lru_cache(maxsize=None)
def _detector(locale: str):
    """(lingua detector over {English, native}, native Language) for a locale; None for
    English or locales lingua does not know."""
    from lingua import IsoCode639_1, Language, LanguageDetectorBuilder

    try:
        native = Language.from_iso_code_639_1(getattr(IsoCode639_1, locale.split("-")[0].upper()))
    except AttributeError:
        return None
    if native == Language.ENGLISH:
        return None
    detector = (
        LanguageDetectorBuilder.from_languages(Language.ENGLISH, native)
        .with_minimum_relative_distance(MIN_RELATIVE_DISTANCE)
        .build()
    )
    return detector, native


def _norm_name(name: str) -> str:
    # The API may rewrite '.' in function names to '_'; compare on a common form.
    return name.replace(".", "_")


def _is_free_text(schema: dict) -> bool:
    if "enum" in schema:
        return False
    if schema.get("type") == "string":
        return True
    if schema.get("type") == "array":
        items = schema.get("items") or {}
        return items.get("type") == "string" and "enum" not in items
    return False


def _strings(value: Any) -> list[str]:
    """All non-empty strings in a value (a string, or a possibly nested list of them)."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _joined(value: Any) -> str:
    return ", ".join(_strings(value))


def _detect(detector, text: str):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return None
    if all(ord(ch) < 0x250 for ch in letters) and len(letters) < MIN_LATIN_LETTERS:
        return None
    return detector.detect_language_of(text)


def _norm_value(v: Any) -> Any:
    """Normalize a value exactly as BFCL's checker does before comparing it with the
    possible answers (standardize_string: drop spaces and ,./-_*^, lowercase, ' -> "),
    element-wise for lists, so "already accepted" here agrees with the AST checker."""
    if isinstance(v, list):
        return [_norm_value(x) for x in v]
    return standardize_string(v) if isinstance(v, str) else v


def param_language_mismatch(
    call_name: str,
    call_args: dict,
    functions: list[dict],
    ground_truth: list[dict],
    locale: str | None,
    localization_level: str | None,
    counterpart_ground_truth: list[dict] | None = None,
) -> list[dict]:
    """Free-text parameters of one model call whose language contradicts the ground
    truth. Returns [] when nothing is (confidently) in the wrong language. Each item:
    {param, value, method, value_lang, expected_langs}, where method is
    "counterpart_gt" (the value is an accepted answer of the other-language variant of
    this entry, see counterpart_experiment) or "lingua" (language detection)."""
    if not locale or not localization_level or not isinstance(call_args, dict):
        return []
    built = _detector(locale)
    if built is None:
        return []
    from lingua import Language

    detector, native = built
    target = native if has_translated_target(localization_level) else Language.ENGLISH
    other = Language.ENGLISH if target == native else native

    fn = next((f for f in functions if _norm_name(f["name"]) == _norm_name(call_name)), None)
    if fn is None:
        return []
    props = (fn.get("parameters") or {}).get("properties") or {}
    gt_calls = _calls_named(ground_truth, call_name)
    counterpart_calls = _calls_named(counterpart_ground_truth or [], call_name)

    def code(lang) -> str:
        return lang.iso_code_639_1.name.lower()

    mismatches = []
    for param, value in call_args.items():
        if not _is_free_text(props.get(param, {})):
            continue
        accepted_lists = [p[param] for p in gt_calls if p.get(param)]
        if not accepted_lists:
            continue
        norm = _norm_value(value)
        if any(_norm_value(a) == norm for acc in accepted_lists for a in acc):
            continue  # the model's value is an accepted answer: not a language problem

        # 1) Exact match against the other-language variant's accepted answers: the model
        #    gave the right answer, in the wrong language.
        if norm != "" and any(
            _norm_value(a) == norm for p in counterpart_calls for a in (p.get(param) or [])
        ):
            mismatches.append({
                "param": param, "value": value, "method": "counterpart_gt",
                "value_lang": [code(other)], "expected_langs": [code(target)],
            })
            continue

        # 2) Fall back to language detection. One detection per accepted alternative and
        #    one for the model's value; an array is detected as a single joined text, so
        #    short items get context.
        for accepted in accepted_lists:
            expected = {_detect(detector, _joined(a)) for a in accepted} - {None}
            model_lang = _detect(detector, _joined(value))
            if target not in expected or model_lang is None:
                continue
            if model_lang not in expected:
                mismatches.append({
                    "param": param, "value": value, "method": "lingua",
                    "value_lang": [code(model_lang)],
                    "expected_langs": sorted(code(lang) for lang in expected),
                })
                break
    return mismatches


def _calls_named(ground_truth: list[dict], call_name: str) -> list[dict]:
    """Parameter dicts of every ground-truth call to `call_name`."""
    return [
        params for gt in ground_truth for name, params in gt.items()
        if _norm_name(name) == _norm_name(call_name)
    ]


ENGLISH_TARGET_COUNTERPART = "native_query_native_target"
TRANSLATED_TARGET_COUNTERPART = "native_query_eng_target"


def counterpart_experiment(localization_level: str | None) -> str | None:
    """The variant whose ground truth holds the same answers in the other language:
    native_query_eng_target for translated-target variants (native_query_native_target,
    native_qt_*), native_query_native_target for English-target ones
    (native_query_eng_target, native_query_func_descs). None for the English baseline."""
    if not localization_level:
        return None
    if has_translated_target(localization_level):
        return TRANSLATED_TARGET_COUNTERPART
    return ENGLISH_TARGET_COUNTERPART
