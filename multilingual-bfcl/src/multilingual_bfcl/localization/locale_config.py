"""
Per-language locale configuration.

A Locale captures everything that differs between languages when we localize
a BFCL test case: the BCP-47 language tag, text direction, date/number format
conventions, and the Anthropic model best suited for translation work.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Locale:
    # BCP-47 language tag (used as file suffix, e.g. "he", "zh-CN")
    code: str
    # Human-readable English name
    name: str
    # Right-to-left script
    rtl: bool = False
    # ISO date format string (strftime-style)
    date_format: str = "%Y-%m-%d"
    # Decimal separator used in this locale
    decimal_separator: str = "."
    # Thousands separator
    thousands_separator: str = ","
    # Any extra system-prompt notes for the model (e.g. calendar system hints)
    model_hints: list[str] = field(default_factory=list)


# Registry of supported locales.
# Add new languages here; the rest of the pipeline picks them up automatically.
SUPPORTED_LOCALES: dict[str, Locale] = {
    "en": Locale(
        code="en",
        name="English",
    ),
    "he": Locale(
        code="he",
        name="Hebrew",
        rtl=True,
        date_format="%d/%m/%Y",
        model_hints=["Use Modern Israeli Hebrew (not Biblical). Prefer right-to-left punctuation conventions."],
    ),
    "ar": Locale(
        code="ar",
        name="Arabic",
        rtl=True,
        date_format="%d/%m/%Y",
        decimal_separator=".",
        thousands_separator=",",
        model_hints=["Use Modern Standard Arabic (MSA). Avoid dialect-specific expressions."],
    ),
    "zh-CN": Locale(
        code="zh-CN",
        name="Chinese (Simplified)",
        date_format="%Y年%m月%d日",
        thousands_separator=",",
        model_hints=["Use Simplified Chinese characters. Use Chinese punctuation conventions (，。？！)."],
    ),
    "fr": Locale(
        code="fr",
        name="French",
        date_format="%d/%m/%Y",
        decimal_separator=",",
        thousands_separator=" ",
    ),
    "de": Locale(
        code="de",
        name="German",
        date_format="%d.%m.%Y",
        decimal_separator=",",
        thousands_separator=".",
    ),
    "es": Locale(
        code="es",
        name="Spanish",
        date_format="%d/%m/%Y",
        decimal_separator=",",
        thousands_separator=".",
    ),
}


def get_locale(code: str) -> Locale:
    if code not in SUPPORTED_LOCALES:
        raise ValueError(
            f"Unsupported locale '{code}'. "
            f"Available: {', '.join(SUPPORTED_LOCALES)}"
        )
    return SUPPORTED_LOCALES[code]
