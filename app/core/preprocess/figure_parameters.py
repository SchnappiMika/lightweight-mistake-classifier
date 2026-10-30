from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

_HSPACE = r"[^\S\r\n]*"
_SIGN = r"[+\-−－＋﹣±]?"
_DECIMAL = r"(?:\d+(?:\.\d+)?|\.\d+)"
_EXPONENT = r"(?:\^" + _HSPACE + r"(?:\{[+\-−－＋﹣]?\d+\}|[+\-−－＋﹣]?\d+)|[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
_SCIENTIFIC = (r"(?:[eE][+\-−－＋﹣]?\d+|" + _HSPACE +
               r"[×·⋅*]" + _HSPACE + r"10" + _EXPONENT + r")?")
_NUMBER = _SIGN + _DECIMAL + _SCIENTIFIC
_LETTER = r"A-Za-zΑ-Ωα-ωϑϕϖϱϵℓ"
_SUFFIX = r"(?:_[A-Za-z0-9Α-Ωα-ω]+|_\{[A-Za-z0-9Α-Ωα-ω]+\}|[₀₁₂₃₄₅₆₇₈₉₊₋ₐₑₒₓₔₕₖₗₘₙₚₛₜᵢⱼᵣᵤᵥ⁰¹²³⁴⁵⁶⁷⁸⁹⃗′″])"
_SYMBOL = r"[" + _LETTER + r"][" + _LETTER + r"0-9]{0,11}(?:" + _SUFFIX + r")*"
_LEFT_BOUNDARY = r"(?<![" + _LETTER + r"0-9_.₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹])"
_ASSIGNMENT = re.compile(_LEFT_BOUNDARY + r"(?P<symbol>" + _SYMBOL + r")" + _HSPACE +
                         r"[=＝≈]" + _HSPACE + r"(?P<value>" + _NUMBER + r")")
_QUANTITY = re.compile(_LEFT_BOUNDARY + r"(?P<value>" + _NUMBER + r")")
_PREFIX = r"(?:da|[fpnumcdhkMGTμµ])?"
_BASE_UNIT = r"(?:Ω|Ω|ohm|Pa|Hz|Wb|[mgsAVWJNCFHTK]|mol|cd|rad|sr|[Llℓ])"
_UNIT_EXPONENT = r"(?:\^" + _HSPACE + r"(?:\{[+\-−－]?\d+\}|[+\-−－]?\d+)|[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)?"
_UNIT_ATOM = r"(?:°C|°F|℃|°|%|" + _PREFIX + _BASE_UNIT + r")" + _UNIT_EXPONENT
_UNIT_FACTOR = _UNIT_ATOM + r"(?![" + _LETTER + r"0-9₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹])(?!" + _HSPACE + r"[=＝≈])"
_UNIT = re.compile(_HSPACE + r"(?P<unit>" + _UNIT_FACTOR +
                   r"(?:(?:" + _HSPACE + r"[·⋅*/]" + _HSPACE + r"|[^\S\r\n]+)" + _UNIT_FACTOR + r")*)")
_BROKEN_NUMERIC = re.compile(r"[0-9A-Za-zΑ-Ωα-ω_.⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻×⋅*+\-−－＋﹣]")


def _extract_match(text: str, match, source: str, require_unit: bool) -> Dict[str, Any]:
    start, end = match.span()
    symbol = match.groupdict().get("symbol") or ""
    value = match.group("value")
    tail = text[end:]
    if tail.startswith((".", ",")) and len(tail) > 1 and tail[1].isdigit():
        return {}
    unit_match = _UNIT.match(text, end)
    unit = unit_match.group("unit").rstrip() if unit_match else ""
    if unit:
        end = unit_match.start("unit") + len(unit)
    else:
        if require_unit or (tail and _BROKEN_NUMERIC.match(tail)):
            return {}
        remainder = re.match(_HSPACE + r"([eE]|[×⋅*])", tail)
        if remainder:
            return {}
    return {"symbol": symbol, "value": value, "unit": unit, "raw": text[start:end],
            "source": source, "start": start, "end": end}


def extract_parameters(text: str, source: str = "text") -> List[Dict[str, Any]]:
    """Return literal parameter spans; no OCR repair, unit conversion or inference."""
    if not isinstance(text, str) or not text:
        return []
    results = []
    occupied = []
    for match in _ASSIGNMENT.finditer(text):
        item = _extract_match(text, match, source, False)
        if item:
            results.append(item)
            occupied.append((item["start"], item["end"]))
        else:
            delimiter = re.search(r"[,，;；\r\n]", text[match.end():])
            invalid_end = match.end() + delimiter.start() if delimiter else len(text)
            occupied.append((match.start(), invalid_end))
    for match in _QUANTITY.finditer(text):
        if match.start() >= 2 and text[match.start() - 1] in ",." and text[match.start() - 2].isdigit():
            continue
        if any(start <= match.start() < end for start, end in occupied):
            continue
        item = _extract_match(text, match, source, True)
        if item and not any(item["start"] < end and item["end"] > start for start, end in occupied):
            results.append(item)
    return sorted(results, key=lambda item: (item["start"], item["end"]))


_UNIT_TOKENS = {
    "kg": "质量", "g": "质量", "mg": "质量", "N": "力", "mN": "力", "kN": "力",
    "A": "电流", "mA": "电流", "μA": "电流", "µA": "电流",
    "V": "电压", "mV": "电压", "kV": "电压",
    "Ω": "电阻", "Ω": "电阻", "ohm": "电阻", "kΩ": "电阻", "MΩ": "电阻",
    "°": "角度", "rad": "角度", "Hz": "频率", "kHz": "频率", "MHz": "频率",
    "F": "电容", "μF": "电容", "µF": "电容", "nF": "电容", "pF": "电容",
    "H": "电感", "mH": "电感", "T": "磁感应强度", "mT": "磁感应强度",
    "Pa": "压强", "kPa": "压强", "MPa": "压强", "J": "能量", "W": "功率",
    "m": "长度", "cm": "长度", "mm": "长度", "s": "时间", "ms": "时间",
    "m/s": "速度", "m/s²": "加速度", "m/s^2": "加速度",
    "m·s⁻¹": "速度", "m·s⁻²": "加速度",
}


def parameter_tokens(parameters: Iterable[Dict[str, Any]]) -> List[str]:
    """Derive only explicit unit semantics, keeping numeric strings out of tokens."""
    tokens = []
    for item in parameters:
        unit = item.get("unit", "")
        if not isinstance(unit, str):
            continue
        token = _UNIT_TOKENS.get(re.sub(r"\s+", "", unit))
        if token and token not in tokens:
            tokens.append(token)
        if re.sub(r"\s+", "", unit) in ("N·m", "N⋅m", "N*m", "Nm") and item.get("symbol") in ("τ", "M"):
            if "力矩" not in tokens:
                tokens.append("力矩")
    return tokens
