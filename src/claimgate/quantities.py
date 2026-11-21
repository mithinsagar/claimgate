"""Deterministic quantity and unit checking with proofs.

Every number in a claim (with its unit, if any) is compared with the numbers in the evidence. Values are converted with
Pint, so "25 ft-lb" and "33.9 N·m" are recognised as the same torque, while "25 N·m" against "25 ft-lb" is a mismatch.
Outputs are verdicts per quantity plus a human-readable proof; no learned model is involved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import pint

UREG = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

# surface forms seen in manuals -> Pint expressions (lower-case keys; Bosch tokenisation drops symbols like ° and ·)
UNIT_ALIASES = {
    "nm": "newton * meter", "n·m": "newton * meter", "n-m": "newton * meter", "n.m": "newton * meter", "newton-meters": "newton * meter",
    "ft-lb": "foot * force_pound", "ft-lbs": "foot * force_pound", "ft. lbs": "foot * force_pound", "lb-ft": "foot * force_pound",
    "lb-ft.": "foot * force_pound", "lbf-ft": "foot * force_pound", "foot-pounds": "foot * force_pound", "in-lb": "inch * force_pound",
    "in-lbs": "inch * force_pound", "psi": "psi", "kpa": "kilopascal", "bar": "bar", "mph": "mile / hour", "km/h": "kilometer / hour",
    "kph": "kilometer / hour", "°f": "degF", "°c": "degC", "f": "degF", "c": "degC", "degrees": "degF", "v": "volt", "volt": "volt",
    "volts": "volt", "amp": "ampere", "amps": "ampere", "a": "ampere", "w": "watt", "watt": "watt", "watts": "watt", "kw": "kilowatt",
    "hp": "horsepower", "mm": "millimeter", "cm": "centimeter", "m": "meter", "meters": "meter", "in": "inch", "in.": "inch", "inch": "inch",
    "inches": "inch", "ft": "foot", "ft.": "foot", "feet": "foot", "foot": "foot", "yd": "yard", "yards": "yard", "km": "kilometer",
    "mile": "mile", "miles": "mile", "mi": "mile", "lb": "pound", "lbs": "pound", "lbs.": "pound", "pound": "pound", "pounds": "pound",
    "kg": "kilogram", "g": "gram", "oz": "ounce", "l": "liter", "liter": "liter", "liters": "liter", "litre": "liter", "litres": "liter",
    "ml": "milliliter", "qt": "quart", "qts": "quart", "quart": "quart", "quarts": "quart", "gal": "gallon", "gallon": "gallon",
    "gallons": "gallon", "s": "second", "sec": "second", "secs": "second", "second": "second", "seconds": "second", "min": "minute",
    "mins": "minute", "minute": "minute", "minutes": "minute", "hr": "hour", "hrs": "hour", "hour": "hour", "hours": "hour",
    "day": "day", "days": "day", "week": "week", "weeks": "week", "month": "month", "months": "month", "year": "year", "years": "year",
    "%": "percent", "percent": "percent", "rpm": "revolution / minute", "mpg": "mile / gallon", "db": None, "hz": "hertz", "khz": "kilohertz",
    "mhz": "megahertz", "ghz": "gigahertz",
}
WORD_NUMBERS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
                "half": 0.5, "once": 1, "twice": 2}

_UNIT_RE = "|".join(sorted((re.escape(k) for k in UNIT_ALIASES), key=len, reverse=True))
_NUM = r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?|\.\d+|\d+/\d+)(?![\w/])"
QTY_RE = re.compile(_NUM + r"(?:\s*(?:-|to|–)\s*(\d+(?:\.\d+)?))?\s*(" + _UNIT_RE + r")?(?![a-z])", re.I)
WORD_RE = re.compile(r"\b(" + "|".join(WORD_NUMBERS) + r")\b\s*(" + _UNIT_RE + r")?(?![a-z])", re.I)


@dataclass(frozen=True)
class Quantity:
    value: float
    high: float | None
    unit: str | None          # Pint expression or None for a bare number
    text: str

    def magnitude_in(self, unit: str) -> float | None:
        if self.unit is None:
            return None
        try:
            return UREG.Quantity(self.value, self.unit).to(unit).magnitude
        except Exception:
            return None


@lru_cache(maxsize=4096)
def _dimensionality(unit: str):
    try:
        return UREG.parse_expression(unit).dimensionality
    except Exception:
        return None


def _to_float(s: str) -> float:
    s = s.replace(",", "")
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b) if float(b) else float("nan")
    return float(s)


def extract(text: str) -> list[Quantity]:
    out = []
    for m in QTY_RE.finditer(text):
        num, high, unit = m.group(1), m.group(2), m.group(3)
        u = UNIT_ALIASES.get(unit.lower()) if unit else None
        out.append(Quantity(_to_float(num), float(high) if high else None, u, m.group(0).strip()))
    for m in WORD_RE.finditer(text):
        word, unit = m.group(1).lower(), m.group(2)
        u = UNIT_ALIASES.get(unit.lower()) if unit else None
        if u is None and word in {"one", "once", "half"}:
            continue  # "one of the", "once the engine" are rarely quantities without a unit
        out.append(Quantity(float(WORD_NUMBERS[word]), None, u, m.group(0).strip()))
    return out


def _close(a: float, b: float, rel: float = 0.02) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-9)


@dataclass
class QuantityVerdict:
    quantity: Quantity
    status: str            # supported | mismatch | unit_mismatch | absent
    proof: str


def check(claim: str, evidence: str) -> list[QuantityVerdict]:
    """Verdict for every quantity in the claim against the evidence text."""
    ev = extract(evidence)
    out = []
    for q in extract(claim):
        same_value = [e for e in ev if _close(e.value, q.value) or (e.high is not None and e.value <= q.value <= e.high)]
        if q.unit is None:
            status = "supported" if same_value else "absent"
            proof = f"'{q.text}' appears in the evidence" if same_value else f"the number {q.value:g} does not appear in the evidence"
            out.append(QuantityVerdict(q, status, proof))
            continue
        dim = _dimensionality(q.unit)
        comparable = [e for e in ev if e.unit and _dimensionality(e.unit) == dim]
        match = next((e for e in comparable if _close(e.magnitude_in(q.unit) or float("nan"), q.value)
                      or (e.high is not None and (e.magnitude_in(q.unit) or 0) <= q.value <= UREG.Quantity(e.high, e.unit).to(q.unit).magnitude)), None)
        if match:
            out.append(QuantityVerdict(q, "supported", f"'{q.text}' equals evidence '{match.text}'"))
        elif any(_close(e.value, q.value) for e in comparable):
            e = next(e for e in comparable if _close(e.value, q.value))
            conv = e.magnitude_in(q.unit)
            out.append(QuantityVerdict(q, "unit_mismatch", f"claim '{q.text}' but evidence '{e.text}' = {conv:.3g} {q.unit}; same number, different unit"))
        elif comparable:
            e = comparable[0]
            conv = e.magnitude_in(q.unit)
            diff = abs(conv - q.value) / max(abs(conv), 1e-9) if conv is not None else float("nan")
            out.append(QuantityVerdict(q, "mismatch", f"claim '{q.text}' but evidence '{e.text}' = {conv:.3g} {q.unit} ({diff:.0%} difference)"))
        elif same_value:
            out.append(QuantityVerdict(q, "supported", f"value {q.value:g} appears in the evidence without a comparable unit"))
        else:
            out.append(QuantityVerdict(q, "absent", f"no quantity like '{q.text}' in the evidence"))
    return out


def features(claim: str, evidence: str) -> dict:
    v = check(claim, evidence)
    return {"q_n": len(v), "q_supported": sum(x.status == "supported" for x in v), "q_mismatch": sum(x.status == "mismatch" for x in v),
            "q_unit_mismatch": sum(x.status == "unit_mismatch" for x in v), "q_absent": sum(x.status == "absent" for x in v)}


def canonicalize(claim: str, evidence: str) -> tuple[str, list[str]]:
    """Rewrite claim quantities that equal an evidence quantity (after unit conversion) into the evidence's wording.

    NLI models judge "34 N·m" against "25 ft-lb" as unsupported although the values agree. Rewriting the claim to the
    evidence's own surface form lets the NLI model judge only the non-numeric content. Returns the new claim and a list of
    the rewrites, which become part of the proof.
    """
    out, notes = claim, []
    ev = [e for e in extract(evidence) if e.unit]
    for q in extract(claim):
        if q.unit is None:
            continue
        dim = _dimensionality(q.unit)
        for e in ev:
            if _dimensionality(e.unit) != dim or e.unit == q.unit:
                continue
            conv = e.magnitude_in(q.unit)
            if conv is not None and _close(conv, q.value) and q.text in out:
                out = out.replace(q.text, e.text, 1)
                notes.append(f"'{q.text}' = '{e.text}' after unit conversion")
                break
    return out, notes
