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
    "kph": "kilometer / hour", "°f": "degF", "°c": "degC", "f": "degF", "c": "degC", "degrees": "degF", "degree": "degF",
    "degrees f": "degF", "degrees fahrenheit": "degF", "fahrenheit": "degF", "° f": "degF", "degrees c": "degC", "degrees celsius": "degC",
    "celsius": "degC", "° c": "degC", "degrees centigrade": "degC", "v": "volt", "volt": "volt",
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
# no letter or "." / ":" before the number ("A4", "7:00"); a unit may be attached after it ("250F", "10mm", "12V")
_NUM = r"(?<![\w.:/])(\d+/\d+|\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?|\.\d+)(?![\d/:]|\.\d)"
# the unit must not continue into a compound unit ("kg/m³", "m^2"): those are not handled and are skipped as unitless
QTY_RE = re.compile(_NUM + r"(?:\s*(?:-|to|–)\s*(?:a\s+)?(\d+/\d+|\d+(?:\.\d+)?))?\s*-?\s*(" + _UNIT_RE + r")?(?![a-z/^²³])", re.I)
WORD_RE = re.compile(r"\b(" + "|".join(WORD_NUMBERS) + r")\b\s*-?\s*(" + _UNIT_RE + r")?(?![a-z])", re.I)


@dataclass(frozen=True)
class Quantity:
    value: float
    high: float | None
    unit: str | None          # Pint expression or None for a bare number
    text: str
    context: frozenset = frozenset()   # topic words within a few words of the number: what the number is about

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


_CTX_WORDS = 6
_LOWER_NOT_UNITS = {"a", "c", "f", "v", "w"}


_CLAUSE = re.compile(r"[,;()\n]|[.!?](?:\s|$)|\s(?:and|or|but|while|whereas)\s|\d")  # not ":" ("Maximum voltage: 12 V")


def _local_context(text: str, start: int, end: int) -> frozenset:
    """Topic words near the number, within its own clause (stops at , ; ( ) sentence ends, conjunctions and other numbers).

    "N years old" adds the marker "<age>", so an age is never compared with a duration.
    """
    left = _CLAUSE.split(text[:start])[-1] if start else ""
    right = _CLAUSE.split(text[end:])[0]
    before = _WORD.findall(left.lower())[-_CTX_WORDS:]
    after = _WORD.findall(right.lower())[:_CTX_WORDS]
    words = {w.rstrip("s") for w in before + after if w not in _STOP and w not in UNIT_ALIASES and w not in WORD_NUMBERS}
    if after[:1] == ["old"] or after[:2] == ["year", "old"]:
        words.add("<age>")
    return frozenset(words)


def extract(text: str) -> list[Quantity]:
    out = []
    for m in QTY_RE.finditer(text):
        num, high, unit = m.group(1), m.group(2), m.group(3)
        if unit and unit in _LOWER_NOT_UNITS and re.search(r"\s", text[m.end(2) if high else m.end(1):m.start(3)]):
            unit = None  # "7 a.m.", "2 a day": a lower-case a, c, f, v or w after a space is a word; attached ("10f", "12v") it is a unit
        u = UNIT_ALIASES.get(unit.lower()) if unit else None
        span_text = m.group(0) if unit else (m.group(1) + (f"-{high}" if high else ""))
        out.append(Quantity(_to_float(num), _to_float(high) if high else None, u, span_text.strip(), _local_context(text, m.start(), m.end())))
    for m in WORD_RE.finditer(text):
        word, unit = m.group(1).lower(), m.group(2)
        if unit and len(unit) <= 2:
            unit = None  # "once a month", "twice in a day": short words after a number word are not units
        u = UNIT_ALIASES.get(unit.lower()) if unit else None
        if u is None and word in {"one", "once", "half"}:
            continue  # "one of the", "once the engine" are rarely quantities without a unit
        text_ = m.group(0).strip() if unit else m.group(1)
        end = m.end() if unit else m.end(1)
        out.append(Quantity(float(WORD_NUMBERS[word]), None, u, text_, _local_context(text, m.start(), end)))
    return out


def _close(a: float, b: float, rel: float = 0.02) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-9)


@dataclass
class QuantityVerdict:
    quantity: Quantity
    status: str            # supported | mismatch | unit_mismatch | absent
    proof: str
    against: str = ""      # the evidence quantity it was compared with, if any
    overlap: int = 0       # words shared by the claim's and the evidence quantity's local context


_EV_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z][a-z\-']+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "be", "can", "should", "will", "your", "you",
         "it", "this", "that", "at", "by", "from", "up", "not", "do", "does", "every", "after", "before", "about", "than", "more", "less"}


def _topic_words(text: str) -> set[str]:
    return {w.rstrip("s") for w in _WORD.findall(text.lower()) if w not in _STOP and w not in UNIT_ALIASES}


def _related_evidence(claim: str, evidence: str) -> str:
    """Evidence sentences about the same thing as the claim: they share at least min(2, n) topic words with it."""
    cw = _topic_words(claim)
    need = min(2, len(cw))
    sents = [x for x in _EV_SENT.split(evidence) if x.strip()]
    return " ".join(x for x in sents if len(cw & _topic_words(x)) >= need)


def _unit_text(q: Quantity) -> str:
    m = re.match(r"[\d.,/]+(?:\s*(?:-|to|–)\s*[\d.]+)?\s*", q.text)
    return q.text[m.end():] if m else (q.unit or "")


def _matches(e: Quantity, q: Quantity) -> bool:
    """Evidence quantity e states the claim's value: equal after conversion, or one range contains the other's value."""
    lo = e.magnitude_in(q.unit)
    if lo is None:
        return False
    hi = UREG.Quantity(e.high, e.unit).to(q.unit).magnitude if e.high is not None else lo
    if _close(lo, q.value) or lo <= q.value <= hi:
        return True
    return q.high is not None and q.value <= lo <= q.high  # claim range "20-30 ft" covers evidence "30 feet"


def check(claim: str, evidence: str) -> list[QuantityVerdict]:
    """Verdict for every quantity in the claim against the evidence text.

    A value anywhere in the evidence can support a claim, but only a value in an evidence sentence about the same thing
    can contradict it: "rotate tires every 5,000 miles" is not contradicted by "check the torque after 25 miles".
    """
    ev = extract(evidence)
    related = extract(_related_evidence(claim, evidence))
    out = []
    for q in extract(claim):
        same_value = [e for e in ev if _close(e.value, q.value) or (e.high is not None and e.value <= q.value <= e.high)
                      or (q.high is not None and q.value <= e.value <= q.high)]
        if q.unit is None:
            status = "supported" if same_value else "absent"
            proof = f"'{q.text}' appears in the evidence" if same_value else f"the number {q.value:g} does not appear in the evidence"
            out.append(QuantityVerdict(q, status, proof))
            continue
        dim = _dimensionality(q.unit)
        anywhere = [e for e in ev if e.unit and _dimensionality(e.unit) == dim]
        # a number can only contradict another number about the same thing: same dimension, a related sentence, and
        # at least one shared word next to the two numbers ("married for 10 years" vs "seven years in prison" is not)
        comparable = sorted((e for e in related if e.unit and _dimensionality(e.unit) == dim and e.context & q.context
                             and ("<age>" in e.context) == ("<age>" in q.context)),
                            key=lambda e: -len(e.context & q.context))

        match = next((e for e in anywhere if _matches(e, q)), None)
        if match:
            how = "" if match.unit == q.unit else " after unit conversion"
            out.append(QuantityVerdict(q, "supported", f"'{q.text}' matches evidence '{match.text}'{how}"))
        elif any(_close(e.value, q.value) for e in comparable):
            e = next(e for e in comparable if _close(e.value, q.value))
            conv = e.magnitude_in(q.unit)
            out.append(QuantityVerdict(q, "unit_mismatch", f"claim '{q.text}' but evidence '{e.text}', which is {conv:.3g} {_unit_text(q)}: "
                                                           "same number, different unit", e.text, len(e.context & q.context)))
        elif comparable:
            e = comparable[0]
            conv = e.magnitude_in(q.unit)
            diff = abs(conv - q.value) / max(abs(conv), 1e-9) if conv is not None else float("nan")
            said = f"'{e.text}'" if e.unit == q.unit else f"'{e.text}', which is {conv:.3g} {_unit_text(q)}"
            out.append(QuantityVerdict(q, "mismatch", f"claim '{q.text}' but evidence says {said} ({diff:.0%} difference)", e.text,
                                       len(e.context & q.context)))
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
