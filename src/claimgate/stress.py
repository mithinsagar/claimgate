"""QuantStress: controlled edits of human-verified supported claims.

Source claims are DelucionQA answer sentences that annotators marked Supported. Each edit has a known label:
  value_shift      number changed, unit kept                     -> should be flagged
  unit_swap        number kept, unit changed (same dimension)    -> should be flagged
  negation         "not" inserted or removed                     -> should be flagged
  condition_swap   before/after, above/below, front/rear, ...    -> should be flagged
  converted        value correctly converted to another unit      -> should pass
  original         unchanged claim                                -> should pass
Edits are rule-generated with a fixed seed, so the set is reproducible and needs no manual labelling. It is a stress test
of specific failure types, not a substitute for a human-labelled benchmark.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from claimgate import quantities
from claimgate.data import Example
from claimgate.text import context_for_nli

SWAP = {"foot * force_pound": ("newton * meter", "N·m"), "newton * meter": ("foot * force_pound", "ft-lb"), "psi": ("kilopascal", "kPa"),
        "kilopascal": ("psi", "psi"), "degF": ("degC", "°C"), "degC": ("degF", "°F"), "mile / hour": ("kilometer / hour", "km/h"),
        "kilometer / hour": ("mile / hour", "mph"), "mile": ("kilometer", "km"), "kilometer": ("mile", "miles"), "quart": ("liter", "L"),
        "liter": ("quart", "quarts"), "inch": ("centimeter", "cm"), "centimeter": ("inch", "in"), "millimeter": ("inch", "in"),
        "pound": ("kilogram", "kg"), "kilogram": ("pound", "lbs"), "foot": ("meter", "m"), "meter": ("foot", "ft"), "gallon": ("liter", "L"),
        "minute": ("second", "seconds"), "hour": ("minute", "minutes"), "second": ("minute", "minutes")}
ANTONYMS = [("before", "after"), ("above", "below"), ("increase", "decrease"), ("maximum", "minimum"), ("front", "rear"), ("left", "right"),
            ("forward", "rearward"), ("clockwise", "counterclockwise"), ("higher", "lower"), ("more", "less"), ("lock", "unlock"),
            ("enable", "disable"), ("press", "release"), ("inside", "outside"), ("upward", "downward"), ("first", "last"),
            ("hot", "cold"), ("open", "close"), ("raise", "lower"), ("faster", "slower")]
AUX = re.compile(r"\b(is|are|can|should|will|does|do|must|may|could|would|was|were)\b(?! not)", re.I)


@dataclass
class StressItem:
    kind: str
    claim: str
    source: str
    label: int          # 1 = should be flagged
    uid: str
    note: str = ""


def _fmt(v: float) -> str:
    if abs(v) < 1e-6:
        return "0"  # 32 °F is 0 °C, not 5.68e-14
    return f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"


def _quantity_edits(claim: str, evidence: str, rng: random.Random) -> list[tuple[str, str, int, str]]:
    out = []
    for v in quantities.check(claim, evidence):
        q = v.quantity
        if v.status != "supported" or not v.proof.startswith(f"'{q.text}' matches evidence") or q.unit is None or q.unit not in SWAP \
                or q.high is not None or q.text not in claim:
            continue  # only quantities the evidence states with a unit, so a correct conversion stays verifiable
        num = re.match(r"[\d.,/]+", q.text)
        if not num:
            continue
        unit_txt = q.text[num.end():].strip()
        factor = rng.choice([0.5, 1.5, 2.0, 3.0]) if q.value >= 4 else rng.choice([2.0, 3.0])
        shifted = q.value * factor
        out.append(("value_shift", claim.replace(q.text, f"{_fmt(shifted)} {unit_txt}".strip(), 1), 1, f"{q.text} -> {_fmt(shifted)} {unit_txt}"))
        new_unit, new_txt = SWAP[q.unit]
        out.append(("unit_swap", claim.replace(q.text, f"{num.group(0)} {new_txt}", 1), 1, f"{q.text} -> {num.group(0)} {new_txt}"))
        conv = quantities.UREG.Quantity(q.value, q.unit).to(new_unit).magnitude
        out.append(("converted", claim.replace(q.text, f"{_fmt(conv)} {new_txt}", 1), 0, f"{q.text} -> {_fmt(conv)} {new_txt}"))
        break
    return out


def _negation(claim: str) -> tuple[str, str] | None:
    if re.search(r"\bnot\b", claim):
        return re.sub(r"\s*\bnot\b", "", claim, count=1), "removed 'not'"
    if re.search(r"\bnever\b", claim, re.I):
        return re.sub(r"\bnever\b", "always", claim, count=1, flags=re.I), "never -> always"
    m = AUX.search(claim)
    if m:
        return claim[: m.end()] + " not" + claim[m.end():], f"inserted 'not' after '{m.group(0)}'"
    return None


def _condition(claim: str) -> tuple[str, str] | None:
    for a, b in ANTONYMS:
        for x, y in ((a, b), (b, a)):
            m = re.search(rf"\b{x}\b", claim, re.I)
            if m:
                rep = y.capitalize() if m.group(0)[0].isupper() else y
                return claim[: m.start()] + rep + claim[m.end():], f"{m.group(0)} -> {rep}"
    return None


def generate(examples: list[Example], seed: int = 0, per_kind_cap: int = 400) -> list[tuple[StressItem, Example]]:
    rng = random.Random(seed)
    items: dict[str, list] = {k: [] for k in ("original", "value_shift", "unit_swap", "converted", "negation", "condition_swap")}
    for ex in examples:
        evidence = context_for_nli(ex)
        for s, lab in zip(ex.sentences, ex.sentence_labels, strict=True):
            if lab != "Supported" or len(s.split()) < 5:
                continue
            items["original"].append(StressItem("original", s, s, 0, ex.uid))
            for kind, new, label, note in _quantity_edits(s, evidence, rng):
                items[kind].append(StressItem(kind, new, s, label, ex.uid, note))
            if (n := _negation(s)) and n[0] != s:
                items["negation"].append(StressItem("negation", n[0], s, 1, ex.uid, n[1]))
            if (c := _condition(s)) and c[0] != s:
                items["condition_swap"].append(StressItem("condition_swap", c[0], s, 1, ex.uid, c[1]))
    by_uid = {ex.uid: ex for ex in examples}
    out = []
    for lst in items.values():
        rng.shuffle(lst)
        out.extend((it, by_uid[it.uid]) for it in lst[:per_kind_cap])
    return out


def as_example(item: StressItem, ex: Example, i: int) -> Example:
    return Example(uid=f"stress{i}", split="stress", question=ex.question, context=ex.context, context_cased=ex.context_cased, answer=item.claim,
                   sentences=[item.claim], sentence_labels=["Supported" if item.label == 0 else "Neither"], label=item.label, coverage=ex.coverage,
                   meta=dict(ex.meta))


EDIT_KINDS = ("value_shift", "unit_swap", "negation", "condition_swap")


def augmentation(examples: list[Example], seed: int = 11, per_kind_cap: int = 400, exclude: set[str] | None = None) -> list[Example]:
    """Rule-made contradictions of supported train sentences, used as extra training rows (label: Conflicted).

    Built from the train split only, and sentences in `exclude` (those that also occur verbatim in dev or test answers)
    are skipped, so the dev/test QuantStress sets stay a held-out evaluation.
    """
    out = []
    exclude = exclude or set()
    edits = [x for x in generate(examples, seed=seed, per_kind_cap=per_kind_cap) if x[0].kind in EDIT_KINDS and x[0].source not in exclude]
    for i, (item, ex) in enumerate(edits):
        e = as_example(item, ex, i)
        e.uid, e.split, e.sentence_labels = f"aug{i}", "train_aug", ["Conflicted"]
        e.meta.update({"source_uid": ex.uid, "kind": item.kind})
        out.append(e)
    return out


def train_augmentation() -> list[Example]:
    from claimgate.data import load_delucionqa
    held = {s for split in ("dev", "test") for e in load_delucionqa(split) for s in e.sentences}
    return augmentation(load_delucionqa("train"), exclude=held)
